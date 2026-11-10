import argparse
import json
import math
import random
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev

from deucelab import constants as c
from deucelab.backtest import synthetic_sets
from deucelab.constants import STANDARD_OU_LINES
from deucelab.logic import SERVE_EDGE, BettingLogic
from deucelab.logic import direct_point_probability as point_probability
from deucelab.markets import no_margin
from deucelab.replay import SetReplay, read_replays
from deucelab.totals import CAP, totals_step

PRIOR_STRENGTH = 24
STRENGTH_GRID = tuple(index / 20 for index in range(1, 20))
HOLDOUT_SEED = 731203


@lru_cache(maxsize=24_000)
def total_distribution(
    p1: int,
    p2: int,
    first_server: int,
    strength: float,
    serve_edge: float = SERVE_EDGE,
) -> tuple[float, ...]:
    return totals_step(
        p1,
        p2,
        first_server,
        lambda server: point_probability(strength, server, serve_edge),
        lambda a, b: total_distribution(a, b, first_server, strength, serve_edge),
    )


def posterior(prior_mean: float, points: list[dict]) -> tuple[float, ...]:
    alpha = prior_mean * PRIOR_STRENGTH
    beta = (1 - prior_mean) * PRIOR_STRENGTH
    logs = []
    for strength in STRENGTH_GRID:
        log_weight = (alpha - 1) * math.log(strength) + (beta - 1) * math.log1p(-strength)
        for point in points:
            probability = point_probability(strength, point["server"])
            likelihood = probability if point["winner"] == 1 else 1 - probability
            log_weight += math.log(likelihood)
        logs.append(log_weight)

    peak = max(logs)
    weights = [math.exp(value - peak) for value in logs]
    normalizer = sum(weights)
    return tuple(weight / normalizer for weight in weights)


def posterior_distribution(model: BettingLogic) -> tuple[float, ...]:
    weights = posterior(model.p1_neutral_point_win_prob_current, model.point_history)
    distributions = (
        total_distribution(
            model.score_p1, model.score_p2, model.initial_server_current_set, strength
        )
        for strength in STRENGTH_GRID
    )
    mixed = [0.0] * (CAP + 1)
    for weight, distribution in zip(weights, distributions):
        for total, probability in enumerate(distribution):
            mixed[total] += weight * probability
    return tuple(mixed)


def confidence_interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    center = mean(values)
    radius = 1.96 * stdev(values) / math.sqrt(len(values))
    return [center - radius, center + radius]


def snapshot_probabilities(replay: SetReplay, after_points: int) -> dict[str, float]:
    if type(after_points) is not int or not 0 <= after_points < len(replay.rallies):
        raise ValueError("forecast snapshot must precede set completion")
    model = BettingLogic()
    model.start_new_set(replay.first_server)
    if replay.prematch is not None:
        model.set_prematch_odds(*replay.prematch)
    for winner in replay.rallies[:after_points]:
        model.record_point(winner)

    split = math.floor(replay.line) + 1
    direct = total_distribution(
        model.score_p1,
        model.score_p2,
        replay.first_server,
        model.p1_neutral_point_win_prob_current,
    )
    return {
        "v2": sum(model._run_simulation()[split:]),
        "direct_prior": sum(direct[split:]),
        "posterior": sum(posterior_distribution(model)[split:]),
        "market": no_margin(replay.over, replay.under)[0],
    }


def evaluate(
    replays: Iterable[SetReplay],
    after_points: int,
    *,
    independent: bool = True,
) -> dict:
    if type(after_points) is not int or after_points < 0:
        raise ValueError("after_points must be non-negative")
    errors = {name: [] for name in ("v2", "direct_prior", "posterior", "market")}
    returns = {name: [] for name in ("v2", "direct_prior", "posterior")}
    wins = dict.fromkeys(returns, 0)
    for replay in replays:
        if len(replay.rallies) <= after_points:
            continue
        actual_over = len(replay.rallies) > replay.line
        probabilities = snapshot_probabilities(replay, after_points)
        for name, probability in probabilities.items():
            errors[name].append((probability - actual_over) ** 2)
            if name == "market":
                continue
            candidates = (
                (probability, replay.over, actual_over),
                (1 - probability, replay.under, not actual_over),
            )
            p, odds, won = max(candidates, key=lambda entry: entry[0] * entry[1] - 1)
            if (
                c.DEFAULT_CONFIG["min_prob_suggestion_base"]
                <= p
                <= c.DEFAULT_CONFIG["max_prob_filter_base"]
                and odds >= c.DEFAULT_CONFIG["min_estimated_odds_base"]
                and p * odds - 1 >= c.DEFAULT_CONFIG["min_acceptable_ev_base"]
            ):
                wins[name] += won
                returns[name].append(odds - 1 if won else -1)

    summary = {}
    for name, losses in errors.items():
        if not losses:
            return {"eligible": 0, "models": {}}
        summary[name] = {
            "brier": mean(losses),
            "brier_delta_vs_v2_ci95": confidence_interval(
                [a - b for a, b in zip(losses, errors["v2"])]
            )
            if independent
            else None,
        }
        if name in returns:
            profits = returns[name]
            summary[name].update(
                signals=len(profits),
                wins=wins[name],
                win_rate=wins[name] / len(profits) if profits else None,
                roi=mean(profits) if profits else None,
                roi_ci95=confidence_interval(profits) if independent else None,
                coverage=len(profits) / len(losses),
            )
    return {"eligible": len(errors["v2"]), "models": summary}


def stress_sets(count: int, seed: int, after_points: int, scenario: str) -> Iterator[SetReplay]:
    rng = random.Random(seed)
    for _ in range(count):
        first_server = rng.choice((1, 2))
        line = rng.choice(STANDARD_OU_LINES)
        strength = rng.uniform(0.30, 0.70)
        serve_edge = rng.uniform(0.01, 0.15)
        quote_noise = 0.12 if scenario == "stale_prior" else 0.06
        quoted_strength = min(0.78, max(0.22, strength + rng.gauss(0, quote_noise)))
        asymmetric_edge = rng.uniform(-0.07, 0.07) if scenario == "asymmetric_serve" else 0.0
        implied_match = min(0.9, max(0.1, 0.5 + 2 * (quoted_strength - 0.5)))
        prematch = (1 / (1.04 * implied_match), 1 / (1.04 * (1 - implied_match)))
        p1 = p2 = 0
        rallies = []
        snapshot = (0, 0)
        while not BettingLogic.is_finished(p1, p2):
            server = BettingLogic.determine_next_server(p1, p2, first_server)
            form = 0.0
            if scenario == "shifting_form" and len(rallies) >= 3:
                form = 0.045 * (2 * rallies[-3:].count(1) / 3 - 1)
            p1_won = rng.random() < point_probability(
                strength + form + asymmetric_edge, server, serve_edge
            )
            rallies.append(1 if p1_won else 2)
            p1 += p1_won
            p2 += not p1_won
            if len(rallies) == after_points:
                snapshot = (p1, p2)

        market_strength = strength if scenario == "oracle_market" else quoted_strength
        market_serve = serve_edge if scenario == "oracle_market" else 0.08
        totals = total_distribution(*snapshot, first_server, market_strength, market_serve)
        over = min(0.95, max(0.05, sum(totals[math.floor(line) + 1 :])))
        yield SetReplay(
            first_server,
            tuple(rallies),
            line,
            1 / (1.04 * over),
            1 / (1.04 * (1 - over)),
            prematch,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare point models on paired, out-of-sample set replays."
    )
    parser.add_argument("--sets", type=int, default=3000, help="sets per synthetic scenario")
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    parser.add_argument("--after-points", type=int, default=8)
    parser.add_argument(
        "--stress-suite",
        action="store_true",
        help="test unseen asymmetric service and noisier pre-match prices",
    )
    parser.add_argument("--data", type=Path, help="historical JSONL with synchronized market odds")
    args = parser.parse_args()
    if args.sets < 1 or args.after_points < 0:
        parser.error("--sets must be positive and --after-points non-negative")

    if args.data is not None:
        try:
            result = evaluate(
                read_replays(args.data, args.after_points), args.after_points, independent=False
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        result = {
            "seed": args.seed,
            "after_points": args.after_points,
            "legacy": evaluate(
                synthetic_sets(args.sets, args.seed, args.after_points), args.after_points
            ),
        }
        scenarios = (
            ("asymmetric_serve", "stale_prior")
            if args.stress_suite
            else ("shifting_form", "oracle_market")
        )
        for scenario in scenarios:
            replays = stress_sets(args.sets, args.seed, args.after_points, scenario)
            result[scenario] = evaluate(replays, args.after_points)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
