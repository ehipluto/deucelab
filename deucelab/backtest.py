import argparse
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from deucelab.logic import FORECAST_MODELS, BettingLogic
from deucelab.markets import no_margin
from deucelab.replay import SetReplay, read_replays


@dataclass
class Results:
    wins: int = 0
    losses: int = 0
    net_units: float = 0.0
    squared_error: float = 0.0

    def record(self, won: bool, price: float, probability: float) -> None:
        self.wins += won
        self.losses += not won
        self.net_units += price - 1 if won else -1
        self.squared_error += (probability - won) ** 2

    def summary(self) -> dict:
        count = self.wins + self.losses
        return {
            "count": count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / count if count else None,
            "return_per_unit": self.net_units / count if count else None,
            "brier_score": self.squared_error / count if count else None,
        }


def evaluate(
    replays: Iterable[SetReplay],
    after_points: int,
    forecast_model: str = "legacy",
) -> dict:
    if type(after_points) is not int or after_points < 0:
        raise ValueError("after_points must be non-negative")
    if forecast_model not in FORECAST_MODELS:
        raise ValueError(f"unknown model: {forecast_model}")
    all_estimates = Results()
    signals = Results()
    market_baseline = Results()
    eligible = 0
    total = 0
    for replay in replays:
        total += 1
        if len(replay.rallies) <= after_points:
            continue
        eligible += 1
        model = BettingLogic(forecast_model)
        model.start_new_set(replay.first_server)
        if replay.prematch is not None:
            model.set_prematch_odds(*replay.prematch)
        for winner in replay.rallies[:after_points]:
            model.record_point(winner)
        model.set_market(replay.line, replay.over, replay.under)
        estimate, _, _ = model.get_suggestions(record=False, include_elaborated=False)
        if estimate is None or estimate.get("type") == "info":
            raise RuntimeError("the selected market did not return an estimate")
        actual_over = len(replay.rallies) > replay.line
        won = actual_over == (estimate["side"] == "Over")
        price = replay.over if estimate["side"] == "Over" else replay.under
        all_estimates.record(won, price, estimate["prob"])
        if estimate["meets_filters"]:
            signals.record(won, price, estimate["prob"])
        market_side_over = replay.over <= replay.under
        market_price = min(replay.over, replay.under)
        market_probability = no_margin(replay.over, replay.under)[not market_side_over]
        market_baseline.record(actual_over == market_side_over, market_price, market_probability)
    return {
        "sets": total,
        "eligible_sets": eligible,
        "after_points": after_points,
        "all_estimates": all_estimates.summary(),
        "filtered_signals": {
            **signals.summary(),
            "coverage": (signals.wins + signals.losses) / eligible if eligible else None,
        },
        "market_favorite": market_baseline.summary(),
    }


def synthetic_sets(count: int, seed: int, after_points: int) -> Iterable[SetReplay]:
    rng = random.Random(seed)
    line = 18.5
    for _ in range(count):
        first_server = rng.choice((1, 2))
        strength = rng.uniform(0.34, 0.66)
        serve_edge = rng.uniform(0.03, 0.15)
        market_strength = min(0.75, max(0.25, strength + rng.gauss(0, 0.035)))
        prematch_p1 = min(0.9, max(0.1, 0.5 + 2 * (market_strength - 0.5)))
        prematch = (1 / (1.04 * prematch_p1), 1 / (1.04 * (1 - prematch_p1)))
        p1 = p2 = 0
        rallies = []
        snapshot = (0, 0)
        while not BettingLogic.is_finished(p1, p2):
            server = BettingLogic.determine_next_server(p1, p2, first_server)
            p1_won = rng.random() < strength + (serve_edge if server == 1 else -serve_edge)
            rallies.append(1 if p1_won else 2)
            p1 += p1_won
            p2 += not p1_won
            if len(rallies) == after_points:
                snapshot = (p1, p2)
        market_over = synthetic_market_probability(*snapshot, line, first_server, market_strength)
        market_over = min(0.95, max(0.05, market_over))
        yield SetReplay(
            first_server,
            tuple(rallies),
            line,
            1 / (1.04 * market_over),
            1 / (1.04 * (1 - market_over)),
            prematch,
        )


def synthetic_market_probability(
    p1: int,
    p2: int,
    line: float,
    first_server: int,
    strength: float,
) -> float:

    @cache
    def over(a: int, b: int) -> float:
        if a + b > line:
            return 1.0
        if BettingLogic.is_finished(a, b):
            return 0.0
        server = BettingLogic.determine_next_server(a, b, first_server)
        point_probability = strength + (0.08 if server == 1 else -0.08)
        return point_probability * over(a + 1, b) + (1 - point_probability) * over(a, b + 1)

    return over(p1, p2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible point-by-point evaluation of DeuceLab.")
    parser.add_argument(
        "--data", type=Path, help="JSONL of completed sets with historical snapshot odds"
    )
    parser.add_argument("--sets", type=int, default=5000, help="number of synthetic sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=FORECAST_MODELS, default="legacy")
    parser.add_argument("--after-points", type=int, default=8)
    args = parser.parse_args()
    if args.sets < 1 or args.after_points < 0:
        parser.error("--sets must be positive and --after-points non-negative")
    replays = (
        read_replays(args.data, args.after_points)
        if args.data
        else synthetic_sets(args.sets, args.seed, args.after_points)
    )
    try:
        summary = evaluate(replays, args.after_points, args.model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summary["model"] = args.model
    summary["source"] = str(args.data) if args.data else "synthetic"
    if not args.data:
        summary["seed"] = args.seed
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
