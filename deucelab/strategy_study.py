import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

from deucelab.backtest import synthetic_sets
from deucelab.constants import DEFAULT_CONFIG
from deucelab.forecast_study import confidence_interval, snapshot_probabilities, stress_sets
from deucelab.replay import SetReplay, read_replay_rows

FEATURES = ("market", "direct_prior", "posterior")
SCENARIOS = ("original", "shifting_form", "oracle_market")
TRAIN_SEED = 163703
TEST_SEEDS = (541873, 984197)
HIGH_CONFIDENCE = 0.70
MIN_EV = DEFAULT_CONFIG["min_acceptable_ev_base"]
MIN_PRICE = DEFAULT_CONFIG["min_estimated_odds_base"]
Weights = tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    over_won: bool
    prices: tuple[float, float]
    v2: float
    features: tuple[float, float, float]


def observations(sets: Iterable[SetReplay], after_points: int) -> list[Observation]:
    result = []
    for replay in sets:
        if len(replay.rallies) <= after_points:
            continue
        forecast = snapshot_probabilities(replay, after_points)
        result.append(
            Observation(
                len(replay.rallies) > replay.line,
                (replay.over, replay.under),
                forecast["v2"],
                tuple(forecast[name] for name in FEATURES),
            )
        )
    return result


def fit_blend(training: list[Observation]) -> Weights:
    if not training:
        raise ValueError("no eligible calibration sets")
    candidates = (
        (market / 10, direct / 10, (10 - market - direct) / 10)
        for market in range(10, -1, -1)
        for direct in range(10 - market, -1, -1)
    )
    return min(
        candidates,
        key=lambda weights: mean((blend(row, weights) - row.over_won) ** 2 for row in training),
    )


def blend(row: Observation, weights: Weights) -> float:
    return sum(weight * probability for weight, probability in zip(weights, row.features))


@dataclass
class Performance:
    wins: int = 0
    returns: list[float] = field(default_factory=list)
    market_probabilities: list[float] = field(default_factory=list)
    forecast_probabilities: list[float] = field(default_factory=list)
    market_residuals: list[float] = field(default_factory=list)
    price_total: float = 0.0

    def record(self, row: Observation, side: int, probability: float) -> None:
        won = row.over_won if side == 0 else not row.over_won
        price = row.prices[side]
        self.wins += won
        self.returns.append(price - 1 if won else -1)
        market_probability = on_side(row.features[0], side)
        self.market_probabilities.append(market_probability)
        self.market_residuals.append(won - market_probability)
        self.forecast_probabilities.append(probability)
        self.price_total += price

    def summary(self, eligible: int, *, independent: bool = True) -> dict:
        count = len(self.returns)
        return {
            "signals": count,
            "coverage": count / eligible,
            "wins": self.wins,
            "win_rate": self.wins / count if count else None,
            "mean_price": self.price_total / count if count else None,
            "mean_market_probability_on_same_picks": (
                mean(self.market_probabilities) if count else None
            ),
            "mean_forecast_probability": mean(self.forecast_probabilities) if count else None,
            "roi": mean(self.returns) if count else None,
            "roi_ci95": confidence_interval(self.returns) if independent else None,
            "win_rate_minus_market_ci95": confidence_interval(self.market_residuals)
            if independent
            else None,
        }


def on_side(over_probability: float, side: int) -> float:
    return over_probability if side == 0 else 1 - over_probability


def select_value(row: Observation, probability: float) -> int | None:
    side = max((0, 1), key=lambda i: on_side(probability, i) * row.prices[i])
    p = on_side(probability, side)
    if (
        DEFAULT_CONFIG["min_prob_suggestion_base"] <= p <= DEFAULT_CONFIG["max_prob_filter_base"]
        and row.prices[side] >= MIN_PRICE
        and p * row.prices[side] - 1 >= MIN_EV
    ):
        return side
    return None


def evaluate_policies(
    test: list[Observation], weights: Weights, *, independent: bool = True
) -> dict:
    if not test:
        raise ValueError("no eligible test sets")

    policies = {
        name: Performance()
        for name in (
            "v2_value",
            "direct_value",
            "posterior_value",
            "blend_value",
            "market_favorite_70",
            "blend_favorite_70",
            "consensus_value_70",
        )
    }
    errors = {name: [] for name in ("v2", *FEATURES, "blend")}
    for row in test:
        forecasts = dict(zip(FEATURES, row.features))
        forecasts["v2"] = row.v2
        forecasts["blend"] = blend(row, weights)
        for name, probability in forecasts.items():
            errors[name].append((probability - row.over_won) ** 2)

        for name, model in (
            ("v2_value", "v2"),
            ("direct_value", "direct_prior"),
            ("posterior_value", "posterior"),
            ("blend_value", "blend"),
        ):
            probability = forecasts[model]
            side = select_value(row, probability)
            if side is not None:
                policies[name].record(row, side, on_side(probability, side))

        for model in ("market", "blend"):
            probability = forecasts[model]
            side = int(probability < 0.5)
            if on_side(probability, side) >= HIGH_CONFIDENCE:
                policies[f"{model}_favorite_70"].record(row, side, on_side(probability, side))

        for side in (0, 1):
            conservative = min(
                on_side(forecasts[name], side)
                for name in (
                    "blend",
                    "direct_prior",
                    "posterior",
                )
            )
            price = row.prices[side]
            if (
                HIGH_CONFIDENCE <= conservative <= DEFAULT_CONFIG["max_prob_filter_base"]
                and price >= MIN_PRICE
                and conservative * price - 1 >= MIN_EV
            ):
                policies["consensus_value_70"].record(row, side, conservative)
                break

    return {
        "eligible": len(test),
        "weights": dict(zip(FEATURES, weights)),
        "brier": {name: mean(values) for name, values in errors.items()},
        "blend_brier_delta_vs_market_ci95": confidence_interval(
            [
                blend_error - market_error
                for blend_error, market_error in zip(
                    errors["blend"],
                    errors["market"],
                )
            ]
        )
        if independent
        else None,
        "policies": {
            name: policy.summary(len(test), independent=independent)
            for name, policy in policies.items()
        },
    }


def sample(scenario: str, count: int, seed: int, after_points: int) -> list[Observation]:
    source = (
        synthetic_sets(count, seed, after_points)
        if scenario == "original"
        else stress_sets(count, seed, after_points, scenario)
    )
    return observations(source, after_points)


def read_historical(
    path: Path, after_points: int
) -> tuple[list[SetReplay], set[str], list[tuple[datetime, datetime]]]:
    replays = []
    matches = set()
    times = []
    for replay, metadata in read_replay_rows(path, after_points):
        if metadata is None:
            raise ValueError(f"{path}: chronological splits require match IDs and timestamps")
        replays.append(replay)
        matches.add(metadata.match_id)
        times.append((metadata.quote_at, metadata.settled_at))
    if not replays:
        raise ValueError(f"{path}: no replay rows")
    return replays, matches, times


def historical_split(
    train_path: Path,
    test_path: Path,
    after_points: int,
) -> tuple[list[SetReplay], list[SetReplay]]:
    train, train_matches, train_times = read_historical(train_path, after_points)
    test, test_matches, test_times = read_historical(test_path, after_points)
    if train_matches & test_matches:
        raise ValueError("the splits share match IDs")
    if max(settled for _, settled in train_times) >= min(quoted for quoted, _ in test_times):
        raise ValueError("all calibration sets must settle before the first test quote")
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a market-aware forecast on one sample, then score fixed policies on another."
    )
    parser.add_argument(
        "--sets",
        type=int,
        default=2500,
        help="calibration sets per scenario (also the default test size)",
    )
    parser.add_argument("--test-sets", type=int, help="sets per scenario and test seed")
    parser.add_argument("--after-points", type=int, default=8)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=TEST_SEEDS)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--test-data", type=Path)
    args = parser.parse_args()
    test_sets = args.sets if args.test_sets is None else args.test_sets
    if min(args.sets, test_sets) < 1 or args.after_points < 0:
        parser.error("--sets must be positive and --after-points non-negative")
    if TRAIN_SEED in args.test_seeds or len(set(args.test_seeds)) != len(args.test_seeds):
        parser.error("test seeds must be unique and exclude the calibration seed")
    if bool(args.train_data) != bool(args.test_data):
        parser.error("provide both historical files or neither")
    if args.train_data:
        if args.train_data.resolve() == args.test_data.resolve():
            parser.error("use separate chronological calibration and test files")
        try:
            train_replays, test_replays = historical_split(
                args.train_data,
                args.test_data,
                args.after_points,
            )
            training = observations(train_replays, args.after_points)
            test = observations(test_replays, args.after_points)
            result = evaluate_policies(test, fit_blend(training), independent=False)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    else:
        training = {
            name: sample(name, args.sets, TRAIN_SEED, args.after_points) for name in SCENARIOS
        }
        result = {}
        for name in SCENARIOS:
            weights = fit_blend(
                [row for other in SCENARIOS if other != name for row in training[other]]
            )
            test = [
                row
                for seed in args.test_seeds
                for row in sample(name, test_sets, seed, args.after_points)
            ]
            result[name] = evaluate_policies(test, weights)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
