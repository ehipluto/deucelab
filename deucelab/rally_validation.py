import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from deucelab.constants import STANDARD_OU_LINES
from deucelab.logic import FORECAST_MODELS, BettingLogic
from deucelab.replay import validate_rallies


def evaluate_rallies(path: Path, after_points: int, line: float) -> dict:
    if after_points < 0 or line not in STANDARD_OU_LINES:
        raise ValueError("expected a nonnegative checkpoint and a supported O/U line")

    errors = {name: [] for name in FORECAST_MODELS}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            identity = row["source_line"]
            if not identity or identity in seen:
                raise ValueError(f"missing or duplicate sequence identifier: {identity!r}")
            seen.add(identity)
            try:
                server = int(row["first_server"])
                raw_rallies = row["rallies"]
                if server not in (1, 2) or not raw_rallies or set(raw_rallies) - {"1", "2"}:
                    raise ValueError("invalid first server or rally sequence")
                rallies = tuple(map(int, raw_rallies))
                validate_rallies(rallies)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{identity}: {error}") from error
            if len(rallies) <= after_points:
                continue
            for name in FORECAST_MODELS:
                model = BettingLogic(name)
                model.start_new_set(server)
                for winner in rallies[:after_points]:
                    model.record_point(winner)
                probability = sum(model._run_simulation()[int(line) + 1 :])
                errors[name].append((probability - (len(rallies) > line)) ** 2)

    return {
        "sets": len(seen),
        "eligible_sets": len(errors["legacy"]),
        "after_points": after_points,
        "line": line,
        "brier": {name: mean(losses) if losses else None for name, losses in errors.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score set-total probabilities on observed rallies without market prices.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--after-points", type=int, default=8)
    parser.add_argument("--line", type=float, default=18.5)
    args = parser.parse_args()
    try:
        report = evaluate_rallies(args.data, args.after_points, args.line)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
