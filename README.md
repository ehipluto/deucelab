# DeuceLab

[![Quality checks](https://github.com/ehipluto/deucelab/actions/workflows/tests.yml/badge.svg)](https://github.com/ehipluto/deucelab/actions/workflows/tests.yml)

DeuceLab is an open-source Python desktop application for real-time probabilistic analysis of table tennis set totals. It runs locally and takes rally results and market prices entered by the user.

## Features

- Track rallies, serving order, score corrections, and set transitions.
- Forecast Over/Under outcomes on half-point lines from 14.5 to 22.5.
- Compare fair odds, implied market probabilities, and expected value. Market prices expire when the relevant match state changes.
- Review forecasts and qualified signals, and manually record the price and stake of an execution.
- Export completed sets as JSONL for replay. Run synthetic backtests and compare forecasting and signal policies from the command line.

## Setup

Use Python 3.12 with Tkinter. The desktop uses CustomTkinter and native Tk charts.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
deucelab
```

From a source checkout, `python main.py` also launches the application after installing its runtime dependencies. Use `deucelab --smoke` to check imports and scoring without opening a window.

## Try a set

1. Choose the initial server and enter the winner of each rally.
2. Select a totals line and apply the Over and/or Under price available at the current score. Enter both prices to see proportional margin removal.
3. Watch **Rally analysis** update as you enter points. Switch between **Probability**, **Score**, and **Details**, or open a separate analysis window. Use **Session history** to review forecasts and manually record an execution.
4. The final rally settles the forecasts. Use **Export replay** to save the completed set locally, then **Next Set** to clear its totals prices and continue.

## Architecture and methods

| Modules | Responsibility |
| --- | --- |
| `rules`, `logic` | Scoring, service, match state, and forecast orchestration |
| `models`, `totals` | Point probabilities and memoized final-total distributions |
| `markets`, `ledger` | Quote validation, market metrics, prediction identity, and settlement |
| `replay`, `backtest`, study commands | Replay validation, simulation, and model and policy comparison |
| `gui`, `history`, `visualization` | Desktop controls, session history, and charts |

The desktop defaults to the legacy heuristic forecast. Choose `deucelab --model direct` to hold the neutral-strength estimate fixed instead of applying the legacy momentum adjustment. The forecast study also compares a Bayesian alternative. The totals recurrence groups all results of 23 points or more in one bucket, which is above every supported line.

For probability `p` and decimal price `d`, fair odds are `1 / p` when `p > 0`, raw implied probability is `1 / d`, and expected value per unit is `p * d - 1`. The two-sided market baseline normalizes the raw implied probabilities to sum to one.

## Reproducible checks

```text
python -m pip install ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m coverage run -m unittest discover
python -m coverage report
python tools/check_public_tree.py
deucelab --smoke
deucelab-backtest --sets 250 --seed 42 --after-points 8
deucelab-forecast-study --sets 100 --seed 42 --after-points 8
```

Use `deucelab-backtest --data path/to/sets.jsonl --after-points 8` to replay completed sets with market prices captured at the declared rally index. Backtests distinguish all estimates, filtered signals, and the market-favorite baseline. Simulated returns use one unit per selection. Use `--help` on the strategy study and rally validation commands for their input requirements.

Each JSONL row contains `first_server`, `rallies` (a list of rally winners, encoded as 1 or 2), `line`, decimal `over` and `under` prices, and `after_points`. A row can include a two-price `prematch` array in player order. For chronological calibration and testing, include `match_id`, `set_number`, `quote_at`, and `settled_at`, with timezone-aware ISO 8601 timestamps. The two splits must have disjoint match IDs, and every calibration result must have settled before the first test quote.

## Limitations

The heuristic parameters have yet to be calibrated on real match data. Synthetic runs test the implementation and assumptions. They provide no estimate of real-world performance. External replay timestamps and price provenance depend on the supplied data. Multiple sets from one match can be statistically dependent. Session history lives in memory, and replay export contains completed set data rather than a persistent execution journal.

## License

[MIT](LICENSE)
