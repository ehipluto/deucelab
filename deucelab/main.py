import argparse
import math

from deucelab import __version__
from deucelab.gui import App
from deucelab.logic import FORECAST_MODELS, BettingLogic


def main() -> None:
    parser = argparse.ArgumentParser(description="Live table tennis analysis and headless checks.")
    parser.add_argument("--model", choices=FORECAST_MODELS, default="legacy")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--smoke", action="store_true", help="verify imports and scoring without opening a window"
    )
    args = parser.parse_args()

    logic = BettingLogic(args.model)
    if args.smoke:
        logic.record_point(1)
        logic.set_market(18.5, over=2.0)
        estimate, _, _ = logic.get_suggestions()
        if (
            estimate is None
            or estimate.get("type") == "info"
            or not math.isclose(sum(logic._run_simulation()), 1.0)
        ):
            raise SystemExit("DeuceLab smoke check failed")
        print(f"DeuceLab {__version__} smoke check passed ({args.model}).")
        return
    App(logic).mainloop()


if __name__ == "__main__":
    main()
