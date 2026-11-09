import io
import unittest
from pathlib import Path
from unittest.mock import patch

from backtest import SetReplay
from forecast_study import (
    STRENGTH_GRID,
    evaluate,
    posterior,
    posterior_distribution,
    snapshot_probabilities,
)
from logic import BettingLogic
from rally_validation import evaluate_rallies


class ForecastStudyTests(unittest.TestCase):
    def test_balanced_prior_is_symmetric(self):
        weights = posterior(0.5, [])
        self.assertAlmostEqual(sum(weights), 1.0)
        for left, right in zip(weights, reversed(weights)):
            self.assertAlmostEqual(left, right)

    def test_winning_on_receive_updates_more_than_winning_on_serve(self):
        serving = posterior(0.5, [{"winner": 1, "server": 1}])
        receiving = posterior(0.5, [{"winner": 1, "server": 2}])
        self.assertGreater(
            sum(weight * strength for weight, strength in zip(receiving, STRENGTH_GRID)),
            sum(weight * strength for weight, strength in zip(serving, STRENGTH_GRID)),
        )

    def test_distribution_has_unit_mass_and_stays_above_impossible_totals(self):
        model = BettingLogic()
        for winner in (1, 2) * 10:
            model.record_point(winner)
        distribution = posterior_distribution(model)
        self.assertAlmostEqual(sum(distribution), 1.0)
        self.assertEqual(sum(distribution[:22]), 0.0)

    def test_direct_desktop_engine_matches_research_model(self):
        replay = SetReplay(2, (1, 2) * 10 + (1, 1), 18.5, 1.9, 1.95, (1.5, 2.8))
        model = BettingLogic("direct")
        model.start_new_set(replay.first_server)
        model.set_prematch_odds(*replay.prematch)
        for winner in replay.rallies[:8]:
            model.record_point(winner)
        self.assertAlmostEqual(
            sum(model._run_simulation()[19:]),
            snapshot_probabilities(replay, 8)["direct_prior"],
        )

    def test_synthetic_rally_sample_runs_without_market_prices(self):
        sample = "source_line,first_server,rallies\nartificial-1,1,1212121212121212121211\n"
        with patch.object(Path, "open", return_value=io.StringIO(sample)):
            result = evaluate_rallies(Path("synthetic.csv"), 8, 18.5)
        self.assertEqual(result["eligible_sets"], 1)
        self.assertEqual(set(result["brier"]), {"legacy", "direct"})
        for score in result["brier"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_same_snapshot_and_quote_produce_same_forecast_for_two_outcomes(self):
        opening = (1, 2) * 4
        over = SetReplay(1, opening + (1, 2) * 6 + (1, 1), 18.5, 1.9, 1.9)
        under = SetReplay(1, opening + (1,) * 7, 18.5, 1.9, 1.9)
        over_error = evaluate([over], 8)["models"]["posterior"]["brier"]
        under_error = evaluate([under], 8)["models"]["posterior"]["brier"]
        self.assertAlmostEqual(over_error**0.5 + under_error**0.5, 1.0)


if __name__ == "__main__":
    unittest.main()
