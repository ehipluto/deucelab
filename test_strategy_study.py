import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from strategy_study import Observation, evaluate_policies, fit_blend, historical_split, select_value


class StrategyStudyTests(unittest.TestCase):
    def test_training_prefers_the_best_calibrated_feature(self):
        rows = [
            Observation(True, (1.9, 1.9), 0.5, (0.8, 0.5, 0.6)),
            Observation(False, (1.9, 1.9), 0.5, (0.2, 0.5, 0.4)),
        ]
        self.assertEqual(fit_blend(rows), (1, 0, 0))

    def test_high_confidence_can_still_fail_value_gate(self):
        row = Observation(True, (1.15, 7.0), 0.8, (0.8, 0.8, 0.8))
        result = evaluate_policies([row], (1, 0, 0))["policies"]
        self.assertEqual(result["blend_favorite_70"]["signals"], 1)
        self.assertEqual(result["consensus_value_70"]["signals"], 0)
        self.assertIsNone(select_value(row, 0.8))

    def test_consensus_abstains_when_point_models_disagree(self):
        row = Observation(True, (1.85, 1.95), 0.73, (0.52, 0.85, 0.60))
        result = evaluate_policies([row], (0, 1, 0))["policies"]
        self.assertEqual(result["blend_value"]["signals"], 1)
        self.assertEqual(result["consensus_value_70"]["signals"], 0)

    def test_market_comparison_uses_exact_same_selected_outcomes(self):
        row = Observation(False, (2.15, 1.70), 0.25, (0.44, 0.21, 0.25))
        result = evaluate_policies([row], (0, 1, 0))["policies"]["consensus_value_70"]
        self.assertEqual(result["win_rate"], 1)
        self.assertAlmostEqual(result["mean_market_probability_on_same_picks"], 0.56)
        self.assertAlmostEqual(result["roi"], 0.70)

    def test_historical_split_rejects_unsettled_training_labels(self):
        row = {
            "first_server": 1,
            "rallies": [1] * 11,
            "line": 18.5,
            "over": 1.90,
            "under": 1.95,
            "after_points": 8,
            "set_number": 1,
        }
        train = dict(
            row, match_id="a", quote_at="2026-01-01T10:00:00Z", settled_at="2026-01-01T10:05:00Z"
        )
        test = dict(
            row, match_id="b", quote_at="2026-01-01T10:04:00Z", settled_at="2026-01-01T10:09:00Z"
        )
        with patch.object(
            Path,
            "open",
            side_effect=[io.StringIO(json.dumps(train)), io.StringIO(json.dumps(test))],
        ):
            with self.assertRaisesRegex(ValueError, "settle before"):
                historical_split(Path("train.jsonl"), Path("test.jsonl"), 8)

        test["quote_at"] = "2026-01-01T10:06:00Z"
        with patch.object(
            Path,
            "open",
            side_effect=[io.StringIO(json.dumps(train)), io.StringIO(json.dumps(test))],
        ):
            train_sets, test_sets = historical_split(Path("train.jsonl"), Path("test.jsonl"), 8)
        self.assertEqual((len(train_sets), len(test_sets)), (1, 1))


if __name__ == "__main__":
    unittest.main()
