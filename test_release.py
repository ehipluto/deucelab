import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from deucelab.backtest import synthetic_sets
from deucelab.forecast_study import evaluate, snapshot_probabilities
from deucelab.ledger import Settlement
from deucelab.logic import BettingLogic
from deucelab.markets import expected_value, fair_odds, no_margin, valid_price
from deucelab.replay import encode_replay, read_replays, write_replay
from deucelab.rules import is_finished, is_score, next_server
from deucelab.strategy_study import historical_split


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.model = BettingLogic()

    def play(self, rallies):
        for winner in rallies:
            self.assertTrue(self.model.record_point(winner))

    def test_standard_and_extended_service_order(self):
        self.assertEqual([next_server(score, 0, 1) for score in range(8)], [1, 1, 2, 2] * 2)
        self.assertTrue(is_score(201, 200))
        self.assertFalse(is_finished(201, 200))
        self.assertTrue(is_finished(202, 200))
        self.assertEqual(next_server(200, 200, 2), 2)
        self.assertEqual(next_server(201, 200, 2), 1)
        self.play((1, 2) * 40)
        self.assertEqual(self.model._run_simulation()[-1], 1.0)

    def test_invalid_scoring_leaves_state_unchanged(self):
        for p1, p2 in ((-1, 0), (True, 1), (12, 8), (2.5, 2)):
            self.assertFalse(is_score(p1, p2))
            with self.assertRaises(ValueError):
                next_server(p1, p2, 1)
        for winner in (True, 1.0, 0, 3):
            with self.assertRaises(ValueError):
                self.model.record_point(winner)
        with self.assertRaises(ValueError):
            self.model.update_current_set_state(2, 0, 2, 1)
        with self.assertRaises(ValueError):
            self.model.start_new_set(0)
        self.assertEqual((self.model.score_p1, self.model.score_p2), (0, 0))

    def test_decimal_prices_and_no_margin(self):
        for invalid in (math.nan, math.inf, 1.0, 0.0, True, "2.0", 10**400):
            self.assertFalse(valid_price(invalid))
        first, second = no_margin(1.8, 2.1)
        self.assertAlmostEqual(first + second, 1)
        self.assertAlmostEqual(first, 2.1 / 3.9)
        self.assertIsNone(fair_odds(0.0))
        self.assertEqual(fair_odds(0.5), 2.0)
        self.assertAlmostEqual(expected_value(0.6, 2), 0.2)
        self.model.set_market(18.5, over=2.5)
        prediction, _, _ = self.model.get_suggestions()
        self.assertIsNone(prediction["market_probability"])

    def test_quote_expiry_and_cache_on_rally_and_correction(self):
        self.model.set_prematch_odds(1.5, 2.5)
        prior = self.model.p1_neutral_point_win_prob_current
        self.model.set_live_odds(3.0, 1.2)
        self.model.set_market(18.5, under=4.0)
        live = self.model._run_simulation()
        self.assertNotEqual(self.model.p1_neutral_point_win_prob_current, prior)
        self.play((1,))
        self.assertIsNone(self.model.total_quote)
        self.assertIsNone(self.model.live_quote)
        self.assertEqual(self.model.p1_neutral_point_win_prob_current, prior)
        self.assertNotEqual(self.model._run_simulation(), live)
        self.model.set_market(18.5, under=4.0)
        self.model.set_live_odds(3.0, 1.2)
        self.model.revise_points(1, remove_player=1)
        self.assertIsNone(self.model.total_quote)
        self.assertIsNone(self.model.live_quote)
        self.assertEqual(self.model.p1_neutral_point_win_prob_current, prior)

    def test_history_distinguishes_snapshot_and_quote_from_refresh(self):
        self.model.set_market(18.5, under=4.0)
        self.model.get_suggestions(record=False)
        self.assertEqual(self.model.history.entries, [])
        self.model.get_suggestions()
        self.model.get_suggestions()
        self.model.set_market(18.5, under=4.0)
        self.model.get_suggestions()
        self.assertEqual(len(self.model.history.entries), 1)
        self.model.set_market(18.5, over=2.0, under=4.0)
        self.model.get_suggestions()
        self.play((1,))
        self.model.set_market(18.5, under=4.0)
        self.model.get_suggestions()
        entries = self.model.history.entries
        self.assertEqual(len(entries), 3)
        self.assertEqual([entry.prediction.point_number for entry in entries], [0, 0, 1])
        self.assertEqual(len({entry.prediction.quote_id for entry in entries}), 3)

    def test_execution_requires_a_signal_and_a_manual_record(self):
        self.model.get_suggestions()
        entry = self.model.history.entries[0]
        self.assertIsNone(entry.signal)
        with self.assertRaises(ValueError):
            self.model.history.record_execution(0, 2.0, 2.0)
        self.model.set_market(17.5, over=3.5)
        self.model.get_suggestions()
        entry = self.model.history.entries[1]
        self.assertIsNotNone(entry.signal)
        self.assertIsNone(entry.execution)
        self.assertIsNone(self.model.history.realized_roi())
        self.model.history.record_execution(1, 2.5, 3.0)
        with self.assertRaises(ValueError):
            self.model.history.record_execution(1, 2.5, 3.0)
        self.play((1, 2) * 10 + (1, 1))
        self.assertEqual(entry.settlement, Settlement.WON)
        self.assertAlmostEqual(self.model.history.realized_roi(), 1.5)
        self.model.revise_points(1, remove_player=1)
        self.assertEqual(entry.settlement, Settlement.INVALIDATED)
        self.assertIsNone(self.model.history.realized_roi())
        self.assertTrue(self.model.record_bet_outcome(1, "null"))
        self.assertEqual(self.model.history.realized_roi(), 0.0)

    def test_signal_qualification_does_not_duplicate_a_prediction(self):
        self.model.get_suggestions()
        history = self.model.history
        (entry,) = history.entries
        qualifying_estimate = entry.estimate.copy()
        qualifying_estimate["meets_filters"] = True
        history.add(entry.prediction, qualifying_estimate)
        self.assertEqual(history.entries, [entry])
        self.assertIsNotNone(entry.signal)

    def test_elaboration_uses_current_score_and_market(self):
        self.play((1, 2) * 4)
        self.model.set_market(18.5, over=2.0)
        _, elaborated, _ = self.model.get_suggestions()
        self.assertEqual(elaborated["market_odds"], 2.0)
        self.play((1,))
        self.model.set_market(19.5, under=3.0)
        _, elaborated, _ = self.model.get_suggestions()
        self.assertEqual(elaborated["line"], 19.5)
        self.assertEqual(elaborated["market_odds"], 3.0)
        self.assertEqual(len(self.model.history.entries), 2)

    def test_configuration_is_atomic(self):
        original = self.model.config.copy()
        with self.assertRaises(ValueError):
            self.model.update_config(
                {"min_prob_suggestion_base": 0.99, "max_prob_filter_base": 0.50}
            )
        self.assertEqual(self.model.config, original)
        with self.assertRaises(ValueError):
            self.model.update_config({"adaptive_elab_base_points": 0})
        self.assertTrue(self.model.update_config({"momentum_impact_scale": 0.0}))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "match_id": "synthetic-match",
            "set_number": 1,
            "quote_at": "2026-01-01T10:01:00Z",
            "settled_at": "2026-01-01T10:05:00Z",
            "after_points": 8,
            "first_server": 1,
            "rallies": [1, 2] * 10 + [1, 1],
            "line": 18.5,
            "over": 2.0,
            "under": 2.1,
        }

    def test_export_import_and_overwrite_protection(self):
        row = encode_replay(self.payload, 8)
        with TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "replay.jsonl"
            write_replay(path, row)
            (replay,) = read_replays(path, 8)
            self.assertEqual(replay.rallies, tuple(self.payload["rallies"]))
            with self.assertRaises(FileExistsError):
                write_replay(path, "bad replacement")
            self.assertEqual(path.read_text(encoding="utf-8"), row)

    def test_duplicate_and_noncausal_replay_rows_are_rejected(self):
        row = encode_replay(self.payload, 8)
        with TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "replay.jsonl"
            path.write_text(row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate replay"):
                list(read_replays(path, 8))
            path.write_text(
                json.dumps(self.payload | {"quote_at": self.payload["settled_at"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must precede settlement"):
                list(read_replays(path, 8))

    def test_calibration_and_test_cannot_share_matches(self):
        row = encode_replay(self.payload, 8)
        later = self.payload | {
            "set_number": 2,
            "quote_at": "2026-01-01T10:06:00Z",
            "settled_at": "2026-01-01T10:10:00Z",
        }
        with TemporaryDirectory(dir=".") as directory:
            train, test = Path(directory) / "train.jsonl", Path(directory) / "test.jsonl"
            write_replay(train, row)
            write_replay(test, encode_replay(later, 8))
            with self.assertRaisesRegex(ValueError, "share match IDs"):
                historical_split(train, test, 8)

    def test_synthetic_regression_matches_original_models(self):
        replay = next(iter(synthetic_sets(1, 87, 8)))
        forecast = snapshot_probabilities(replay, 8)
        for key, expected in (
            ("v2", 0.6162606980223937),
            ("direct_prior", 0.6749634762596868),
            ("posterior", 0.576969213908162),
        ):
            self.assertAlmostEqual(forecast[key], expected, places=13)

    def test_historical_forecast_excludes_independent_set_intervals(self):
        result = evaluate(synthetic_sets(40, 87, 8), 8, independent=False)
        for model in result["models"].values():
            self.assertIsNone(model["brier_delta_vs_v2_ci95"])
            if "roi" in model:
                self.assertIsNone(model["roi_ci95"])


if __name__ == "__main__":
    unittest.main()
