import json
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from backtest import SetReplay, evaluate, read_replays, synthetic_sets


class BacktestTests(unittest.TestCase):
    def test_known_set_counts_signal_and_return(self):
        result = evaluate([SetReplay(1, (1, 2) * 10 + (1, 1), 17.5, 3.5, 1.1)], 0)
        self.assertEqual(result["eligible_sets"], 1)
        signals = result["filtered_signals"]
        self.assertEqual((signals["wins"], signals["losses"]), (1, 0))
        self.assertEqual(signals["return_per_unit"], 2.5)

    def test_finished_sets_have_no_forecast(self):
        result = evaluate([SetReplay(1, (1,) * 11, 18.5, 2.0, 2.0)], 11)
        self.assertEqual(result["eligible_sets"], 0)
        self.assertIsNone(result["filtered_signals"]["win_rate"])

    def test_rejects_incomplete_sets_and_rallies_after_finish(self):
        for rallies in ((1,) * 10, (1,) * 12):
            with self.subTest(rallies=rallies), self.assertRaises(ValueError):
                SetReplay(1, rallies, 18.5, 2.0, 2.0)

    def test_historical_snapshot_must_match_requested_point(self):
        row = {
            "first_server": 1,
            "rallies": [1] * 11,
            "line": 18.5,
            "over": 2.0,
            "under": 2.0,
            "after_points": 8,
        }
        with patch.object(Path, "open", side_effect=lambda **_: StringIO(json.dumps(row))):
            path = Path("sets.jsonl")
            self.assertEqual(len(list(read_replays(path, 8))), 1)
            with self.assertRaisesRegex(ValueError, "snapshot"):
                list(read_replays(path, 7))

    def test_desktop_export_metadata_is_accepted_by_regular_backtest(self):
        row = {
            "first_server": 1,
            "rallies": [1] * 11,
            "line": 18.5,
            "over": 1.9,
            "under": 1.95,
            "after_points": 8,
            "match_id": "example",
            "set_number": 1,
            "quote_at": "2026-01-01T10:00:00Z",
            "settled_at": "2026-01-01T10:05:00Z",
        }
        with patch.object(Path, "open", return_value=StringIO(json.dumps(row))):
            replays = list(read_replays(Path("export.jsonl"), 8))
        self.assertEqual(len(replays), 1)
        self.assertEqual(evaluate(replays, 8, "direct")["eligible_sets"], 1)

    def test_seed_reproduces_synthetic_input(self):
        self.assertEqual(list(synthetic_sets(10, 42, 8)), list(synthetic_sets(10, 42, 8)))


if __name__ == "__main__":
    unittest.main()
