import unittest

from logic import BettingLogic


class BettingLogicTests(unittest.TestCase):
    def setUp(self):
        self.logic = BettingLogic()

    def play(self, *winners):
        for winner in winners:
            self.assertTrue(self.logic.record_point(winner))

    def test_scoring_stops_at_set_end(self):
        self.play(*([1] * 11))
        self.assertFalse(self.logic.record_point(2))
        self.assertEqual(len(self.logic.point_history), 11)

    def test_deleting_past_set_invalidates_affected_estimates(self):
        self.logic.get_suggestions()
        self.play(*([1] * 11))
        self.logic.start_new_set(2)
        self.logic.get_suggestions()
        self.assertTrue(self.logic.delete_past_set(0))
        self.assertEqual(self.logic.past_sets, [])
        self.assertEqual(self.logic.set_stats_history, [])
        self.assertTrue(all(r["outcome"] == "hist_cleared" for r in self.logic.get_bet_history()))
        self.assertFalse(self.logic.delete_past_set(0))
        self.logic.get_suggestions()
        self.assertEqual(self.logic.get_bet_history()[-1]["outcome"], "pending")

    def test_clearing_one_models_cache_preserves_other_models_cache(self):
        other = BettingLogic()
        other.get_suggestions()
        before = other._simulate_totals.cache_info()
        self.logic.clear_caches()
        self.assertEqual(other._simulate_totals.cache_info(), before)

    def test_distribution_preserves_probability_mass(self):
        self.assertAlmostEqual(sum(self.logic._run_simulation()), 1.0)
        self.logic.update_config({"momentum_impact_scale": 0.0})
        self.play(*((1, 2) * 10))
        totals = self.logic._run_simulation()
        self.assertAlmostEqual(sum(totals), 1.0)
        self.assertAlmostEqual(totals[22], 2 * 0.61 * 0.39)
        self.assertAlmostEqual(totals[23], 1 - totals[22])
        self.assertEqual(sum(totals[:22]), 0)

    def test_ev_requires_market_price_for_same_side_and_line(self):
        estimate, _, _ = self.logic.get_suggestions()
        self.assertIsNone(estimate["ev"])
        self.assertAlmostEqual(estimate["odds"], 1 / estimate["prob"])
        self.logic.set_market(18.5, over=1.20)
        estimate, _, _ = self.logic.get_suggestions()
        self.assertEqual(estimate["label"], "Over 18.5")
        self.assertLess(estimate["ev"], 0)
        self.assertAlmostEqual(estimate["ev"], estimate["prob"] * 1.20 - 1)
        self.logic.set_market(18.5, under=3.20)
        estimate, _, _ = self.logic.get_suggestions()
        self.assertEqual(estimate["label"], "Under 18.5")
        self.assertGreater(estimate["ev"], 0)
        self.assertAlmostEqual(estimate["ev"], estimate["prob"] * 3.20 - 1)

    def test_invalid_prices_preserve_state(self):
        self.assertTrue(self.logic.set_prematch_odds(1.5, 2.5))
        before = self.logic.p1_neutral_point_win_prob_current
        self.assertFalse(self.logic.set_prematch_odds(float("nan"), 2.0))
        self.assertFalse(self.logic.set_live_odds(float("inf"), 2.0))
        self.assertEqual(self.logic.p1_neutral_point_win_prob_current, before)
        self.logic.set_market(18.5, over=2.0)
        with self.assertRaises(ValueError):
            self.logic.set_market(19.5, under=float("inf"))
        self.assertEqual(self.logic.market_line, 18.5)

    def test_correction_replays_serves_and_score_history(self):
        self.play(1, 2)
        self.assertTrue(self.logic.revise_points(1, remove_player=1))
        self.assertEqual((self.logic.score_p1, self.logic.score_p2), (0, 1))
        self.assertEqual([point["winner"] for point in self.logic.point_history], [2])
        self.assertEqual(
            (self.logic.p1_serve_points_played, self.logic.p1_serve_points_won), (1, 0)
        )
        self.assertTrue(self.logic.revise_points(2))
        self.assertEqual(
            (self.logic.p2_serve_points_played, self.logic.p2_serve_points_won), (1, 1)
        )
        self.assertEqual(self.logic.get_visualization_data()["p2_score_history"], [0, 1])

    def test_completed_set_settles_estimates_and_reset_drops_state(self):
        self.logic.set_market(18.5, under=3.0)
        estimate, _, _ = self.logic.get_suggestions()
        self.play(*([1] * 11))
        self.logic.start_new_set(2)
        self.assertEqual(self.logic.past_sets, [(11, 0)])
        self.assertEqual(self.logic.get_bet_history()[0], {"bet": estimate, "outcome": "won"})
        self.assertIsNone(self.logic.market_line)
        self.play(*([2] * 11))
        self.logic.reset_match()
        self.assertEqual(self.logic.past_sets, [])
        self.assertEqual(self.logic.get_bet_history(), [])

    def test_manual_past_set_does_not_settle_current_estimates(self):
        self.logic.get_suggestions()
        self.assertFalse(self.logic.add_past_set(12, 8))
        self.assertTrue(self.logic.add_past_set(12, 10))
        self.assertEqual(self.logic.get_bet_history()[0]["bet"]["set_number"], 2)
        self.assertEqual(self.logic.get_bet_history()[0]["outcome"], "pending")

    def test_deuce_server_alternates_every_point(self):
        self.assertEqual(self.logic.determine_next_server(10, 10, 2), 2)
        self.assertEqual(self.logic.determine_next_server(11, 10, 2), 1)
        self.assertEqual(self.logic.determine_next_server(11, 11, 2), 2)


if __name__ == "__main__":
    unittest.main()
