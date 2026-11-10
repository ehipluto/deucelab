import logging
import math
from functools import lru_cache
from statistics import mean, pstdev
from typing import Any, TypedDict
from uuid import uuid4

from deucelab import MODEL_VERSION
from deucelab import constants as c
from deucelab.ledger import (
    Estimate as BetMetrics,
)
from deucelab.ledger import (
    EstimateKind,
    FinishedSet,
    LegacyRecord,
    Prediction,
    PredictionHistory,
    Settlement,
)
from deucelab.markets import Quote, Side, expected_value, fair_odds, no_margin, valid_price
from deucelab.models import (
    SERVE_EDGE as SERVE_EDGE,
)
from deucelab.models import (
    direct_point_probability as direct_point_probability,
)
from deucelab.models import (
    exponential_momentum,
    legacy_point_probabilities,
)
from deucelab.rules import Score, is_finished, is_player, is_score, next_server
from deucelab.totals import totals_step

logger = logging.getLogger(__name__)

PastSet = Score


class PointData(TypedDict):
    winner: int
    server: int
    p1s: int
    p2s: int


class SetStats(TypedDict, total=False):
    set_num: int
    p1s: int
    p2s: int
    p1_sp: int
    p1_sw: int
    p2_sp: int
    p2_sw: int
    p1_hold: float | None
    p2_hold: float | None
    avg_momentum: float
    avg_game_state: float


FORECAST_MODELS = ("legacy", "direct")


class BettingLogic:
    def __init__(self, forecast_model: str = "legacy"):
        if forecast_model not in FORECAST_MODELS:
            raise ValueError(f"Unknown forecast model: {forecast_model}")
        self.forecast_model = forecast_model
        self.config = c.DEFAULT_CONFIG.copy()

        self.match_id = uuid4().hex
        self.set_id = 0
        self._set_revision = 0
        self._quote_sequence = 0
        self.total_quote: Quote | None = None
        self.live_quote: Quote | None = None
        self.prematch_odds: tuple[float, float] | None = None
        self.history = PredictionHistory()

        self.score_p1: int = 0
        self.score_p2: int = 0
        self.server: int = 1
        self.initial_server_current_set: int = 1
        self.point_history: list[PointData] = []
        self.point_probability_history: list[float] = []
        self.past_sets: list[PastSet] = []
        self.set_stats_history: list[SetStats] = []
        self.p1_sets_won: int = 0
        self.p2_sets_won: int = 0
        self.p1_match_win_prob_true_base: float = 0.5
        self.p1_neutral_point_win_prob_base: float = 0.5
        self.p1_neutral_point_win_prob_current: float = 0.5
        self.live_odds_active: bool = False
        self.p1_neutral_point_win_prob_history_adjusted: float = 0.5
        self.p1_serve_points_played: int = 0
        self.p1_serve_points_won: int = 0
        self.p2_serve_points_played: int = 0
        self.p2_serve_points_won: int = 0
        self.momentum_scores_in_set: list[float] = []
        self.game_state_scores_in_set: list[float] = []
        self.current_momentum_score: float = 0.0
        self.current_game_state_score: float = 0.0
        self.elaboration_progress: int = 0
        self.elaborated_bet_ready: bool = False
        self.proposed_elaborated_bet: BetMetrics | None = None

        self.required_points_for_elaboration: int = c.ADAPTIVE_ELAB_BASE_POINTS
        self._get_point_probabilities_cached = lru_cache(maxsize=c.CACHE_SIZE_POINT_PROB)(
            self._calculate_point_probabilities
        )
        self._simulate_totals = lru_cache(maxsize=c.CACHE_SIZE_SIMULATION)(self._calculate_totals)

    def update_config(self, new_config: dict[str, Any]) -> bool:
        changes = {
            key: value
            for key, value in new_config.items()
            if key in c.DEFAULT_CONFIG and self.config.get(key) != value
        }
        if not changes:
            return False

        candidate = self.config | changes
        if not c.valid_config(candidate):
            raise ValueError("invalid forecasting or signal configuration")
        self.config = candidate

        self.clear_caches()
        self.refresh_elaboration()
        return True

    def reset_match(self):
        self.score_p1 = self.score_p2 = 0
        self.match_id = uuid4().hex
        self.set_id = 0
        self.total_quote = self.live_quote = None
        self.prematch_odds = None
        self.live_odds_active = False

        self.p1_match_win_prob_true_base = 0.5
        self.p1_neutral_point_win_prob_base = 0.5
        self.past_sets = []
        self.set_stats_history = []
        self.p1_sets_won = 0
        self.p2_sets_won = 0
        self.history = PredictionHistory()

        self.point_probability_history = []
        self.required_points_for_elaboration = c.ADAPTIVE_ELAB_BASE_POINTS
        if not self.live_odds_active:
            self.p1_neutral_point_win_prob_current = self.p1_neutral_point_win_prob_base
        self._calculate_history_adjusted_skill()
        self.start_new_set(initial_srv=1)
        logger.debug("--- MATCH RESET ---")

    def clear_caches(self):
        self._get_point_probabilities_cached.cache_clear()
        self._simulate_totals.cache_clear()

    is_finished = staticmethod(is_finished)
    determine_next_server = staticmethod(next_server)

    @property
    def market_line(self) -> float | None:
        return self.total_quote.line if self.total_quote else None

    @property
    def market_odds(self) -> dict[Side, float]:
        quote = self.total_quote
        if quote is None:
            return {}
        prices: dict[Side, float] = {}
        if quote.first is not None:
            prices["Over"] = quote.first
        if quote.second is not None:
            prices["Under"] = quote.second
        return prices

    def _new_quote(self, line: float | None, first: float | None, second: float | None) -> Quote:
        self._quote_sequence += 1
        return Quote(self._quote_sequence, (self.score_p1, self.score_p2), line, first, second)

    def set_market(
        self,
        line: float | None,
        over: float | None = None,
        under: float | None = None,
    ) -> None:
        if line is None and over is None and under is None:
            self.total_quote = None
        elif (
            self.is_finished(self.score_p1, self.score_p2)
            or line not in c.STANDARD_OU_LINES
            or (over is None and under is None)
            or any(not valid_price(price) for price in (over, under) if price is not None)
        ):
            raise ValueError("Choose an active market and enter decimal odds greater than 1.")
        else:
            current = self.total_quote
            if current is None or (current.line, current.first, current.second) != (
                line,
                over,
                under,
            ):
                self.total_quote = self._new_quote(line, over, under)
        self.refresh_elaboration()

    def refresh_elaboration(self) -> None:
        self.elaborated_bet_ready = False
        self.proposed_elaborated_bet = None
        self._update_required_elaboration_points()
        self._update_elaboration_progress()

    def _update_set_score_display(self):
        self.p1_sets_won = sum(1 for p1, p2 in self.past_sets if p1 > p2)
        self.p2_sets_won = sum(1 for p1, p2 in self.past_sets if p2 > p1)

    def add_past_set(self, p1s: int, p2s: int, set_stats: SetStats | None = None) -> bool:
        if not is_score(p1s, p2s):
            return False
        set_num_finished = len(self.past_sets) + 1
        if self.is_finished(p1s, p2s):
            self.past_sets.append((p1s, p2s))
            self.set_stats_history.append(set_stats or {})
            self._update_set_score_display()
            self._calculate_history_adjusted_skill()
            if set_stats is None:
                for entry in self.history.entries:
                    if entry.prediction.set_id == self.set_id:
                        entry.estimate["set_number"] = set_num_finished + 1
            self.refresh_elaboration()

            logger.debug(
                f"Past set:{p1s}-{p2s} added. Stats:{'Yes' if set_stats else 'No'}. "
                f"Match:{self.p1_sets_won}-{self.p2_sets_won}"
            )
            return True
        logger.debug(f"Err: Invalid set {p1s}-{p2s}")
        return False

    def delete_past_set(self, index: int) -> bool:
        if not 0 <= index < len(self.past_sets):
            return False
        self.past_sets.pop(index)
        self.set_stats_history.pop(index)
        self._update_set_score_display()
        self._calculate_history_adjusted_skill()
        self.history.invalidate(lambda entry: entry.estimate["set_number"] >= index + 1)
        self._set_revision += 1
        self.refresh_elaboration()
        return True

    def clear_past_sets(self):
        self.past_sets.clear()
        self.set_stats_history.clear()
        self._update_set_score_display()
        self._calculate_history_adjusted_skill()
        self.history.invalidate(lambda entry: True)
        self._set_revision += 1
        self.refresh_elaboration()

    def _calc_neut_prob(self, p1mp: float) -> float:
        sd = p1mp - 0.5
        np = 0.5 + sd * (c.SKILL_DIFF_POINT_IMPACT / 0.5)
        return max(
            0.5 - c.SKILL_DIFF_POINT_IMPACT,
            min(0.5 + c.SKILL_DIFF_POINT_IMPACT, np),
        )

    def set_prematch_odds(self, o1: float, o2: float) -> bool:
        if not valid_price(o1) or not valid_price(o2):
            return False
        self.prematch_odds = (o1, o2)
        self.p1_match_win_prob_true_base, _ = no_margin(o1, o2)
        self.p1_neutral_point_win_prob_base = self._calc_neut_prob(self.p1_match_win_prob_true_base)
        self._calculate_history_adjusted_skill()
        self.refresh_elaboration()
        return True

    def set_live_odds(self, o1: float, o2: float) -> bool:
        if (
            not valid_price(o1)
            or not valid_price(o2)
            or self.is_finished(self.score_p1, self.score_p2)
        ):
            return False
        if self.live_quote is None or (self.live_quote.first, self.live_quote.second) != (o1, o2):
            self.live_quote = self._new_quote(None, o1, o2)
        self.p1_neutral_point_win_prob_current = self._calc_neut_prob(no_margin(o1, o2)[0])
        if not self.live_odds_active:
            self.clear_caches()
        self.live_odds_active = True
        self.refresh_elaboration()
        return True

    def clear_live_odds(self) -> None:
        if self.live_odds_active:
            self.live_odds_active = False
            self.live_quote = None
            self.p1_neutral_point_win_prob_current = self.p1_neutral_point_win_prob_history_adjusted
            self.clear_caches()
            self.refresh_elaboration()

    def _calculate_history_adjusted_skill(self):
        if not self.past_sets:
            target_prob = self.p1_neutral_point_win_prob_base
            self.p1_neutral_point_win_prob_history_adjusted = target_prob
            if not self.live_odds_active:
                needs_clear = not math.isclose(self.p1_neutral_point_win_prob_current, target_prob)
                self.p1_neutral_point_win_prob_current = target_prob
                if needs_clear:
                    logger.debug("History Adj (Empty): -> CURR updated to Base. Clearing caches.")
                    self.clear_caches()
            return

        nps = len(self.past_sets)
        p1sw = self.p1_sets_won
        p1se = self.p1_match_win_prob_true_base * nps
        pds = p1sw - p1se
        wpd = 0
        tw = 0
        apd = 0
        if nps > 0:
            recency = self.config.get("history_weighting_recency", c.HISTORY_WEIGHTING_RECENCY)
            if recency > 0:
                for i, (p1, p2) in enumerate(reversed(self.past_sets)):
                    w = max(0, 1.0 - (i / recency) * 0.5)
                    if w > 0:
                        wpd += (p1 - p2) * w
                        tw += w
                    else:
                        break
                if tw > 0:
                    apd = wpd / tw
            else:
                apd = sum((p1 - p2) for p1, p2 in self.past_sets) / nps
        base_adj = 0.0
        if nps > 0:
            base_adj += (pds / nps) * 0.5
        base_adj += (apd / 6.0) * 0.5
        serve_hold_trend = 0.0
        state_score_trend = 0.0
        num_hist_stats = len(self.set_stats_history)
        if num_hist_stats > 0:
            avg_p1h = self._average_set_stat("p1_hold")
            avg_p2h = self._average_set_stat("p2_hold")
            avg_state = self._average_set_stat("avg_game_state")
            if avg_p1h is not None and avg_p2h is not None:
                serve_hold_trend = max(-1.0, min(1.0, (avg_p1h - avg_p2h) * 1.5))
            if avg_state is not None:
                state_score_trend = avg_state
            final_adj = (
                base_adj * 0.6 + serve_hold_trend * 0.2 + state_score_trend * 0.2
            ) * c.PERFORMANCE_ADJUSTMENT_FACTOR
        else:
            final_adj = base_adj * c.PERFORMANCE_ADJUSTMENT_FACTOR

        new_hist_adj_prob = self.p1_neutral_point_win_prob_base + final_adj
        min_p = max(0.25, self.p1_neutral_point_win_prob_base - 0.12)
        max_p = min(0.75, self.p1_neutral_point_win_prob_base + 0.12)
        old_hist_adj = self.p1_neutral_point_win_prob_history_adjusted
        self.p1_neutral_point_win_prob_history_adjusted = max(min_p, min(max_p, new_hist_adj_prob))

        if not self.live_odds_active:
            if not math.isclose(
                self.p1_neutral_point_win_prob_current,
                self.p1_neutral_point_win_prob_history_adjusted,
                abs_tol=1e-4,
            ):
                if not math.isclose(
                    old_hist_adj, self.p1_neutral_point_win_prob_history_adjusted, abs_tol=1e-4
                ):
                    self.p1_neutral_point_win_prob_current = (
                        self.p1_neutral_point_win_prob_history_adjusted
                    )
                    logger.debug(
                        f"History Adj: -> CURR updated from History Adj ({self.p1_neutral_point_win_prob_current:.4f}). Clearing caches."
                    )
                    self.clear_caches()

    def _calculate_exponential_momentum(self, history: tuple[int, ...]) -> float:
        return exponential_momentum(
            history, self.config["momentum_decay_factor"], self.config["momentum_impact_scale"]
        )

    def _update_game_state(self, winner: int):
        min_srv = 3
        if self.server == 1:
            self.p1_serve_points_played += 1
            self.p1_serve_points_won += 1 if winner == 1 else 0
        else:
            self.p2_serve_points_played += 1
            self.p2_serve_points_won += 1 if winner == 2 else 0

        recent_winners_tuple = tuple(p["winner"] for p in self.point_history[-c.MOMENTUM_WINDOW :])
        momentum_effect = self._calculate_exponential_momentum(recent_winners_tuple)
        self.current_momentum_score = momentum_effect

        perf_vs_odds = (
            self.p1_neutral_point_win_prob_history_adjusted - self.p1_neutral_point_win_prob_base
        )
        s_perf_odds = perf_vs_odds / 0.12 if abs(perf_vs_odds) > 1e-3 else 0.0
        pressure = 0.0
        ps1 = self.point_history[-1]["p1s"] if self.point_history else 0
        ps2 = self.point_history[-1]["p2s"] if self.point_history else 0
        if ps1 >= 10 or ps2 >= 10:
            pressure += 0.3
        if (ps1 == 10 and ps2 <= 9) or (ps1 >= 11 and ps1 == ps2 + 1):
            pressure += 0.7
        if (ps2 == 10 and ps1 <= 9) or (ps2 >= 11 and ps2 == ps1 + 1):
            pressure -= 0.7
        pressure = max(-1.0, min(1.0, pressure))
        match_p1_sp = (
            sum(s.get("p1_sp", 0) for s in self.set_stats_history) + self.p1_serve_points_played
        )
        match_p1_sw = (
            sum(s.get("p1_sw", 0) for s in self.set_stats_history) + self.p1_serve_points_won
        )
        match_p2_sp = (
            sum(s.get("p2_sp", 0) for s in self.set_stats_history) + self.p2_serve_points_played
        )
        match_p2_sw = (
            sum(s.get("p2_sw", 0) for s in self.set_stats_history) + self.p2_serve_points_won
        )
        serve_perf = 0.0
        if match_p1_sp >= min_srv and match_p2_sp >= min_srv:
            p1h = match_p1_sw / match_p1_sp
            p2h = match_p2_sw / match_p2_sp
            serve_perf = max(-1.0, min(1.0, ((p1h - 0.5) - (p2h - 0.5)) * 2.0))

        current_state_score = (
            (
                self.current_momentum_score / c.MOMENTUM_IMPACT_SCALE
                if c.MOMENTUM_IMPACT_SCALE > 0
                else 0.0
            )
            * self.config.get("W_MOMENTUM", c.W_MOMENTUM)
            + s_perf_odds * self.config.get("W_PERF_VS_ODDS", c.W_PERF_VS_ODDS)
            + pressure * self.config.get("W_PRESSURE", c.W_PRESSURE)
            + serve_perf * self.config.get("W_SERVE_PERF", c.W_SERVE_PERF)
        )
        self.current_game_state_score = max(-1.5, min(1.5, current_state_score))
        self.momentum_scores_in_set.append(self.current_momentum_score)
        self.game_state_scores_in_set.append(self.current_game_state_score)

    def _apply_point_state(self, p1: int, p2: int, server: int, winner: int | None) -> None:
        if winner is not None:
            self.point_history.append(
                {
                    "winner": winner,
                    "server": self.server,
                    "p1s": self.score_p1,
                    "p2s": self.score_p2,
                }
            )
            self._update_game_state(winner)
        self.score_p1, self.score_p2, self.server = p1, p2, server
        if winner is not None:
            probability, _, _ = self._get_dynamic_point_probabilities()
            self.point_probability_history.append(probability)

    def update_current_set_state(
        self, p1s: int, p2s: int, srv: int, win: int | None = None
    ) -> None:
        if not is_score(p1s, p2s) or not is_player(srv):
            raise ValueError("invalid score or server")
        if win is None:
            if (p1s, p2s, srv) != (self.score_p1, self.score_p2, self.server):
                raise ValueError("use rally entry or score correction to change the set")
            return
        if (
            not is_player(win)
            or self.is_finished(self.score_p1, self.score_p2)
            or (p1s, p2s) != (self.score_p1 + (win == 1), self.score_p2 + (win == 2))
            or srv != next_server(p1s, p2s, self.initial_server_current_set)
        ):
            raise ValueError("the score and server must follow the recorded rally")
        self.total_quote = None
        self.clear_live_odds()
        self._apply_point_state(p1s, p2s, srv, win)
        self.refresh_elaboration()
        if self.is_finished(p1s, p2s):
            self.history.settle(self.set_id, p1s + p2s)

    def record_point(self, winner: int) -> bool:
        if not is_player(winner):
            raise ValueError("winner must be 1 or 2")
        if self.is_finished(self.score_p1, self.score_p2):
            return False
        p1 = self.score_p1 + (winner == 1)
        p2 = self.score_p2 + (winner == 2)
        self.update_current_set_state(
            p1, p2, next_server(p1, p2, self.initial_server_current_set), winner
        )
        return True

    def revise_points(self, initial_server: int, remove_player: int | None = None) -> bool:
        if not is_player(initial_server) or (
            remove_player is not None and not is_player(remove_player)
        ):
            raise ValueError("player must be 1 or 2")
        if remove_player is None and initial_server == self.initial_server_current_set:
            return True
        winners = [point["winner"] for point in self.point_history]
        if remove_player is not None:
            if remove_player not in winners:
                return False
            index = len(winners) - 1 - winners[::-1].index(remove_player)
            del winners[index]
            p1 = p2 = 0
            for winner in winners[:-1]:
                p1 += winner == 1
                p2 += winner == 2
                if self.is_finished(p1, p2):
                    return False

        self.history.invalidate(lambda entry: entry.prediction.set_id == self.set_id)
        self._set_revision += 1
        self._reset_set_state(initial_server)

        for winner in winners:
            p1 = self.score_p1 + (winner == 1)
            p2 = self.score_p2 + (winner == 2)
            server = self.determine_next_server(p1, p2, initial_server)
            self._apply_point_state(p1, p2, server, winner)
        self.refresh_elaboration()
        return True

    def start_new_set(self, initial_srv: int):
        if not is_player(initial_srv):
            raise ValueError("first server must be 1 or 2")
        self.clear_live_odds()
        final_p1 = self.score_p1
        final_p2 = self.score_p2
        set_complete = self.is_finished(final_p1, final_p2)
        set_num_finished = len(self.past_sets) + 1

        if set_complete:
            logger.debug(f"Recording set {set_num_finished}:{final_p1}-{final_p2}")
            p1_hold = (
                self.p1_serve_points_won / self.p1_serve_points_played
                if self.p1_serve_points_played > 0
                else None
            )
            p2_hold = (
                self.p2_serve_points_won / self.p2_serve_points_played
                if self.p2_serve_points_played > 0
                else None
            )
            avg_mom = mean(self.momentum_scores_in_set) if self.momentum_scores_in_set else 0.0
            avg_state = (
                mean(self.game_state_scores_in_set) if self.game_state_scores_in_set else 0.0
            )
            avg_mom = max(-c.MOMENTUM_IMPACT_SCALE, min(c.MOMENTUM_IMPACT_SCALE, avg_mom))
            avg_state = max(-1.5, min(1.5, avg_state))

            set_stats: SetStats = {
                "set_num": set_num_finished,
                "p1s": final_p1,
                "p2s": final_p2,
                "p1_sp": self.p1_serve_points_played,
                "p1_sw": self.p1_serve_points_won,
                "p2_sp": self.p2_serve_points_played,
                "p2_sw": self.p2_serve_points_won,
                "p1_hold": p1_hold,
                "p2_hold": p2_hold,
                "avg_momentum": avg_mom,
                "avg_game_state": avg_state,
            }
            self.history.settle(self.set_id, final_p1 + final_p2)
            self.add_past_set(final_p1, final_p2, set_stats)
        else:
            self.history.invalidate(lambda entry: entry.prediction.set_id == self.set_id)
        self.set_id += 1
        self._set_revision = 0
        self._reset_set_state(initial_srv)

    def _reset_set_state(self, initial_srv: int) -> None:
        self.total_quote = None
        self.clear_live_odds()

        self.score_p1 = 0
        self.score_p2 = 0
        self.initial_server_current_set = initial_srv
        self.server = initial_srv
        self.point_history = []
        self.point_probability_history = []
        self.p1_serve_points_played = 0
        self.p1_serve_points_won = 0
        self.p2_serve_points_played = 0
        self.p2_serve_points_won = 0
        self.current_momentum_score = 0.0
        self.current_game_state_score = 0.0
        self.momentum_scores_in_set = []
        self.game_state_scores_in_set = []
        self.elaboration_progress = 0
        self.elaborated_bet_ready = False
        self.proposed_elaborated_bet = None
        self.required_points_for_elaboration = c.ADAPTIVE_ELAB_BASE_POINTS

        self.clear_caches()
        logger.debug(f"--- Start New Set {len(self.past_sets) + 1} --- Init Srv: P{initial_srv}")

    def _calculate_point_probabilities(
        self, srv: int, hist_win_tup: tuple[int, ...], p1_curr_npp: float
    ) -> tuple[float, float, list[str]]:
        if self.forecast_model == "direct":
            probability = direct_point_probability(p1_curr_npp, srv)
            return probability, 1 - probability, [f"P{srv} Srv", "Direct neutral strength"]

        momentum_effect = self._calculate_exponential_momentum(hist_win_tup)
        p1_skill_adv = p1_curr_npp - 0.5
        p1w, p2w = legacy_point_probabilities(p1_curr_npp, srv, momentum_effect)

        r_mods = []
        r_mods.append(f"P{srv} Srv")
        skill_prefix = "(Live)" if self.live_odds_active else "(Base)"
        if abs(p1_skill_adv) > 0.02:
            skill_level = "++" if abs(p1_skill_adv) > 0.08 else "+"
            skill_sign = "+" if p1_skill_adv > 0 else "-"
            r_mods.append(f"Skill P1:{skill_sign}{skill_level}{skill_prefix}")
        if (
            abs(momentum_effect)
            > self.config.get("momentum_impact_scale", c.MOMENTUM_IMPACT_SCALE) * 0.66
        ):
            mom_level = (
                "++"
                if abs(momentum_effect)
                > self.config.get("momentum_impact_scale", c.MOMENTUM_IMPACT_SCALE) * 0.9
                else "+"
            )
            mom_player = "P1" if momentum_effect > 0 else "P2"
            r_mods.append(f"Mom:{mom_player}{mom_level}")
        elif (
            abs(momentum_effect)
            > self.config.get("momentum_impact_scale", c.MOMENTUM_IMPACT_SCALE) * 0.25
        ):
            mom_player = "P1" if momentum_effect > 0 else "P2"
            r_mods.append(f"Mom:{mom_player}-")

        return p1w, p2w, r_mods

    def _get_dynamic_point_probabilities(self) -> tuple[float, float, list[str]]:
        hist_win_tup = self._forecast_history()
        return self._get_point_probabilities_cached(
            self.server, hist_win_tup, self.p1_neutral_point_win_prob_current
        )

    def _calculate_totals(
        self,
        p1: int,
        p2: int,
        initial_server: int,
        neutral_prob: float,
        recent_winners: tuple[int, ...],
    ) -> tuple[float, ...]:
        return totals_step(
            p1,
            p2,
            initial_server,
            lambda server: self._get_point_probabilities_cached(
                server, recent_winners, neutral_prob
            )[0],
            lambda a, b: self._simulate_totals(a, b, initial_server, neutral_prob, recent_winners),
        )

    def _forecast_history(self) -> tuple[int, ...]:
        if self.forecast_model == "direct":
            return ()
        return tuple(point["winner"] for point in self.point_history[-c.MOMENTUM_WINDOW :])

    def _run_simulation(self) -> tuple[float, ...]:
        recent_winners = self._forecast_history()
        return self._simulate_totals(
            self.score_p1,
            self.score_p2,
            self.initial_server_current_set,
            self.p1_neutral_point_win_prob_current,
            recent_winners,
        )

    def _evaluate_over_under(
        self,
        line: float,
        totals: tuple[float, ...],
        reasons: list[str],
        set_num: int,
        min_ev: float,
        min_odds: float,
    ) -> list[BetMetrics]:
        split = math.floor(line) + 1
        estimates = []
        probabilities: tuple[tuple[Side, float], ...] = (
            ("Over", sum(totals[split:])),
            ("Under", sum(totals[:split])),
        )
        prices = self.market_odds if self.market_line == line else {}
        quote = self.total_quote if self.market_line == line else None
        market_probabilities = quote.market_probability if quote else None
        for index, (side, probability) in enumerate(probabilities):
            probability = min(1.0, probability)
            market_price = prices.get(side)
            ev = expected_value(probability, market_price) if market_price is not None else None
            if self.market_line != line and not (
                self.config["min_prob_suggestion_base"]
                <= probability
                <= self.config["max_prob_filter_base"]
            ):
                continue
            meets_filters = (
                ev is not None
                and market_price is not None
                and ev >= min_ev
                and market_price >= min_odds
                and self.config["min_prob_suggestion_base"]
                <= probability
                <= self.config["max_prob_filter_base"]
            )
            estimates.append(
                {
                    "prob": probability,
                    "odds": fair_odds(probability),
                    "market_probability": market_probabilities[index]
                    if market_probabilities
                    else None,
                    "market_odds": market_price,
                    "ev": ev,
                    "meets_filters": meets_filters,
                    "reason": ", ".join(reasons),
                    "label": f"{side} {line}",
                    "side": side,
                    "line": line,
                    "set_number": set_num,
                }
            )
        return estimates

    @staticmethod
    def _estimate_rank(estimate: BetMetrics) -> tuple[bool, float]:
        ev = estimate["ev"]
        return ev is not None, ev if ev is not None else estimate["prob"]

    def _update_required_elaboration_points(self):
        if not self.config.get(
            "adaptive_elab_threshold_enabled", c.ADAPTIVE_ELAB_THRESHOLD_ENABLED
        ):
            self.required_points_for_elaboration = self.config.get(
                "adaptive_elab_base_points", c.ADAPTIVE_ELAB_BASE_POINTS
            )
            return

        base_points = self.config.get("adaptive_elab_base_points", c.ADAPTIVE_ELAB_BASE_POINTS)
        min_points = c.ADAPTIVE_ELAB_MIN_POINTS
        max_points = c.ADAPTIVE_ELAB_MAX_POINTS
        hist_len = c.ADAPTIVE_ELAB_HISTORY_LEN
        vol_scale = self.config.get(
            "adaptive_elab_volatility_scale", c.ADAPTIVE_ELAB_VOLATILITY_SCALE
        )

        current_points_played = len(self.point_history)

        if current_points_played < min_points:
            self.required_points_for_elaboration = base_points
        elif current_points_played < base_points:
            self.required_points_for_elaboration = base_points
        else:
            if len(self.point_probability_history) >= hist_len:
                recent_probs = self.point_probability_history[-hist_len:]
                volatility = pstdev(recent_probs) if len(recent_probs) > 1 else 0.0
                volatility_adjustment = volatility * vol_scale
                req_points = base_points + volatility_adjustment
                self.required_points_for_elaboration = int(
                    round(max(min_points, min(max_points, req_points)))
                )
            else:
                self.required_points_for_elaboration = base_points

    def _update_elaboration_progress(self):
        current_points = len(self.point_history)
        required_points = self.required_points_for_elaboration
        self.elaboration_progress = min(100, 100 * current_points // required_points)

    def _calculate_elaborated_bet(self):
        self.elaborated_bet_ready = True
        if self.is_finished(self.score_p1, self.score_p2):
            return

        center = self.calculate_suggested_ou_line(self.score_p1, self.score_p2)
        radius = self.config["ou_evaluation_range"]
        lines = (
            [self.market_line]
            if self.market_line is not None
            else [line for line in c.STANDARD_OU_LINES if abs(line - center) <= radius]
        )
        totals = self._run_simulation()
        reasons = self._get_dynamic_point_probabilities()[2]
        set_num = len(self.past_sets) + 1
        estimates = [
            estimate
            for line in lines
            for estimate in self._evaluate_over_under(
                line,
                totals,
                reasons,
                set_num,
                self.config["min_acceptable_ev_base"],
                self.config["min_estimated_odds_base"],
            )
        ]
        self.proposed_elaborated_bet = max(estimates, key=self._estimate_rank, default=None)

    def calculate_suggested_ou_line(self, p1_score: int, p2_score: int) -> float:
        a = p1_score
        b = p2_score
        p = a + b
        max_score = max(a, b)

        if (a >= 11 and a >= b + 2) or (b >= 11 and b >= a + 2):
            points_needed_to_win = 0
        elif a >= 10 and b >= 10:
            points_needed_to_win = 2
        else:
            points_needed_to_win = max(0, 11 - max_score)

        if p <= 11:
            af_base = max(2.5, 6.5 - (4.0 / 7.0) * max(0, p - 4))
        else:
            af_base = 2.5

        server_factor = self.config["ou_line_server_factor"]
        p1w_dyn, _, _ = self._get_dynamic_point_probabilities()
        p_srv_win = p1w_dyn if self.server == 1 else 1.0 - p1w_dyn
        af_final = af_base + (p_srv_win - 0.5) * server_factor * 2.0
        raw_line = float(p + points_needed_to_win + af_final)
        return min(c.STANDARD_OU_LINES, key=lambda line: abs(line - raw_line))

    def get_suggestions(
        self,
        *,
        record: bool = True,
        include_elaborated: bool = True,
    ) -> tuple[BetMetrics | FinishedSet | None, BetMetrics | None, int]:
        if self.is_finished(self.score_p1, self.score_p2):
            return (
                {"label": f"Set Finished {self.score_p1}-{self.score_p2}", "type": "info"},
                None,
                100,
            )

        line = self.market_line or self.calculate_suggested_ou_line(self.score_p1, self.score_p2)
        estimates = self._evaluate_over_under(
            line,
            self._run_simulation(),
            self._get_dynamic_point_probabilities()[2],
            len(self.past_sets) + 1,
            self.config["min_acceptable_ev_base"],
            self.config["min_estimated_odds_base"],
        )
        current = max(estimates, key=self._estimate_rank, default=None)
        if (
            include_elaborated
            and self.elaboration_progress == 100
            and not self.elaborated_bet_ready
        ):
            self._calculate_elaborated_bet()
        elaborated = self.proposed_elaborated_bet if include_elaborated else None
        for estimate, kind in ((current, "dynamic"), (elaborated, "elaborated")):
            if estimate is not None:
                estimate["type"] = kind
                if record:
                    self._record_estimate(estimate, kind)
        return current, elaborated, self.elaboration_progress

    def _record_estimate(self, estimate: BetMetrics, kind: EstimateKind) -> None:
        quote = self.total_quote if self.market_line == estimate["line"] else None
        self.history.add(
            Prediction(
                match_id=self.match_id,
                set_id=self.set_id,
                revision=self._set_revision,
                score=(self.score_p1, self.score_p2),
                point_number=len(self.point_history),
                model=f"{self.forecast_model}:{MODEL_VERSION}",
                kind=kind,
                side=estimate["side"],
                line=estimate["line"],
                probability=estimate["prob"],
                quote_id=quote.id if quote else None,
                market_odds=estimate["market_odds"],
                market_probability=estimate["market_probability"],
                live_quote_id=self.live_quote.id if self.live_quote else None,
                prematch_odds=self.prematch_odds,
                market_ev=estimate["ev"],
            ),
            estimate,
        )

    def get_bet_history(self) -> list[LegacyRecord]:
        return [
            {"bet": entry.estimate.copy(), "outcome": entry.settlement.value}
            for entry in self.history.entries
        ]

    @property
    def bet_outcomes(self) -> list[LegacyRecord]:
        return self.get_bet_history()

    def record_bet_outcome(self, bet_idx: int, outcome: str) -> bool:
        if not 0 <= bet_idx < len(self.history.entries) or outcome not in ("won", "lost", "null"):
            return False
        entry = self.history.entries[bet_idx]
        if entry.settlement in (Settlement.WON, Settlement.LOST):
            return False
        if entry.settlement is Settlement.INVALIDATED and entry.execution is None:
            return False
        entry.settlement = Settlement(outcome)
        return True

    def get_visualization_data(self) -> dict[str, Any]:
        pts_x = [0]
        p1_y = [0]
        p2_y = [0]
        plot_history = self.point_history
        current_p1s = self.score_p1
        current_p2s = self.score_p2
        if not plot_history:
            pts_x = [0]
            p1_y = [current_p1s]
            p2_y = [current_p2s]
        else:
            pts_x = [0]
            p1_y = [plot_history[0]["p1s"]]
            p2_y = [plot_history[0]["p2s"]]
            for i, p in enumerate(plot_history):
                pts_x.append(i + 1)
                p1a = p["p1s"] + (1 if p["winner"] == 1 else 0)
                p2a = p["p2s"] + (1 if p["winner"] == 2 else 0)
                p1_y.append(p1a)
                p2_y.append(p2a)

        p1p, _, _ = self._get_dynamic_point_probabilities()
        p1sh = (
            self.p1_serve_points_won / self.p1_serve_points_played
            if self.p1_serve_points_played > 2
            else None
        )
        p2sh = (
            self.p2_serve_points_won / self.p2_serve_points_played
            if self.p2_serve_points_played > 2
            else None
        )
        spv = 0.0
        if isinstance(p1sh, float) and isinstance(p2sh, float):
            spv = max(-1.0, min(1.0, (p1sh - 0.5) - (p2sh - 0.5)) * 2.0)
        spvo = (
            (self.p1_neutral_point_win_prob_history_adjusted - self.p1_neutral_point_win_prob_base)
            / 0.12
            if abs(
                self.p1_neutral_point_win_prob_history_adjusted
                - self.p1_neutral_point_win_prob_base
            )
            > 1e-3
            else 0.0
        )
        mom_score_for_viz = (
            self.current_momentum_score
            / self.config.get("momentum_impact_scale", c.MOMENTUM_IMPACT_SCALE)
            if self.config.get("momentum_impact_scale", c.MOMENTUM_IMPACT_SCALE) > 0
            else 0.0
        )
        mom_score_for_viz = max(-1.0, min(1.0, mom_score_for_viz))

        avg_hist_p1h = self._average_set_stat("p1_hold")
        avg_hist_p2h = self._average_set_stat("p2_hold")
        avg_hist_state = self._average_set_stat("avg_game_state")

        return {
            "points_played_in_set": pts_x,
            "p1_score_history": p1_y,
            "p2_score_history": p2_y,
            "current_p1_point_win_prob": p1p,
            "point_probability_history": self.point_probability_history[:],
            "p1_neutral_point_win_prob_base": self.p1_neutral_point_win_prob_base,
            "p1_neutral_point_win_prob_current": self.p1_neutral_point_win_prob_current,
            "p1_neutral_point_win_prob_history_adjusted": self.p1_neutral_point_win_prob_history_adjusted,
            "live_odds_active": self.live_odds_active,
            "game_state_score": self.current_game_state_score,
            "momentum_score": mom_score_for_viz,
            "scaled_perf_vs_odds": spvo,
            "serve_perf_score": spv,
            "p1_serve_hold_pct": p1sh if p1sh is not None else "N/A(<3)",
            "p2_serve_hold_pct": p2sh if p2sh is not None else "N/A(<3)",
            "performance_delta_sets": self._calculate_perf_delta_sets(),
            "avg_point_diff": self._calculate_avg_point_diff(),
            "avg_hist_p1_hold": avg_hist_p1h,
            "avg_hist_p2_hold": avg_hist_p2h,
            "avg_hist_state": avg_hist_state,
            "p1_sets_won": self.p1_sets_won,
            "p2_sets_won": self.p2_sets_won,
        }

    def _average_set_stat(self, key: str) -> float | None:
        values = [stats[key] for stats in self.set_stats_history if stats.get(key) is not None]
        return mean(values) if values else None

    def _calculate_perf_delta_sets(self) -> float | None:
        if not self.past_sets:
            return None
        n = len(self.past_sets)
        e = self.p1_match_win_prob_true_base * n
        a = self.p1_sets_won
        return a - e

    def _calculate_avg_point_diff(self) -> float | None:
        if not self.past_sets:
            return None
        n = len(self.past_sets)
        wpd = 0
        tw = 0
        apd = 0
        recency = self.config.get("history_weighting_recency", c.HISTORY_WEIGHTING_RECENCY)
        if recency > 0:
            for i, (p1, p2) in enumerate(reversed(self.past_sets)):
                w = max(0, 1.0 - (i / recency) * 0.5)
                if w > 0:
                    wpd += (p1 - p2) * w
                    tw += w
                else:
                    break
            if tw > 0:
                apd = wpd / tw
        else:
            apd = sum((p1 - p2) for p1, p2 in self.past_sets) / n if n > 0 else 0.0
        return apd
