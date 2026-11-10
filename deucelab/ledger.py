import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from deucelab.markets import Side, valid_price
from deucelab.rules import Score

EstimateKind = Literal["dynamic", "elaborated"]


class Estimate(TypedDict):
    prob: float
    odds: float | None
    market_odds: float | None
    market_probability: float | None
    ev: float | None
    meets_filters: bool
    reason: str
    label: str
    side: Side
    line: float
    set_number: int
    type: NotRequired[EstimateKind]


class FinishedSet(TypedDict):
    label: str
    type: Literal["info"]


class Settlement(StrEnum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    VOID = "null"
    INVALIDATED = "hist_cleared"


class LegacyRecord(TypedDict):
    bet: Estimate
    outcome: str


@dataclass(frozen=True)
class Prediction:
    match_id: str
    set_id: int
    revision: int
    score: Score
    point_number: int
    model: str
    kind: EstimateKind = field(compare=False)
    side: Side
    line: float
    probability: float
    quote_id: int | None
    market_odds: float | None
    market_probability: float | None
    live_quote_id: int | None
    prematch_odds: tuple[float, float] | None
    market_ev: float | None


@dataclass(frozen=True)
class Signal:
    prediction: Prediction


@dataclass(frozen=True)
class Execution:
    signal: Signal
    price: float
    stake: float

    def net_units(self, settlement: Settlement) -> float | None:
        if settlement is Settlement.WON:
            return (self.price - 1) * self.stake
        if settlement is Settlement.LOST:
            return -self.stake
        if settlement is Settlement.VOID:
            return 0.0
        return None


@dataclass
class HistoryEntry:
    prediction: Prediction
    estimate: Estimate
    signal: Signal | None
    settlement: Settlement = Settlement.PENDING
    execution: Execution | None = None


@dataclass
class PredictionHistory:
    entries: list[HistoryEntry] = field(default_factory=list)
    _by_prediction: dict[Prediction, HistoryEntry] = field(default_factory=dict, init=False)

    def add(self, prediction: Prediction, estimate: Estimate) -> None:
        is_signal = estimate["meets_filters"]
        entry = self._by_prediction.get(prediction)
        if entry is not None:
            if is_signal and entry.signal is None and entry.settlement is Settlement.PENDING:
                entry.signal = Signal(entry.prediction)
                entry.estimate = estimate.copy()
            return
        entry = HistoryEntry(prediction, estimate.copy(), Signal(prediction) if is_signal else None)
        self._by_prediction[prediction] = entry
        self.entries.append(entry)

    def settle(self, set_id: int, total: int) -> None:
        for entry in self.entries:
            if entry.prediction.set_id == set_id and entry.settlement is Settlement.PENDING:
                prediction = entry.prediction
                won = (total > prediction.line) == (prediction.side == "Over")
                entry.settlement = Settlement.WON if won else Settlement.LOST

    def invalidate(self, affected: Callable[[HistoryEntry], bool]) -> None:
        for entry in self.entries:
            if affected(entry):
                entry.settlement = Settlement.INVALIDATED

    def record_execution(self, index: int, price: float, stake: float) -> None:
        if not 0 <= index < len(self.entries):
            raise ValueError("unknown prediction")
        entry = self.entries[index]
        if entry.signal is None or entry.execution is not None:
            raise ValueError("select a signal without an existing execution")
        if entry.settlement is not Settlement.PENDING:
            raise ValueError("only pending signals can have an execution recorded")
        if (
            not valid_price(price)
            or not isinstance(stake, (int, float))
            or isinstance(stake, bool)
            or not 0 < stake <= float.fromhex("0x1.fffffffffffffp1023")
            or not math.isfinite((price - 1) * stake)
        ):
            raise ValueError("enter valid decimal odds and a positive, finite stake")
        entry.execution = Execution(entry.signal, price, stake)

    def realized_roi(self) -> float | None:
        settled = [
            (entry.execution.stake, net)
            for entry in self.entries
            if entry.execution is not None
            and (net := entry.execution.net_units(entry.settlement)) is not None
        ]
        if not settled:
            return None
        scale = max(stake for stake, _ in settled)
        return sum((net / scale) / len(settled) for _, net in settled) / (
            sum(stake / scale for stake, _ in settled) / len(settled)
        )
