import math
from collections import Counter
from tkinter import simpledialog

import customtkinter as ctk

from deucelab.ledger import HistoryEntry, Settlement
from deucelab.logic import BettingLogic

STATUS = {
    Settlement.PENDING: "Pending",
    Settlement.WON: "Won",
    Settlement.LOST: "Lost",
    Settlement.VOID: "Void",
    Settlement.INVALIDATED: "Needs review",
}


class BetHistoryWindow(ctk.CTkToplevel):
    def __init__(self, master: ctk.CTk, logic_ref: BettingLogic):
        super().__init__(master)
        self.logic = logic_ref
        self.title("Session History")
        self.geometry("940x550")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.summary = ctk.CTkLabel(self, anchor="w", justify="left")
        self.summary.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 2))
        self.status = ctk.CTkLabel(self, anchor="w")
        self.status.grid(row=1, column=0, sticky="ew", padx=10)
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.populate_history()

    def populate_history(self) -> None:
        history = self.logic.history
        entries = history.entries
        outcomes = Counter(entry.settlement for entry in entries)
        settled = outcomes[Settlement.WON] + outcomes[Settlement.LOST]
        rate = f"{outcomes[Settlement.WON] / settled:.1%}" if settled else "N/A"
        roi = history.realized_roi()
        roi_label = f"{roi:+.1%}" if roi is not None else "N/A"
        execution_count = sum(entry.execution is not None for entry in entries)
        reviewed = sum(
            entry.execution is not None and entry.execution.net_units(entry.settlement) is not None
            for entry in entries
        )
        self.summary.configure(
            text=(
                f"Predictions {len(entries)}  |  Signal-qualified {sum(entry.signal is not None for entry in entries)}"
                f"  |  Settled predictions {settled}  |  Prediction win rate {rate}\n"
                f"Manual executions {execution_count}  |  Settled or void {reviewed}  |  Recorded ROI {roi_label}"
            ),
        )
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        if not entries:
            ctk.CTkLabel(self.scroll_frame, text="No predictions recorded yet.").pack(pady=10)
        for index in reversed(range(len(entries))):
            self._create_row(index, entries[index])

    def _create_row(self, index: int, entry: HistoryEntry) -> None:
        row = ctk.CTkFrame(self.scroll_frame)
        row.pack(fill="x", pady=2)
        row.grid_columnconfigure(0, weight=1)
        prediction = entry.prediction
        p1, p2 = prediction.score
        kind = "Signal" if entry.signal is not None else "Prediction"
        details = f"S{entry.estimate['set_number']}  {p1}-{p2}  |  {entry.estimate['label']}  |  {prediction.probability:.1%}"
        if prediction.market_odds is not None and prediction.market_ev is not None:
            details += f"  |  Price {prediction.market_odds:.2f}  |  EV {prediction.market_ev:+.1%}"
        if entry.execution is not None:
            execution = entry.execution
            details += f"  |  Executed {execution.stake:g} @ {execution.price:.2f}"
        ctk.CTkLabel(
            row, text=f"{details}\n{kind} · {STATUS[entry.settlement]}", justify="left", anchor="w"
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=5,
        )

        if (
            entry.settlement is Settlement.PENDING
            and entry.signal is not None
            and entry.execution is None
        ):
            ctk.CTkButton(
                row,
                text="Record execution",
                width=112,
                command=lambda: self._record_execution(index),
            ).grid(row=0, column=1, padx=5)
        if entry.settlement is Settlement.INVALIDATED and entry.execution is not None:
            for column, outcome in enumerate(
                (Settlement.WON, Settlement.LOST, Settlement.VOID), start=2
            ):
                ctk.CTkButton(
                    row,
                    text=STATUS[outcome],
                    width=50,
                    command=lambda result=outcome: self._mark_outcome(index, result),
                ).grid(row=0, column=column, padx=2)

    def _mark_outcome(self, index: int, outcome: Settlement) -> None:
        if self.logic.record_bet_outcome(index, outcome):
            self.status.configure(text="Execution outcome recorded.")
            self.populate_history()

    def _record_execution(self, index: int) -> None:
        price = simpledialog.askfloat("Record execution", "Actual decimal odds", parent=self)
        if price is None:
            return
        stake = simpledialog.askfloat("Record execution", "Actual stake in units", parent=self)
        if stake is None:
            return
        try:
            if not math.isfinite(price) or not math.isfinite(stake):
                raise ValueError("Price and stake must be finite.")
            self.logic.history.record_execution(index, price, stake)
        except ValueError as error:
            self.status.configure(text=str(error))
        else:
            self.status.configure(text="Execution recorded.")
            self.populate_history()

    def refresh_history(self) -> None:
        if self.winfo_exists() and self.winfo_viewable():
            self.populate_history()
