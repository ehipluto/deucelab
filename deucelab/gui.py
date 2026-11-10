import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from tkinter import BooleanVar, Menu, TclError, filedialog, messagebox
from typing import Optional

import customtkinter as ctk

from deucelab import __version__
from deucelab import constants as c
from deucelab.history import BetHistoryWindow
from deucelab.logic import BetMetrics, BettingLogic
from deucelab.replay import encode_replay, write_replay
from deucelab.visualization import (
    BG,
    BORDER,
    FONT,
    MUTED,
    P1,
    P2,
    PANEL,
    SURFACE,
    TEXT,
    LiveAnalysis,
    VisualizationWindow,
)

APP_NAME = "DeuceLab"
logger = logging.getLogger(__name__)


class ToolTip(ctk.CTkToplevel):
    def __init__(self, widget, text="Widget info"):
        super().__init__(widget)
        self.widget = widget
        self.text = text
        self.withdraw()
        self.overrideredirect(True)
        self.label = ctk.CTkLabel(
            self,
            text=self.text if self.text else " ",
            justify="left",
            fg_color=("grey75", "grey25"),
            corner_radius=4,
            padx=5,
            pady=2,
            font=ctk.CTkFont(size=10),
        )
        self.label.pack()
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)
        self.widget.bind("<ButtonPress>", self.hidetip)

    def set_text(self, text):
        self.text = text
        self.label.configure(text=self.text if self.text else " ")

    def showtip(self, event=None):
        if not self.text or not self.text.strip():
            return
        if not self.widget.winfo_exists() or not self.widget.winfo_viewable():
            return
        try:
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 20
            self.geometry(f"+{x}+{y}")
            self.deiconify()
        except (TclError, TypeError):
            try:
                x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
                y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
                self.geometry(f"+{x}+{y}")
                self.deiconify()
            except TclError:
                logger.debug("Tooltip positioning failed", exc_info=True)

    def hidetip(self, event=None):
        self.withdraw()


class SettingsWindow(ctk.CTkToplevel):
    def __init__(
        self, master=None, logic_ref: BettingLogic | None = None, app_ref: Optional["App"] = None
    ):
        super().__init__(master)
        if logic_ref is None or app_ref is None:
            raise ValueError("Settings needs Logic and App refs")
        self.logic = logic_ref
        self.app = app_ref
        self.title("Settings - O/U Focus")
        self.geometry("450x300")
        self.transient(master)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.tooltips: list[ToolTip] = []
        row = 0
        ctk.CTkLabel(self, text="Filtering Thresholds", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, columnspan=3, pady=(10, 5), sticky="w", padx=10
        )
        row += 1
        params_base = {
            "min_prob_suggestion_base": "Min Prob",
            "max_prob_filter_base": "Max Prob",
            "min_estimated_odds_base": "Min Odds",
            "min_acceptable_ev_base": "Min EV",
        }
        for k, lbl in params_base.items():
            self._create_entry_row(k, lbl, row, c.TOOLTIPS.get(k, ""))
            row += 1
        bf = ctk.CTkFrame(self, fg_color="transparent")
        bf.grid(row=row, column=0, columnspan=3, pady=(15, 10))
        bf.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(bf, text="Save & Apply", command=self.save_settings).grid(
            row=0, column=0, padx=5
        )
        ctk.CTkButton(bf, text="Cancel", command=self.destroy, fg_color="grey").grid(
            row=0, column=1, padx=5
        )

    def _create_entry_row(self, cfg_key: str, lbl_txt: str, grid_row: int, tt_txt: str):
        lbl = ctk.CTkLabel(self, text=lbl_txt + ":")
        lbl.grid(row=grid_row, column=0, padx=(10, 5), pady=3, sticky="w")
        current_val = self.logic.config.get(cfg_key, 0.0)
        placeholder_format = "{:.2f}" if isinstance(current_val, (int, float)) else "{}"
        insert_format = "{:.3f}" if isinstance(current_val, (int, float)) else "{}"
        try:
            entry = ctk.CTkEntry(
                self, width=80, placeholder_text=placeholder_format.format(current_val)
            )
            entry.insert(0, insert_format.format(current_val))
        except (ValueError, TypeError):
            entry = ctk.CTkEntry(self, width=80, placeholder_text=str(current_val))
            entry.insert(0, str(current_val))

        entry.grid(row=grid_row, column=1, padx=(0, 5), pady=3, sticky="ew")
        self.entries[cfg_key] = entry
        info = ctk.CTkLabel(self, text="?", width=15, fg_color="gray", corner_radius=5)
        info.grid(row=grid_row, column=2, padx=(0, 10), pady=3, sticky="w")
        self.tooltips.append(ToolTip(info, tt_txt))

    def save_settings(self):
        values = {}
        for key, entry in self.entries.items():
            try:
                value = float(entry.get().replace(",", "."))
                if not math.isfinite(value) or not (
                    value > 1 if "odds" in key else 0 <= value <= 1
                ):
                    raise ValueError
            except ValueError:
                entry.focus_set()
                self.app.show_status(f"Invalid setting: {key}")
                return
            values[key] = value
        if values["min_prob_suggestion_base"] > values["max_prob_filter_base"]:
            self.app.show_status("Minimum probability must not exceed maximum probability.")
            return
        self.logic.update_config(values)
        self.app.update_suggestion_display(record=True)
        self.app.show_status("Settings applied.")
        self.destroy()


class App(ctk.CTk):
    def __init__(self, logic: BettingLogic):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        theme = ctk.ThemeManager.theme
        theme["CTkFrame"].update(fg_color=PANEL, top_fg_color=SURFACE, corner_radius=12)
        theme["CTkLabel"]["text_color"] = TEXT
        theme["CTkButton"].update(fg_color=BORDER, hover_color="#37516e", corner_radius=8)
        theme["CTkEntry"].update(fg_color=BG, border_color=BORDER, text_color=TEXT)
        theme["CTkProgressBar"].update(fg_color=BORDER, progress_color=P1)
        super().__init__(fg_color=BG)
        self.logic = logic
        self.replay_started = False
        self.replay_finished_at = None
        self.replay_invalidated = False
        self.replay_quote = None
        self.title(f"{APP_NAME} · Live workspace")
        self.geometry("1120x790")
        self.minsize(1000, 770)
        self.visualization_window: VisualizationWindow | None = None
        self.bet_history_window: BetHistoryWindow | None = None
        self.settings_window: SettingsWindow | None = None

        self.dyn_sugg1_tooltip: ToolTip | None = None
        self.elab_sugg_tooltip: ToolTip | None = None

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(12, 0))
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left")
        ctk.CTkLabel(brand, text=APP_NAME, font=(FONT, 27, "bold")).pack(anchor="w")
        ctk.CTkLabel(
            brand, text="TABLE TENNIS  /  LIVE WORKSPACE", text_color=MUTED, font=(FONT, 10, "bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text=f"{logic.forecast_model.upper()} MODEL   ·   v{__version__}",
            fg_color=SURFACE,
            corner_radius=8,
            text_color=MUTED,
            padx=12,
            pady=4,
            font=(FONT, 11, "bold"),
        ).pack(side="right")

        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(padx=18, pady=12, fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=5)
        main_frame.grid_columnconfigure(1, weight=4)
        main_frame.grid_rowconfigure(6, weight=1)

        self._build_match_panel(main_frame)
        self._build_market_panel(main_frame)
        sidebar = ctk.CTkFrame(main_frame, fg_color="transparent")
        sidebar.grid(row=0, column=1, rowspan=8, sticky="nsew", padx=(9, 0))
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(0, weight=1)
        self.analysis = LiveAnalysis(sidebar)
        self.analysis.grid(row=0, column=0, sticky="nsew")
        self._build_history_panel(sidebar)

        self.status_bar = ctk.CTkFrame(self, fg_color=SURFACE)
        self.status_bar.pack(side="bottom", fill="x", padx=18, pady=(0, 10), before=main_frame)
        self.status_label = ctk.CTkLabel(
            self.status_bar, text="Ready.", anchor="w", text_color=MUTED
        )
        self.status_label.pack(side="left", padx=12)
        ctk.CTkLabel(self.status_bar, text="Ctrl+1  P1   ·   Ctrl+2  P2", text_color=MUTED).pack(
            side="right", padx=12
        )
        self._build_menu()
        self.update_match_score_display()
        self.update_history_display()
        self.dyn_sugg1_tooltip = ToolTip(self.dyn_sugg1_label, "")
        self.elab_sugg_tooltip = ToolTip(self.elab_sugg_label, "")
        self.update_suggestion_display(record=True)

    def _build_menu(self):
        menu = Menu(self)
        match = Menu(menu, tearoff=False)
        match.add_command(
            label="Award point to P1", accelerator="Ctrl+1", command=lambda: self.update_score(1)
        )
        match.add_command(
            label="Award point to P2", accelerator="Ctrl+2", command=lambda: self.update_score(2)
        )
        match.add_command(label="Next set", command=self.start_next_set)
        match.add_separator()
        match.add_command(label="Export replay…", command=self.export_replay)
        match.add_command(label="Reset match…", command=self.confirm_reset)
        menu.add_cascade(label="Match", menu=match)
        view = Menu(menu, tearoff=False)
        view.add_command(label="Analysis window", command=self.toggle_visualization)
        view.add_command(label="Session history", command=self.toggle_bet_history)
        view.add_command(label="Forecast settings", command=self.open_settings)
        self.motion_enabled = BooleanVar(self, True)
        view.add_checkbutton(
            label="Motion effects", variable=self.motion_enabled, command=self._set_motion
        )
        menu.add_cascade(label="View", menu=view)
        self.configure(menu=menu)
        self.bind("<Control-Key-1>", lambda _: self.update_score(1))
        self.bind("<Control-Key-2>", lambda _: self.update_score(2))

    def _set_motion(self):
        self.analysis.motion = self.motion_enabled.get()
        if self.visualization_window is not None and self.visualization_window.winfo_exists():
            self.visualization_window.analysis.motion = self.analysis.motion
        self.update_visualization_data()

    def confirm_reset(self):
        if messagebox.askyesno("Reset match", "Clear the score and session history?", parent=self):
            self.reset_match()

    def _build_match_panel(self, main_frame: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(main_frame, fg_color="transparent")
        panel.grid(row=0, column=0, rowspan=7, sticky="nsew", padx=(0, 5))
        panel.grid_columnconfigure((0, 1, 2, 3), weight=1)
        panel.grid_rowconfigure(6, weight=1)

        prematch = ctk.CTkFrame(panel)
        prematch.grid(row=0, column=0, columnspan=4, padx=0, pady=(0, 5), sticky="ew")
        prematch.grid_columnconfigure((1, 2), weight=1)
        ctk.CTkLabel(prematch, text="Pre-match odds").grid(
            row=0, column=0, padx=(10, 5), pady=2, sticky="w"
        )
        self.odds1_entry = ctk.CTkEntry(prematch, width=50, placeholder_text="P1")
        self.odds1_entry.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.odds2_entry = ctk.CTkEntry(prematch, width=50, placeholder_text="P2")
        self.odds2_entry.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        self.apply_odds_button = ctk.CTkButton(
            prematch, text="Apply", width=60, command=self.apply_prematch_odds
        )
        self.apply_odds_button.grid(row=0, column=3, padx=(5, 10), pady=2, sticky="e")
        live_odds = ctk.CTkFrame(panel)
        live_odds.grid(row=1, column=0, columnspan=4, padx=0, pady=(0, 5), sticky="ew")
        live_odds.grid_columnconfigure((1, 2), weight=1)
        ctk.CTkLabel(live_odds, text="Live match odds").grid(
            row=0, column=0, padx=(10, 5), pady=2, sticky="w"
        )
        self.live_odds1_entry = ctk.CTkEntry(live_odds, width=50, placeholder_text="P1")
        self.live_odds1_entry.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.live_odds2_entry = ctk.CTkEntry(live_odds, width=50, placeholder_text="P2")
        self.live_odds2_entry.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        self.apply_live_odds_button = ctk.CTkButton(
            live_odds, text="Apply", width=60, command=self.apply_live_odds
        )
        self.apply_live_odds_button.grid(row=0, column=3, padx=(5, 0), pady=2, sticky="e")
        self.clear_live_odds_button = ctk.CTkButton(
            live_odds, text="X", width=25, command=self.clear_live_odds, fg_color="grey"
        )
        self.clear_live_odds_button.grid(row=0, column=4, padx=(2, 10), pady=2, sticky="e")
        score_frame = ctk.CTkFrame(panel, fg_color=SURFACE, border_width=1, border_color=BORDER)
        score_frame.grid(row=2, column=0, columnspan=4, padx=0, pady=5, sticky="ew")
        score_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.match_score_label = ctk.CTkLabel(
            score_frame, text="Sets: 0-0", font=ctk.CTkFont(size=14)
        )
        self.match_score_label.grid(row=0, column=0, columnspan=4, pady=(5, 0))
        ctk.CTkLabel(score_frame, text="PLAYER 1", text_color=P1, font=(FONT, 12, "bold")).grid(
            row=1, column=0, columnspan=2
        )
        ctk.CTkLabel(score_frame, text="PLAYER 2", text_color=P2, font=(FONT, 12, "bold")).grid(
            row=1, column=2, columnspan=2
        )
        self.p1_score_var = ctk.IntVar(value=0)
        self.p2_score_var = ctk.IntVar(value=0)
        self.p1_score_label = ctk.CTkLabel(
            score_frame, textvariable=self.p1_score_var, font=(FONT, 52, "bold"), text_color=P1
        )
        self.p1_score_label.grid(row=2, column=0, columnspan=2, pady=(0, 4))
        self.p2_score_label = ctk.CTkLabel(
            score_frame, textvariable=self.p2_score_var, font=(FONT, 52, "bold"), text_color=P2
        )
        self.p2_score_label.grid(row=2, column=2, columnspan=2, pady=(0, 4))
        ctk.CTkButton(
            score_frame,
            text="+  P1 point",
            width=95,
            height=36,
            fg_color=P1,
            text_color=BG,
            hover_color="#9ff4ee",
            command=lambda: self.update_score(1),
        ).grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            score_frame,
            text="−1 P1",
            width=64,
            command=lambda: self.update_score(1, delta=-1),
            fg_color=BORDER,
        ).grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            score_frame,
            text="+  P2 point",
            width=95,
            height=36,
            fg_color=P2,
            text_color=BG,
            hover_color="#ffdab7",
            command=lambda: self.update_score(2),
        ).grid(row=3, column=2, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            score_frame,
            text="−1 P2",
            width=64,
            command=lambda: self.update_score(2, delta=-1),
            fg_color=BORDER,
        ).grid(row=3, column=3, padx=5, pady=5, sticky="ew")
        serve_frame = ctk.CTkFrame(panel)
        serve_frame.grid(row=3, column=0, columnspan=4, padx=0, pady=5, sticky="ew")
        serve_frame.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(serve_frame, text="First server").grid(
            row=0, column=0, padx=(10, 5), pady=5, sticky="w"
        )
        self.initial_server_var = ctk.IntVar(value=1)
        ctk.CTkRadioButton(
            serve_frame,
            text="P1",
            variable=self.initial_server_var,
            value=1,
            command=self.update_initial_server,
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkRadioButton(
            serve_frame,
            text="P2",
            variable=self.initial_server_var,
            value=2,
            command=self.update_initial_server,
        ).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.current_server_label = ctk.CTkLabel(serve_frame, text="Current Server: P1", anchor="w")
        self.current_server_label.grid(row=0, column=3, padx=(10, 10), pady=5, sticky="ew")
        controls = ctk.CTkFrame(panel, fg_color="transparent")
        controls.grid(row=4, column=0, columnspan=4, padx=0, pady=(5, 0), sticky="ew")
        controls.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.next_set_button = ctk.CTkButton(
            controls,
            text="→  Next set",
            width=100,
            command=self.start_next_set,
            fg_color="#287570",
            hover_color="#338b84",
        )
        self.next_set_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.viz_button = ctk.CTkButton(
            controls, text="↗  Analysis", width=98, command=self.toggle_visualization
        )
        self.viz_button.grid(row=0, column=1, padx=2, sticky="ew")
        self.history_button = ctk.CTkButton(
            controls, text="≡  Session history", width=112, command=self.toggle_bet_history
        )
        self.history_button.grid(row=0, column=2, padx=2, sticky="ew")
        self.settings_button = ctk.CTkButton(
            controls, text="⚙  Settings", width=96, command=self.open_settings
        )
        self.settings_button.grid(row=0, column=3, padx=(5, 0), sticky="ew")

        progress_frame = ctk.CTkFrame(panel, fg_color="transparent")
        progress_frame.grid(row=5, column=0, columnspan=4, padx=0, pady=(5, 5), sticky="ew")
        progress_frame.grid_columnconfigure(1, weight=1)
        self.elaboration_progress_label = ctk.CTkLabel(
            progress_frame, text="Elaborated forecast: 0/X Pts", font=ctk.CTkFont(size=11)
        )
        self.elaboration_progress_label.grid(row=0, column=0, padx=(10, 5), sticky="w")
        self.elaboration_progressbar = ctk.CTkProgressBar(progress_frame, height=10)
        self.elaboration_progressbar.grid(row=0, column=1, padx=(0, 10), sticky="ew")
        self.elaboration_progressbar.set(0)

        estimates = ctk.CTkFrame(panel, fg_color="transparent")
        estimates.grid(row=6, column=0, columnspan=4, padx=0, pady=5, sticky="nsew")
        estimates.grid_columnconfigure(0, weight=1)
        estimates.grid_rowconfigure(0, weight=1)
        estimates.grid_rowconfigure(1, weight=1)
        current_estimate = ctk.CTkFrame(estimates)
        current_estimate.grid(row=0, column=0, padx=0, pady=(5, 2), sticky="nsew")
        current_estimate.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            current_estimate, text="Current O/U Estimate", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(5, 2), sticky="w")
        self.dyn_sugg1_label = ctk.CTkLabel(
            current_estimate,
            text="N/A",
            font=ctk.CTkFont(size=14),
            wraplength=460,
            justify="left",
            anchor="nw",
        )
        self.dyn_sugg1_label.grid(row=1, column=0, padx=10, pady=(0, 2), sticky="ew")
        self.dyn_sugg1_details = ctk.CTkLabel(
            current_estimate,
            text="",
            font=ctk.CTkFont(size=11),
            wraplength=460,
            justify="left",
            text_color="gray",
            anchor="nw",
        )
        self.dyn_sugg1_details.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew")

        self.elab_frame = ctk.CTkFrame(estimates, border_width=1, border_color=BORDER)
        self.elab_frame.grid(row=1, column=0, padx=0, pady=(5, 5), sticky="nsew")
        self.elab_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.elab_frame,
            text="Elaborated Estimate (snapshot)",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(5, 2))
        self.elab_sugg_label = ctk.CTkLabel(
            self.elab_frame,
            text="Calculating...",
            font=ctk.CTkFont(size=16),
            wraplength=460,
            justify="left",
            anchor="nw",
        )
        self.elab_sugg_label.grid(row=1, column=0, sticky="ew", padx=10)
        self.elab_sugg_details = ctk.CTkLabel(
            self.elab_frame,
            text="",
            font=ctk.CTkFont(size=12),
            wraplength=460,
            justify="left",
            text_color="gray",
            anchor="nw",
        )
        self.elab_sugg_details.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))

    def _build_history_panel(self, main_frame: ctk.CTkFrame) -> None:
        history_panel = ctk.CTkFrame(main_frame, height=172)
        history_panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        history_panel.grid_propagate(False)
        history_panel.grid_columnconfigure(0, weight=1)
        history_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(history_panel, text="Set log", font=(FONT, 15, "bold")).grid(
            row=0, column=0, pady=(10, 0)
        )
        self.history_scroll_frame = ctk.CTkScrollableFrame(history_panel, height=38)
        self.history_scroll_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.history_scroll_frame.grid_columnconfigure(0, weight=1)
        hist_if = ctk.CTkFrame(history_panel)
        hist_if.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 5))
        hist_if.grid_columnconfigure((1, 2), weight=1)
        hist_if.grid_columnconfigure(3, weight=0)
        ctk.CTkLabel(hist_if, text="Add Set:").grid(row=0, column=0, padx=(5, 2))
        self.past_set_p1_entry = ctk.CTkEntry(hist_if, placeholder_text="P1", width=40)
        self.past_set_p1_entry.grid(row=0, column=1, padx=(0, 5), sticky="ew")
        self.past_set_p2_entry = ctk.CTkEntry(hist_if, placeholder_text="P2", width=40)
        self.past_set_p2_entry.grid(row=0, column=2, padx=(0, 5), sticky="ew")
        self.add_set_button = ctk.CTkButton(
            hist_if, text="+", width=30, command=self.add_past_set_manual
        )
        self.add_set_button.grid(row=0, column=3, padx=(0, 5), sticky="e")
        clear_frame = ctk.CTkFrame(history_panel, fg_color="transparent")
        clear_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=(0, 10))
        clear_frame.grid_columnconfigure((0, 1), weight=1)
        self.clear_history_button = ctk.CTkButton(
            clear_frame, text="Clear sets", command=self.clear_match_history
        )
        self.clear_history_button.grid(row=0, column=0, padx=(0, 2), sticky="ew")
        self.reset_match_button = ctk.CTkButton(
            clear_frame,
            text="Reset Match",
            fg_color="#502b38",
            hover_color="#693647",
            command=self.confirm_reset,
        )
        self.reset_match_button.grid(row=0, column=1, padx=(2, 0), sticky="ew")

    def _build_market_panel(self, main_frame: ctk.CTkFrame) -> None:
        market = ctk.CTkFrame(main_frame)
        market.grid(row=7, column=0, sticky="ew", pady=(5, 0), padx=(0, 5))
        self.market_line = ctk.CTkOptionMenu(
            market,
            values=[str(line) for line in c.STANDARD_OU_LINES],
            width=72,
        )
        self.market_line.set("18.5")
        self.market_line.pack(side="left", padx=5, pady=5)
        self.over_price = ctk.CTkEntry(market, placeholder_text="Over odds", width=90)
        self.over_price.pack(side="left", padx=3)
        self.under_price = ctk.CTkEntry(market, placeholder_text="Under odds", width=90)
        self.under_price.pack(side="left", padx=3)
        self.market_button = ctk.CTkButton(
            market,
            text="Apply O/U odds",
            command=self.apply_market,
            width=105,
        )
        self.market_button.pack(side="left", padx=3)
        ctk.CTkButton(market, text="Clear", command=self.clear_market, width=50).pack(
            side="left", padx=3
        )
        ctk.CTkButton(market, text="Export replay", command=self.export_replay, width=95).pack(
            side="left", padx=3
        )

    def apply_prematch_odds(self):
        try:
            o1 = float(self.odds1_entry.get().replace(",", "."))
        except ValueError:
            self.show_status("Error: Invalid Base P1 Odds")
            return
        try:
            o2 = float(self.odds2_entry.get().replace(",", "."))
        except ValueError:
            self.show_status("Error: Invalid Base P2 Odds")
            return
        if self.replay_started:
            self.show_status("Pre-match odds can only be entered before the first rally.")
            return
        if self.logic.set_prematch_odds(o1, o2):
            self.update_suggestion_display(record=True)
            self.show_status("Base odds applied.")
        else:
            self.show_status("Error applying base odds.")

    def apply_live_odds(self):
        try:
            o1 = float(self.live_odds1_entry.get().replace(",", "."))
        except ValueError:
            self.show_status("Error: Invalid Live P1 Odds")
            return
        try:
            o2 = float(self.live_odds2_entry.get().replace(",", "."))
        except ValueError:
            self.show_status("Error: Invalid Live P2 Odds")
            return
        if self.logic.set_live_odds(o1, o2):
            self.update_suggestion_display(record=True)
            self.show_status("Live odds applied.")
        else:
            self.show_status("Error applying live odds.")

    def clear_live_odds(self):
        self.live_odds1_entry.delete(0, "end")
        self.live_odds2_entry.delete(0, "end")
        self.logic.clear_live_odds()
        self.update_suggestion_display(record=True)
        self.show_status("Live odds cleared.")

    def apply_market(self):
        try:
            prices = [
                float(entry.get().replace(",", ".")) if entry.get().strip() else None
                for entry in (self.over_price, self.under_price)
            ]
            self.logic.set_market(float(self.market_line.get()), *prices)
        except ValueError:
            self.show_status("Enter Over and/or Under odds greater than 1.")
            return
        if (
            not self.replay_invalidated
            and self.replay_quote is None
            and len(self.logic.point_history) == 8
            and all(price is not None for price in prices)
        ):
            self.replay_quote = {
                "line": self.logic.market_line,
                "over": prices[0],
                "under": prices[1],
                "quote_at": datetime.now(UTC).isoformat(),
            }
        self.update_suggestion_display(record=True)
        self.show_status(f"O/U odds applied for {self.market_line.get()}.")

    def export_replay(self):
        if (
            not self.logic.is_finished(self.logic.score_p1, self.logic.score_p2)
            or not self.replay_quote
        ):
            self.show_status("Finish a set with both O/U prices recorded at eight points.")
            return
        payload = {
            **self.replay_quote,
            "match_id": self.logic.match_id,
            "set_number": self.logic.set_id + 1,
            "after_points": 8,
            "settled_at": self.replay_finished_at,
            "first_server": self.logic.initial_server_current_set,
            "rallies": [point["winner"] for point in self.logic.point_history],
            "prematch": self.logic.prematch_odds,
        }
        try:
            row = encode_replay(payload, 8)
            destination = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".jsonl",
                filetypes=[("JSON Lines", "*.jsonl")],
                initialfile=f"{self.logic.match_id}-set-{self.logic.set_id + 1}.jsonl",
            )
            if not destination:
                return
            write_replay(Path(destination), row)
        except (OSError, ValueError) as error:
            self.show_status(f"Replay export failed: {error}", 5000)
            return
        self.show_status("Replay exported.")

    def clear_market(self):
        self.over_price.delete(0, "end")
        self.under_price.delete(0, "end")
        self.logic.set_market(None)
        self.update_suggestion_display(record=True)

    def update_score(self, pnum: int, delta: int = 1):
        if delta < 0:
            if not self.logic.revise_points(self.logic.initial_server_current_set, pnum):
                self.show_status("Cannot remove that point. Try undoing the last point first.")
                return
            self.replay_quote = None
            self.replay_invalidated = True
        elif not self.logic.record_point(pnum):
            self.show_status("Set complete. Correct the score or start the next set.")
            return
        else:
            self.replay_started = True
            if self.logic.is_finished(self.logic.score_p1, self.logic.score_p2):
                self.replay_finished_at = datetime.now(UTC).isoformat()
        self._clear_quote_inputs()
        self.p1_score_var.set(self.logic.score_p1)
        self.p2_score_var.set(self.logic.score_p2)
        self.update_current_server_display(self.logic.server)
        self.update_suggestion_display(record=True)
        self.show_status(f"Score updated: {self.logic.score_p1}-{self.logic.score_p2}")

    def _clear_quote_inputs(self) -> None:
        for entry in (
            self.over_price,
            self.under_price,
            self.live_odds1_entry,
            self.live_odds2_entry,
        ):
            entry.delete(0, "end")

    def update_initial_server(self):
        initial_server = self.initial_server_var.get()
        if initial_server != self.logic.initial_server_current_set:
            self.logic.revise_points(initial_server)
            self._clear_quote_inputs()
            if self.logic.point_history:
                self.replay_invalidated = True
                self.replay_quote = None
        self.update_current_server_display(self.logic.server)
        self.update_suggestion_display(record=True)

    def update_current_server_display(self, srv: int):
        self.current_server_label.configure(text=f"Current Server: P{srv}")

    def start_next_set(self):
        if not self.logic.is_finished(self.logic.score_p1, self.logic.score_p2):
            self.show_status("Complete the current set before starting the next one.")
            return
        next_server = 3 - self.logic.initial_server_current_set
        self.logic.start_new_set(next_server)
        self.replay_finished_at = None
        self.replay_quote = None
        self.replay_invalidated = False
        self.initial_server_var.set(next_server)
        self._clear_quote_inputs()

        self.p1_score_var.set(self.logic.score_p1)
        self.p2_score_var.set(self.logic.score_p2)
        self.update_match_score_display()
        self.update_history_display()
        self.update_current_server_display(self.logic.server)
        self.update_suggestion_display(record=True)
        self.show_status(f"Starting Set {len(self.logic.past_sets) + 1}.")

    def add_past_set_manual(self):
        try:
            p1 = int(self.past_set_p1_entry.get())
            p2 = int(self.past_set_p2_entry.get())
        except ValueError:
            self.show_status("Score must be numeric.")
            return
        if not self.logic.add_past_set(p1, p2):
            self.show_status("Invalid past set score.")
            return
        self.logic.refresh_elaboration()
        self.past_set_p1_entry.delete(0, "end")
        self.past_set_p2_entry.delete(0, "end")
        self.update_history_display()
        self.update_match_score_display()
        self.update_suggestion_display(record=True)
        self.show_status("Manual past set added.")

    def delete_past_set(self, idx: int):
        if not self.logic.delete_past_set(idx):
            self.show_status(f"Invalid set index: {idx}.")
            return
        self.update_history_display()
        self.update_match_score_display()
        self.update_suggestion_display(record=True)
        self.show_status(f"Deleted Set {idx + 1}.")

    def clear_match_history(self):
        self.logic.clear_past_sets()
        self.update_history_display()
        self.update_match_score_display()
        self.update_suggestion_display(record=True)
        self.show_status("Match history cleared.")

    def reset_match(self):
        self.logic.reset_match()
        self.replay_started = False
        self.replay_finished_at = None
        self.replay_invalidated = False
        self.replay_quote = None
        self._clear_quote_inputs()
        for entry in (
            self.odds1_entry,
            self.odds2_entry,
            self.past_set_p1_entry,
            self.past_set_p2_entry,
        ):
            entry.delete(0, "end")
        self.p1_score_var.set(0)
        self.p2_score_var.set(0)
        self.initial_server_var.set(1)
        self.update_match_score_display()
        self.update_history_display()
        self.update_current_server_display(self.logic.server)
        self.update_suggestion_display(record=True)
        self.show_status("New match ready.")
        if self.bet_history_window and self.bet_history_window.winfo_exists():
            self.bet_history_window.populate_history()

    def update_history_display(self):
        for w in self.history_scroll_frame.winfo_children():
            w.destroy()
        if not self.logic.past_sets:
            ctk.CTkLabel(
                self.history_scroll_frame, text="No completed sets", text_color=MUTED, height=24
            ).pack()
        else:
            for i, (p1, p2) in enumerate(self.logic.past_sets):
                rf = ctk.CTkFrame(self.history_scroll_frame, fg_color="transparent")
                rf.pack(fill="x", pady=1)
                rf.grid_columnconfigure(0, weight=1)
                rf.grid_columnconfigure(1, weight=0)
                lbl = f"Set {i + 1}: {p1}-{p2}"
                sl = ctk.CTkLabel(rf, text=lbl, anchor="w")
                sl.grid(row=0, column=0, padx=(5, 0), sticky="ew")
                db = ctk.CTkButton(
                    rf,
                    text="X",
                    width=25,
                    fg_color="grey",
                    command=lambda idx=i: self.delete_past_set(idx),
                )
                db.grid(row=0, column=1, padx=(0, 5))

    def update_match_score_display(self):
        self.match_score_label.configure(
            text=f"Sets: {self.logic.p1_sets_won} - {self.logic.p2_sets_won}"
        )

    def toggle_visualization(self):
        if self.visualization_window is None or not self.visualization_window.winfo_exists():
            self.visualization_window = VisualizationWindow(self)
            self.visualization_window.analysis.motion = self.analysis.motion
            self.visualization_window.show()
            self.update_visualization_data()
        elif self.visualization_window.state() == "withdrawn":
            self.visualization_window.show()
            self.update_visualization_data()
        else:
            self.visualization_window.withdraw()

    def toggle_bet_history(self):
        if self.bet_history_window is None or not self.bet_history_window.winfo_exists():
            self.bet_history_window = BetHistoryWindow(self, self.logic)
            self.bet_history_window.lift()
            self.bet_history_window.focus_force()
        elif self.bet_history_window.state() == "withdrawn":
            self.bet_history_window.deiconify()
            self.bet_history_window.populate_history()
            self.bet_history_window.lift()
            self.bet_history_window.focus_force()
        else:
            self.bet_history_window.withdraw()

    def open_settings(self):
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.logic, self)
        self.settings_window.lift()
        self.settings_window.focus_force()

    def update_visualization_data(self):
        viz_data = self.logic.get_visualization_data()
        viz_data["finished"] = self.logic.is_finished(self.logic.score_p1, self.logic.score_p2)
        self.analysis.update_plots_and_summary(viz_data)
        if self.visualization_window and self.visualization_window.state() == "normal":
            self.visualization_window.update_plots_and_summary(viz_data)

    def show_status(self, message: str, duration: int = 3000):
        self.status_label.configure(text=message)
        if hasattr(self, "_status_clear_timer"):
            self.status_label.after_cancel(self._status_clear_timer)
        self._status_clear_timer = self.status_label.after(
            duration,
            lambda: (
                self.status_label.configure(text="Ready.")
                if self.status_label.cget("text") == message
                else None
            ),
        )

    @staticmethod
    def format_estimate(estimate: BetMetrics) -> str:
        fair_odds = estimate["odds"]
        fair = f"{fair_odds:.2f}" if fair_odds is not None else "N/A"
        details = f"P: {estimate['prob']:.1%} | Fair: {fair}"
        if estimate["ev"] is None:
            return f"{details} | EV: awaiting market odds"
        market_probability = estimate["market_probability"]
        if market_probability is not None:
            details += f" | No-margin P: {market_probability:.1%}"
        threshold = "signal" if estimate["meets_filters"] else "outside filters"
        return (
            f"{details} | Market: {estimate['market_odds']:.2f} | "
            f"EV: {estimate['ev']:+.1%} ({threshold})"
        )

    def update_suggestion_display(self, *, record: bool = False) -> None:
        try:
            dynamic_sugg_ou, elaborated_sugg_ou, elaboration_prog = self.logic.get_suggestions(
                record=record
            )
        except Exception:
            logger.exception("Could not calculate forecasts")
            dynamic_sugg_ou, elaborated_sugg_ou, elaboration_prog = (
                None,
                None,
                self.logic.elaboration_progress,
            )
            self.show_status("Error calculating suggestions!", 5000)

        req_pts = self.logic.required_points_for_elaboration
        self.elaboration_progress_label.configure(
            text="Elaborated forecast ready"
            if self.logic.elaborated_bet_ready
            else f"Elaborated forecast · {len(self.logic.point_history)}/{req_pts} rallies"
        )
        self.elaboration_progressbar.set(elaboration_prog / 100.0)

        default_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        value_color = P1

        if dynamic_sugg_ou and dynamic_sugg_ou.get("type") != "info":
            reason = dynamic_sugg_ou["reason"]
            self.dyn_sugg1_label.configure(text=dynamic_sugg_ou["label"], text_color=value_color)
            self.dyn_sugg1_details.configure(
                text=self.format_estimate(dynamic_sugg_ou), text_color=default_color
            )

            if self.dyn_sugg1_tooltip:
                self.dyn_sugg1_tooltip.set_text(f"Reason: {reason}")

        elif dynamic_sugg_ou and dynamic_sugg_ou.get("type") == "info":
            self.dyn_sugg1_label.configure(
                text=dynamic_sugg_ou.get("label", "Set Finished"), text_color="cyan"
            )
            self.dyn_sugg1_details.configure(text="The set is complete.", text_color=default_color)
            if self.dyn_sugg1_tooltip:
                self.dyn_sugg1_tooltip.set_text("")
        else:
            self.dyn_sugg1_label.configure(text="Dynamic O/U: N/A", text_color=default_color)
            self.dyn_sugg1_details.configure(text="", text_color=default_color)
            if self.dyn_sugg1_tooltip:
                self.dyn_sugg1_tooltip.set_text("")

        if self.logic.elaborated_bet_ready:
            if elaborated_sugg_ou:
                reason_e = elaborated_sugg_ou["reason"]
                self.elab_sugg_label.configure(
                    text=elaborated_sugg_ou["label"], text_color=value_color
                )
                self.elab_sugg_details.configure(
                    text=self.format_estimate(elaborated_sugg_ou), text_color=default_color
                )

                if self.elab_sugg_tooltip:
                    self.elab_sugg_tooltip.set_text(f"Reason: {reason_e}")

            else:
                self.elab_sugg_label.configure(
                    text="No elaborated O/U bet.", text_color=default_color
                )
                self.elab_sugg_details.configure(text="Criteria not met.", text_color=default_color)
                if self.elab_sugg_tooltip:
                    self.elab_sugg_tooltip.set_text("")

        else:
            self.elab_sugg_label.configure(text="Waiting for rally data", text_color=default_color)
            self.elab_sugg_details.configure(
                text=f"Requires {req_pts} points (adaptive).", text_color=default_color
            )
            if self.elab_sugg_tooltip:
                self.elab_sugg_tooltip.set_text("")

        if self.bet_history_window and self.bet_history_window.winfo_viewable():
            self.bet_history_window.refresh_history()
        self.update_visualization_data()
