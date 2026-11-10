import tkinter as tk
from typing import Any

import customtkinter as ctk

BG = "#0b111b"
PANEL = "#121c2b"
SURFACE = "#182538"
BORDER = "#26374c"
TEXT = "#f0f5ff"
MUTED = "#a9b9cf"
P1 = "#65e5dd"
P2 = "#ffbd80"
FONT = "Segoe UI"


class LiveChart(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, bg=PANEL, highlightthickness=0, height=200, width=380)
        self.series = []
        self.percent = True
        self.bind("<Configure>", self.draw)
        self.bind("<Motion>", self.inspect)
        self.bind("<Leave>", lambda _: self.delete("cursor"))

    def set_series(self, series, percent):
        self.series = series
        self.percent = percent
        self.draw()

    def bounds(self):
        left, top = 44, 12
        right = max(left + 1, self.winfo_width() - 14)
        bottom = max(top + 1, self.winfo_height() - 30)
        count = max((len(values) for values, _, _ in self.series), default=1)
        limit = 1 if self.percent else max([11, *(max(s[0], default=0) for s in self.series)])
        return left, top, right, bottom, max(1, count - 1), limit

    def draw(self, _event=None):
        self.delete("all")
        left, top, right, bottom, steps, limit = self.bounds()
        if right - left < 40 or bottom - top < 40:
            return
        ticks = (0, 0.25, 0.5, 0.75, 1) if self.percent else (0, 0.5, 1)
        for fraction in ticks:
            y = bottom - fraction * (bottom - top)
            label = f"{fraction:.0%}" if self.percent else f"{fraction * limit:.0f}"
            self.create_line(left, y, right, y, fill=BORDER)
            self.create_text(left - 8, y, text=label, fill=MUTED, anchor="e", font=(FONT, 9))
        for step in sorted({round(steps * fraction / 4) for fraction in range(5)}):
            x = left + step * (right - left) / steps
            self.create_text(x, bottom + 14, text=str(step), fill=MUTED, font=(FONT, 9))
        for values, color, _ in self.series:
            coords = [
                (left + index * (right - left) / steps, bottom - value / limit * (bottom - top))
                for index, value in enumerate(values)
                if value is not None
            ]
            if len(coords) > 1:
                self.create_line(coords, fill=color, width=2, capstyle="round", joinstyle="round")
            if coords:
                x, y = coords[-1]
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

    def inspect(self, event):
        self.delete("cursor")
        left, top, right, bottom, steps, _ = self.bounds()
        if not self.series or not (left <= event.x <= right and top <= event.y <= bottom):
            return
        index = round((event.x - left) * steps / (right - left))
        values = [
            f"{name} {value:.1%}" if self.percent else f"{name} {value}"
            for series, _, name in self.series
            if index < len(series)
            for value in (series[index],)
            if value is not None
        ]
        if not values:
            return
        x = left + index * (right - left) / steps
        self.create_line(x, top, x, bottom, fill=MUTED, dash=(2, 3), tags="cursor")
        self.create_text(
            right - 4,
            top + 8,
            text=f"After {index} rallies  /  " + "   ".join(values),
            fill=TEXT,
            anchor="ne",
            font=(FONT, 9, "bold"),
            tags="cursor",
        )


class LiveAnalysis(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(
            master, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.data: dict[str, Any] = {}
        self._draw_job = None
        self._animation = None
        self.motion = True
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 0))
        ctk.CTkLabel(header, text="Rally analysis", font=(FONT, 20, "bold")).pack(side="left")
        self.live_label = ctk.CTkLabel(
            header, text="●  LIVE", text_color=P1, font=(FONT, 11, "bold")
        )
        self.live_label.pack(side="right")

        self.readout = ctk.CTkLabel(self, text="", font=(FONT, 18, "bold"), anchor="w")
        self.readout.grid(row=1, column=0, sticky="ew", padx=16, pady=(2, 0))
        self.bar = ctk.CTkProgressBar(self, height=5, fg_color=P2, progress_color=P1)
        self.bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 10))
        self.bar.set(0.5)

        self.selector = ctk.CTkSegmentedButton(
            self,
            values=["Probability", "Score", "Details"],
            command=self._select,
            fg_color=SURFACE,
            selected_color=BORDER,
            selected_hover_color=BORDER,
            unselected_color=SURFACE,
            unselected_hover_color=BORDER,
        )
        self.selector.set("Probability")
        self.selector.grid(row=3, column=0, sticky="ew", padx=16)

        self.chart = LiveChart(self)
        self.chart.grid(row=4, column=0, sticky="nsew", padx=12, pady=(6, 0))
        self.details = ctk.CTkLabel(self, text="", anchor="nw", justify="left", font=(FONT, 13))
        self.details.grid(row=4, column=0, sticky="nsew", padx=18, pady=10)
        self.details.grid_remove()
        self.caption = ctk.CTkLabel(self, text="", text_color=MUTED, font=(FONT, 11), anchor="w")
        self.caption.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 8))

    def _select(self, _value):
        self.render_chart()

    def update_plots_and_summary(self, data: dict[str, Any]):
        self.data = data
        if self._draw_job is None:
            self._draw_job = self.after_idle(self._render)

    def _render(self):
        self._draw_job = None
        probability = self.data["current_p1_point_win_prob"]
        finished = self.data["finished"]
        self.live_label.configure(text="SET COMPLETE" if finished else "●  LIVE")
        self.readout.configure(
            text="Final score recorded"
            if finished
            else f"P1  {probability:.1%}    /    P2  {1 - probability:.1%}"
        )
        if finished:
            self.bar.grid_remove()
        else:
            self.bar.grid()
            self._animate(probability)
        self.render_chart()

    def render_chart(self):
        if not self.data:
            return
        data = self.data
        view = self.selector.get()
        if view == "Details":
            self.chart.grid_remove()
            self.details.grid()
            self.details.configure(text=self._format_details(data))
            self.caption.configure(
                text="Neutral probabilities refer to a point without a serve advantage."
            )
            return
        self.details.grid_remove()
        self.chart.grid()
        if view == "Score":
            series = [(data["p1_score_history"], P1, "P1"), (data["p2_score_history"], P2, "P2")]
            caption = "Score after each rally  ·  P1 teal  /  P2 amber"
        else:
            history = data["point_probability_history"]
            if data["finished"]:
                history = history[:-1]
            elif not history:
                history = [data["current_p1_point_win_prob"]]
            else:
                history = [*history[:-1], data["current_p1_point_win_prob"]]
            series = (
                [([None, *history], P1, "P1")]
                if len(data["points_played_in_set"]) > 1
                else [(history, P1, "P1")]
            )
            caption = "P1 next-point estimate  ·  Hover to inspect a rally"
        self.chart.set_series(series, view == "Probability")
        self.caption.configure(text=caption)

    @staticmethod
    def _format_details(data):
        def metric(key, spec):
            value = data[key]
            return format(value, spec) if isinstance(value, (int, float)) else "—"

        return "\n".join(
            (
                f"P1 neutral   {metric('p1_neutral_point_win_prob_current', '.1%')}",
                f"Base   {metric('p1_neutral_point_win_prob_base', '.1%')}    /    Set-adjusted   {metric('p1_neutral_point_win_prob_history_adjusted', '.1%')}",
                f"Serve points won   P1 {metric('p1_serve_hold_pct', '.1%')}    /    P2 {metric('p2_serve_hold_pct', '.1%')}",
                f"Momentum   {metric('momentum_score', '+.2f')}    /    Game state   {metric('game_state_score', '+.2f')}",
                f"Serve factor   {metric('serve_perf_score', '+.2f')}    /    Odds factor   {metric('scaled_perf_vs_odds', '+.2f')}",
                f"Set difference   {metric('performance_delta_sets', '+.1f')}    /    Point difference   {metric('avg_point_diff', '+.1f')}",
            )
        )

    def _animate(self, target):
        if self._animation is not None:
            self.after_cancel(self._animation)
            self._animation = None
        start = self.bar.get()
        if not self.motion or abs(target - start) < 0.001:
            self.bar.set(target)
            return

        def step(index):
            self.bar.set(start + (target - start) * (1 - (1 - index / 8) ** 3))
            self._animation = self.after(15, step, index + 1) if index < 8 else None

        step(1)

    def destroy(self):
        for job in (self._draw_job, self._animation):
            if job is not None:
                self.after_cancel(job)
        super().destroy()


class VisualizationWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("DeuceLab · Rally analysis")
        self.geometry("570x440")
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.analysis = LiveAnalysis(self)
        self.analysis.pack(fill="both", expand=True, padx=12, pady=12)
        self.withdraw()

    def show(self):
        self.deiconify()
        self.lift()

    def update_plots_and_summary(self, data):
        self.analysis.update_plots_and_summary(data)
