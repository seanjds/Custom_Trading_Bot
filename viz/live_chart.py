# viz/live_chart.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib
# Try a GUI backend first; fall back cleanly if unavailable
try:
    matplotlib.use("TkAgg")          # or "Qt5Agg" if you have PyQt5 installed
except Exception:
    pass

class LiveChart:
    """
    Live, non-blocking chart with:
      - Top: Close + BB(upper/middle/lower) + entry/exit markers
      - Bottom: RSI with 70/50/30 guides
    Call update(df, sig, trades) each time you close a bar.
    """
    def __init__(self, symbol: str, max_bars: int = 1200):
        self.symbol = symbol
        self.max_bars = max_bars

        plt.ion()
        self.fig = plt.figure(figsize=(12, 7))
        self.ax_price = self.fig.add_subplot(2, 1, 1)
        self.ax_rsi   = self.fig.add_subplot(2, 1, 2, sharex=self.ax_price)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)         # <= ensure the window actually appears

        # line placeholders
        (self.line_close,)  = self.ax_price.plot([], [], linewidth=1.2, label="Close")
        (self.line_bb_u,)   = self.ax_price.plot([], [], linewidth=0.9, label="BB Upper")
        (self.line_bb_m,)   = self.ax_price.plot([], [], linewidth=0.9, linestyle="--", label="BB Middle")
        (self.line_bb_l,)   = self.ax_price.plot([], [], linewidth=0.9, label="BB Lower")

        (self.line_rsi,)    = self.ax_rsi.plot([], [], linewidth=1.0, label="RSI")

        # entry/exit scatter—will be redrawn each update
        self.entry_scatter = None
        self.exit_scatter  = None

        # axes cosmetics
        self.ax_price.set_ylabel("Price")
        self.ax_price.set_title(f"{self.symbol} — Live Price / BBands")
        self.ax_price.grid(True, alpha=0.2)
        self.ax_price.legend(loc="best")

        for h in (70, 50, 30):
            self.ax_rsi.axhline(h, linewidth=0.8)
        self.ax_rsi.set_ylabel("RSI")
        self.ax_rsi.set_xlabel("Time")
        self.ax_rsi.grid(True, alpha=0.2)
        self.ax_rsi.legend(loc="best")

        # time formatter
        self.ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, df: pd.DataFrame, sig: pd.DataFrame, trades: list):
        if df is None or df.empty:
            return

        # restrict for readability
        plot_df = df.tail(self.max_bars).copy()
        plot_sig = sig.tail(self.max_bars).copy()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)              # <= yield to the GUI event loop


        t = pd.to_datetime(plot_df["time"], utc=True)

        # Convert datetime series → array of matplotlib float dates
        x = mdates.date2num(t.dt.to_pydatetime())

        # ensure required columns exist
        def col(name): return plot_sig[name] if name in plot_sig.columns else pd.Series([float("nan")] * len(plot_sig))

        # update lines
        self.line_close.set_data(x, plot_df["close"].values)
        self.line_bb_u.set_data(x, col("bb_upper").values)
        self.line_bb_m.set_data(x, col("bb_middle").values)
        self.line_bb_l.set_data(x, col("bb_lower").values)

        # RSI panel
        self.line_rsi.set_data(x, col("rsi").values)

        # rescale axes
        self.ax_price.relim(); self.ax_price.autoscale_view()
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.set_xlim(x[0], x[-1] if len(x) else 1)

        # redraw entry/exit markers
        if self.entry_scatter: self.entry_scatter.remove()
        if self.exit_scatter:  self.exit_scatter.remove()

        if trades:
            # only draw markers that fall within current window
            t0, t1 = t.iloc[0], t.iloc[-1]
            e_x, e_y, s_x, s_y = [], [], [], []
            for tr in trades:
                tt = pd.to_datetime(tr["time"], utc=True)
                if tt < t0 or tt > t1:
                    continue
                if tr["side"].lower() == "buy":
                    e_x.append(mdates.date2num(tt.to_pydatetime())); e_y.append(float(tr["price"]))
                else:
                    s_x.append(mdates.date2num(tt.to_pydatetime())); s_y.append(float(tr["price"]))
            if e_x:
                self.entry_scatter = self.ax_price.scatter(e_x, e_y, marker="^", s=60, label="Entry")
            if s_x:
                self.exit_scatter  = self.ax_price.scatter(s_x, s_y, marker="v", s=60, label="Exit")

        # draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()