# live/live_loop_poll.py
import time
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import SYMBOL, DRY_RUN, POLL_INTERVAL
from live.broker import Broker
from core.trade_logger import TradeLogger
from strategies.macd_bb_rsi import generate_signals
from data.fetcher import init_exchange, fetch_ohlcv  # warm-up full fetch

from viz.live_chart import LiveChart
MAX_BARS_ON_CHART = 1200

# =========================
# Tunable knobs
# =========================
TIMEFRAME = "1m"               # polling timeframe
WARMUP_BARS = 1500             # initial bars to load
HISTORY_BARS = 3000            # keep last N bars in memory (>= WARMUP_BARS)
ENTRY_COOLDOWN = 3             # bars after entry/exit before next action

# Risk & sizing (quote-currency)
USE_CONSTANT_RISK = True
RISK_PER_TRADE_PCT = 0.005     # 0.5% of free quote balance per entry / scale
STOP_LOSS_PCT = 0.006          # 0.6% below entry ⇒ stop
TAKE_PROFIT_PCT = 0.010        # 1.0% above entry ⇒ take-profit
TRAILING_STOP_PCT = 0.006      # trailing from peak; None to disable
FEE_BPS = 5.0                  # optional: 0.05% per side for cash P&L

# Scale-in on Bollinger mid-touch
SCALE_COOLDOWN = 8             # bars between scale-ins
MAX_SCALES_PER_POSITION = 3    # None for unlimited
SCALE_RISK_MULT = 1.0          # scale-in risk relative to entry (1.0 = same)
MID_TOLERANCE_BPS = 5.0        # treat |close-mid|/mid <= 5 bps as a touch

# Session filter (entries only; exits allowed anytime)
SESSION_FILTER = True
SESSION_TZ = "America/New_York"
SESSION_START_H = 9            # 9:00 ET
SESSION_END_H = 16             # 16:00 ET
WEEKDAYS_ONLY = True

# Summary dashboard cadence
SUMMARY_EVERY_N = 5            # print summary every N closed bars

# Incremental fetch knobs
INCR_OVERLAP_BARS = 3          # re-fetch last N bars to avoid gaps
EX_LIMIT_SOFT = 300            # safety cap per request (Coinbase ~350)
BACKOFF_SLEEP = 3              # seconds to sleep on transient errors

# ANSI colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
RESET  = "\033[0m"


# =========================
# Helpers
# =========================
def base_quote(symbol: str) -> Tuple[str, str]:
    base, quote = symbol.split("/")
    return base, quote

def timeframe_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("ms"): return int(tf[:-2])
    if tf.endswith("s"):  return int(tf[:-1]) * 1000
    if tf.endswith("m"):  return int(tf[:-1]) * 60_000
    if tf.endswith("h"):  return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):  return int(tf[:-1]) * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")

def in_session(t, tz: str, start_h: int, end_h: int, weekdays_only: bool) -> bool:
    ts = pd.to_datetime(t, unit="ms", utc=True) if isinstance(t, (int, float, np.integer, np.floating)) else pd.to_datetime(t, utc=True)
    local = ts.tz_convert(tz)
    if weekdays_only and local.weekday() >= 5:  # 5=Sat,6=Sun
        return False
    return start_h <= local.hour < end_h

def mid_touch(prev_close: float, prev_mid: float, close_px: float, mid: float, tol_bps: float) -> bool:
    if any(map(lambda x: x is None or np.isnan(x), (prev_close, prev_mid, close_px, mid))):
        return False
    eps = tol_bps / 1e4
    close_is_near = (mid != 0) and (abs(close_px - mid) / mid <= eps)
    crossed_mid = (prev_close - prev_mid) * (close_px - mid) <= 0
    return close_is_near or crossed_mid


# =========================
# Live (REST polling) runner with incremental fetching
# =========================
class RestLive:
    def __init__(self, symbol: str, timeframe: str = TIMEFRAME, poll_interval: int = POLL_INTERVAL):
        self.symbol = symbol
        self.base, self.quote = base_quote(symbol)
        self.timeframe = timeframe
        self.poll_interval = poll_interval
        self.tf_ms = timeframe_to_ms(timeframe)

        self.ex = init_exchange("coinbase")
        self.broker = Broker("coinbase", dry_run=DRY_RUN)
        self.logger = TradeLogger("logs/trade_log.csv")

        self.chart = LiveChart(symbol=self.symbol, max_bars=MAX_BARS_ON_CHART)
        self.trades = []   # keep the in-memory trade list for markers

        # Position / risk state
        self.in_position = False
        self.position_qty = 0.0
        self.entry_px: Optional[float] = None
        self.avg_entry_px: Optional[float] = None
        self.peak_px: Optional[float] = None

        # Cooldowns
        self.cooldown = 0
        self.scale_cd = 0
        self.scales_in_pos = 0

        # Dashboard / P&L state (quote currency)
        self.cash_quote = float(self._free_quote_balance())  # DRY uses default 10k
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.bar_counter = 0

        # Data
        self.df: pd.DataFrame = pd.DataFrame(columns=["timestamp","open","high","low","close","volume","time"])
        self.last_bar_time = None  # ms timestamp of last seen closed bar

    def run(self):
        # Warm-up
        self._warmup(WARMUP_BARS)

        while True:
            try:
                self._poll_incremental()
                time.sleep(self.poll_interval)
            except Exception as e:
                logging.exception(f"Live polling error: {e}")
                time.sleep(BACKOFF_SLEEP)

    # ---------- warm-up (full fetch once) ----------
    def _warmup(self, limit: int):
        try:
            df = fetch_ohlcv(self.ex, self.symbol, timeframe=self.timeframe, limit=limit)
            if df.empty:
                print("Warm-up: no data returned.")
                return
            self.df = df.tail(HISTORY_BARS).copy()
            self.df = self.df.drop_duplicates("timestamp").sort_values("timestamp")
            self.last_bar_time = int(self.df["timestamp"].iloc[-1])
            newest = pd.to_datetime(self.df["time"].iloc[-1])
            print(f"Warm-up complete: {len(self.df)} bars cached (through {newest}).")
        except Exception as e:
            logging.exception(f"Warm-up failed: {e}")

    # ---------- incremental polling (only latest chunk) ----------
    def _poll_incremental(self):
        """
        Fetch only the newest candles:
          since = last_bar_time - INCR_OVERLAP_BARS * tf_ms (small overlap)
          limit = EX_LIMIT_SOFT
        """
        if self.last_bar_time is None:
            # Should not happen after warm-up, but guard anyway
            return self._warmup(WARMUP_BARS)

        since = int(self.last_bar_time - INCR_OVERLAP_BARS * self.tf_ms)
        since = max(since, 0)

        # Use ccxt native fetch_ohlcv with since+limit if possible
        try:
            # Some exchanges ignore 'since' granularity; overlap absorbs it
            ohlcv = self.ex.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since, limit=EX_LIMIT_SOFT)
            if not ohlcv:
                return
            inc = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            inc["time"] = pd.to_datetime(inc["timestamp"], unit="ms", utc=True)
            # Merge with current buffer
            merged = pd.concat([self.df, inc], ignore_index=True)
            merged = merged.drop_duplicates("timestamp").sort_values("timestamp")
            if len(merged) > HISTORY_BARS:
                merged = merged.iloc[-HISTORY_BARS:]
            self.df = merged

            new_last = int(self.df["timestamp"].iloc[-1])

            # New-bar-only: act only when we see a NEW closed bar
            if new_last != self.last_bar_time:
                # compute signals & act on PREV bar's signal
                self._on_new_closed_bar()
                self.last_bar_time = new_last

        except Exception as e:
            logging.exception(f"Incremental fetch failed: {e}")
            time.sleep(BACKOFF_SLEEP)

    # ---------- core logic on each newly closed bar ----------
    def _on_new_closed_bar(self):
        df = self.df.copy()
        if df.empty or len(df) < 50:
            return

        sig = generate_signals(df)
        last, prev = sig.iloc[-1], sig.iloc[-2]

        trade_sig = int(prev["trade"])  # act on previous bar's signal
        prev_close = float(prev["close"]); prev_mid = float(prev.get("bb_middle", np.nan))
        close_px   = float(last["close"]); mid       = float(last.get("bb_middle", np.nan))
        mid_hit = mid_touch(prev_close, prev_mid, close_px, mid, MID_TOLERANCE_BPS)

        # ---- exits first (allowed anytime) ----
        self._maybe_exit(trade_sig, close_px)

        # recompute signals to ensure columns (bb_upper, bb_middle, bb_lower, rsi) exist
        sig_for_plot = generate_signals(self.df.copy())
        self.chart.update(self.df, sig_for_plot, self.trades)

        # ---- entries (session-gated) ----
        if not self.in_position and trade_sig == 1 and self.cooldown == 0:
            if (not SESSION_FILTER) or in_session(last["timestamp"], SESSION_TZ, SESSION_START_H, SESSION_END_H, WEEKDAYS_ONLY):
                qty = self._size_for_entry(close_px)
                if qty > 0:
                    self._buy(close_px, qty, reason="entry")
                    self.cooldown = ENTRY_COOLDOWN
                    self.scales_in_pos = 0
                    self.scale_cd = SCALE_COOLDOWN  # short delay after entry

        # ---- repeatable scale-ins on BB mid-touch ----
        if self.in_position and self.scale_cd == 0 and mid_hit:
            if MAX_SCALES_PER_POSITION is None or self.scales_in_pos < int(MAX_SCALES_PER_POSITION):
                if (not SESSION_FILTER) or in_session(last["timestamp"], SESSION_TZ, SESSION_START_H, SESSION_END_H, WEEKDAYS_ONLY):
                    qty = self._size_for_entry(close_px, scale=True)
                    if qty > 0:
                        self._buy(close_px, qty, reason="scale_in_mid")
                        self.scales_in_pos += 1
                        self.scale_cd = SCALE_COOLDOWN

        # --- colorized per-bar print with live price ---
        ts = pd.to_datetime(last["time"]).strftime("%Y-%m-%d %H:%M:%S UTC")
        if trade_sig == 1:
            color = GREEN
        elif trade_sig == -1:
            color = RED
        elif mid_hit:
            color = YELLOW
        else:
            color = GRAY

        print(
            f"{color}[{ts}] {self.symbol} | "
            f"Price: {close_px:,.4f} | "
            f"MACD: {last['macd']:+.5f} | "
            f"RSI: {last.get('rsi', np.nan):.1f} | "
            f"Trade={trade_sig} | "
            f"Scale={1 if mid_hit else 0} | "
            f"Pos={self.in_position} | "
            f"Qty={self.position_qty:.6f}{RESET}"
        )

        # cooldown tickers after acting on new bar
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.scale_cd > 0:
            self.scale_cd -= 1

        # --- summary dashboard every N bars ---
        self.bar_counter += 1
        if SUMMARY_EVERY_N > 0 and (self.bar_counter % SUMMARY_EVERY_N == 0):
            self._print_summary(close_px)

    # ---------- sizing ----------
    def _size_for_entry(self, price: float, scale: bool = False) -> float:
        """
        Constant-risk sizing (quote-based). qty = (risk% * free_quote) / (price * stop%)
        Capped by available quote cash; returns 0 on failure.
        """
        if not USE_CONSTANT_RISK or not STOP_LOSS_PCT or STOP_LOSS_PCT <= 0:
            return 0.001  # fallback tiny qty

        risk_mult = SCALE_RISK_MULT if scale else 1.0
        risk_pct = max(0.0, RISK_PER_TRADE_PCT * risk_mult)

        free_quote = self._free_quote_balance()
        risk_dollars = free_quote * risk_pct
        per_unit_risk = price * float(STOP_LOSS_PCT)
        if per_unit_risk <= 0:
            return 0.0
        qty = risk_dollars / per_unit_risk

        # cap to cash (no leverage)
        notional = qty * price
        if notional > free_quote:
            qty = free_quote / price

        return float(qty) if qty > 0 else 0.0

    def _free_quote_balance(self) -> float:
        """
        Pull free quote balance from broker (DRY returns a stub).
        """
        bal = self.broker.fetch_balance()
        q = self.quote
        try:
            if isinstance(bal, dict):
                if "free" in bal and isinstance(bal["free"], dict) and q in bal["free"]:
                    return float(bal["free"][q])
                if q in bal and isinstance(bal[q], dict) and "free" in bal[q]:
                    return float(bal[q]["free"])
        except Exception:
            pass
        return 10_000.0  # DRY default

    # ---------- orders & accounting ----------
    def _buy(self, price: float, qty: float, reason: str):
        self.broker.place_market_order(self.symbol, "buy", qty)
        self.logger.log_trade(self.symbol, "buy", price, qty, strategy="MACD+RSI+BB", reason=reason)

        t = self.df["time"].iloc[-1] if not self.df.empty else pd.Timestamp.utcnow()
        self.trades.append({"time": t, "side": "buy", "price": price})

        fee = (qty * price) * (FEE_BPS / 1e4) if FEE_BPS else 0.0
        cost = qty * price
        self.cash_quote -= (cost + fee)

        if not self.in_position or (self.position_qty <= 0):
            self.position_qty = qty
            self.avg_entry_px = price
            self.entry_px = price
            self.peak_px = price
            self.in_position = True
        else:
            total_qty = self.position_qty + qty
            self.avg_entry_px = (self.avg_entry_px * self.position_qty + price * qty) / total_qty
            self.position_qty = total_qty

    def _sell_all(self, price: float, reason: str):
        qty = self.position_qty
        if qty <= 0:
            return

        self.broker.place_market_order(self.symbol, "sell", qty)
        self.logger.log_trade(self.symbol, "sell", price, qty, strategy="MACD+RSI+BB", reason=reason)

        t = self.df["time"].iloc[-1] if not self.df.empty else pd.Timestamp.utcnow()
        self.trades.append({"time": t, "side": "sell", "price": price}) 

        proceeds = qty * price
        fee = proceeds * (FEE_BPS / 1e4) if FEE_BPS else 0.0
        self.cash_quote += (proceeds - fee)

        trade_pnl = (price - (self.avg_entry_px or price)) * qty
        self.realized_pnl += trade_pnl
        self.trade_count += 1
        if trade_pnl >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        # reset position state
        self.in_position = False
        self.position_qty = 0.0
        self.entry_px = None
        self.avg_entry_px = None
        self.peak_px = None
        self.cooldown = ENTRY_COOLDOWN
        self.scale_cd = 0
        self.scales_in_pos = 0

    def _maybe_exit(self, trade_sig: int, close_px: float):
        if not self.in_position:
            return

        # update trailing peak
        if TRAILING_STOP_PCT:
            self.peak_px = max(self.peak_px or close_px, close_px)

        # Risk exits first (execute on this bar close; REST polling executes at market now)
        if STOP_LOSS_PCT and self.entry_px and close_px <= self.entry_px * (1 - STOP_LOSS_PCT):
            self._sell_all(close_px, "stop_loss");  return
        if TAKE_PROFIT_PCT and self.entry_px and close_px >= self.entry_px * (1 + TAKE_PROFIT_PCT):
            self._sell_all(close_px, "take_profit");  return
        if TRAILING_STOP_PCT and self.peak_px and close_px <= self.peak_px * (1 - TRAILING_STOP_PCT):
            self._sell_all(close_px, "trailing_stop");  return

        # Strategy exit (MACD/RSI combined) when signal is -1 (and not on cooldown)
        if trade_sig == -1 and self.cooldown == 0:
            self._sell_all(close_px, "signal_exit")

    # ---------- summary ----------
    def _equity_now(self, price: float) -> float:
        position_val = (self.position_qty or 0.0) * price
        return float(self.cash_quote) + position_val

    def _print_summary(self, price: float):
        equity = self._equity_now(price)
        unrealized = 0.0 if not self.in_position or not self.avg_entry_px else (price - self.avg_entry_px) * self.position_qty
        total_pnl = self.realized_pnl + unrealized
        win_rate = (100.0 * self.win_count / self.trade_count) if self.trade_count else 0.0
        pnl_color = GREEN if total_pnl >= 0 else RED

        print(
            f"{CYAN}== SUMMARY == "
            f"Price: {price:,.4f} | "
            f"Equity: {equity:,.2f} | "
            f"Realized: {self.realized_pnl:+,.2f} | "
            f"Unrealized: {unrealized:+,.2f} | "
            f"Total P&L: {pnl_color}{total_pnl:+,.2f}{RESET}{CYAN} | "
            f"Trades: {self.trade_count} (Win%: {win_rate:.1f}%) | "
            f"Cash: {self.cash_quote:,.2f} | "
            f"PosQty: {self.position_qty:.6f}{RESET}"
        )


def run_polling_loop(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, poll_interval: int = POLL_INTERVAL):
    logging.basicConfig(level=logging.INFO)
    live = RestLive(symbol=symbol, timeframe=timeframe, poll_interval=poll_interval)
    live.run()


def main():
    run_polling_loop()


if __name__ == "__main__":
    main()