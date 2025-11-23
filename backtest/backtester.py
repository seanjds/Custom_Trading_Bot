import os, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from config import INITIAL_CAPITAL
from core.trade_logger import TradeLogger
from data.fetcher import init_exchange, fetch_ohlcv
from strategies.macd_bb_rsi import generate_signals

def run_backtest_example(symbol="BTC/USDT", timeframe="1m", limit=5000,
                         risk_pct=0.25, stop_loss_pct=0.006, take_profit_pct=0.008,
                         scale_cooldown_bars=10):
    ex = init_exchange("coinbase")
    df = fetch_ohlcv(ex, symbol, timeframe=timeframe, limit=limit)
    signals = generate_signals(df)

    capital = INITIAL_CAPITAL
    position, in_position = 0.0, False
    cooldown, scale_cd, scales_in_pos = 0, 0, 0
    equity_rows, trades = [], []
    logger = TradeLogger("logs/trade_log.csv")

    for i in range(1, len(signals)):
        row, prev = signals.iloc[i], signals.iloc[i-1]
        open_px, close_px = float(row["open"]), float(row["close"])
        t = row["time"]

        if cooldown > 0: cooldown -= 1
        if scale_cd > 0: scale_cd -= 1

        prev_sig = int(prev["trade"])

        # --- Exit rules ---
        if in_position:
            if (stop_loss_pct and close_px <= entry_px*(1-stop_loss_pct)) or \
               (take_profit_pct and close_px >= entry_px*(1+take_profit_pct)) or \
               (prev_sig == -1):
                capital += position * open_px
                trades.append({"time": t, "side": "sell", "price": open_px, "qty": position})
                logger.log_trade(symbol, "sell", open_px, position)
                position, in_position = 0.0, False
                cooldown, scale_cd, scales_in_pos = 3, 0, 0

        # --- Entry ---
        if not in_position and prev_sig == 1 and cooldown == 0:
            spend = capital * risk_pct
            qty = spend / open_px
            capital -= spend
            position, in_position, entry_px = qty, True, open_px
            trades.append({"time": t, "side": "buy", "price": open_px, "qty": qty})
            logger.log_trade(symbol, "buy", open_px, qty)

        # --- Scale-in (mid touch) ---
        if in_position and scale_cd == 0 and row.get("scale_in", 0) == 1:
            spend = capital * risk_pct * 0.5
            if spend > 0:
                qty = spend / open_px
                capital -= spend
                position += qty
                scales_in_pos += 1
                scale_cd = scale_cooldown_bars
                trades.append({"time": t, "side": "buy", "price": open_px, "qty": qty, "reason": "scale_in"})
                logger.log_trade(symbol, "buy", open_px, qty, reason="scale_in")

        equity_rows.append({"time": t, "equity": capital + position * close_px})

    eq = pd.DataFrame(equity_rows)
    eq.to_csv("reports/backtest_equity.csv", index=False)

    plt.plot(eq["time"], eq["equity"])
    plt.title(f"Equity Curve {symbol}")
    plt.savefig("reports/backtest_equity.png", dpi=150)
    plt.close()

    print(f"Final equity: {eq['equity'].iloc[-1]:.2f}")
    return eq, trades