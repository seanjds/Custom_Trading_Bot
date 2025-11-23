import pandas as pd
import numpy as np
from strategies.bbands import generate_signals
from data.fetcher import init_exchange, fetch_ohlcv

# === Metrics ===
def calculate_metrics(equity_curve: pd.DataFrame):
    returns = equity_curve["equity"].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = (equity_curve["equity"].cummax() - equity_curve["equity"]).max()
    total_return = equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1
    return {"Sharpe": sharpe, "Max Drawdown": max_dd, "Total Return": total_return}

# === Backtest loop ===
def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, initial_capital=10000, trade_size=1.0):
    capital = initial_capital
    position = 0
    equity = []

    for i in range(1, len(signals)):
        row = signals.iloc[i]
        prev = signals.iloc[i - 1]

        # BUY signal
        if prev["trade"] == 1 and capital > 0:
            position = (capital * trade_size) / row["open"]
            capital = 0

        # SELL signal
        elif prev["trade"] == -1 and position > 0:
            capital = position * row["open"]
            position = 0

        equity.append({
            "time": row["time"],
            "equity": capital + position * row["close"]
        })

    return pd.DataFrame(equity)

# === Main entrypoint ===
if __name__ == "__main__":
    ex = init_exchange("binance")
    df = fetch_ohlcv(ex, "BTC/USDT", timeframe="1h", limit=1000)

    # Generate signals with Bollinger Bands
    signals = generate_signals(df, period=20, std_dev=2)

    # Run backtest
    equity_curve = run_backtest(df, signals)
    metrics = calculate_metrics(equity_curve)

    print("=== Bollinger Bands Backtest ===")
    print(metrics)

    # Optional: plot equity curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(equity_curve["time"], equity_curve["equity"], label="Equity Curve")
    plt.title("Equity Curve - Bollinger Bands Strategy")
    plt.legend()
    plt.show()