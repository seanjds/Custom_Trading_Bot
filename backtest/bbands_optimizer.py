import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

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

        if prev["trade"] == 1 and capital > 0:   # BUY
            position = (capital * trade_size) / row["open"]
            capital = 0
        elif prev["trade"] == -1 and position > 0:  # SELL
            capital = position * row["open"]
            position = 0

        equity.append({
            "time": row["time"],
            "equity": capital + position * row["close"]
        })

    return pd.DataFrame(equity)

# === Grid search optimization ===
def optimize_bbands(df, period_range=range(10, 40, 5), stddev_range=[1, 1.5, 2, 2.5, 3]):
    results = []

    for period, std_dev in itertools.product(period_range, stddev_range):
        signals = generate_signals(df, period=period, std_dev=std_dev)
        equity_curve = run_backtest(df, signals)
        metrics = calculate_metrics(equity_curve)
        results.append({"period": period, "std_dev": std_dev, **metrics})

    return pd.DataFrame(results)

# === Heatmap plot ===
def plot_heatmap(results, metric="Sharpe"):
    pivot = results.pivot(index="period", columns="std_dev", values=metric)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f"Bollinger Bands Optimization ({metric})")
    plt.show()

# === Main entrypoint ===
if __name__ == "__main__":
    ex = init_exchange("binance")
    df = fetch_ohlcv(ex, "BTC/USDT", timeframe="1h", limit=1000)

    results = optimize_bbands(df)
    print("Top results:")
    print(results.sort_values("Sharpe", ascending=False).head(10))

    plot_heatmap(results, metric="Sharpe")