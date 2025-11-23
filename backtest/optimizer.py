# backtest/optimizer.py
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from strategies.registry import STRATEGIES

# --- Metrics ---
def calculate_metrics(equity_curve: pd.DataFrame):
    returns = equity_curve["equity"].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = (equity_curve["equity"].cummax() - equity_curve["equity"]).max()
    total_return = equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1
    return {"Sharpe": sharpe, "Max Drawdown": max_dd, "Total Return": total_return}

# --- Backtest ---
def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, initial_capital=10000):
    capital = initial_capital
    position = 0
    equity = []

    for i in range(1, len(signals)):
        row = signals.iloc[i]
        prev = signals.iloc[i - 1]

        if prev["trade"] == 1 and capital > 0:  # Buy
            position = capital / row["open"]
            capital = 0
        elif prev["trade"] == -1 and position > 0:  # Sell
            capital = position * row["open"]
            position = 0

        equity.append({
            "time": row["timestamp"],
            "equity": capital + position * row["close"]
        })

    return pd.DataFrame(equity)

# --- Multi-strategy optimizer ---
def optimize_all_strategies(df, n_jobs=-1):
    results = []

    def test_params(strategy_name, func, params):
        signals = func(df, **params)
        equity = run_backtest(df, signals)
        metrics = calculate_metrics(equity)
        return {"strategy": strategy_name, **params, **metrics}

    tasks = []
    for strat_name, strat_info in STRATEGIES.items():
        func = strat_info["func"]
        param_grid = strat_info["params"]

        keys, values = zip(*param_grid.items())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            tasks.append((strat_name, func, params))

    results = Parallel(n_jobs=n_jobs)(
        delayed(test_params)(s, f, p) for (s, f, p) in tasks
    )

    return pd.DataFrame(results)

# --- Plot strategy comparison ---
def plot_strategy_comparison(results, metric="Sharpe"):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="strategy", y=metric, data=results)
    plt.title(f"Strategy Comparison by {metric}")
    plt.show()