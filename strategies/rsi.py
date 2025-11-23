# strategies/rsi.py
import pandas as pd

def generate_signals(df, **kwargs):
    period = kwargs.get("period", kwargs.get("rsi_period", 14))
    overbought = kwargs.get("overbought", 70)
    oversold = kwargs.get("oversold", 30)

    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(int(period)).mean()
    loss = (-delta).clip(lower=0).rolling(int(period)).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["trade"] = 0
    df.loc[df["rsi"] < oversold, "trade"] = 1
    df.loc[df["rsi"] > overbought, "trade"] = -1
    return df