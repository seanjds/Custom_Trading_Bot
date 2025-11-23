# strategies/sma_crossover.py
import pandas as pd

def generate_signals(df, **kwargs):
    """
    Accept flexible parameter names:
      - short or short_window
      - long or long_window
    """
    short = kwargs.get("short", kwargs.get("short_window", 20))
    long  = kwargs.get("long",  kwargs.get("long_window", 40))

    df = df.copy()
    df["sma_short"] = df["close"].rolling(int(short)).mean()
    df["sma_long"]  = df["close"].rolling(int(long)).mean()
    df["signal"] = 0
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
    df.loc[df["sma_short"] <= df["sma_long"], "signal"] = 0
    df["trade"] = df["signal"].diff().fillna(0)  # +1 = buy, -1 = sell/exit
    return df