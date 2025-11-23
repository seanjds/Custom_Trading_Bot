# strategies/bbands.py
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, mode: str = "mean_revert"):
    """
    Expects df with columns: ['time','open','high','low','close','volume'] (fetcher returns this).
    Adds: 'middle','upper','lower','trade' where trade ∈ {1 (buy), -1 (sell), 0 (hold)}.

    mode:
      - "mean_revert": BUY when close crosses UP above LOWER band; SELL when close crosses DOWN below UPPER
      - "breakout"   : BUY when close crosses UP above UPPER band; SELL when close crosses DOWN below LOWER
    """
    df = df.copy()
    c = df["close"].astype(float)

    # rolling mean/std
    middle = c.rolling(window=period, min_periods=period).mean()
    rolling_std = c.rolling(window=period, min_periods=period).std(ddof=0)

    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    df["middle"] = middle
    df["upper"]  = upper
    df["lower"]  = lower

    # cross logic (use previous values to detect true crossing)
    prev_c = c.shift(1)
    prev_u = upper.shift(1)
    prev_l = lower.shift(1)

    if mode == "breakout":
        buy  = (prev_c <= prev_u) & (c > upper)         # cross above upper
        sell = (prev_c >= prev_l) & (c < lower)         # cross below lower
    else:
        # mean reversion: fade extremes back to middle
        buy  = (prev_c <= prev_l) & (c > lower)         # rebound above lower
        sell = (prev_c >= prev_u) & (c < upper)         # drop below upper

    df["trade"] = np.where(buy, 1, np.where(sell, -1, 0))

    # --- Regime filters (trend & volatility) ---
    df = add_regime_filters(df)
    return df


def add_regime_filters(df: pd.DataFrame, trend_len: int = 200, vol_len: int = 50,
                       vol_low: float = 0.002, vol_high: float = 0.01) -> pd.DataFrame:
    """
    Trend: price vs EMA(trend_len) => uptrend (1), downtrend (-1), flat (0).
    Volatility: mean of (high-low)/close over vol_len bars.
    Produces boolean gates for different modes:
      - filter_mean_revert: higher-vol chop allowed
      - filter_breakout   : trend present AND mid-range vol
    """
    df = df.copy()
    c = df["close"].astype(float)

    df["ema"] = c.ewm(span=trend_len, adjust=False).mean()
    df["trend"] = 0
    df.loc[c > df["ema"], "trend"] = 1
    df.loc[c < df["ema"], "trend"] = -1

    rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["volatility"] = rng.rolling(vol_len, min_periods=vol_len).mean().fillna(0)

    df["filter_mean_revert"] = df["volatility"] >= vol_low
    df["filter_breakout"]    = (df["trend"] != 0) & (df["volatility"].between(vol_low, vol_high))

    return df