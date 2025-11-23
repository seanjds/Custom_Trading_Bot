import pandas as pd
import numpy as np

def _rsi(series: pd.Series, period: int = 14, wilder: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    if wilder:
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    else:
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype(float)

    # --- MACD ---
    ema_fast = c.ewm(span=12, adjust=False).mean()
    ema_slow = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger ---
    mid = c.rolling(20).mean()
    std = c.rolling(20).std(ddof=0)
    df["bb_middle"], df["bb_upper"], df["bb_lower"] = mid, mid + 2*std, mid - 2*std

    # --- RSI ---
    df["rsi"] = _rsi(c)

    # --- Base MACD trades ---
    prev_macd = df["macd"].shift(1)
    trade = np.where((prev_macd <= 0) & (df["macd"] > 0), 1,
                     np.where((prev_macd >= 0) & (df["macd"] < 0), -1, 0))

    # --- RSI override ---
    macd_neg = df["macd"] < 0
    rsi_above = df["rsi"] > 50
    override_sell = (trade == -1) & macd_neg & rsi_above
    trade = np.where(override_sell, 0, trade)

    prev_rsi = df["rsi"].shift(1)
    rsi_cross_down = (prev_rsi >= 50) & (df["rsi"] < 50)
    trade = np.where((trade != 1) & rsi_cross_down & macd_neg, -1, trade)

    df["trade"] = trade

    # --- Mid-touch scale-in signal ---
    prev_c, prev_mid = c.shift(1), mid.shift(1)
    df["scale_in"] = np.where(
        (prev_c > prev_mid) & (c <= mid) & (df["macd"] > 0), 1, 0
    )
    return df