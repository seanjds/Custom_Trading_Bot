# data/fetcher.py
import ccxt
import pandas as pd

def init_exchange(exchange_id="coinbase", **kwargs):
    ex_class = getattr(ccxt, exchange_id)
    return ex_class({"enableRateLimit": True, **kwargs})

def fetch_ohlcv(ex, symbol, timeframe="1m", limit=1000):
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["timestamp","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df