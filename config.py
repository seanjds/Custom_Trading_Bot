# config.py
import os
from dotenv import load_dotenv
load_dotenv()


INITIAL_CAPITAL = 100.0
SYMBOL = "SOL/USD"
POLL_INTERVAL = 60        # seconds
DRY_RUN = True            # disable real trades

TIMEFRAME = "1m"             # OHLCV timeframe for backtest
BACKTEST_START = "2025-05-01"

EXCHANGE_ID = "coinbase"

# Risk
RISK_PER_TRADE = 0.01  # fraction of account equity to risk


# API keys (put real keys into a .env file)
COINBASE_API_KEY = os.getenv("organizations/0ec855d6-aaa5-4d34-a26a-6434e2ead588/apiKeys/b4dee7c2-34cd-4af4-bdbe-126a20028ee8")
COINBASE_SECRET = os.getenv("""
-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIDHcv4ktvpnA6DsS70mar/boNuSZXrDts92LFEUmWKCRoAoGCCqGSM49\nAwEHoUQDQgAE26a9XOXOBIIF/JvScmwqa4z8yaboO9QpcPh/IHLsZjNpldZMD3VD\nOb7CU+XYKEmOrNxePUPN79BmtsfG97G8vA==
-----END EC PRIVATE KEY-----
""")
COINBASE_PASSPHRASE = os.getenv("COINBASE_PASSPHRASE")


# Optional log files
log_file = "trade_log.csv"
equity_file = "equity_curve.csv"