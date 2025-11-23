import ccxt, logging, time
from config import COINBASE_API_KEY, COINBASE_SECRET

class Broker:
    def __init__(self, exchange_id="coinbase", api_key=None, secret=None, dry_run=True):
        self.dry_run = dry_run
        ex_class = getattr(ccxt, exchange_id)
        self.ex = ex_class({
            "apiKey": api_key or COINBASE_API_KEY,
            "secret": secret or COINBASE_SECRET,
            "enableRateLimit": True,
        })

    def fetch_ticker(self, symbol):
        return self.ex.fetch_ticker(symbol)

    def place_market_order(self, symbol, side, amount):
        if self.dry_run:
            logging.info(f"[DRY RUN] {side.upper()} {amount} {symbol}")
            return {"id": f"dry-{time.time()}", "status": "closed"}
        return self.ex.create_order(symbol, "market", side, amount)

    def fetch_balance(self):
        if self.dry_run:
            return {"USD": {"free": 10000}}
        return self.ex.fetch_balance()