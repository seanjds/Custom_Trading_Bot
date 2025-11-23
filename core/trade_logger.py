import csv
import os
from datetime import datetime

class TradeLogger:
    def __init__(self, filename='logs/trade_log.csv'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp", "symbol", "side", "price", "quantity",
                    "strategy", "reason", "balance_after_trade"
                ])

    def log_trade(self, symbol, side, price, quantity, strategy="", reason="", balance_after_trade=None):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                symbol,
                side,
                round(price, 6),
                quantity,
                strategy,
                reason,
                balance_after_trade if balance_after_trade is not None else ""
            ])
        print(f"✅ Logged {side.upper()} trade for {symbol} at {price}")