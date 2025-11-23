import time, logging
from config import SYMBOL, POLL_INTERVAL, DRY_RUN
from data.fetcher import init_exchange, fetch_ohlcv
from strategies.macd_bb_rsi import generate_signals
from live.broker import Broker
from core.trade_logger import TradeLogger

def main():
    trade_logger = TradeLogger("logs/trade_log.csv")
    broker = Broker("coinbase", dry_run=DRY_RUN)
    ex = init_exchange("coinbase")

    in_position = False
    last_bar_time = None
    cooldown = 0

    while True:
        try:
            df = fetch_ohlcv(ex, SYMBOL, "1m", 200)
            df = generate_signals(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]

            # Run new-bar-only
            if last_bar_time == last["time"]:
                time.sleep(POLL_INTERVAL)
                continue
            last_bar_time = last["time"]

            trade_sig = int(prev["trade"])
            if cooldown > 0:
                cooldown -= 1

            # --- Sell logic ---
            if in_position and trade_sig == -1:
                broker.place_market_order(SYMBOL, "sell", 0.001)
                trade_logger.log_trade(SYMBOL, "sell", last["close"], 0.001)
                in_position = False
                cooldown = 3

            # --- Buy logic ---
            elif not in_position and trade_sig == 1 and cooldown == 0:
                broker.place_market_order(SYMBOL, "buy", 0.001)
                trade_logger.log_trade(SYMBOL, "buy", last["close"], 0.001)
                in_position = True
                cooldown = 3

            print(f"[{last['time']}] {SYMBOL} MACD={last['macd']:.4f} RSI={last['rsi']:.2f} Trade={trade_sig}")

        except Exception as e:
            logging.exception(f"Live loop error: {e}")

        time.sleep(POLL_INTERVAL)