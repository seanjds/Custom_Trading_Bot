# strategies/registry.py
from strategies.sma_crossover import generate_signals as sma_signals
from strategies.rsi import generate_signals as rsi_signals
from strategies.bbands import generate_signals as bbands_signals

STRATEGIES = {
    "sma_crossover": {
        "func": sma_signals,
        "params": {
            "short": range(5, 50, 5),
            "long": range(20, 200, 20),
        }
    },
    "rsi": {
        "func": rsi_signals,
        "params": {
            "period": range(10, 30, 5),
            "overbought": [70],
            "oversold": [30],
        }
    },
    "bollinger_bands": {
        "func": bbands_signals,
        "params": {
            "period": range(10, 30, 5),
            "std_dev": [1.5, 2, 2.5],
        }
    }
}