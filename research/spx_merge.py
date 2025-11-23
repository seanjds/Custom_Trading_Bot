import pandas as pd
import yfinance as yf

# Load rainy days
df = pd.read_csv("manhattan_rain_days_2024.csv", parse_dates=["Rain Date"])

# Pull SPY daily data (add buffer from 2023)
spy = yf.download("SPY", start="2023-12-15", end="2025-01-01")

# Use Adj Close if available, else Close
price_col = "Adj Close" if "Adj Close" in spy.columns else "Close"

# Keep close price
spy = spy[[price_col]].rename(columns={price_col: "SPY_Close"})

# Compute daily returns
spy["SPY_Return"] = spy["SPY_Close"].pct_change()

# Compute cumulative return normalized to first 2024 trading day
spy["SPY_CumReturn"] = (1 + spy["SPY_Return"]).cumprod()
first2024 = spy.loc[spy.index >= "2024-01-01"].index[0]
baseline = spy.loc[first2024, "SPY_CumReturn"]
spy["SPY_CumReturn"] = spy["SPY_CumReturn"] / baseline - 1

# Reset index for merging
spy = spy.reset_index().rename(columns={"Date": "Rain Date"})

# Merge onto rain days
merged = pd.merge(
    df,
    spy[["Rain Date", "SPY_Close", "SPY_Return", "SPY_CumReturn"]],
    on="Rain Date",
    how="left"
)

# Save to CSV
merged.to_csv("manhattan_rain_days_2024_with_spy_returns.csv", index=False)

print("✅ File saved: manhattan_rain_days_2024_with_spy_returns.csv")
print(merged.head())