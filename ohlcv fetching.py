# 1. Install dependencies (run only once)
!pip install yfinance pandas

# 2. Imports
import yfinance as yf
import pandas as pd

# 3. Configuration
TICKER = "PSEI.PS"
START_DATE = "2024-01-01"
END_DATE = "2025-06-30"
OUT_CSV = "PSEI_OHLCV_2024_2025.csv"

# 4. Fetch OHLCV data
print(f"Fetching OHLCV data for {TICKER} from {START_DATE} to {END_DATE}...")
ticker = yf.Ticker(TICKER)
data = ticker.history(start=START_DATE, end=END_DATE)

# 5. Check and save
if data.empty:
    raise RuntimeError("No data fetched. Check ticker or date range.")
else:
    data.to_csv(OUT_CSV)
    print(f"✅ Saved {len(data)} rows of OHLCV data to {OUT_CSV}")

# 6. Preview
display(data.head())
