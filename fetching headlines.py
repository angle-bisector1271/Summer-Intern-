# 1. Install dependencies
!pip install yfinance feedparser pandas numpy python-dateutil

# 2. Imports
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from dateutil.relativedelta import relativedelta

# 3. Configuration

# 3.1 Index ticker for PSE Composite Index on Yahoo Finance
INDEX_TICKER = "PSEI.PS"

# 3.2 Global date range: adjust to your desired full range
GLOBAL_START = "2024-01-01"   # inclusive
GLOBAL_END   = "2025-05-30"   # exclusive: loop until before this date

# 3.3 News queries: general financial & political topics in Philippines
QUERIES = [
    "Philippines stock market",
    "PSE Composite Index",
    "Philippine economy",
    "Bangko Sentral ng Pilipinas policy",
    "Philippine inflation",
    "Philippine GDP data",
    "Philippines politics economy",
    "Philippine central bank",
    "Philippines interest rate decision",
    "Philippines election impact economy",
    "Philippines financial market",
    "Philippines budget announcement",
    "Philippine fiscal policy",
    "Philippine trade balance",
    "Philippines foreign investment",
    # add more if desired
]

# 3.4 Google News RSS URL template for Philippines edition
# We will inject date filters via Google News search operators "after:YYYY-MM-DD before:YYYY-MM-DD"
# Note: effectiveness may vary; test if RSS returns historical items.
RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}"

# 3.5 Labeling threshold for index returns
RETURN_THRESHOLD = 0.005  # 0.5%

# 3.6 Limits and sleeps
MAX_RSS_ENTRIES_PER_QUERY = 1000
SLEEP_BETWEEN_RSS = 1.0     # seconds between RSS fetches
SLEEP_BETWEEN_MONTHS = 2.0  # seconds pause between month loops (optional)

# 4. Utility: fetch index history for a given month range
def fetch_index_history_month(ticker_obj, month_start: datetime, month_end: datetime):
    """
    Fetch index history from month_start (inclusive) to month_end (exclusive).
    Returns a DataFrame with DateOnly index and 'Close' column.
    """
    start_str = month_start.strftime("%Y-%m-%d")
    end_str   = month_end.strftime("%Y-%m-%d")
    hist = ticker_obj.history(start=start_str, end=end_str)
    if hist.empty:
        # No data in this month
        return pd.DataFrame(columns=['Close']).set_index(pd.Index([], name='DateOnly'))
    df_idx = hist[['Close']].copy()
    df_idx['DateOnly'] = df_idx.index.date
    df_idx = df_idx.set_index('DateOnly').sort_index()
    return df_idx

# 5. Utility: fetch RSS headlines for a single query within a date window
def fetch_rss_headlines_for_period(query: str, start_date: datetime, end_date: datetime, max_entries=MAX_RSS_ENTRIES_PER_QUERY):
    """
    Fetch headlines via Google News RSS for given query, filtering by date window:
    Uses search operators: after:YYYY-MM-DD before:YYYY-MM-DD
    Returns list of dicts: {'text': str, 'date': datetime UTC}
    """
    date_filter = f" after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
    full_query = f"{query}{date_filter}"
    url = RSS_TEMPLATE.format(query=full_query.replace(" ", "%20"))
    feed = feedparser.parse(url)
    items = []
    count = 0
    for entry in feed.entries:
        if count >= max_entries:
            break
        title = entry.get("title", "").strip()
        pub_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        pub_dt = None
        if pub_parsed:
            try:
                pub_dt = datetime(*pub_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pub_dt = None
        if title and pub_dt:
            pub_date_local = pub_dt.date()
            if start_date.date() <= pub_date_local < end_date.date():
                # Rename keys to 'text' and 'date'
                items.append({"text": title, "date": pub_dt})
                count += 1
    print(f"[RSS] Query='{query}' period {start_date.date()} to {end_date.date()}: fetched {len(items)} items")
    return items

# 6. Utility: get prev and next index close for a publish_datetime
def get_prev_next_close(df_index: pd.DataFrame, publish_dt: datetime):
    """
    Given df_index indexed by datetime.date with column 'Close',
    find:
      prev_close = close on last trading day <= publish_date
      next_close = close on first trading day > publish_date
    Returns (prev_close, next_close) or (None, None) if not found.
    """
    pub_date = publish_dt.date()
    dates = df_index.index
    if len(dates) == 0:
        return None, None
    if pub_date < dates[0] or pub_date >= dates[-1]:
        return None, None
    next_dates = dates[dates > pub_date]
    if len(next_dates) == 0:
        return None, None
    next_date = next_dates[0]
    prev_dates = dates[dates <= pub_date]
    if len(prev_dates) == 0:
        return None, None
    prev_date = prev_dates[-1]
    prev_close = df_index.loc[prev_date, "Close"]
    next_close = df_index.loc[next_date, "Close"]
    return prev_close, next_close

# 7. Main loop: month-by-month
def run_monthly_collection(global_start: str, global_end: str, output_csv="PSE_index_news_labeled_all.csv"):
    """
    Loops from global_start (inclusive) to global_end (exclusive) month by month,
    fetches index history and RSS headlines per month, labels them, and merges into one DataFrame.
    Saves combined CSV at the end with columns renamed to include 'date' and 'text'.
    """
    start_dt = datetime.fromisoformat(global_start)
    end_dt   = datetime.fromisoformat(global_end)
    if start_dt >= end_dt:
        raise ValueError("GLOBAL_START must be before GLOBAL_END")

    ticker = yf.Ticker(INDEX_TICKER)
    combined_records = []
    month_start = start_dt.replace(day=1)

    while month_start < end_dt:
        next_month = month_start + relativedelta(months=1)
        month_end = next_month

        print(f"\n=== Processing period: {month_start.date()} to {month_end.date()} ===")
        # Fetch index data for this month
        df_idx_month = fetch_index_history_month(ticker, month_start, month_end)
        print(f"  Index data: {len(df_idx_month)} trading days")
        if df_idx_month.empty:
            print("  No index data for this month, skipping.")
            month_start = next_month
            time.sleep(SLEEP_BETWEEN_MONTHS)
            continue

        # Fetch RSS headlines for each query in this month
        all_items = []
        for q in QUERIES:
            items = fetch_rss_headlines_for_period(q, month_start, month_end)
            all_items.extend(items)
            time.sleep(SLEEP_BETWEEN_RSS)

        # Deduplicate by (text, date)
        df_news = pd.DataFrame(all_items)
        if df_news.empty:
            print("  No headlines fetched this month.")
            month_start = next_month
            time.sleep(SLEEP_BETWEEN_MONTHS)
            continue
        df_news = df_news.drop_duplicates(subset=["text", "date"]).reset_index(drop=True)
        print(f"  Unique headlines collected: {len(df_news)}")

        # Label each headline based on index data
        for idx, row in df_news.iterrows():
            pub_dt = row["date"]
            prev_c, next_c = get_prev_next_close(df_idx_month, pub_dt)
            if prev_c is None or next_c is None:
                continue
            ret = (next_c - prev_c) / prev_c
            if ret > RETURN_THRESHOLD:
                label = "Bullish"
            elif ret < -RETURN_THRESHOLD:
                label = "Bearish"
            else:
                label = "Neutral"
            combined_records.append({
                "text": row["text"],
                "date": pub_dt.isoformat(),   # store full timestamp; or use pub_dt.date().isoformat() for date-only
                "period_start": month_start.date().isoformat(),
                "period_end": (month_end - relativedelta(days=1)).date().isoformat(),
                "prev_close": prev_c,
                "next_close": next_c,
                "return": ret,
                "label": label
            })
        print(f"  Labeled this month: {len(combined_records)} total records so far")

        month_start = next_month
        time.sleep(SLEEP_BETWEEN_MONTHS)

    # After loop: build DataFrame and save
    if not combined_records:
        print("No labeled records collected in entire range.")
        return
    df_all = pd.DataFrame(combined_records)
    # Drop duplicates globally if any
    df_all = df_all.drop_duplicates(subset=["text", "date"]).reset_index(drop=True)
    print(f"\nTotal labeled headlines over full period: {len(df_all)}")
    # Save CSV
    df_all.to_csv(output_csv, index=False)
    print(f"Saved combined labeled dataset to {output_csv}")
    # Show distribution
    summary = df_all['label'].value_counts().rename_axis('label').reset_index(name='count')
    print("Overall label distribution:")
    print(summary)
    display(df_all.head(10))

# 8. Run the monthly collection
if __name__ == "__main__":
    run_monthly_collection(GLOBAL_START, GLOBAL_END)
