
import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_macro():
    """Download USD/RUB historical data and save to CSV."""
    ticker = "RUB=X"
    print(f"Fetching historical data for {ticker}...")
    
    # Download entire history (or enough to cover 2021+)
    df = yf.download(ticker, start="2020-01-01", progress=False)
    
    if df.empty:
        print("Error: No data downloaded.")
        return

    # yfinance returns MultiIndex columns sometimes, fix it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to get Date as column
    df = df.reset_index()
    
    # Keep only relevant columns
    # 'Close' is the exchange rate at end of day
    df = df[["Date", "Close"]].copy()
    
    # Rename for clarity
    df = df.rename(columns={"Date": "date", "Close": "usd_rub"})
    
    # Ensure date format
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    output_path = Path("data/raw/macro.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print(df.tail())

if __name__ == "__main__":
    fetch_macro()
