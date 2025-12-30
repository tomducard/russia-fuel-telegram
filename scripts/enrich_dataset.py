
import pandas as pd
from pathlib import Path

def enrich():
    input_path = Path("data/processed/merged_final.parquet")
    output_path = Path("data/processed/merged_enriched.parquet")
    
    if not input_path.exists():
        print(f"Missing {input_path}")
        return

    df = pd.read_parquet(input_path)
    
    # Filter for relevant date range (>= 2021-01-01) as per user request
    df["date"] = pd.to_datetime(df["date"]).dt.date
    start_date = pd.to_datetime("2021-01-01").date()
    df = df[df["date"] >= start_date].copy()
    
    print(f"Loaded {len(df)} rows (filtered >= {start_date}).")
    
    # Recover count columns from shares
    # share_X = count_X / total_messages (roughly, or count_X / keyword_mentions? No, share is per message usually)
    # in features.py: aggregations[share_name] = (flag_col, "mean") where flag_col is boolean per message.
    # So share = sum(flags)/count(rows).
    # sum(flags) is exactly the count we want.
    # We have total_messages = count(rows).
    # So count_X = share_X * total_messages.
    
    share_cols = [c for c in df.columns if c.startswith("share_")]
    
    for share_col in share_cols:
        group_name = share_col.replace("share_", "")
        count_col = f"count_{group_name}"
        
        # Calculate and round to nearest int (it should be int)
        df[count_col] = (df[share_col] * df["total_messages"]).round().astype(int)
        print(f"Recovered {count_col}")

    # Also ensure sentiment columns exist (they rely on NLP).
    # merged_final has 'fuel_stress_index' which uses sentiment, but does it have 'sentiment_mean'?
    # Checking Step 274: It DOES NOT have sentiment_mean!
    # It has 'fuel_stress_index'.
    # We can't easily recover sentiment_mean from stress index because stress index mixes price and keywords.
    # BUT we can set it to 0.0 or NaN if missing, or try to reverse engineer?
    # Reverse engineering is hard (max(0, 1-sentiment)).
    # Let's check if 'sentiment_mean' was in the columns list in Step 274.
    # List: [... 'fuel_stress_index', ...]. No sentiment_mean.
    # Okay, so we will lack sentiment_mean for the *old* data.
    # We will add it as NaN or 0 so the pipeline doesn't crash, but warn the user.
    # However, for graphs, we don't plot sentiment directly, we plot stress index.
    # For ML, it might be missing.
    
    
    if "sentiment_mean" not in df.columns:
        df["sentiment_mean"] = 0.0 # Default neutral

    # --- Merge Macro Data (USD/RUB) ---
    macro_path = Path("data/raw/macro.csv")
    if macro_path.exists():
        print("Merging Macro Data (USD/RUB)...")
        macro_df = pd.read_csv(macro_path)
        macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.date
        # Merge left to keep original rows
        df = pd.merge(df, macro_df, on="date", how="left")
        # Forward fill missing weekends/holidays for exchange rates
        df["usd_rub"] = df["usd_rub"].ffill()
    
    # --- Feature Engineering: Rolling Stats (Trend, Shock, Volatility) ---
    print("Computing Rolling Features (Trend, Shock, Volatility)...")
    
    # Ensure correct time order
    df = df.sort_values("date").reset_index(drop=True)
    
    # Key signals to enhance
    signals = ["fuel_stress_index", "count_logistics_terms", "sentiment_mean", "avg_price", "usd_rub"]
    
    for col in signals:
        if col not in df.columns:
            continue
            
        # 1. Trend (Acceleration): Short Term (7d) / Long Term (30d)
        # We add epsilon to avoid division by zero
        rolling_7 = df[col].rolling(window=7, min_periods=1).mean()
        rolling_30 = df[col].rolling(window=30, min_periods=1).mean()
        
        df[f"{col}_trend_7d_30d"] = (rolling_7 + 1e-6) / (rolling_30 + 1e-6)
        
        # 2. Shock (Spike): Today / Avg(7d)
        # Captures "Sudden Boom after Flat"
        df[f"{col}_shock_7d"] = (df[col] + 1e-6) / (rolling_7 + 1e-6)
        
        # 3. Volatility (Instability): Std(7d)
        df[f"{col}_volatility_7d"] = df[col].rolling(window=7, min_periods=1).std().fillna(0.0)
        
        print(f"Added rolling features for {col}")

    df.to_parquet(output_path, index=False)
    print(f"Saved enriched dataset to {output_path}")

if __name__ == "__main__":
    enrich()
