#!/usr/bin/env python3
"""Quick sanity checks for scraped messages and daily feature parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_outputs(raw_path: Path, daily_path: Path) -> None:
    raw_df = pd.read_parquet(raw_path)
    if "date" not in raw_df.columns:
        raise ValueError("Raw parquet must contain a 'date' column.")
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    print(f"Channels covered: {raw_df['channel'].nunique()} unique handles")
    print(f"Total messages: {len(raw_df)}")
    print(f"Coverage window: {raw_df['date'].min().date()} -> {raw_df['date'].max().date()}")

    daily_df = pd.read_parquet(daily_path)
    share_cols = [c for c in daily_df.columns if c.startswith("share_")]
    if share_cols:
        means = daily_df[share_cols].mean().sort_index()
        print("\nMean keyword shares:")
        for col, val in means.items():
            print(f"  {col}: {val:.3f}")
    else:
        print("\nNo share_* columns found in daily features.")

    preview_cols = ["date", "total_messages", "keyword_mentions", "price_mentions", "fuel_stress_index"]
    preview_cols = [c for c in preview_cols if c in daily_df.columns]
    if preview_cols:
        print("\nDaily feature preview:")
        print(daily_df[preview_cols].head())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Telegram scraping outputs.")
    parser.add_argument("--raw", required=True, type=Path, help="Path to raw messages parquet.")
    parser.add_argument("--daily", required=True, type=Path, help="Path to daily features parquet.")
    args = parser.parse_args()

    summarize_outputs(args.raw, args.daily)


if __name__ == "__main__":
    main()
