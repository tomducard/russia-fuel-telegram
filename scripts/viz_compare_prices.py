import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

def plot_official_prices(df: pd.DataFrame, output_dir: Path):
    """Plot official dataset prices over time."""
    cols = ["Diesel_RUB", "Regular92_RUB"]
    if not all(c in df.columns for c in cols):
        print(f"Skipping official plot: Missing columns {cols}")
        return

    plt.figure()
    for col in cols:
        plt.plot(df["date"], df[col], label=col.replace("_RUB", ""), linewidth=2)
    
    plt.title("Official Russian Fuel Prices (Central Bank/Rosstat)")
    plt.ylabel("Price (RUB / Liter)")
    plt.xlabel("Date")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = output_dir / "official_prices_plot.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")

def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare Official vs Scraped Prices."""
    # Plot Diesel Comparison
    fig, ax = plt.subplots()
    
    # Official Diesel
    ax.plot(df["date"], df["Diesel_RUB"], label="Official Diesel", color="blue", linewidth=2)
    
    # Scraped Diesel (avg_price_diesel)
    # Filter out zeros or noise if needed
    scraped = df[df["avg_price_diesel"] > 0].copy()
    # Smoothen scraped data
    scraped["diesel_ma"] = scraped["avg_price_diesel"].rolling(window=14, min_periods=1).mean()
    
    ax.plot(scraped["date"], scraped["diesel_ma"], label="Telegram Diesel (Smoothed)", color="orange", linestyle="--", linewidth=2)
    
    ax.set_title("Price Comparison: Official vs Telegram (Diesel)")
    ax.set_ylabel("Price (RUB)")
    ax.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = output_dir / "comparison_diesel_plot.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")

    # Plot Gasoline Comparison (Regular92 vs avg_price_gasoline)
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["Regular92_RUB"], label="Official Gasoline (92)", color="red", linewidth=2)
    
    scraped_gas = df[df["avg_price_gasoline"] > 0].copy()
    scraped_gas["gas_ma"] = scraped_gas["avg_price_gasoline"].rolling(window=14, min_periods=1).mean()
    
    ax.plot(scraped_gas["date"], scraped_gas["gas_ma"], label="Telegram Gasoline (Smoothed)", color="green", linestyle="--", linewidth=2)
    
    ax.set_title("Price Comparison: Official vs Telegram (Gasoline)")
    ax.set_ylabel("Price (RUB)")
    ax.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = output_dir / "comparison_gasoline_plot.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")

def main():
    input_path = Path("data/processed/merged_enriched.parquet")
    output_dir = Path("data/processed")
    
    if not input_path.exists():
        print(f"Not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    
    setup_plot_style()
    plot_official_prices(df, output_dir)
    plot_comparison(df, output_dir)

if __name__ == "__main__":
    main()
