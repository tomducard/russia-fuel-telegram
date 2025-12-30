import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_logistics():
    input_path = Path("data/processed/merged_enriched.parquet")
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    
    df_zoom = df.copy()
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Logistics Share
    ax.plot(df_zoom["date"], df_zoom["share_logistics_terms"], label="Logistics Terms Share", color="green", linewidth=2)
    
    # Add rolling average for smoother trend
    df_zoom["logistics_ma"] = df_zoom["share_logistics_terms"].rolling(window=7).mean()
    ax.plot(df_zoom["date"], df_zoom["logistics_ma"], label="7-Day Moving Average", color="darkgreen", linestyle="--", linewidth=1.5)

    ax.set_title("Logistics Keyword Frequency (Deep Scrape)")
    ax.set_ylabel("Share of Messages containing Logistics Terms")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = "data/processed/logistics_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_logistics()
