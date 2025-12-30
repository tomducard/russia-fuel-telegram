import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_telegram_prices():
    input_path = Path("data/processed/merged_final.parquet")
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # helper for plotting trend
    def plot_trend(col, label, color):
        series = df[df[col] > 0].set_index("date")[col]
        if series.empty:
            return
        # Interpolate and roll
        full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
        s_interp = series.reindex(full_idx).interpolate(method='time', limit=7)
        rolling = s_interp.rolling(window=14, min_periods=1).mean()
        
        # Plot Scatter (Raw)
        ax.scatter(series.index, series, color=color, alpha=0.2, s=15)
        # Plot Trend (Line)
        ax.plot(rolling.index, rolling, label=f"{label} (14d MA)", color=color, linewidth=2.5)

    plot_trend("avg_price_diesel", "Diesel", "blue")
    plot_trend("avg_price_gasoline", "Gasoline (AI)", "red")

    ax.set_title("Telegram Fuel Prices: Diesel vs Gasoline")
    ax.set_ylabel("Price (RUB / Liter)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    
    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = "data/processed/telegram_prices_fueltype.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_telegram_prices()
