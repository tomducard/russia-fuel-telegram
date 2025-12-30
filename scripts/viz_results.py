import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_pipeline_results():
    input_path = Path("data/processed/merged_enriched.parquet")
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # --- PRINT DATASET STRUCTURE FOR USER ---
    print("\n" + "="*50)
    print(" DATASET DISPONIBLE POUR LE ML (Toutes les colonnes)")
    print("="*50)
    print(f"Nombre total de lignes : {len(df)}")
    print("Colonnes disponibles :")
    for col in sorted(df.columns):
        print(f" - {col}")
    print("="*50 + "\n")
    # ----------------------------------------
    # Filter for relevant period - disabled to show full history
    # start_date = "2023-01-01" 
    # df = df[df["date"] >= start_date].copy()
    
    # Plot ranges: Show full available data
    
    plt.style.use('ggplot')
    # Create 4 subplots sharing X axis
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    
    # Plot 0: Official Price (Target)
    ax0.plot(df["date"], df["Diesel_RUB"], label="Official Price (Diesel RUB)", color="navy", linewidth=2)
    ax0.set_title("1. Target: Official Diesel Price (Crisis Event = Spike)")
    ax0.set_ylabel("Price (RUB)")
    ax0.legend(loc="upper left")
    ax0.grid(True)
    
    # Plot 1: Macro Driver (USD/RUB)
    # Check if column exists (it is in enriched dataset)
    if "usd_rub" in df.columns:
        ax1.plot(df["date"], df["usd_rub"], label="USD/RUB Exchange Rate", color="darkgreen", linewidth=2)
    else:
        ax1.text(0.5, 0.5, "USD/RUB Data Missing", ha='center')
    ax1.set_title("2. Top Predictor (Macro): USD/RUB Exchange Rate")
    ax1.set_ylabel("RUB per USD")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    
    # Plot 2: Volume Signal (Buzz)
    ax2.bar(df["date"], df["unique_messages"], label="Daily Message Volume", color="purple", alpha=0.6, width=1.0)
    # Add trend line for volume
    if "total_messages" in df.columns: # fallback
         pass
    ax2.set_title("3. Top Predictor (Social): Telegram Volume (Buzz)")
    ax2.set_ylabel("Messages/Day")
    ax2.legend(loc="upper left")
    ax2.grid(True)
    
    # Plot 3: Logistics Specifics (The "Ground Truth")
    ax3.plot(df["date"], df["share_logistics_terms"], label="Share of Logistics/Trucker Terms", color="firebrick", linewidth=1.5)
    ax3.set_title("4. Confirmation Signal: Logistics Complaints (Truckers)")
    ax3.set_ylabel("Share of Discussion")
    ax3.legend(loc="upper left")
    ax3.grid(True)
    
    # Formatting
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = "data/processed/results_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_pipeline_results()
