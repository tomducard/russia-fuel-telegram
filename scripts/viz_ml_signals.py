import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
DATA_PATH = Path("data/processed/merged_enriched.parquet")
OUTPUT_DIR = Path("data/processed/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)

# Ensure date is datetime
df["date"] = pd.to_datetime(df["date"])

# Select relevant columns for ML visualization
ml_cols = [
    "crisis_7d",
    "Diesel_RUB_change",
    "fuel_stress_index",
    "count_logistics_terms",
    "count_shortage_terms",
    "sentiment_mean",
    "unique_messages"
]

# 1. Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
corr = df[ml_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix: Telegram Signals vs Crisis Targets")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ml_correlation_matrix.png")
print(f"Saved {OUTPUT_DIR / 'ml_correlation_matrix.png'}")
plt.close()

# 2. Time Series Overlay: Logistics vs Crisis
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot Crisis Periods (Background)
# We fill areas where crisis_7d == 1
# Need to resample to find contiguous blocks or just bar plot
crisis_dates = df[df["crisis_7d"] == 1]["date"]
# Simple approach: Plot crisis as a bar at the bottom or shaded regions?
# Let's plot Diesel Price Change
sns.lineplot(data=df, x="date", y="Diesel_RUB_change", ax=ax1, color="black", alpha=0.3, label="Daily Price Change")
ax1.set_ylabel("Diesel Price Variation")

# Plot Signal (Logistics)
sns.lineplot(data=df, x="date", y="count_logistics_terms", ax=ax2, color="red", label="Logistics Mentions (Signal)")
ax2.set_ylabel("Logistics Mentions")

# Highlight Crisis
ylim = ax1.get_ylim()
ax1.fill_between(df["date"], ylim[0], ylim[1], where=df["crisis_7d"] == 1, color='red', alpha=0.1, label="Crisis Period (Target)")

plt.title("Leading Indicator Analysis: Logistics Mentions vs Price Crisis")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ml_signal_logistics.png")
print(f"Saved {OUTPUT_DIR / 'ml_signal_logistics.png'}")
plt.close()

# 3. Sentiment Analysis Check
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

sns.lineplot(data=df, x="date", y="Diesel_RUB", ax=ax1, color="black", label="Diesel Price ($RUB)")
sns.lineplot(data=df, x="date", y="sentiment_mean", ax=ax2, color="blue", alpha=0.6, label="Sentiment Score (Neg=-1, Pos=1)")
ax2.axhline(0, color="gray", linestyle="--")
ax2.set_ylabel("Telegram Sentiment")

plt.title("Sentiment Reality Check: Does Negativity correlate with Price hikes?")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ml_signal_sentiment.png")
print(f"Saved {OUTPUT_DIR / 'ml_signal_sentiment.png'}")
plt.close()
