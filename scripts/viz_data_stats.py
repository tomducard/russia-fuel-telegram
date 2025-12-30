import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_data_stats():
    # Load dataset
    input_path = Path("data/raw/messages_final.parquet")
    if not input_path.exists():
        # Fallback for dev if final not ready
        input_path = Path("data/raw/messages_combined.parquet")
    
    if not input_path.exists():
        print("No message data found.")
        return

    df = pd.read_parquet(input_path)
    
    # Load channel metadata to attach categories
    channels_df = pd.read_csv("channels/channels_seed.csv")
    # Map channel -> category
    cat_map = dict(zip(channels_df["channel"], channels_df["category"]))
    
    # Prepare stats
    df["category"] = df["channel"].map(cat_map).fillna("unknown")
    
    # 1. Message Count per Channel
    channel_counts = df["channel"].value_counts().reset_index()
    channel_counts.columns = ["channel", "count"]
    
    # 2. Message Count per Category
    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    
    # Plotting
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: By Channel
    sns.barplot(data=channel_counts, x="count", y="channel", ax=axes[0], palette="viridis")
    axes[0].set_title("Messages Scraped by Channel")
    axes[0].set_xlabel("Count")
    
    # Plot 2: By Category
    sns.barplot(data=cat_counts, x="count", y="category", ax=axes[1], palette="rocket")
    axes[1].set_title("Messages Scraped by Category")
    axes[1].set_xlabel("Count")
    
    plt.tight_layout()
    output_path = "data/processed/data_stats_plot.png"
    plt.savefig(output_path)
    print(f"Stats Plot saved to {output_path}")

    # Explicitly save the sector plot alone for clarity
    fig_cat, ax_cat = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cat_counts, x="count", y="category", ax=ax_cat, palette="rocket")
    ax_cat.set_title("Messages Scraped by Sector (Category)")
    ax_cat.set_xlabel("Count")
    plt.tight_layout()
    output_path_sector = "data/processed/data_stats_sector.png"
    plt.savefig(output_path_sector)
    print(f"Sector Plot saved to {output_path_sector}")

    # Print summary specific to "unknown" to identify mapping issues
    if "unknown" in df["category"].values:
        print("Warning: Some channels missing from seed CSV mapping:")
        print(df[df["category"] == "unknown"]["channel"].unique())

if __name__ == "__main__":
    plot_data_stats()
