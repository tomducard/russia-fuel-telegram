
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from rft import model

def viz_predictions(data_path: str, target_col: str = "crisis_7d"):
    """
    Train model on past, predict probability on future, and plot Risk Curve vs Danger Zones.
    """
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Setup Data
    # Use the logic from model.py to get features
    features = model.get_feature_selection(df)
    
    # Drop rows without target
    data = df.dropna(subset=[target_col]).sort_values("date")
    
    # Time Split (80/20)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    X_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col].astype(int)
    X_test = test_df[features].fillna(0.0)
    y_test = test_df[target_col].astype(int)
    
    # 2. Train Model (Using Best Known Params + Scale Pos Weight)
    print("Training XGBoost Probability Model...")
    clf = XGBClassifier(
        n_estimators=133,
        max_depth=4,
        learning_rate=0.015,
        subsample=0.78,
        colsample_bytree=0.82,
        gamma=0.2,
        scale_pos_weight=12,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", clf)
    ])
    pipeline.fit(X_train, y_train)
    
    # 3. Predict Probabilities
    # We predict on BOTH Train and Test to show the full history fit
    full_proba = pipeline.predict_proba(data[features].fillna(0.0))[:, 1]
    data["crisis_risk"] = full_proba
    
    # 4. Plot
    plt.style.use('bmh') # Clean aesthetic
    fig, ax = plt.subplots(figsize=(15, 8))
    
    dates = data["date"]
    
    # A. The Risk Curve (Probability)
    ax.plot(dates, data["crisis_risk"], color="#2c3e50", linewidth=2, label="Model Risk Assessment (0-100%)")
    
    # Fill area under the curve for better visual
    ax.fill_between(dates, data["crisis_risk"], color="#3498db", alpha=0.2)
    
    # B. The Danger Zones (Real Crisis)
    # We want to highlight where crisis_7d == 1
    # Create a boolean mask
    crisis_mask = data[target_col] == 1
    # We use a secondary axis or just vertical spans?
    # Vertical spans are cleaner for "Zones"
    
    # Find contiguous regions of crisis
    # Simple hack: bar chart with 100% height where crisis exists
    ax.fill_between(dates, 0, 1, where=crisis_mask, color="#e74c3c", alpha=0.3, label="Real Crisis Event (Target)")
    
    # C. Threshold Line
    threshold = 0.60 # The one we found optimal
    ax.axhline(y=threshold, color="#e67e22", linestyle="--", linewidth=1.5, label=f"Decision Threshold ({int(threshold*100)}%)")
    
    ax.set_title("Russian Fuel Crisis: Early Warning System (Probability vs Reality)", fontsize=16, pad=20)
    ax.set_ylabel("Risk Limit (Probability)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    
    # Formatting
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add annotations for Split
    split_date = test_df["date"].iloc[0]
    ax.axvline(x=split_date, color="black", linestyle=":", linewidth=2, label="Train / Test Split")
    ax.text(split_date, 1.05, "  Predicting the Future  -->", ha="left", va="bottom", fontsize=10, color="black")
    
    plt.tight_layout()
    output_file = "data/processed/probability_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    viz_predictions("data/processed/merged_enriched.parquet")
