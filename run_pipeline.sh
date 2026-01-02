#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🚀 Starting Full Russian Fuel Crisis Pipeline..."

# 1. Scraping (Incremental Check)
# Limit 0 means 'Scrape Everything'. Since means 'From 2021'.
echo -e "\n📡 [1/5] Deep Scraping Telegram Channels (2021-2025)..."
python3 -m rft.cli scrape --channels channels/channels_seed.csv --output data/raw/messages.parquet --limit 0 --since 2021-01-01

# 2. Features (NLP)
# Converts raw text into mathematical signals
echo -e "\n🧠 [2/5] Extracting NLP Features & Stress Index..."
python3 -m rft.cli features --raw data/raw/messages.parquet --output data/interim/daily_features.parquet

# 3. Enrichment (Macro)
# Merges with USD/RUB and official stats
echo -e "\n💰 [3/5] Enriching with Macro-Economic Data..."
python3 scripts/enrich_dataset.py --input data/interim/daily_features.parquet --output data/processed/merged_enriched.parquet

# 4. Training (All Models)
echo -e "\n🤖 [4/5] Training Models..."

# 4a. Random Forest (Baseline)
echo "   > Training Random Forest..."
python3 -m rft.cli train --data data/processed/merged_enriched.parquet --mode classification --model-type rf --target crisis_7d

# 4b. LSTM (Deep Learning)
echo "   > Training LSTM (Sequence Model)..."
python3 -m rft.cli train --data data/processed/merged_enriched.parquet --mode classification --model-type lstm --target crisis_7d

# 4c. XGBoost (Final Hybrid)
echo "   > Training XGBoost (Final Production Model)..."
python3 -m rft.cli train --data data/processed/merged_enriched.parquet --mode classification --model-type xgb --target crisis_7d

# 5. Visualization (Report)
# Updates the decision support graph
echo -e "\n📈 [5/5] Generating Prediction Graphs..."
python3 scripts/viz_predictions.py

echo -e "\n✅ PIPELINE SUCCESS! Final report graph updated in data/processed/probability_plot.png"
