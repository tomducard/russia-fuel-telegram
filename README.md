# Russia Fuel Telegram (rft)

Pipeline to scrape public Telegram channels, engineer fuel-related signals, merge with official data, and train a baseline model.

## Requirements
- Python 3.11+
- Telegram API credentials in environment: `TG_API_ID`, `TG_API_HASH`, optional `TG_SESSION_NAME`
- Dependencies from `pyproject.toml` (`pip install -e .[dev]` recommended)

## Project layout
- `src/rft`: package modules (scraping, normalization, keywords/prices, features, official merge, modeling, CLI)
- `channels/`: seed channel list + keyword YAML
- `data/raw|interim|processed`: storage for artifacts (gitignored)

## Quickstart (Full Pipeline)

This project has been updated to handle large-scale historical data (2021-2025). Follow these steps to reproduce the "Hybrid Model" results.

### 1. Scraping (Deep Mine)
Collect all messages since 2021 from channels listed in `channels/channels_seed_extended.csv`.
```bash
python -m rft.cli scrape --channels channels/channels_seed.csv --output data/raw/messages.parquet --limit 0 --since 2021-01-01
```
*Note: This saves incremental chunks in `data/raw/chunks/` to prevent data loss.*

### 2. Feature Extraction (NLP + Indexes)
Compute sentiment, text embeddings, and specific keyword indices (Stress, Logistics).
```bash
python -m rft.cli features --raw data/raw/messages.parquet --output data/interim/daily_features.parquet
```

### 3. Data Enrichment (Macro + Rolling Stats)
Merge with official macro-economic data (USD/RUB) and compute rolling volatility/trends.
```bash
python scripts/enrich_dataset.py --input data/interim/daily_features.parquet
```
*Output: `data/processed/merged_enriched.parquet`*

### 4. Training (XGBoost Hybrid)
Train the final model using the optimized hyperparameters and "Crisis" target (7-day ahead).
```bash
python -m rft.cli train --data data/processed/merged_enriched.parquet --mode classification --model-type xgb --target crisis_7d
```

### 5. Visualization (Probability Mode)
Generate the "Risk Curve" graph showing probabilities vs real crisis events.
```bash
python scripts/viz_predictions.py
```
*Output: `data/processed/probability_plot.png`*

## Data collection
- **.env setup**: Copy `.env.example` to `.env`, then paste the values from https://my.telegram.org/apps into `TG_API_ID` and `TG_API_HASH`. `TG_SESSION_NAME` is optional; if omitted the CLI uses `rft_session`.
- **First login flow**: On the first `rft scrape` run Telethon will prompt in the terminal for your phone number and a verification code sent through official Telegram channels. Authenticate once and the session becomes trusted for future runs.
- **Session artifact**: Telethon creates `<session_name>.session` (e.g., `rft_session.session`) in the project root. Keep this file locally so you do not need to log in every run.
- **Do not commit**: `.env`, any `*.session` files, and everything under `data/raw`, `data/interim`, or `data/processed` must stay out of git (they are gitignored by default).

To sanity-check freshly scraped data after running the pipeline, execute:
```
python scripts/check_outputs.py \
  --raw data/raw/messages_sample.parquet \
  --daily data/interim/daily_features_sample.parquet
```
The script reports total message counts, date coverage, and mean daily keyword shares.

## Testing
Run unit tests:
```
pytest
```

## Notes
- Scraping targets public channels only; no private data or PII.
- Raw artifacts live under `data/raw` (gitignored); intermediate and processed under `data/interim` and `data/processed`.
