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

## Quickstart
1) Install deps: `pip install -e .[dev]`
2) Copy `.env.example` to `.env` and fill in `TG_API_ID`, `TG_API_HASH`, and optionally `TG_SESSION_NAME`.
3) Update `channels/channels_seed.csv` with public channels and `channels/keywords_ru.yaml` with keywords.

### CLI usage (`rft`)
- Scrape messages to Parquet:
  - `rft scrape --channels channels/channels_seed.csv --output data/raw/messages.parquet --limit 300`
- Build daily features + Fuel Stress Index:
  - `rft features --raw data/raw/messages.parquet --keywords channels/keywords_ru.yaml --output data/interim/daily_features.parquet`
- Merge with official CSV (must include `date` column):
  - `rft merge-official --features data/interim/daily_features.parquet --official-csv path/to/official.csv --output data/processed/merged.parquet`
- Train baseline Ridge model (time split):
  - `rft train --data data/processed/merged.parquet --target official_metric`

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
