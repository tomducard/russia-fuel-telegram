Objective: scaffold a Python repo that scrapes public Telegram channels (Telethon), deduplicates text, extracts RU keywords + ruble prices, builds daily features + a Fuel Stress Index, merges with an official CSV, and trains a baseline model with time split.

Constraints:
- Python 3.11+
- Public channels only, no user PII
- Raw data under data/raw (gitignored)
- Provide CLI command `rft` with subcommands: scrape, features, merge-official, train
- Keep code modular under src/rft
