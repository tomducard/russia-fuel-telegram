# Project context (summary)

Goal:
Build a Telegram-based Domestic Fuel Stress Index for Russia.

Rationale:
- Telegram captures micro-level signals (prices, shortages, rationing) not fully reflected in official statistics.
- Focus is on domestic fuel (diesel/essence), NOT on predicting international gas prices.

Pipeline:
- Scrape ~50–150 public Telegram channels (regional news, logistics, trucking, agriculture).
- Deduplicate reposts.
- Extract RU keywords (shortages, rationing, queues) and ruble prices.
- Aggregate daily features.
- Build a Fuel Stress Index (weighted z-scores).
- Merge with official weekly/monthly fuel CPI (Rosstat CSV).
- Baseline modeling with time-split evaluation (Ridge / simple ML).

Constraints:
- Public channels only.
- No PII, no private chats.
- Telegram = daily signal; official data = validation / anchor.
