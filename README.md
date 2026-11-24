# Event prediction model

Forecasts unscheduled, checkable world events (country + event + date) for the
Polymarket Insider Model Competition (live run 8–21 Dec 2026).

## How it works

```
GDELT (every news event worldwide, 2016→now) ─┐
Wikipedia pageviews per country ──────────────┼─► daily panel  F[day, country, 168 features]
                                              │         │
                                              │         ├─► LightGBM (tabular rolling features)
                                              │         └─► Spatio-temporal transformer
                                              │              (56-day history per country +
                                              │               cross-country attention)
                                              │                     │ calibrated P(event type t,
                                              │                     ▼ country c, day D)
                                              │         expected judge score per candidate
                                              │         (exact day / ±1 / ±2-3 / ±4-7 credit)
News RSS + Polymarket movers (live) ──────────┴─► Claude: picks ≤5, phrases them checkably,
                                                   vetoes scheduled/unphraseable ones
                                                          │
                                                          ▼
                                     predictions/<issue-date>/predictions.json
```

**Label ("did it happen?").** For each country × event type × day, the
share of that country's news coverage about that event type (from GDELT CAMEO
codes). A day is *notable* when the share is above the noise floor that
peaceful, heavily-covered countries show on 95% of days. Thresholds were fitted on data up to mid-2024 and then
frozen (`models/label_thresholds.json`).

**Event types** (in `config.py`): protest, strike, riot, arrest, deport/expel,
repression, sanctions, diplomatic break, force posture, armed clash, airstrike,
ceasefire violation, abduction, bombing, assassination, mass violence,
de-escalation, agreement. 74 countries.

**Selection.** A prediction scores (judge − 3), so we only predict when the
expected judge score is ≥ `SCORE_THRESHOLD` (default 5). At most 5 per day,
at most 2 per country, and 1 per country+type. Choosing zero is allowed.

**No look-ahead.** Predictions on issue day X use only GDELT data up to the
end of X−1 (GDELT's daily file is published the next morning). Each hourly run
saves the live inputs it used (headlines, markets) to `data/snapshots/` with a timestamp,
and saves the shortlist and LLM output next to each day's predictions.

## Polymarket insider model (`src/event_prediction/poly/`)

This is a separate deep model that reads Polymarket trade tapes and asks whether the
flow knows something the price doesn't.

- **Data:** resolved geopolitics, conflict and diplomacy markets since late 2023 (Gamma
  keyset API) and every trade of $100 or more in them (Data API v2 `/v2/trades`).
- **Per trade:** size, direction (bet on YES or NO), price, the price of the side
  bought (longshot bets), and the wallet's *point-in-time* reputation: prior trades,
  age, win rate and realised PnL counting only markets that had resolved before
  that trade, plus whether this is the wallet's first trade in the market.
- **Samples:** a snapshot every 12h on a fixed UTC grid over each market's life.
  Input is the last 256 trades plus 15 market-level flow features.
- **Heads:** `soon` (the market resolves YES within 72h, i.e. the event happens),
  `yes` (final outcome), `up` (price +10 points within 24h).
- **Split by close date:** train before 2025-10, validation 2025-10 to 2026-03,
  test after. Compared against the price-only and context-only baselines.

At prediction time, Claude maps active markets to (country, event type), with a
cache in `data/poly/market_map.json`. Scheduled-event markets are dropped. Each
market's daily event probability, from the insider model for the next 72h and
from the price-implied hazard after that, is combined by noisy-OR into the main model's
probabilities. The top signals are also shown to the LLM.

```bash
uv run python -m event_prediction.poly.data       # markets + trade tapes
uv run python -m event_prediction.poly.insider    # train/val/test + calibration
uv run python -c "from event_prediction.poly.live import build_wallet_table; build_wallet_table()"
uv run python -m event_prediction.poly.insider --final --epochs N
```

## Commands

```bash
uv sync
uv run python -m event_prediction.gdelt          # backfill GDELT 2016→yesterday (~1h first time)
uv run python -m event_prediction.wiki           # pageviews
uv run python -m event_prediction.train_lgbm     # train/val/test, writes models/lgbm*
uv run python -m event_prediction.train_seq      # transformer, writes models/seq*
uv run python -m event_prediction.backtest       # simulated competitions on held-out data
bash scripts/train_all.sh                        # all of the above + final models

# dry run of the live pipeline at any date (writes predictions_dryrun/)
uv run python -m event_prediction.predict --dry-run [--no-llm] [--as-of 2026-11-25T07:00]
```

Bigger transformer on a rented GPU: `bash scripts/gpu_box.sh` (rsync `data/`
over first to skip the backfill). Sizes are set with
`--d-model/--t-layers/--c-layers/--heads`.

macOS: LightGBM needs OpenMP (`brew install libomp`).

## GitHub Actions

`.github/workflows/predict.yml` runs every hour at :07 UTC. Each run updates
`predictions/status.json` (a heartbeat). The first run each day after
yesterday's GDELT file is out (usually 06:00–08:00 UTC) issues the day's
predictions. Set the repo secret `ANTHROPIC_API_KEY`. Without it, predictions
are phrased from templates instead of by the LLM.

Before lock-in, commit `models/` (the final LightGBM, transformer,
calibrators and label thresholds).
