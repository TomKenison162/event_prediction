#!/usr/bin/env bash
# Full training pipeline. Run locally or on a rented GPU box.
#   bash scripts/train_all.sh            # default model sizes
#   SEQ_ARGS="--d-model 512 --t-layers 6 --c-layers 4 --heads 8 --bs 32" bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m event_prediction.gdelt            # backfill / top up GDELT history
uv run python -m event_prediction.wiki             # pageviews

# 1) evaluation models (train <= 2024-06, val to 2025-06, test after)
uv run python -m event_prediction.train_lgbm
uv run python -m event_prediction.train_seq ${SEQ_ARGS:-}

# 2) honest backtest of the full system on the unseen test period
uv run python -m event_prediction.backtest

# 3) final models on all data for the live run
uv run python -m event_prediction.train_lgbm --final
EPOCHS=$(uv run python -c "import json;print(json.load(open('models/seq_meta.json'))['epochs_trained'])")
uv run python -m event_prediction.train_seq ${SEQ_ARGS:-} --final --epochs "$EPOCHS" --tag seq_final

# 4) Polymarket insider model
uv run python -m event_prediction.poly.data --skip-markets   # top up trade tapes
uv run python -m event_prediction.poly.insider ${INSIDER_ARGS:-}
uv run python -c "from event_prediction.poly.live import build_wallet_table; build_wallet_table()"
IEP=$(uv run python -c "import json;print(json.load(open('models/insider_meta.json'))['epochs_trained'])")
uv run python -m event_prediction.poly.insider ${INSIDER_ARGS:-} --final --epochs "$IEP"
