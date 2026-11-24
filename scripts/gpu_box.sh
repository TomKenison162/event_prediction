#!/usr/bin/env bash
# One-shot setup on a rented Linux GPU box (RunPod / Lambda / vast.ai, CUDA image).
# Copy the repo (or git clone your private repo), then:  bash scripts/gpu_box.sh
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
uv sync
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# copy data/gdelt_agg + data/wiki from your laptop first (rsync) to skip the ~1h backfill
SEQ_ARGS="${SEQ_ARGS:---d-model 512 --t-layers 6 --c-layers 4 --heads 8 --bs 32}" bash scripts/train_all.sh
echo "Now copy models/ back to your laptop and commit it."
