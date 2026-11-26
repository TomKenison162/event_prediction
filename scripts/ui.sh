#!/usr/bin/env bash
# Local technical dashboard. Usage: scripts/ui.sh [--refresh]   then open http://localhost:8000/ui/
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--refresh" ]]; then
  uv run python scripts/forecast_calendar.py      # live 14-day forecast (~10 min)
fi
uv run python scripts/build_ui_summary.py         # model metrics, backtests, data inventory
echo "open http://localhost:8000/ui/"
exec uv run python -m http.server 8000 --bind 127.0.0.1
