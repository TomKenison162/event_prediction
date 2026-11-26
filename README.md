# Event prediction model

Forecasts unscheduled, checkable world events (country + event + single date) for the
Polymarket Insider Model Competition (live run 8–21 Dec 2026, events up to 28 Dec).
Each prediction scores (judge score − 3), so the system only predicts when the expected
judge score clears the cost, and predicting nothing on a day is allowed.

## Where predictions come from

```
Polymarket + Manifold event markets ──► filters ──► price → daily hazard → best date → E[score] ──┐
   (trade / bet tapes)                    │                                                       │ candidates
                                          └──► insider-flow transformer ── alert ───────────────┐ │ E ≥ 3.5
GDELT daily events, FCDO travel advice,                                                       │ │
State Dept advisories, threat spikes ──► LightGBM + spatio-temporal transformer (gated, E≥5) ─┤ │
headlines, military flights ───────────► early-warning context ───────────────────────────────┤ │
                                                                                              ▼ ▼
                                       Claude (claude-opus-5-5): picks ≤5, words them checkably,
                                       rejects scheduled / no-event-day / not-BBC-worthy markets
                                                               │
                                                               ▼
                                         predictions/<issue-date>/predictions.json (22:00 UK)
```

**The decision core is not a learned model.** For a market priced *p* with *N* days to its
deadline: daily hazard λ = 1 − (1 − p)^(1/N); P(first occurrence on day k) = (1 − λ)^(k−1)·λ;
expected judge score of predicting day d = 0.9 · Σ_k P(k) · credit(|k − d|), credit 10 / 8.5 /
6.5 / 6.5 / 4.5 ×4 for offsets 0–7. Predict the best d when E ≥ 3.5. Every learned adjustment
we tried (price calibration, insider blend, date-ladder term structure, FCDO flags) lost
points out of sample, so none is used in selection.

| Component | Role live | Out-of-sample result |
|---|---|---|
| Market hazard + date optimiser (`poly/strategy.py`, `poly/live.py`, `poly/manifold.py`) | selects predictions | +27 per 14 days, 79% of windows positive (Oct 2025 – Nov 2026, Polymarket + Manifold, 22:00 issue) |
| Insider-flow transformer (`poly/insider.py`) | alert in Claude's prompt | price-jump head AUC 0.85 vs 0.79 baseline; hurt the score when blended in |
| GDELT surprise-onset LightGBM + spatio-temporal transformer | gated candidates (E ≥ 5) and context | AUC 0.967 / 0.942, but ~1.4 points per prediction vs the 3-point cost |
| Early warning (`precursors/`): FCDO notes, State Dept, threat spikes, OpenSky | context | FCDO "leave now" → 8.7× onset rate within 7 days |
| Claude analyst (`llm.py`) | final pick + wording | – |

Naive "just use Polymarket" strategies (top markets, predict tomorrow / the deadline / mid-window)
lose 7–123 points per 14 days on the same data; the edge is the date choice and the bar.

## Rules compliance

- **No look-ahead.** Market features use trades strictly before the issue time; wallet reputation
  only counts markets resolved before each trade; GDELT uses the previous day's file.
- **Issue time.** Nothing is issued before 22:00 UTC (= UK time in December); targets are ≥
  `MIN_LEAD_DAYS` (default 1 calendar day) after the issue day and ≤ 28 Dec; at most 5 per day.
- **Audit trail.** Each day's folder holds the shortlist, Claude's output and rejections, the
  early-warning signals, and `data/snapshots/` keeps every live input with a timestamp.

## Commands

```bash
uv sync
uv run python -m event_prediction.predict --dry-run [--no-llm] [--force]   # full live pipeline now
uv run python -m event_prediction.poly.strategy --source poly_wide+manifold  # market backtest
uv run python -m event_prediction.poly.data --wide                           # Polymarket history
uv run python -m event_prediction.poly.manifold                              # Manifold history
uv run python -m event_prediction.poly.insider                               # insider model
bash scripts/train_all.sh                                                    # GDELT models
scripts/ui.sh [--refresh]                                                    # local console, http://localhost:8000/ui/
```

macOS: LightGBM needs OpenMP (`brew install libomp`, or point it at torch's bundled copy).

## GitHub Actions

`.github/workflows/predict.yml` runs hourly at :07 UTC plus a 22:37 backup. Every run commits
`predictions/status.json` (heartbeat); the first run at or after 22:00 UTC on each live day
issues and commits that day's predictions. Claude runs through the Claude Code CLI with the
repo secret `CLAUDE_CODE_OAUTH_TOKEN` (or the API with `ANTHROPIC_API_KEY`); without either,
the model's default picks are issued with template wording. No secrets are stored in the repo.
