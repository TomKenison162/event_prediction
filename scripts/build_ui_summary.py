"""Collect model metadata, backtest reports and data inventory into ui/data/summary.json
for the local technical dashboard (ui/index.html). Re-run after any retrain/backtest."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M, R, D = ROOT / "models", ROOT / "reports", ROOT / "data"


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def nfiles(p, pat="*.parquet"):
    p = Path(p)
    return len(list(p.glob(pat))) if p.exists() else 0


def span(p):
    fs = sorted(Path(p).glob("*.parquet")) if Path(p).exists() else []
    return [fs[0].stem, fs[-1].stem] if fs else None


def size_mb(p):
    p = Path(p)
    if not p.exists():
        return 0
    return round(sum(f.stat().st_size for f in (p.rglob("*") if p.is_dir() else [p]) if f.is_file()) / 1e6, 1)


models = {}
for f in M.glob("*"):
    models[f.name] = {"mb": round(f.stat().st_size / 1e6, 2),
                      "modified": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="minutes")}

summary = {
    "built_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
    "meta": {"lgbm": jload(M / "lgbm_meta.json"), "seq": jload(M / "seq_meta.json"),
             "insider": jload(M / "insider_meta.json"), "label_thresholds": jload(M / "label_thresholds.json")},
    "model_files": models,
    "backtests": {
        "gdelt_onset": jload(R / "backtest.json"),
        "poly_07utc_test": jload(R / "poly_strategy.json"),
        "poly_07utc_test_calibrated": jload(R / "poly_strategy_cal.json"),
        "manifold_2024_2026": jload(R / "poly_strategy_manifold.json"),
        "poly_wide_22utc": jload(R / "poly_strategy_poly_wide.json"),
        "poly_wide_plus_manifold_22utc": jload(R / "poly_strategy_poly_wide+manifold.json"),
    },
    "data": [
        {"name": "GDELT 1.0 daily aggregates", "path": "data/gdelt_agg", "files": nfiles(D / "gdelt_agg"), "span": span(D / "gdelt_agg"), "mb": size_mb(D / "gdelt_agg")},
        {"name": "GDELT top events (for context)", "path": "data/gdelt_top", "files": nfiles(D / "gdelt_top"), "span": span(D / "gdelt_top"), "mb": size_mb(D / "gdelt_top")},
        {"name": "GDELT threat / ultimatum codes", "path": "data/gdelt_precursor", "files": nfiles(D / "gdelt_precursor"), "span": span(D / "gdelt_precursor"), "mb": size_mb(D / "gdelt_precursor")},
        {"name": "GDELT 2013-2015 (not yet in training)", "path": "data/gdelt_agg_early", "files": nfiles(D / "gdelt_agg_early"), "span": span(D / "gdelt_agg_early"), "mb": size_mb(D / "gdelt_agg_early")},
        {"name": "Wikipedia pageviews, 74 countries", "path": "data/wiki", "files": nfiles(D / "wiki"), "span": None, "mb": size_mb(D / "wiki")},
        {"name": "UK FCDO travel-advice change notes", "path": "data/precursors/fcdo_changes.parquet", "files": 1, "span": None, "mb": size_mb(D / "precursors")},
        {"name": "Polymarket trade tapes (resolved markets)", "path": "data/poly/trades", "files": nfiles(D / "poly" / "trades"), "span": None, "mb": size_mb(D / "poly" / "trades")},
        {"name": "Manifold bet tapes (resolved markets)", "path": "data/manifold/bets", "files": nfiles(D / "manifold" / "bets"), "span": None, "mb": size_mb(D / "manifold")},
        {"name": "Wallet reputation table", "path": "models/wallet_rep.parquet", "files": 1, "span": None, "mb": size_mb(M / "wallet_rep.parquet")},
        {"name": "Hourly live snapshots", "path": "data/snapshots", "files": nfiles(D / "snapshots", "*/*.json"), "span": None, "mb": size_mb(D / "snapshots")},
    ],
    # Experiments run interactively (25-26 Nov 2026); numbers copied from the run output.
    # 14d net = mean over rolling 14-day windows of sum(score - 3); test period Oct 2025 - Nov 2026 unless noted.
    "experiments": [
        {"name": "Issue hour", "verdict": "adopted", "detail": "Polymarket, bar 3.5: 07:00 +13.5 · 14:00 +12.7 · 19:00 +15.6 · 22:00 +17.5 (70% windows positive)", "change": "ISSUE_HOUR_UTC = 22"},
        {"name": "Wider Polymarket set", "verdict": "adopted", "detail": "39 tags, min volume $10k: +18.6 vs +17.5, positive windows 77% vs 70%", "change": "live.signals(tags=TAGS+EXTRA_TAGS, min_volume=10k)"},
        {"name": "Add Manifold markets", "verdict": "adopted", "detail": "Manifold alone +12.6; combined +29.2; with cross-site duplicates removed +27.0 (79% positive)", "change": "manifold.live_signals merged, dedupe on (country, date)"},
        {"name": "Insider model in the score", "verdict": "rejected", "detail": "Linear blend of insider hazard into E[score] (fit Apr-Jul, test Aug-Nov 2026): +4.1 -> -0.1 at bar 3.5", "change": "insider kept as LLM alert only"},
        {"name": "Price calibration (isotonic)", "verdict": "rejected", "detail": "Fit pre-Oct 2025, lowered the test-period score", "change": "models/poly_price_iso.pkl unused"},
        {"name": "Date-ladder term structure", "verdict": "rejected", "detail": "Piecewise hazard from sibling deadline markets: bar 3.5 +7.0 vs constant hazard +8.7", "change": "none"},
        {"name": "FCDO flag on markets", "verdict": "rejected", "detail": "Flagged markets resolve YES 85% vs 63%, but realised score = expected: price already includes it", "change": "context only"},
        {"name": "GDELT onset model + precursor features", "verdict": "rejected for picks", "detail": "AP 0.0126 -> 0.0137, AUC 0.967; still ~1.4 realised per prediction vs 3-point cost", "change": "bar 5.0, context only"},
        {"name": "Persistence model (v1)", "verdict": "rejected", "detail": "Scored well by predicting ongoing violence (+448 per window) but these are not surprises", "change": "label switched to onset"},
        {"name": "Kalshi", "verdict": "rejected", "detail": "Open events are economics/sports/elections; almost no geopolitical event markets", "change": "none"},
    ],
    "fcdo_lifts": {"leave_now": 8.7, "staff_withdrawn": 4.7, "travel_upgrade": 2.9, "tensions": 2.8, "airspace": 2.5, "unrest": 2.2},
}
out = ROOT / "ui" / "data"
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print("wrote", out / "summary.json")
