"""Replay the competition on held-out history.

For every 14-day window in the test period: each day, take the cutoff C (data to
end of yesterday), score candidates, select <=5, then score them against what
actually happened. Reports the distribution of competition totals (sum of
score-3) and the per-prediction hit profile, for a sweep of thresholds.

Uses the non-final (train/val only) models so the test period is truly unseen.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import HORIZONS, MIN_HORIZON, REPORTS, TYPE_NAMES
from .dataset import TabularBuilder, build_panel
from .infer import candidates, ensemble, load_models
from .scoring import PENALTY, realised_score, select


def run(start: str, end: str, thresholds: list[float], live_days: int = 14,
        max_lead: int = 7) -> dict:
    p = build_panel()
    tb = TabularBuilder(p)
    m = load_models(final=False)
    cut = np.where((p.days >= pd.Timestamp(start)) & (p.days <= pd.Timestamp(end)))[0]
    cut = cut[cut + max(HORIZONS) < len(p.days)]
    print(f"backtest over {len(cut)} cutoff days")

    allc = []
    for i in range(0, len(cut), 20):
        chunk = cut[i:i + 20]
        P = ensemble(m, p, tb, chunk)
        for j, c in enumerate(chunk):
            cd = candidates(p, c, P[j], MIN_HORIZON, min(max(HORIZONS), max_lead + 1))
            cd["cutoff"] = c
            # keep only plausible ones to save memory
            allc.append(cd[cd["exp_score"] >= min(thresholds)])
    cands = pd.concat(allc, ignore_index=True)
    ci = {c: i for i, c in enumerate(p.countries)}
    ti = {t: i for i, t in enumerate(TYPE_NAMES)}
    cands["realised"] = [
        realised_score(p.Y[:, ci[r.country], ti[r.type]], r.cutoff + r.h) for r in cands.itertuples()
    ]

    out = {}
    for th in thresholds:
        picks = (cands.groupby("cutoff", group_keys=False)
                      .apply(lambda g: select(g, threshold=th)))
        picks["net"] = picks["realised"] - PENALTY
        daily = picks.groupby("cutoff")["net"].sum().reindex(cut, fill_value=0.0)
        # every possible 14-day competition window
        win = daily.rolling(live_days).sum().dropna()
        out[th] = {
            "n_preds_per_day": float(picks.groupby("cutoff").size().reindex(cut, fill_value=0).mean()),
            "mean_realised": float(picks["realised"].mean()) if len(picks) else 0.0,
            "mean_expected": float(picks["exp_score"].mean()) if len(picks) else 0.0,
            "hit_exact": float((picks["realised"] == 10).mean()) if len(picks) else 0.0,
            "miss_rate": float((picks["realised"] == 0).mean()) if len(picks) else 0.0,
            "window_total_mean": float(win.mean()),
            "window_total_p10": float(win.quantile(0.1)),
            "window_total_p90": float(win.quantile(0.9)),
            "top_types": picks["type"].value_counts().head(8).to_dict(),
            "top_countries": picks["country_name"].value_counts().head(8).to_dict(),
        }
        print(f"\nthreshold {th}: " + json.dumps({k: v for k, v in out[th].items()
                                                  if not k.startswith("top")}, indent=1))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps({str(k): v for k, v in out.items()}, indent=2))
    cands.to_parquet(REPORTS / "backtest_candidates.parquet", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-07-01")
    ap.add_argument("--end", default="2026-11-15")
    ap.add_argument("--thresholds", default="4,5,6,7,8")
    a = ap.parse_args()
    run(a.start, a.end, [float(x) for x in a.thresholds.split(",")])


if __name__ == "__main__":
    main()
