"""Forward 14-day forecast snapshot for the calendar UI: market candidates (Polymarket +
Manifold), GDELT surprise-model risks and early-warning signals -> reports/calendar_data.json."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from event_prediction.config import REPORTS
from event_prediction.poly import live as poly_live
from event_prediction.poly.strategy import first_occurrence_probs
from event_prediction.precursors import live_signals

HORIZON = 14
now = datetime.now(timezone.utc)
today = pd.Timestamp(now).normalize()
out = {"generated_utc": now.isoformat(timespec="minutes"), "horizon_days": HORIZON,
       "market_bar": poly_live.MARKET_THRESHOLD}

sig = poly_live.signals(now, max_lead=HORIZON)
mk = []
for r in sig.itertuples():
    px = first_occurrence_probs(r.p_now, int(r.days_left))[:HORIZON]
    mk.append({"question": r.question, "site": getattr(r, "source", "polymarket"), "country": r.country_name,
               "target": str(r.target), "p_now": round(float(r.p_now), 3), "days_left": int(r.days_left),
               "exp_score": round(float(r.exp_score), 2), "net": round(float(r.exp_score) - 3, 2),
               "daily_p": [round(float(x), 4) for x in px],
               "p_jump24": None if pd.isna(getattr(r, "p_jump24", np.nan)) else round(float(r.p_jump24), 3),
               "p_soon72": None if pd.isna(getattr(r, "p_soon72", np.nan)) else round(float(r.p_soon72), 3),
               "lam_price": round(float(r.lam_price), 4),
               "lam_insider": None if pd.isna(getattr(r, "lam_insider", np.nan)) else round(float(r.lam_insider), 4),
               "family": r.family, "condition_id": r.condition_id,
               "volume": None if pd.isna(r.volume) else float(r.volume)})
out["markets"] = sorted(mk, key=lambda x: -x["exp_score"])

# GDELT surprise model: top country/type onset risks for the next week (context, not a pick source)
try:
    from event_prediction import gdelt, wiki
    from event_prediction.dataset import TabularBuilder, build_panel
    from event_prediction.infer import ensemble, load_models
    from event_prediction.config import HORIZONS, TYPE_NAMES, COUNTRIES
    from event_prediction.precursors import fcdo, threats
    cutoff = (today - pd.Timedelta(days=1)).date()
    gdelt.backfill(cutoff - timedelta(days=430), cutoff, workers=8)
    fcdo.update()
    threats.backfill(cutoff - timedelta(days=430), cutoff)
    views = wiki.fetch(start=cutoff - timedelta(days=430), end=cutoff)
    panel = build_panel(start=cutoff - timedelta(days=430), end=cutoff, views=views)
    ci = len(panel.days) - 1
    P = ensemble(load_models(final=True), panel, TabularBuilder(panel), [ci])[0]   # C,T,H
    p7 = 1 - np.prod(1 - P, axis=2)
    base = panel.onset[max(0, ci - 364):ci + 1].mean(0)
    idx = np.dstack(np.unravel_index(np.argsort(-p7, axis=None)[:15], p7.shape))[0]
    out["gdelt_model"] = {"cutoff": str(cutoff), "panel_days": len(panel.days), "countries": len(panel.countries),
                          "features": len(panel.feat_names), "event_types": len(TYPE_NAMES)}
    out["gdelt_risks"] = [{"country": COUNTRIES[panel.countries[c]], "type": TYPE_NAMES[t],
                           "p_week": round(float(p7[c, t]), 4), "lift": round(float(p7[c, t] / max(base[c, t] * 8, 1e-4)), 1),
                           "peak_day": str((panel.days[ci] + pd.Timedelta(days=HORIZONS[int(np.argmax(P[c, t]))])).date())}
                          for c, t in idx]
except Exception as e:  # noqa: BLE001
    print("gdelt risks unavailable:", repr(e)); out["gdelt_risks"] = []

out["early_warning"] = live_signals.collect(now, (today - pd.Timedelta(days=1)).date())
REPORTS.mkdir(exist_ok=True)
(REPORTS / "calendar_data.json").write_text(json.dumps(out, indent=1, default=str))
print("markets", len(mk), "gdelt", len(out["gdelt_risks"]))
