"""Live runner — called hourly by GitHub Actions.

Each hour:
  * updates predictions/status.json (heartbeat, so every hourly run commits)
  * if today's predictions are not yet issued and yesterday's GDELT daily file
    is out: refresh data, run models, shortlist, ask the LLM, validate, write
    predictions/<issue-date>/predictions.json (+ a readable README.md and the
    exact inputs used).
"""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import collectors, gdelt, llm, wiki
from .config import (MIN_LEAD_DAYS, ISSUE_HOUR_UTC, GDELT_TOP, HORIZONS, LAST_TARGET, LIVE_END, LIVE_START, MAX_PER_DAY,
                     MIN_HORIZON, PREDICTIONS, ROOT, SCORE_THRESHOLD)
from .dataset import TabularBuilder, build_panel
from .infer import candidates, ensemble, load_models
from .scoring import select
from .poly import live as poly_live

HISTORY_DAYS = 430     # enough for the 365-day rolling features + sequence window


def _status(out_root, **kw) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    s = {"last_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw}
    (out_root / "status.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s))


def recent_events(cutoff: date, shortlist: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    frames = []
    for i in range(days):
        f = GDELT_TOP / f"{cutoff - timedelta(days=i):%Y%m%d}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame(columns=["date", "country_name", "type", "actor1", "actor2", "articles", "url"])
    top = pd.concat(frames)
    keys = set(zip(shortlist["country"], shortlist["type"]))
    top = top[[k in keys for k in zip(top["country"], top["type"])]]
    names = dict(zip(shortlist["country"], shortlist["country_name"]))
    top = top.assign(country_name=top["country"].map(names))
    return (top.sort_values("articles", ascending=False)
               .groupby(["country", "type"]).head(4).head(120)
               .fillna(""))


def _gdelt_candidates(issue_day: date, cutoff: date) -> pd.DataFrame:
    gdelt.backfill(cutoff - timedelta(days=HISTORY_DAYS), cutoff, workers=8)
    if not (gdelt.GDELT_AGG / f"{cutoff:%Y%m%d}.parquet").exists():
        raise RuntimeError(f"GDELT daily file for {cutoff} not available")
    # early-warning precursors: FCDO travel-advice history (full refresh, ~1 min) and
    # GDELT threat/ultimatum aggregates for the feature window
    from .precursors import fcdo, threats
    try:
        fcdo.update()
    except Exception as e:  # noqa: BLE001 - fall back to the cached copy
        print(f"fcdo refresh failed: {e!r}")
    threats.backfill(cutoff - timedelta(days=HISTORY_DAYS), cutoff)
    views = wiki.fetch(start=cutoff - timedelta(days=HISTORY_DAYS), end=cutoff)
    panel = build_panel(start=cutoff - timedelta(days=HISTORY_DAYS), end=cutoff, views=views)
    assert panel.days[-1].date() == cutoff, "panel does not end at cutoff"
    ci = len(panel.days) - 1
    P = ensemble(load_models(final=True), panel, TabularBuilder(panel), [ci])[0]
    max_h = min(max(HORIZONS), (LAST_TARGET - cutoff).days)
    cands = candidates(panel, ci, P, MIN_HORIZON, max_h)
    return cands[cands["target"] >= issue_day + timedelta(days=MIN_LEAD_DAYS)]   # rule: >= 1 day after issue


def run(now: datetime | None = None, dry_run: bool = False, force: bool = False,
        use_llm: bool = True) -> None:
    now = now or datetime.now(timezone.utc)
    issue_day = now.date()
    out_root = ROOT / ("predictions_dryrun" if dry_run else "predictions")
    out_dir = out_root / issue_day.isoformat()

    if (out_dir / "predictions.json").exists() and not force:
        return _status(out_root, state="issued", issue_day=str(issue_day))
    if not dry_run and not (LIVE_START <= issue_day <= LIVE_END):
        return _status(out_root, state="outside live window")
    if not dry_run and not force and now.hour < ISSUE_HOUR_UTC:
        return _status(out_root, state=f"waiting for {ISSUE_HOUR_UTC}:00 UTC issue slot")

    cutoff = issue_day - timedelta(days=1)
    # GDELT path (context + gated candidates). Any failure here must not cost the day:
    # the market candidates below don't depend on it.
    cands = pd.DataFrame()
    try:
        cands = _gdelt_candidates(issue_day, cutoff)
    except Exception as e:  # noqa: BLE001
        print(f"GDELT path failed, continuing with markets only: {e!r}")

    insider = pd.DataFrame()
    try:
        insider = poly_live.signals(now)
        # NB: poly_live.boost (market hazard OR'd into the GDELT probabilities) is disabled:
        # never validated, and once Claude maps markets to event types it duplicates the
        # market as inflated "gdelt" candidates (E 7.0 vs the market's own 5.75).
    except Exception as e:  # noqa: BLE001 - polymarket outage must not stop predictions
        print(f"polymarket signals unavailable: {e!r}")

    # Two candidate sources:
    #   gdelt      : onset model (quiet country/type pairs only) — rarely clears the bar
    #   polymarket : active Polymarket + Manifold event markets, best date from price hazard — main source
    g = (select(cands, k=15, threshold=SCORE_THRESHOLD - 1.0, max_per_country=3, max_per_country_type=2)
         .assign(source="gdelt", question="", event_phrase="", market_site="")) if len(cands) else pd.DataFrame()
    mc = pd.DataFrame()
    if not insider.empty:
        mc = insider[(insider["exp_score"] >= poly_live.MARKET_THRESHOLD - 0.5)
                     & (pd.to_datetime(insider["target"]).dt.date >= issue_day + timedelta(days=MIN_LEAD_DAYS))
                     & (pd.to_datetime(insider["target"]).dt.date <= LAST_TARGET)].head(15)
        site = mc["source"].fillna("polymarket") if "source" in mc else "polymarket"
        mc = mc.assign(market_site=site, source="polymarket", p=mc["p_now"], base_rate_365d=float("nan"),
                       h=(pd.to_datetime(mc["target"]) - pd.Timestamp(cutoff)).dt.days)
    cols = ["source", "country", "country_name", "type", "target", "h", "p", "exp_score",
            "base_rate_365d", "question", "event_phrase", "market_site"]
    pool = pd.concat([x[cols] for x in (g, mc) if len(x)], ignore_index=True) if len(g) or len(mc) \
        else pd.DataFrame(columns=cols)
    pool = pool.sort_values("exp_score", ascending=False).reset_index(drop=True)
    pool.insert(0, "cid", range(1, len(pool) + 1))
    from .poly.live import MARKET_THRESHOLD
    bar = pool["source"].map({"gdelt": SCORE_THRESHOLD, "polymarket": MARKET_THRESHOLD})
    default = pool[pool["exp_score"] >= bar].head(MAX_PER_DAY)

    heads = collectors.headlines()
    movers = collectors.market_movers(collectors.polymarket())
    recent = recent_events(cutoff, pool)
    from .precursors import live_signals
    early = live_signals.collect(now, cutoff)

    llm_meta, preds = {}, []
    if use_llm and len(pool):
        try:
            preds, llm_meta = llm.choose(issue_day, pool, recent, heads, movers, insider, early)
        except Exception as e:  # noqa: BLE001 - never miss a day because of the API
            llm_meta = {"error": repr(e), "trace": traceback.format_exc()}
    if not preds and not (llm_meta.get("output") is not None):
        preds = llm.template_fallback(default)   # LLM unavailable -> model's own picks

    issued_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    valid = []
    for p in preds:                       # drop (never crash on) anything that breaks the rules
        d = date.fromisoformat(str(p.get("predicted_date")))
        if not (issue_day + timedelta(days=MIN_LEAD_DAYS) <= d <= LAST_TARGET) or not p.get("countries") or not p.get("event"):
            print(f"dropping invalid prediction: {p}")
            continue
        p["issued_at_utc"] = issued_at
        valid.append(p)
    preds = valid[:MAX_PER_DAY]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(
        {"issue_day": str(issue_day), "issued_at_utc": issued_at, "data_cutoff": str(cutoff),
         "predictions": preds}, indent=2))
    pool.to_csv(out_dir / "shortlist.csv", index=False)
    if not insider.empty:
        insider.to_csv(out_dir / "polymarket_candidates.csv", index=False)
    (out_dir / "early_warning.json").write_text(json.dumps(early, indent=1, default=str))
    (out_dir / "llm.json").write_text(json.dumps(llm_meta, indent=2, default=str))
    lines = [f"# Predictions issued {issued_at} (data to end of {cutoff})", "",
             "| # | Country | Event | Date | model p | E[score] |", "|---|---|---|---|---|---|"]
    lines += [f"| {i} | {', '.join(p['countries'])} | {p['event']} | {p['predicted_date']} | "
              f"{p['model_p']:.2f} | {p['expected_score']:.1f} |" for i, p in enumerate(preds, 1)]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    _status(out_root, state="issued", issue_day=str(issue_day), n=len(preds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="run outside the live window, write to predictions_dryrun/")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--as-of", help="pretend 'now' is this UTC datetime (dry runs only)")
    a = ap.parse_args()
    now = datetime.fromisoformat(a.as_of).replace(tzinfo=timezone.utc) if a.as_of else None
    run(now=now, dry_run=a.dry_run, force=a.force, use_llm=not a.no_llm)


if __name__ == "__main__":
    main()
