"""Polymarket-led surprise strategy + an exact backtest on resolved markets.

Each issue day (07:00 UTC) for every open event market:
  * daily hazard of the event from the price (constant until the deadline), optionally
    sharpened over the next 72h by the insider model's P(event within 72h)
  * distribution of the first-occurrence day X over [today+1, deadline]
  * choose the prediction date d maximising E[judge score] = sum_x P(X=x) * credit(|x-d|)
  * predict when E - 3 clears the threshold; max 5 per day, one per event family
Realised score uses the actual event day (resolution time for YES markets that resolve
early; 0 if the market resolved NO).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from ..config import REPORTS
from ..scoring import JUDGE_FACTOR, OFFSET_SCORE, PENALTY
from .data import POLY, load_trades
from .features import epoch_s

EVENT_WORDS = re.compile(
    r"\b(strike|strikes|attack|attacks|invade|invasion|ceasefire|war\b|missile|bomb|military|troops|"
    r"enter|capture|clash|coup|assassinat|killed|ground operation|airstrike|drone|offensive|"
    r"seize|blockade|declare|annex|nuclear test|martial law|ousted|out as|resign|arrested|detained)", re.I)
SCHEDULED_WORDS = re.compile(r"\b(election|primary|vote|referendum|inaugurat|summit|debate|"
                             r"tariff rate|next .* (prime minister|president|leader|pope))", re.I)
# YES must mean "the event happens": drop negated / continuation / count markets
NOT_EVENT = re.compile(r"(^no |\bnot\b|nothing|survive|continues|continue|remain|still|"
                       r"≤|<=|less than|fewer|\d+\s*times|\d+\s*countries|\bstrike \d+ |extended)", re.I)
CREDIT = np.array([OFFSET_SCORE[k] for k in range(8)])         # credit by |offset| 0..7
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_RE = re.compile(rf"\b(by|on|before)\s+(?:the\s+)?(?:end of\s+)?({MONTHS})\s*(\d{{1,2}})?(?:,?\s*(\d{{4}}))?", re.I)
HIT = 0.90          # price level we treat as "the event has happened"


def is_event_market(q: str) -> bool:
    return bool(EVENT_WORDS.search(q)) and not SCHEDULED_WORDS.search(q) and not NOT_EVENT.search(q)


def question_deadline(q: str, end: pd.Timestamp) -> tuple[pd.Timestamp, bool]:
    """(deadline day, single_day) from the question text, falling back to the market end."""
    m = DATE_RE.search(q)
    if not m:
        return end.normalize(), False
    kind, mon, day, year = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
    year = int(year) if year else end.year
    try:
        if day:
            d = pd.Timestamp(f"{mon} {day} {year}", tz="UTC")
        else:                                           # "by March" -> end of March
            d = (pd.Timestamp(f"{mon} 1 {year}", tz="UTC") + pd.offsets.MonthEnd(0))
    except ValueError:
        return end.normalize(), False
    if kind == "before":
        d = d - pd.Timedelta(days=1)
    return min(d.normalize(), end.normalize()), kind == "on"


def event_day(ts: np.ndarray, yp: np.ndarray, closed: pd.Timestamp, yes_won: int):
    """Day the event actually happened: first trade where the YES price reached HIT
    (markets often resolve hours/days after the event), else the close day."""
    if not yes_won:
        return None
    k = np.argmax(yp >= HIT) if (yp >= HIT).any() else None
    d = pd.Timestamp(int(ts[k]), unit="s", tz="UTC").normalize() if k is not None else closed.normalize()
    return min(d, closed.normalize())


def first_occurrence_probs(p_by_deadline: float, days_left: int, insider_p72: float | None = None) -> np.ndarray:
    """P(X = day k) for k = 1..days_left (first day the event happens), constant hazard,
    optionally replacing the first 3 days' hazard with the insider model's estimate."""
    days_left = max(int(days_left), 1)
    lam = 1 - (1 - np.clip(p_by_deadline, 0, 0.999)) ** (1 / days_left)
    haz = np.full(days_left, lam)
    if insider_p72 is not None:
        haz[:3] = np.maximum(haz[:3], 1 - (1 - insider_p72) ** (1 / 3))
    surv = np.concatenate([[1.0], np.cumprod(1 - haz)[:-1]])
    return surv * haz


def best_date(px: np.ndarray, earliest: int = 1, latest: int | None = None) -> tuple[int, float]:
    """px[k-1] = P(X=k). Returns (day offset, expected judge score) of the best date."""
    n = len(px)
    latest = min(latest or n, n)
    best, bestE = earliest, -1.0
    ks = np.arange(1, n + 1)
    for d in range(earliest, latest + 1):
        off = np.abs(ks - d)
        e = float((px * np.where(off <= 7, CREDIT[np.minimum(off, 7)], 0.0)).sum())
        if e > bestE:
            best, bestE = d, e
    return best, JUDGE_FACTOR * bestE


def realised(event_day: pd.Timestamp | None, pred_day: pd.Timestamp) -> float:
    if event_day is None:
        return 0.0
    off = abs((event_day - pred_day).days)
    return OFFSET_SCORE.get(off, 0.0) if off <= 7 else 0.0


def family(slug: str, q: str) -> str:
    """Group date-ladder markets about the same event (e.g. 'US strikes Iran by ...')."""
    base = re.sub(r"\b(by|on|before|in)\b.*$", "", q.lower()).strip(" ?")
    return (slug or "") + "|" + base


def fit_price_calibration(split="2025-10-01"):
    """Isotonic map from market price -> realised P(YES), fitted only on daily snapshots
    of event markets issued before `split` (corrects the YES over-pricing bias)."""
    from sklearn.isotonic import IsotonicRegression
    C = pd.read_parquet(REPORTS / "poly_strategy_candidates.parquet")
    C = C[C["issue_day"] < pd.Timestamp(split, tz="UTC")]
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(C["p"], C["yes_won"])


def load_source(source: str = "poly"):
    """(event markets, {cid: (ts, yes_price)}) for 'poly', 'poly_wide', 'manifold' or 'a+b'."""
    if "+" in source:
        parts = [load_source(x) for x in source.split("+")]
        mk = pd.concat([m.assign(src=n) for (m, _), n in zip(parts, source.split("+"))], ignore_index=True)
        return mk, {k: v for _, t in parts for k, v in t.items()}
    if source == "manifold":
        from . import manifold
        mk = pd.read_parquet(manifold.MF / "markets.parquet")
        mk = mk[mk["question"].map(is_event_market)].copy()
        return mk, manifold.tapes(mk["condition_id"])
    f = "markets_wide.parquet" if source == "poly_wide" else "markets_resolved.parquet"
    mk = pd.read_parquet(POLY / f)
    mk = mk[mk["question"].map(is_event_market)].copy()
    tr = load_trades(mk["condition_id"])
    tr["yes_price"] = np.where(tr["outcome_index"] == 0, tr["price"], 1 - tr["price"])
    tapes = {c: (g["ts"].to_numpy(), g["yes_price"].to_numpy()) for c, g in tr.sort_values("ts").groupby("condition_id")}
    return mk, tapes


def backtest(start="2024-01-01", end="2026-11-15", max_lead=7, thresholds=(3, 4, 5, 6),
             insider: dict | None = None, calib=None, source: str = "poly", min_trades: int = 10,
             hour: int = 7, save: bool = True, earliest: int = 1) -> dict:
    mk, tapes = load_source(source)
    print(f"{len(mk)} event markets with tapes: {sum(c in tapes for c in mk.condition_id)}")

    rows = []
    for d in pd.date_range(start, end, freq="D", tz="UTC"):
        tau = d + pd.Timedelta(hours=hour)
        ts_tau = int(tau.timestamp())
        live = mk[(mk["created"] < tau) & (mk["closed_time"] > tau) & (mk["end"] > tau)]
        for r in live.itertuples():
            tp = tapes.get(r.condition_id)
            if tp is None:
                continue
            i = np.searchsorted(tp[0], ts_tau)
            if i < min_trades or ts_tau - tp[0][i - 1] > 3 * 86400:      # need a recent, real price
                continue
            p = float(tp[1][i - 1])
            if (tp[1][:i] >= HIT).any():          # already happened (price hit 90%) -> not a prediction
                continue
            deadline, single = question_deadline(r.question, r.end)
            days_left = (deadline - d.normalize()).days
            if days_left < earliest:              # the question's date has passed / too close
                continue
            if calib is not None:
                p = float(calib.predict([p])[0])
            p72 = insider.get((r.condition_id, ts_tau)) if insider else None
            px = first_occurrence_probs(p, days_left, p72)
            if single:                            # "on <date>" market: only that date makes sense
                if days_left > max_lead:
                    continue
                k = days_left
                px = np.zeros(days_left); px[-1] = p
                e = JUDGE_FACTOR * 10 * p
            else:
                k, e = best_date(px, earliest, min(days_left, max_lead))
            pred = (d.normalize() + pd.Timedelta(days=k))
            ev = event_day(tp[0], tp[1], r.closed_time, r.yes_won)
            rows.append((d, r.condition_id, r.question, family(r.event_slug, r.question), p, days_left,
                         k, e, realised(ev, pred), int(r.yes_won)))
    C = pd.DataFrame(rows, columns=["issue_day", "cid", "question", "family", "p", "days_left",
                                    "lead", "exp_score", "realised", "yes_won"])
    out = {}
    days = pd.date_range(start, end, freq="D", tz="UTC")
    for th in thresholds:
        c = C[C["exp_score"] >= th].sort_values("exp_score", ascending=False)
        c = c.drop_duplicates(["issue_day", "family"])
        picks = c.groupby("issue_day").head(5)
        net = (picks["realised"] - PENALTY).groupby(picks["issue_day"]).sum().reindex(days, fill_value=0.0)
        win = net.rolling(14).sum().dropna()
        out[th] = {"preds_per_day": float(len(picks) / len(days)),
                   "mean_expected": float(picks["exp_score"].mean()) if len(picks) else 0,
                   "mean_realised": float(picks["realised"].mean()) if len(picks) else 0,
                   "miss_rate": float((picks["realised"] == 0).mean()) if len(picks) else 0,
                   "window_total_mean": float(win.mean()), "window_p10": float(win.quantile(.1)),
                   "window_p90": float(win.quantile(.9)), "share_windows_positive": float((win > 0).mean())}
        print(th, json.dumps(out[th]))
    if not save:
        return out
    REPORTS.mkdir(exist_ok=True)
    tag = ("_cal" if calib is not None else "") + ("" if source == "poly" else f"_{source}")
    C.to_parquet(REPORTS / f"poly_strategy_candidates{tag}.parquet", index=False)
    (REPORTS / f"poly_strategy{tag}.json").write_text(json.dumps({str(k): v for k, v in out.items()}, indent=2))
    return out


def ladder_px(points: list[tuple[int, float]]) -> np.ndarray:
    """points = [(days_left, P(by that deadline))] for one event family. Monotone
    cumulative curve -> piecewise-constant daily hazard -> P(first occurrence = day k)."""
    points = sorted(points)
    px, F_prev, d_prev, surv = [], 0.0, 0, 1.0
    for dl, F in points:
        F = max(F, F_prev)
        seg = dl - d_prev
        if seg <= 0:
            continue
        q = 0.0 if F_prev >= 0.999 else (F - F_prev) / (1 - F_prev)
        lam = 1 - (1 - min(q, 0.999)) ** (1 / seg)
        for _ in range(seg):
            px.append(surv * lam)
            surv *= 1 - lam
        F_prev, d_prev = F, dl
    return np.array(px)


def ladder_backtest(start="2025-10-01", end="2026-11-15", max_lead=7, thresholds=(2.5, 3, 3.5, 4, 5),
                    source="poly", min_trades=10) -> dict:
    """Per event family: constant-hazard (best single market, = live strategy) vs term-structure
    hazard from all its deadline markets. Same picks rule, same realised scoring."""
    mk, tapes = load_source(source)
    mk["fam"] = [family(a, b) for a, b in zip(mk["event_slug"], mk["question"])]
    rows = []
    for d in pd.date_range(start, end, freq="D", tz="UTC"):
        tau = d + pd.Timedelta(hours=7); ts_tau = int(tau.timestamp())
        live = mk[(mk["created"] < tau) & (mk["closed_time"] > tau) & (mk["end"] > tau)]
        fams: dict[str, list] = {}
        for r in live.itertuples():
            tp = tapes.get(r.condition_id)
            if tp is None:
                continue
            i = np.searchsorted(tp[0], ts_tau)
            if i < min_trades or ts_tau - tp[0][i - 1] > 3 * 86400 or (tp[1][:i] >= HIT).any():
                continue
            deadline, single = question_deadline(r.question, r.end)
            dl = (deadline - d.normalize()).days
            if dl < 1 or single:
                continue
            ev = event_day(tp[0], tp[1], r.closed_time, r.yes_won)
            fams.setdefault(r.fam, []).append((dl, float(tp[1][i - 1]), ev, r.question))
        for fam, ms in fams.items():
            evs = [m[2] for m in ms if m[2] is not None]
            ev = min(evs) if evs else None
            # constant hazard on each market separately, keep the best (live behaviour)
            best = max(((best_date(first_occurrence_probs(p, dl), 1, min(dl, max_lead)), dl, p, q)
                        for dl, p, _, q in ms), key=lambda x: x[0][1])
            (k0, e0), _, p0, q0 = best
            # term structure across the ladder
            px = ladder_px([(dl, p) for dl, p, _, _ in ms])
            k1, e1 = best_date(px, 1, min(len(px), max_lead))
            day = d.normalize()
            rows.append((d, fam, q0, len(ms), p0, k0, e0, realised(ev, day + pd.Timedelta(days=k0)),
                         k1, e1, realised(ev, day + pd.Timedelta(days=k1))))
    C = pd.DataFrame(rows, columns=["issue_day", "family", "question", "n_ladder", "p", "lead_const",
                                    "exp_const", "real_const", "lead_ladder", "exp_ladder", "real_ladder"])
    C.to_parquet(REPORTS / f"ladder_candidates_{source}.parquet", index=False)
    days = pd.date_range(start, end, freq="D", tz="UTC")
    out = {}
    for kind in ("const", "ladder"):
        for th in thresholds:
            c = C[C[f"exp_{kind}"] >= th].sort_values(f"exp_{kind}", ascending=False)
            picks = c.groupby("issue_day").head(5)
            net = (picks[f"real_{kind}"] - PENALTY).groupby(picks["issue_day"]).sum().reindex(days, fill_value=0.0)
            win = net.rolling(14).sum().dropna()
            out[f"{kind}@{th}"] = {"preds_per_day": round(len(picks) / len(days), 2),
                                  "exp": round(float(picks[f"exp_{kind}"].mean()), 2) if len(picks) else 0,
                                  "real": round(float(picks[f"real_{kind}"].mean()), 2) if len(picks) else 0,
                                  "window_mean": round(float(win.mean()), 1),
                                  "share_pos": round(float((win > 0).mean()), 2)}
            print(f"{kind}@{th}", out[f"{kind}@{th}"], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-11-15")
    ap.add_argument("--source", default="poly", choices=["poly", "poly_wide", "manifold", "poly_wide+manifold"])
    ap.add_argument("--ladder", action="store_true", help="compare constant vs term-structure hazard")
    ap.add_argument("--calibrated", action="store_true",
                    help="fit price calibration on pre-2025-10 data, test on 2025-10 onwards")
    a = ap.parse_args()
    if a.ladder:
        ladder_backtest(source=a.source)
    elif a.calibrated:
        iso = fit_price_calibration()
        import pickle
        from ..config import MODELS
        pickle.dump(iso, open(MODELS / "poly_price_iso.pkl", "wb"))
        print("price calibration:", {x: round(float(iso.predict([x])[0]), 3) for x in (0.2, 0.4, 0.6, 0.8, 0.9)})
        print("--- uncalibrated, test period ---")
        backtest("2025-10-01", a.end)
        print("--- calibrated, test period ---")
        backtest("2025-10-01", a.end, calib=iso)
    else:
        backtest(a.start, a.end, thresholds=(2.5, 3, 3.5, 4, 5), source=a.source)
