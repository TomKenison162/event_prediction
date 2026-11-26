"""Learned event-timing model: replaces the constant-hazard formula with an empirical
distribution over the first-occurrence day, learned from resolved markets.

One row per (issue day 22:00 UTC, live event market) with features available at that time
and label = day offset k of the actual event (first YES-price crossing of 0.90), bucketed
into classes 1..K (K = 14) and K+1 = "later or never". The model's class probabilities
replace P(X = k) in the expected-judge-score formula; selection rule unchanged.

Splits by issue day: train < 2025-07-01, val 2025-07-01..2025-09-30 (early stopping and
calibration), test >= 2025-10-01 (same period as every other strategy backtest).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from ..config import MODELS, REPORTS
from ..scoring import JUDGE_FACTOR, OFFSET_SCORE, PENALTY
from .strategy import (HIT, best_date, event_day, family, first_occurrence_probs,
                       load_source, question_deadline)

K = 14
CREDIT = np.array([OFFSET_SCORE[k] for k in range(8)])
CATS = {
    "strike": r"\bstrike|airstrike|bomb|missile|drone|attack",
    "capture": r"captur|enter|seize|take control|fall",
    "ceasefire": r"ceasefire|truce|peace deal|agreement|deal",
    "leader_out": r"\bout\b|resign|ousted|removed|step down|leave office",
    "arrest": r"arrest|detain|indict|charged",
    "invade": r"invade|invasion|ground operation|troops",
    "war": r"\bwar\b|declare|clash|military action",
}
CAT_RE = {k: re.compile(v, re.I) for k, v in CATS.items()}
FEATS = ["logit_p", "p", "days_left", "log_days_left", "single", "src_manifold", "age_days",
         "dp_1d", "dp_3d", "dp_7d", "vol_7d", "max_p_7d", "n_tr_24h", "n_tr_7d", "n_tr_all",
         "hours_since_trade", "lam_const", "p_by_7", *[f"cat_{c}" for c in CATS], "n_siblings",
         "dow"]


def _p_at(ts, yp, t):
    k = np.searchsorted(ts, t) - 1
    return float(yp[k]) if k >= 0 else np.nan


def build(start="2024-01-01", end="2026-11-15", hour=22, sources=("poly_wide", "manifold")) -> pd.DataFrame:
    rows = []
    for src in sources:
        mk, tapes = load_source(src)
        mk = mk.assign(fam=[family(a, b) for a, b in zip(mk["event_slug"], mk["question"])])
        for d in pd.date_range(start, end, freq="D", tz="UTC"):
            tau = d + pd.Timedelta(hours=hour); t = int(tau.timestamp()); day = d.normalize()
            live = mk[(mk["created"] < tau) & (mk["closed_time"] > tau) & (mk["end"] > tau)]
            sib = live["fam"].value_counts()
            for r in live.itertuples():
                tp = tapes.get(r.condition_id)
                if tp is None:
                    continue
                ts, yp = tp
                i = np.searchsorted(ts, t)
                if i < 10 or t - ts[i - 1] > 3 * 86400 or (yp[:i] >= HIT).any():
                    continue
                dl_day, single = question_deadline(r.question, r.end)
                n = (dl_day - day).days
                if n < 1:
                    continue
                p = float(yp[i - 1]); pc = min(max(p, 0.005), 0.995)
                ev = event_day(ts, yp, r.closed_time, r.yes_won)
                k = (ev - day).days if ev is not None else None
                lo7 = np.searchsorted(ts, t - 7 * 86400)
                w7 = yp[lo7:i]
                lam = 1 - (1 - min(p, 0.999)) ** (1 / n)
                rows.append({
                    "issue_day": d, "cid": r.condition_id, "src": src, "question": r.question, "fam": r.fam,
                    "yes_won": int(r.yes_won), "k": k, "label": (k if k is not None and 1 <= k <= K else K + 1),
                    "logit_p": np.log(pc / (1 - pc)), "p": p, "days_left": n, "log_days_left": np.log1p(n),
                    "single": int(single), "src_manifold": int(src == "manifold"),
                    "age_days": (t - (r.created.timestamp() if pd.notna(r.created) else ts[0])) / 86400,
                    "dp_1d": p - _p_at(ts, yp, t - 86400), "dp_3d": p - _p_at(ts, yp, t - 3 * 86400),
                    "dp_7d": p - _p_at(ts, yp, t - 7 * 86400),
                    "vol_7d": float(np.std(w7)) if len(w7) > 1 else 0.0, "max_p_7d": float(w7.max()) if len(w7) else p,
                    "n_tr_24h": int(i - np.searchsorted(ts, t - 86400)), "n_tr_7d": int(i - lo7), "n_tr_all": int(i),
                    "hours_since_trade": (t - ts[i - 1]) / 3600, "lam_const": lam,
                    "p_by_7": 1 - (1 - lam) ** min(n, 7),
                    **{f"cat_{c}": int(bool(rx.search(r.question))) for c, rx in CAT_RE.items()},
                    "n_siblings": int(sib.get(r.fam, 1)), "dow": d.dayofweek,
                })
        print(f"{src}: {len(rows)} rows so far", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(REPORTS / "timing_rows.parquet", index=False)
    return df


# ---------------------------------------------------------------- scoring helpers
def score_dates(px: np.ndarray, earliest: int, latest: int) -> tuple[int, float]:
    """px[j] = P(event on day j+1), j < K (mass beyond K ignored: it can only earn credit
    for d >= K-6). Returns best (d, E)."""
    ks = np.arange(1, len(px) + 1)
    best, bestE = earliest, -1.0
    for d in range(earliest, latest + 1):
        off = np.abs(ks - d)
        e = float((px * np.where(off <= 7, CREDIT[np.minimum(off, 7)], 0.0)).sum())
        if e > bestE:
            best, bestE = d, e
    return best, JUDGE_FACTOR * bestE


def realised_k(k, d):
    if k is None or pd.isna(k):
        return 0.0
    off = abs(int(k) - d)
    return OFFSET_SCORE.get(off, 0.0) if off <= 7 else 0.0


def constant_px(r) -> np.ndarray:
    px = first_occurrence_probs(r.p, int(r.days_left))
    out = np.zeros(K); out[:min(K, len(px))] = px[:K]
    return out


def evaluate(df: pd.DataFrame, px_all: np.ndarray, bar=3.5, max_lead=7, label="") -> dict:
    """Same selection rule as strategy.backtest: E >= bar, one per family per day, top 5/day."""
    d_best, e_best, real = [], [], []
    for r, px in zip(df.itertuples(), px_all):
        n = int(r.days_left)
        if r.single:
            if n > max_lead:
                d_best.append(-1); e_best.append(-1); real.append(0); continue
            d, e = n, JUDGE_FACTOR * 10 * px[n - 1] if n <= K else 0.0
        else:
            px = px.copy(); px[n:] = 0                       # can't happen after the deadline
            d, e = score_dates(px, 1, min(n, max_lead))
        d_best.append(d); e_best.append(e); real.append(realised_k(r.k, d))
    out = df.assign(d=d_best, e=e_best, real=real)
    days = pd.date_range(df.issue_day.min(), df.issue_day.max(), freq="D")
    res = {}
    for b in ([bar] if np.isscalar(bar) else bar):
        c = out[out.e >= b].sort_values("e", ascending=False).drop_duplicates(["issue_day", "fam"])
        picks = c.groupby("issue_day").head(5)
        net = (picks.real - PENALTY).groupby(picks.issue_day).sum().reindex(days, fill_value=0.0)
        w = net.rolling(14).sum().dropna()
        res[b] = {"preds_per_day": round(len(picks) / len(days), 2), "exp": round(float(picks.e.mean()), 2),
                  "real": round(float(picks.real.mean()), 2), "net14": round(float(w.mean()), 1),
                  "pos": round(float((w > 0).mean()), 2)}
        print(f"{label:28s} bar {b}: {res[b]}", flush=True)
    return res


# ---------------------------------------------------------------- models
TRAIN_END, VAL_END = pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC")


def split(df):
    return (df[df.issue_day < TRAIN_END], df[(df.issue_day >= TRAIN_END) & (df.issue_day < VAL_END)],
            df[df.issue_day >= VAL_END])


def within_deadline_label(df):
    """Did the event happen (YES) at all? -> calibrates p by horizon."""
    return df["yes_won"].to_numpy()


V1_FEATS = ["logit_p", "log_days_left", "src_manifold", "single"]


def fit_v1(tr):
    """Horizon-aware calibration: P(YES) = sigmoid(a*logit p + b*log N + c*logit p*log N + ...)."""
    from sklearn.linear_model import LogisticRegression
    X = tr[V1_FEATS].assign(ix=tr.logit_p * tr.log_days_left).to_numpy()
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, within_deadline_label(tr))


def px_v1(m, df):
    X = df[V1_FEATS].assign(ix=df.logit_p * df.log_days_left).to_numpy()
    pc = m.predict_proba(X)[:, 1]
    out = []
    for p, n in zip(pc, df.days_left):
        px = first_occurrence_probs(p, int(n)); z = np.zeros(K); z[:min(K, len(px))] = px[:K]; out.append(z)
    return np.array(out)


def hazard_rows(df):
    """Discrete-time survival expansion: one row per (sample, day k) while still at risk."""
    reps, ks, ys = [], [], []
    lab = df["label"].to_numpy(); n = df["days_left"].to_numpy()
    for i in range(len(df)):
        last = min(K, int(n[i]), int(lab[i]) if lab[i] <= K else K)
        for k in range(1, last + 1):
            reps.append(i); ks.append(k); ys.append(int(lab[i] == k))
    X = df.iloc[reps][FEATS].reset_index(drop=True)
    X["k"] = ks; X["k_frac"] = X["k"] / df["days_left"].to_numpy()[reps]
    X["k_lam"] = 1 - (1 - df["lam_const"].to_numpy()[reps]) ** 1          # constant-hazard prior
    return X, np.array(ys)


def fit_v2(tr, va):
    import lightgbm as lgb
    Xt, yt = hazard_rows(tr); Xv, yv = hazard_rows(va)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
    m = lgb.train(params, lgb.Dataset(Xt, yt), 2000, valid_sets=[lgb.Dataset(Xv, yv)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"v2 hazard model: {m.best_iteration} rounds, {len(Xt):,} train rows")
    return m


def px_v2(m, df):
    rows = []
    for k in range(1, K + 1):
        X = df[FEATS].copy(); X["k"] = k; X["k_frac"] = k / df["days_left"].to_numpy(); X["k_lam"] = df["lam_const"].to_numpy()
        rows.append(m.predict(X, num_iteration=m.best_iteration))
    h = np.vstack(rows).T                                                   # [n, K]
    h = np.where(np.arange(1, K + 1)[None, :] <= df["days_left"].to_numpy()[:, None], h, 0.0)
    surv = np.cumprod(np.hstack([np.ones((len(df), 1)), 1 - h[:, :-1]]), axis=1)
    return surv * h


def train_and_compare(bars=(3.0, 3.5, 4.0, 4.5)):
    df = pd.read_parquet(REPORTS / "timing_rows.parquet")
    tr, va, te = split(df)
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}")
    res = {}
    P0 = np.array([constant_px(r) for r in te.itertuples()])
    res["v0_constant"] = evaluate(te, P0, bars, label="V0 constant hazard")
    m1 = fit_v1(pd.concat([tr, va])); P1 = px_v1(m1, te)
    res["v1_horizon_calib"] = evaluate(te, P1, bars, label="V1 horizon-calibrated p")
    m2 = fit_v2(tr, va); P2 = px_v2(m2, te)
    res["v2_hazard_gbm"] = evaluate(te, P2, bars, label="V2 learned hazard")
    res["v3_blend"] = evaluate(te, 0.5 * P1 + 0.5 * P2, bars, label="V3 blend V1+V2")
    # probabilistic accuracy of the day distribution (log score on first-occurrence day, k<=7)
    def logscore(P):
        lab = te["label"].to_numpy(); s = []
        for p, l in zip(P, lab):
            q = p[l - 1] if l <= K else max(1 - p.sum(), 1e-6)
            s.append(np.log(max(q, 1e-6)))
        return float(np.mean(s))
    for name, P in (("v0", P0), ("v1", P1), ("v2", P2), ("v3", 0.5 * P1 + 0.5 * P2)):
        print(f"log score {name}: {logscore(P):.4f}")
    imp = pd.Series(m2.feature_importance("gain"), index=m2.feature_name()).sort_values(ascending=False)
    print("v2 top features:", (imp / imp.sum()).head(12).round(3).to_dict())
    (REPORTS / "timing_compare.json").write_text(json.dumps({k: {str(b): v for b, v in r.items()} for k, r in res.items()}, indent=1))
    import pickle
    pickle.dump({"v1": m1}, open(MODELS / "timing_v1.pkl", "wb")); m2.save_model(str(MODELS / "timing_v2.txt"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_and_compare()
    if a.build:
        df = build()
        print(df.groupby("src").size(), df.label.value_counts().sort_index().to_dict())
