"""Train the gradient-boosting model + isotonic calibration.

Splits (by cutoff date, never shuffled across time):
  train : <= 2024-06-30
  val   : 2024-07-01 .. 2025-06-30   (early stopping + calibration)
  test  : 2025-07-01 .. end          (untouched; used by backtest)
`--final` retrains on everything up to the last labelled day for live use.
"""
from __future__ import annotations

import argparse
import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import HORIZONS, MODELS, SEQ_LEN
from .dataset import CATEGORICAL, TabularBuilder, build_panel, feature_columns

TRAIN_END = "2024-06-30"
VAL_END = "2025-06-30"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=0)


def cutoff_range(p, lo, hi, step=1):
    idx = np.where((p.days >= pd.Timestamp(lo)) & (p.days <= pd.Timestamp(hi)))[0]
    idx = idx[idx >= SEQ_LEN]
    idx = idx[idx + max(HORIZONS) < len(p.days)]   # label must exist
    return idx[::step]


def sampled_rows(tb, cutoffs, neg_rate, rng, chunk=60):
    """All horizons for every cutoff; keep only quiet (c,t) rows (as at inference),
    every positive, and a random neg_rate share of negatives (weight 1/neg_rate)."""
    from .dataset import ONSET_QUIET
    out = []
    for i in range(0, len(cutoffs), chunk):
        df = tb.rows(cutoffs[i:i + chunk])
        df = df[(df["y"] >= 0) & (df["t_big_since"] >= ONSET_QUIET)]
        keep = (df["y"] == 1) | (rng.random(len(df)) < neg_rate)
        df = df[keep].copy()
        df["w"] = np.where(df["y"] == 1, 1.0, 1.0 / neg_rate)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def to_lgb(df, cols):
    X = df[cols].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="train on all data for live use")
    ap.add_argument("--neg-rate", type=float, default=0.03, help="share of negative rows kept")
    ap.add_argument("--rounds", type=int, default=3000)
    a = ap.parse_args()

    p = build_panel()
    tb = TabularBuilder(p)
    rng = np.random.default_rng(0)
    print(f"panel: {p.F.shape}, onset base rate {p.onset.mean():.5f}")

    first = str(p.days[SEQ_LEN].date())
    last = str(p.days[-1].date())
    if a.final:
        tr = sampled_rows(tb, cutoff_range(p, first, last), a.neg_rate, rng)
        meta = json.loads((MODELS / "lgbm_meta.json").read_text())
        rounds = int(meta["best_iteration"] * 1.1)
        cols = [c for c in feature_columns(tr) if c != "w"]
        m = lgb.train(PARAMS, lgb.Dataset(to_lgb(tr, cols), tr["y"], weight=tr["w"]), rounds)
        m.save_model(str(MODELS / "lgbm_final.txt"))
        print(f"final model saved ({rounds} rounds, {len(tr):,} rows)")
        return

    from .dataset import ONSET_QUIET
    tr = sampled_rows(tb, cutoff_range(p, first, TRAIN_END), a.neg_rate, rng)

    def full(lo, hi):                      # val/test: every row (no sampling), quiet pairs only
        d = pd.concat([tb.rows(c) for c in np.array_split(cutoff_range(p, lo, hi), 8)])
        return d[(d["y"] >= 0) & (d["t_big_since"] >= ONSET_QUIET)].reset_index(drop=True)
    va, te = full("2024-07-01", VAL_END), full("2025-07-01", last)
    cols = [c for c in feature_columns(tr) if c != "w"]
    print(f"positives: train {int(tr.y.sum())} val {int(va.y.sum())} test {int(te.y.sum())}")
    print(f"rows train {len(tr):,} val {len(va):,} test {len(te):,}; {len(cols)} features")

    dtr = lgb.Dataset(to_lgb(tr, cols), tr["y"], weight=tr["w"])
    dva = lgb.Dataset(to_lgb(va, cols), va["y"], reference=dtr)
    m = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

    raw_va = m.predict(to_lgb(va, cols), num_iteration=m.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, va["y"])
    raw_te = m.predict(to_lgb(te, cols), num_iteration=m.best_iteration)
    cal_te = iso.predict(raw_te)

    from sklearn.metrics import average_precision_score
    report = {
        "best_iteration": m.best_iteration,
        "positives_test": int(te["y"].sum()),
        "avg_precision_test": float(average_precision_score(te["y"], raw_te)),
        "base_rate_test": float(te["y"].mean()),
        "auc_test": float(roc_auc_score(te["y"], raw_te)),
        "logloss_test_cal": float(log_loss(te["y"], np.clip(cal_te, 1e-6, 1 - 1e-6))),
        "brier_test_cal": float(brier_score_loss(te["y"], cal_te)),
        "features": cols,
    }
    # reliability of the high-confidence region (what we actually bet on)
    bins = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    te = te.assign(p=cal_te)
    rel = te.groupby(pd.cut(te["p"], bins), observed=True)["y"].agg(["mean", "size"])
    print(rel)
    report["reliability"] = {str(k): v for k, v in rel.to_dict("index").items()}

    MODELS.mkdir(exist_ok=True)
    m.save_model(str(MODELS / "lgbm.txt"), num_iteration=m.best_iteration)
    pickle.dump(iso, open(MODELS / "lgbm_iso.pkl", "wb"))
    (MODELS / "lgbm_meta.json").write_text(json.dumps(report, indent=2))
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print(imp.head(20))
    print({k: v for k, v in report.items() if k not in ("features", "reliability")})


if __name__ == "__main__":
    main()
