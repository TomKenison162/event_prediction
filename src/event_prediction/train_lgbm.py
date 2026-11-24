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
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=0)


def cutoff_range(p, lo, hi, step=1):
    idx = np.where((p.days >= pd.Timestamp(lo)) & (p.days <= pd.Timestamp(hi)))[0]
    idx = idx[idx >= SEQ_LEN]
    idx = idx[idx + max(HORIZONS) < len(p.days)]   # label must exist
    return idx[::step]


def to_lgb(df, cols):
    X = df[cols].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="train on all data for live use")
    ap.add_argument("--step", type=int, default=2, help="use every Nth cutoff day for training")
    ap.add_argument("--rounds", type=int, default=3000)
    a = ap.parse_args()

    p = build_panel()
    tb = TabularBuilder(p)
    rng = np.random.default_rng(0)
    print(f"panel: {p.F.shape}, label base rate {p.Y.mean():.4f}")

    first = str(p.days[SEQ_LEN].date())
    last = str(p.days[-1].date())
    if a.final:
        tr = tb.rows(cutoff_range(p, first, last, a.step), sample_one_h=True, rng=rng)
        meta = json.loads((MODELS / "lgbm_meta.json").read_text())
        rounds = int(meta["best_iteration"] * 1.1)
        cols = feature_columns(tr)
        m = lgb.train(PARAMS, lgb.Dataset(to_lgb(tr, cols), tr["y"]), rounds)
        m.save_model(str(MODELS / "lgbm_final.txt"))
        print(f"final model saved ({rounds} rounds, {len(tr):,} rows)")
        return

    tr = tb.rows(cutoff_range(p, first, TRAIN_END, a.step), sample_one_h=True, rng=rng)
    va = tb.rows(cutoff_range(p, "2024-07-01", VAL_END, 3), sample_one_h=True, rng=rng)
    te = tb.rows(cutoff_range(p, "2025-07-01", last, 3), sample_one_h=True, rng=rng)
    cols = feature_columns(tr)
    print(f"rows train {len(tr):,} val {len(va):,} test {len(te):,}; {len(cols)} features")

    dtr = lgb.Dataset(to_lgb(tr, cols), tr["y"])
    dva = lgb.Dataset(to_lgb(va, cols), va["y"], reference=dtr)
    m = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

    raw_va = m.predict(to_lgb(va, cols), num_iteration=m.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, va["y"])
    raw_te = m.predict(to_lgb(te, cols), num_iteration=m.best_iteration)
    cal_te = iso.predict(raw_te)

    report = {
        "best_iteration": m.best_iteration,
        "base_rate_test": float(te["y"].mean()),
        "auc_test": float(roc_auc_score(te["y"], raw_te)),
        "logloss_test_cal": float(log_loss(te["y"], np.clip(cal_te, 1e-6, 1 - 1e-6))),
        "brier_test_cal": float(brier_score_loss(te["y"], cal_te)),
        "features": cols,
    }
    # reliability of the high-confidence region (what we actually bet on)
    bins = [0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
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
