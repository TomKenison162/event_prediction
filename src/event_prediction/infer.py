"""Model inference -> scored candidate list for a given cutoff day."""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import COUNTRIES, HORIZONS, MODELS, TYPE_NAMES
from .dataset import Panel, TabularBuilder, feature_columns
from .scoring import expected_score
from .train_lgbm import to_lgb

HZ = np.array(HORIZONS)


@dataclass
class Models:
    lgbm: lgb.Booster | None
    lgbm_iso: object | None
    seq: dict | None          # {"model", "iso", "stats"}
    w_seq: float = 0.5


def load_models(final: bool = True) -> Models:
    lg, iso, seq = None, None, None
    path = MODELS / ("lgbm_final.txt" if final and (MODELS / "lgbm_final.txt").exists() else "lgbm.txt")
    if path.exists():
        lg = lgb.Booster(model_file=str(path))
        iso = pickle.load(open(MODELS / "lgbm_iso.pkl", "rb"))
    sp = MODELS / ("seq_final.pt" if final and (MODELS / "seq_final.pt").exists() else "seq.pt")
    if sp.exists():
        import torch
        from .train_seq import STTransformer
        ck = torch.load(sp, map_location="cpu")
        m = STTransformer(**ck["cfg"])
        m.load_state_dict(ck["state"])
        m.eval()
        seq = {"model": m, "stats": ck["stats"],
               "iso": pickle.load(open(MODELS / "seq_iso.pkl", "rb"))}
    if lg is None and seq is None:
        raise FileNotFoundError("no trained model in models/")
    return Models(lg, iso, seq)


def probs_lgbm(m: Models, tb: TabularBuilder, cutoffs: np.ndarray) -> np.ndarray:
    """-> [N, C, T, H] calibrated probabilities."""
    df = tb.rows(cutoffs, with_label=False)
    cols = feature_columns(df)
    raw = m.lgbm.predict(to_lgb(df, cols))
    p = m.lgbm_iso.predict(raw)
    C, T = len(tb.p.countries), len(TYPE_NAMES)
    return p.reshape(len(cutoffs), C, T, len(HZ))


def probs_seq(m: Models, panel: Panel, cutoffs: np.ndarray) -> np.ndarray:
    import torch
    from .train_seq import Data, predict
    data = Data(panel, m.seq["stats"], torch.device("cpu"))
    p = predict(m.seq["model"], data, cutoffs)
    return m.seq["iso"].predict(p.ravel()).reshape(p.shape)


def ensemble(m: Models, panel: Panel, tb: TabularBuilder, cutoffs: np.ndarray) -> np.ndarray:
    parts, ws = [], []
    if m.lgbm is not None:
        parts.append(probs_lgbm(m, tb, cutoffs)); ws.append(1 - m.w_seq if m.seq else 1.0)
    if m.seq is not None:
        parts.append(probs_seq(m, panel, cutoffs)); ws.append(m.w_seq if m.lgbm else 1.0)
    return sum(w * x for w, x in zip(ws, parts)) / sum(ws)


def candidates(panel: Panel, cutoff: int, P: np.ndarray, min_h: int, max_h: int) -> pd.DataFrame:
    """P: [C, T, H] probabilities for this cutoff. Returns one row per
    (country, type, target) with expected judge score. Offsets that fall on
    already-observed days use the observed label; beyond the model horizon we
    fall back to the 28-day base rate."""
    C, T, _ = P.shape
    Y = panel.Y
    base = Y[max(0, cutoff - 27):cutoff + 1].mean(0)            # C,T
    rows = []
    for h in range(min_h, max_h + 1):
        pk = {}
        for k in range(-7, 8):
            hk = h + k
            if hk <= 0:                                           # already happened (or not)
                pk[k] = Y[cutoff + hk].astype(np.float64)
            elif hk in HORIZONS:
                pk[k] = P[:, :, HORIZONS.index(hk)]
            else:
                pk[k] = base
        e = expected_score(pk)
        tgt = panel.days[cutoff] + pd.Timedelta(days=h)
        for c in range(C):
            for t in range(T):
                rows.append((panel.countries[c], COUNTRIES[panel.countries[c]], TYPE_NAMES[t],
                             tgt.date(), h, float(pk[0][c, t]), float(e[c, t]), float(base[c, t])))
    return pd.DataFrame(rows, columns=["country", "country_name", "type", "target", "h",
                                       "p", "exp_score", "base_rate_28d"])
