"""Competition scoring: expected judge score from probabilities, realised score
from labels (for backtests), and the daily selection rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_PER_DAY, SCORE_THRESHOLD

# Score credited when the nearest occurrence is k days away (rules section 7).
OFFSET_SCORE = {0: 10.0, 1: 8.5, 2: 6.5, 3: 6.5, 4: 4.5, 5: 4.5, 6: 4.5, 7: 4.5}
# Haircut for judges reading our wording more strictly than our label does.
JUDGE_FACTOR = 0.9
PENALTY = 3.0


def expected_score(p_by_offset: dict[int, np.ndarray]) -> np.ndarray:
    """p_by_offset[k] = P(event on target+k), k in -7..7 (arrays, same shape).
    Missing offsets should be filled by the caller (e.g. with a base rate)."""
    def any_of(ks):
        q = np.ones_like(p_by_offset[0])
        for k in ks:
            q = q * (1 - p_by_offset[k])
        return 1 - q
    p0 = p_by_offset[0]
    p1 = any_of([-1, 1])
    p23 = any_of([-3, -2, 2, 3])
    p47 = any_of([-7, -6, -5, -4, 4, 5, 6, 7])
    e = 10 * p0 + (1 - p0) * (8.5 * p1 + (1 - p1) * (6.5 * p23 + (1 - p23) * 4.5 * p47))
    return JUDGE_FACTOR * e


def realised_score(Y_ct: np.ndarray, target_idx: int) -> float:
    """Score for predicting the event on target_idx given the label series Y_ct."""
    for k in range(0, 8):
        for s in (target_idx - k, target_idx + k):
            if 0 <= s < len(Y_ct) and Y_ct[s]:
                return OFFSET_SCORE[k]
    return 0.0


def select(cands: pd.DataFrame, k: int = MAX_PER_DAY, threshold: float = SCORE_THRESHOLD,
           max_per_country: int = 2, max_per_country_type: int = 1) -> pd.DataFrame:
    """cands needs columns country, type, target, exp_score. Greedy top-k with
    diversity limits so one hot conflict doesn't eat all five slots."""
    c = cands[cands["exp_score"] >= threshold].sort_values("exp_score", ascending=False)
    picked, per_c, per_ct = [], {}, {}
    for r in c.itertuples():
        if per_c.get(r.country, 0) >= max_per_country:
            continue
        if per_ct.get((r.country, r.type), 0) >= max_per_country_type:
            continue
        picked.append(r.Index)
        per_c[r.country] = per_c.get(r.country, 0) + 1
        per_ct[(r.country, r.type)] = per_ct.get((r.country, r.type), 0) + 1
        if len(picked) == k:
            break
    return c.loc[picked]
