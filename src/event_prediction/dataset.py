"""Turn daily GDELT aggregates + pageviews into model-ready tensors.

Panel layout
------------
F[d, c, k]  daily feature tensor (day, country, feature), float32
Y[d, c, t]  label: was event type t "notable" in country c on day d
Everything at index d only uses data added on day d, so features for a
prediction made with cutoff day C use F[:C+1] only (no look-ahead).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import gdelt, wiki
from .config import COUNTRIES, HORIZONS, TYPE_NAMES

TYPE_FIELDS = ["n_events", "root_articles", "articles", "max_sources", "tone", "goldstein"]
Q_FIELDS = ["n_events", "articles", "tone", "goldstein"]
COUNT_FIELDS = {"n_events", "root_articles", "articles", "max_sources"}

# Label definition
# ----------------
# frac[d,c,t] = share of country c's root-event coverage on day d that is about type t.
# A day is "notable" for (c,t) if frac exceeds the 95th percentile of frac seen in
# peaceful, heavily-covered countries (their GDELT noise floor), AND the event has
# a minimum share of global coverage (so tiny-coverage countries can't trip it).
# Thresholds are fitted on data up to LABEL_FIT_END only and then frozen.
# Surprise target (what we actually predict)
# ----------------------------------------
# big[d,c,t]   : global coverage share >= BIG_SHARE per 10k (top ~0.5% of days) AND
#                >= BIG_RATIO x that country/type's mean over the 28 days before d.
# onset[d,c,t] : big today, and no big day for (c,t) in the previous ONSET_QUIET days.
# The model predicts onset; scoring counts any big day (onset or its continuation),
# which is how a judge would check "did it happen". 'notable' Y stays as a feature.
BIG_SHARE = 80.0
BIG_RATIO = 4.0
ONSET_QUIET = 14

QUIET_COUNTRIES = ["UK", "CA", "AS", "GM", "FR", "JA"]
QUIET_QUANTILE = 0.95
MIN_GLOBAL_SHARE_PER_10K = 0.5
LABEL_FIT_END = "2024-06-30"
THRESH_FILE = "label_thresholds.json"


def fit_thresholds(frac: np.ndarray, days: pd.DatetimeIndex, countries: list[str]) -> dict:
    m = days <= pd.Timestamp(LABEL_FIT_END)
    qi = [countries.index(c) for c in QUIET_COUNTRIES if c in countries]
    out = {}
    for i, t in enumerate(TYPE_NAMES):
        x = frac[m][:, qi, i].ravel()
        x = x[np.isfinite(x)]
        out[t] = float(max(np.quantile(x, QUIET_QUANTILE), 1e-3)) if len(x) else 1.0
    return out


def load_thresholds(frac, days, countries) -> dict:
    from .config import MODELS
    f = MODELS / THRESH_FILE
    if f.exists():
        return json.loads(f.read_text())
    th = fit_thresholds(frac, days, countries)
    if days[0] <= pd.Timestamp("2017-01-01") and days[-1] >= pd.Timestamp(LABEL_FIT_END):
        MODELS.mkdir(exist_ok=True)
        f.write_text(json.dumps(th, indent=2))      # freeze only when fitted on full history
    return th


@dataclass
class Panel:
    days: pd.DatetimeIndex
    countries: list[str]
    feat_names: list[str]
    F: np.ndarray            # [D, C, K]
    Y: np.ndarray            # [D, C, T] bool
    share: np.ndarray        # [D, C, T] root-article share per 10k (label source)
    missing: np.ndarray      # [D] bool: no GDELT data that day (features carried forward,
                             #           never used as a label/target)
    big: np.ndarray          # [D, C, T] bool: big spike day (used for realised scoring)
    onset: np.ndarray        # [D, C, T] bool: start of a big development (the model target)

    def day_index(self, d) -> int:
        return int(self.days.get_loc(pd.Timestamp(d)))


def build_panel(start: date | None = None, end: date | None = None,
                agg: pd.DataFrame | None = None, views: pd.DataFrame | None = None,
                fcdo_df: pd.DataFrame | None = None, threat_df: pd.DataFrame | None = None) -> Panel:
    agg = gdelt.load_agg(start, end) if agg is None else agg
    views = wiki.load() if views is None else views
    countries = list(COUNTRIES)
    days = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    D, C = len(days), len(countries)

    # global volume per day (all countries incl. ones we don't model)
    allrows = agg[agg["type"] == "ALL"]
    g_art = allrows.groupby("date")["articles"].sum().reindex(days).ffill().to_numpy()
    g_root = allrows.groupby("date")["root_articles"].sum().reindex(days).ffill().to_numpy()
    g_art = np.maximum(g_art, 1.0)
    g_root = np.maximum(g_root, 1.0)

    sub = agg[agg["country"].isin(countries)]
    blocks, names = [], []

    def tensor(typ: str, field: str) -> np.ndarray:
        s = sub[sub["type"] == typ].pivot_table(index="date", columns="country",
                                                values=field, aggfunc="sum")
        s = s.reindex(index=days, columns=countries)
        return s.to_numpy(dtype=np.float64)

    raw_root = {}
    for typ in TYPE_NAMES + ["ALL"]:
        for f in TYPE_FIELDS:
            x = tensor(typ, f)
            if f == "root_articles" and typ != "ALL":
                raw_root[typ] = np.nan_to_num(x)
            if f in COUNT_FIELDS:
                norm = g_root if f == "root_articles" else g_art
                x = np.log1p(np.nan_to_num(x) * 1e5 / norm[:, None])
            else:
                x = np.nan_to_num(x)
            blocks.append(x)
            names.append(f"{typ}:{f}")
    for q in (1, 2, 3, 4):
        for f in Q_FIELDS:
            x = np.nan_to_num(tensor(f"Q{q}", f))
            if f in COUNT_FIELDS:
                x = np.log1p(x * 1e5 / g_art[:, None])
            blocks.append(x)
            names.append(f"Q{q}:{f}")

    # pageviews (attention). Missing -> 0 plus a mask feature.
    v = (views.pivot_table(index="date", columns="country", values="views")
              .reindex(index=days, columns=countries))
    blocks.append(np.log1p(np.nan_to_num(v.to_numpy(dtype=np.float64))))
    names.append("wiki:log_views")
    blocks.append(np.broadcast_to(np.log(g_art)[:, None], (D, C)).copy())
    names.append("global:log_articles")

    # --- early-warning precursors (features only) ---
    from .precursors import fcdo, threats
    fc = fcdo.load() if fcdo_df is None else fcdo_df
    if len(fc):
        for cat, a in fcdo.daily_features(fc, days, countries).items():
            blocks.append(np.log1p(a.astype(np.float64)))
            names.append(f"fcdo:{cat}")
    else:
        for cat in ["any", *fcdo.CATEGORIES]:
            blocks.append(np.zeros((D, C)))
            names.append(f"fcdo:{cat}")
    th = threats.load(start, end) if threat_df is None else threat_df
    for kind in threats.CODES:
        if len(th):
            x = (th[th["kind"] == kind].pivot_table(index="date", columns="country", values="articles", aggfunc="sum")
                   .reindex(index=days, columns=countries))
            x = np.log1p(np.nan_to_num(x.to_numpy(dtype=np.float64)) * 1e5 / g_art[:, None])
        else:
            x = np.zeros((D, C))
        blocks.append(x)
        names.append(f"threat:{kind}")

    share = np.stack([raw_root[t] * 1e4 / g_root[:, None] for t in TYPE_NAMES], axis=-1)
    c_root = np.nan_to_num(tensor("ALL", "root_articles"))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.stack([raw_root[t] for t in TYPE_NAMES], -1) / c_root[:, :, None]
    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0)
    th = load_thresholds(frac, days, countries)
    thv = np.array([th[t] for t in TYPE_NAMES], dtype=np.float64)
    Y = (frac >= thv) & (share >= MIN_GLOBAL_SHARE_PER_10K)
    for i, t in enumerate(TYPE_NAMES):
        blocks.append(np.log1p(frac[:, :, i] * 100))
        names.append(f"{t}:frac")
    # label history is also a feature
    for i, t in enumerate(TYPE_NAMES):
        blocks.append(Y[:, :, i].astype(np.float64))
        names.append(f"{t}:label")

    have = set(pd.to_datetime(agg["date"].unique()))
    missing = np.array([d not in have for d in days])
    big, onset = surprise_labels(share.astype(np.float64), missing)
    for i, t in enumerate(TYPE_NAMES):
        blocks.append(big[:, :, i].astype(np.float64))
        names.append(f"{t}:big")
    F = np.stack(blocks, axis=-1).astype(np.float32)
    for d in np.where(missing)[0]:
        if d > 0:
            F[d] = F[d - 1]          # carry forward: an outage is not "nothing happened"
    Y = Y & ~missing[:, None, None]
    return Panel(days, countries, names, F, Y, share.astype(np.float32), missing, big, onset)


def surprise_labels(share: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prev28 = np.concatenate([np.full((1,) + share.shape[1:], np.nan), _rolling_mean(share, 28)[:-1]])
    prev28 = np.where(np.isnan(prev28), np.inf, prev28)      # no history -> can't be a spike
    big = (share >= BIG_SHARE) & (share >= BIG_RATIO * prev28) & ~missing[:, None, None]
    prior = np.zeros_like(big)
    for k in range(1, ONSET_QUIET + 1):
        prior[k:] |= big[:-k]
    return big, big & ~prior


def quiet_mask(p: "Panel", cutoff: int) -> np.ndarray:
    """[C, T] True where (c, t) had no big day in the ONSET_QUIET days up to cutoff,
    i.e. a big event there would be a new development, not a continuation."""
    lo = max(0, cutoff - ONSET_QUIET + 1)
    return ~p.big[lo:cutoff + 1].any(0)


# ---------------------------------------------------------------------------
# Tabular features for gradient boosting
# ---------------------------------------------------------------------------
def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w days (inclusive) along axis 0."""
    cs = np.cumsum(np.concatenate([np.zeros_like(x[:1]), x], axis=0), axis=0, dtype=np.float64)
    out = (cs[w:] - cs[:-w]) / w
    pad = np.full((w - 1,) + x.shape[1:], np.nan)
    return np.concatenate([pad, out], axis=0).astype(np.float32)


def _days_since(y: np.ndarray) -> np.ndarray:
    """Days since last True along axis 0 (capped at 400)."""
    out = np.full(y.shape, 400, dtype=np.float32)
    last = np.full(y.shape[1:], -10_000)
    for d in range(y.shape[0]):
        last = np.where(y[d], d, last)
        out[d] = np.minimum(d - last, 400)
    return out


class TabularBuilder:
    """Precomputes rolling statistics once; then emits rows for any cutoff set."""

    def __init__(self, p: Panel):
        self.p = p
        idx = {n: i for i, n in enumerate(p.feat_names)}
        F = p.F
        self.type_stats = {}
        for t in TYPE_NAMES:
            ra = F[:, :, idx[f"{t}:root_articles"]]
            ne = F[:, :, idx[f"{t}:n_events"]]
            ms = F[:, :, idx[f"{t}:max_sources"]]
            y = p.Y[:, :, TYPE_NAMES.index(t)].astype(np.float32)
            bg = p.big[:, :, TYPE_NAMES.index(t)].astype(np.float32)
            sh = np.log1p(p.share[:, :, TYPE_NAMES.index(t)])
            st = {
                "ra_1": ra, "ra_3": _rolling_mean(ra, 3), "ra_7": _rolling_mean(ra, 7),
                "ra_28": _rolling_mean(ra, 28), "ne_1": ne, "ne_7": _rolling_mean(ne, 7),
                "ne_28": _rolling_mean(ne, 28), "ms_1": ms, "ms_7": _rolling_mean(ms, 7),
                "sh_1": sh, "sh_7": _rolling_mean(sh, 7),
                "y_7": _rolling_mean(y, 7), "y_28": _rolling_mean(y, 28),
                "y_91": _rolling_mean(y, 91), "y_365": _rolling_mean(y, 365),
                "since": _days_since(y > 0),
                "big_365": _rolling_mean(bg, 365), "big_since": _days_since(bg > 0),
                "sh_max_7": np.log1p(np.maximum.reduce([np.roll(p.share[:, :, TYPE_NAMES.index(t)], k, 0)
                                                        for k in range(7)])),
                "tone_7": _rolling_mean(F[:, :, idx[f"{t}:tone"]], 7),
                "gold_7": _rolling_mean(F[:, :, idx[f"{t}:goldstein"]], 7),
            }
            st["ra_trend"] = st["ra_3"] - st["ra_28"]
            self.type_stats[t] = st
        ctry = {}
        for name in ["ALL:articles", "ALL:n_events", "ALL:tone", "ALL:goldstein",
                     "Q4:articles", "Q3:articles", "Q4:n_events", "wiki:log_views"]:
            x = F[:, :, idx[name]]
            key = name.replace(":", "_")
            ctry[f"{key}_1"] = x
            ctry[f"{key}_7"] = _rolling_mean(x, 7)
            ctry[f"{key}_28"] = _rolling_mean(x, 28)
        ctry["wiki_spike"] = ctry["wiki_log_views_1"] - ctry["wiki_log_views_28"]
        # early-warning precursors
        for cat in ["any", "staff_withdrawn", "leave_now", "airspace", "travel_upgrade", "tensions",
                    "unrest", "downgrade"]:
            x = F[:, :, idx[f"fcdo:{cat}"]]
            ctry[f"fcdo_{cat}_7"] = _rolling_mean(x, 7) * 7
        for cat in ["staff_withdrawn", "leave_now", "travel_upgrade", "airspace"]:
            ctry[f"fcdo_since_{cat}"] = _days_since(F[:, :, idx[f"fcdo:{cat}"]] > 0)
        for kind in ["threaten", "threat_force", "threat_attack", "ultimatum", "mobilise",
                     "reduce_relations", "coerce"]:
            x = F[:, :, idx[f"threat:{kind}"]]
            ctry[f"thr_{kind}_1"] = x
            ctry[f"thr_{kind}_3"] = _rolling_mean(x, 3)
            ctry[f"thr_{kind}_trend"] = _rolling_mean(x, 3) - _rolling_mean(x, 28)
        ctry["q4_trend"] = ctry["Q4_articles_7"] - ctry["Q4_articles_28"]
        ctry["n_big_28"] = _rolling_mean(p.big.any(-1).astype(np.float32), 28)
        ctry["global_big_7"] = np.broadcast_to(
            _rolling_mean(p.big.any((1, 2)).astype(np.float32)[:, None], 7), p.big.shape[:2]).copy()
        yall = p.Y.astype(np.float32)
        ctry["n_types_notable_1"] = yall.sum(-1)
        ctry["n_types_notable_7"] = _rolling_mean(yall.sum(-1), 7)
        self.ctry = ctry
        self.glob = F[:, 0, idx["global:log_articles"]]

    def rows(self, cutoffs: np.ndarray, horizons=HORIZONS, sample_one_h: bool = False,
             rng: np.random.Generator | None = None, with_label: bool = True) -> pd.DataFrame:
        """One row per (cutoff, country, type[, h]). cutoffs are day indices."""
        p = self.p
        C, T = len(p.countries), len(TYPE_NAMES)
        cut = np.asarray(cutoffs)
        di, ci, ti = np.meshgrid(cut, np.arange(C), np.arange(T), indexing="ij")
        di, ci, ti = di.ravel(), ci.ravel(), ti.ravel()
        if sample_one_h:
            rng = rng or np.random.default_rng(0)
            hh = rng.choice(np.asarray(horizons), size=len(di))
        else:
            H = len(horizons)
            di, ci, ti = np.repeat(di, H), np.repeat(ci, H), np.repeat(ti, H)
            hh = np.tile(np.asarray(horizons), len(di) // H)
        cols = {"cutoff": di, "country": ci, "type": ti, "h": hh}
        for k in next(iter(self.type_stats.values())):
            arr = np.stack([self.type_stats[t][k] for t in TYPE_NAMES], axis=-1)  # D,C,T
            cols[f"t_{k}"] = arr[di, ci, ti]
        for k, arr in self.ctry.items():
            cols[f"c_{k}"] = arr[di, ci]
        cols["g_log_articles"] = self.glob[di]
        tgt = p.days[0] + pd.to_timedelta(di + hh, unit="D")
        cols["tgt_dow"] = tgt.dayofweek.to_numpy()
        cols["tgt_doy"] = tgt.dayofyear.to_numpy()
        md = tgt.month.to_numpy() * 100 + tgt.day.to_numpy()
        cols["tgt_holiday"] = np.isin(md, [1224, 1225, 1226, 1231, 101]).astype(np.int8)
        df = pd.DataFrame(cols)
        if with_label:
            tgt_i = di + hh
            ok = tgt_i < len(p.days)
            ok[ok] &= ~p.missing[tgt_i[ok]]
            y = np.full(len(df), -1, dtype=np.int8)
            y[ok] = p.onset[tgt_i[ok], ci[ok], ti[ok]]
            df["y"] = y
        return df


FEATURE_EXCLUDE = {"cutoff", "y"}
CATEGORICAL = ["country", "type"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]
