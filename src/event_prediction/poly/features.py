"""Trade-tape features for the insider model.

Per trade we compute (strictly point-in-time):
  direction     +1 = bet on YES (buy Yes / sell No), -1 = bet on NO
  cash          USDC notional
  yes_price     trade price expressed as P(Yes)
  p_bought      price of the side being bet on (longshot bets have low p_bought)
  wallet stats  #prior trades, age, #resolved markets before this trade, shrunk
                win-rate and signed realised PnL on those markets, cumulative
                cash in this market, first-time-in-market flag

Samples are (market, snapshot time tau). Inputs are built lazily from the
last SEQ trades before tau so memory stays small.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEQ = 256
SNAP_EVERY_H = 12
LOOKBACK_DAYS = 30
SOON_H = 72
TOK_FEATS = ["log_cash", "dir", "dir_cash", "yes_price", "log_dt", "w_log_n", "w_log_age",
             "w_log_res", "w_win", "w_pnl", "w_new", "w_log_mkt_cash", "w_first", "p_bought",
             "dir_x_win", "dir_x_pnl"]
CTX_FEATS = ["logit_p", "log_h_to_end", "log_h_age", "log_cash_tot", "log_cash_24h", "log_cash_6h",
             "flow_24h", "flow_6h", "log_wallets_24h", "new_wallet_share_24h", "dp_24h", "dp_72h",
             "smart_flow_24h", "longshot_flow_24h", "max_trade_24h_share"]


def epoch_s(x):
    """Datetime(s) -> integer epoch seconds, independent of pandas' storage unit."""
    if isinstance(x, pd.Series):
        return (x.dt.tz_convert("UTC") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    return int(pd.Timestamp(x).timestamp())


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def enrich_trades(tr: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """tr: trades of many markets (condition_id, ts, wallet, side, outcome_index, price, size).
    markets: needs condition_id, closed_time, yes_won (NaN for unresolved)."""
    tr = tr.sort_values("ts", kind="stable").reset_index(drop=True)
    buy = tr["side"].eq("BUY").to_numpy()
    yes = tr["outcome_index"].eq(0).to_numpy()
    tr["dir"] = np.where(buy == yes, 1.0, -1.0)
    tr["cash"] = tr["size"] * tr["price"]
    tr["yes_price"] = np.where(yes, tr["price"], 1 - tr["price"])
    tr["p_bought"] = np.where(buy, tr["price"], 1 - tr["price"])

    # wallet activity so far
    g = tr.groupby("wallet", sort=False)
    tr["w_n"] = g.cumcount()
    tr["w_first_ts"] = g["ts"].transform("min")
    gm = tr.groupby(["wallet", "condition_id"], sort=False)
    tr["w_mkt_cash"] = gm["cash"].cumsum() - tr["cash"]
    tr["w_first"] = (gm.cumcount() == 0).astype(np.float32)

    # point-in-time track record: per (wallet, resolved market) outcome, known at close time
    mk = markets.dropna(subset=["yes_won"])[["condition_id", "closed_time", "yes_won"]].copy()
    mk["res_ts"] = epoch_s(mk["closed_time"])
    t2 = tr.merge(mk[["condition_id", "res_ts", "yes_won"]], on="condition_id", how="inner")
    won_side = np.where(t2["outcome_index"] == 0, t2["yes_won"], 1 - t2["yes_won"])
    sign = np.where(t2["side"] == "BUY", 1.0, -1.0)
    t2["pnl"] = sign * t2["size"] * (won_side - t2["price"])
    t2["sdir"] = t2["dir"] * t2["cash"]
    pairs = t2.groupby(["wallet", "condition_id"]).agg(res_ts=("res_ts", "first"),
                                                       yes_won=("yes_won", "first"),
                                                       pnl=("pnl", "sum"), sdir=("sdir", "sum")).reset_index()
    pairs["correct"] = ((pairs["sdir"] > 0) == (pairs["yes_won"] == 1)).astype(float)
    pairs = pairs.sort_values("res_ts")
    pg = pairs.groupby("wallet")
    pairs["c_res"] = pg.cumcount() + 1
    pairs["c_cor"] = pg["correct"].cumsum()
    pairs["c_pnl"] = pg["pnl"].cumsum()
    left = tr[["ts", "wallet"]].reset_index().sort_values("ts")
    right = pairs[["res_ts", "wallet", "c_res", "c_cor", "c_pnl"]].rename(columns={"res_ts": "ts"})
    right = right.sort_values("ts")
    j = pd.merge_asof(left, right, on="ts", by="wallet", direction="backward",
                      allow_exact_matches=False).set_index("index").sort_index()
    tr["w_res"] = j["c_res"].fillna(0).to_numpy()
    tr["w_cor"] = j["c_cor"].fillna(0).to_numpy()
    tr["w_pnl_raw"] = j["c_pnl"].fillna(0).to_numpy()

    age_d = (tr["ts"] - tr["w_first_ts"]) / 86400
    win = (tr["w_cor"] + 1) / (tr["w_res"] + 2) - 0.5          # shrunk toward 50%
    out = pd.DataFrame({
        "condition_id": tr["condition_id"], "ts": tr["ts"], "wallet": tr["wallet"],
        "cash": tr["cash"], "dir": tr["dir"], "yes_price": tr["yes_price"],
        "log_cash": np.log1p(tr["cash"]), "dir_cash": tr["dir"] * np.log1p(tr["cash"]),
        "w_log_n": np.log1p(tr["w_n"]), "w_log_age": np.log1p(age_d),
        "w_log_res": np.log1p(tr["w_res"]), "w_win": win, "w_pnl": _slog(tr["w_pnl_raw"]) / 10,
        "w_new": (age_d < 3).astype(np.float32), "w_log_mkt_cash": np.log1p(tr["w_mkt_cash"]),
        "w_first": tr["w_first"], "p_bought": tr["p_bought"],
        "dir_x_win": tr["dir"] * win, "dir_x_pnl": tr["dir"] * _slog(tr["w_pnl_raw"]) / 10,
    })
    return out


@dataclass
class MarketTape:
    cid: str
    ts: np.ndarray        # [n]
    tok: np.ndarray       # [n, len(TOK_FEATS)] (log_dt filled per sample)
    cash: np.ndarray
    dir: np.ndarray
    yes_price: np.ndarray
    new: np.ndarray
    smart: np.ndarray     # dir * shrunk win-rate
    p_bought: np.ndarray
    wallet_codes: np.ndarray


def build_tapes(enriched: pd.DataFrame) -> dict[str, MarketTape]:
    tapes = {}
    codes = enriched["wallet"].astype("category").cat.codes.to_numpy()
    enriched = enriched.assign(_w=codes)
    for cid, g in enriched.groupby("condition_id", sort=False):
        g = g.sort_values("ts", kind="stable")
        tok = g.reindex(columns=TOK_FEATS).fillna(0).to_numpy(np.float32)
        tapes[cid] = MarketTape(cid, g["ts"].to_numpy(np.int64), tok, g["cash"].to_numpy(np.float32),
                                g["dir"].to_numpy(np.float32), g["yes_price"].to_numpy(np.float32),
                                g["w_new"].to_numpy(np.float32), g["dir_x_win"].to_numpy(np.float32),
                                g["p_bought"].to_numpy(np.float32), g["_w"].to_numpy())
    return tapes


def context(tp: MarketTape, i: int, tau: int, end_ts: int, created_ts: int) -> np.ndarray:
    """Market-level features using trades [0, i) (all strictly before tau)."""
    ts = tp.ts[:i]
    p = float(tp.yes_price[i - 1])
    lo24 = np.searchsorted(ts, tau - 86400)
    lo6 = np.searchsorted(ts, tau - 6 * 3600)
    c24, c6 = tp.cash[lo24:i], tp.cash[lo6:i]
    tot24 = c24.sum() + 1e-6

    def p_at(t):
        k = np.searchsorted(ts, t) - 1
        return float(tp.yes_price[k]) if k >= 0 else p
    w24 = tp.wallet_codes[lo24:i]
    pc = np.clip(p, 0.01, 0.99)
    return np.array([
        np.log(pc / (1 - pc)),
        np.log1p(max(end_ts - tau, 0) / 3600),
        np.log1p(max(tau - created_ts, 0) / 3600),
        np.log1p(tp.cash[:i].sum()), np.log1p(c24.sum()), np.log1p(c6.sum()),
        float((tp.dir[lo24:i] * c24).sum() / tot24),
        float((tp.dir[lo6:i] * c6).sum() / (c6.sum() + 1e-6)),
        np.log1p(len(np.unique(w24))),
        float((tp.new[lo24:i] * c24).sum() / tot24),
        p - p_at(tau - 86400), p - p_at(tau - 3 * 86400),
        float((tp.smart[lo24:i] * c24).sum() / tot24),
        float((tp.dir[lo24:i] * (tp.p_bought[lo24:i] < 0.2) * c24).sum() / tot24),
        float(c24.max() / tot24) if len(c24) else 0.0,
    ], dtype=np.float32)


def window(tp: MarketTape, i: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    lo = max(0, i - SEQ)
    tok = tp.tok[lo:i].copy()
    tok[:, TOK_FEATS.index("log_dt")] = np.log1p((tau - tp.ts[lo:i]) / 3600)
    pad = SEQ - len(tok)
    mask = np.zeros(SEQ, dtype=bool)          # True = padding
    if pad:
        tok = np.concatenate([np.zeros((pad, tok.shape[1]), np.float32), tok])
        mask[:pad] = True
    return tok, mask


def make_samples(tapes: dict[str, MarketTape], markets: pd.DataFrame, min_trades: int = 20,
                 max_per_market: int = 60, seed: int = 0) -> pd.DataFrame:
    """One row per (market, tau) with labels. tau on a fixed 12h UTC grid over the
    market's traded life, strictly before close."""
    rng = np.random.default_rng(seed)
    rows = []
    m = markets.set_index("condition_id")
    for cid, tp in tapes.items():
        if cid not in m.index or len(tp.ts) < min_trades:
            continue
        r = m.loc[cid]
        close = epoch_s(r["closed_time"])
        end = epoch_s(r["end"]) if pd.notna(r["end"]) else close
        created = epoch_s(r["created"]) if pd.notna(r["created"]) else int(tp.ts[0])
        # Fixed UTC 12h grid over the market's whole traded life (not anchored to
        # the close, which would over-sample the hours just before resolution
        # and inflate the 'soon' base rate relative to live use).
        step = SNAP_EVERY_H * 3600
        start = int(tp.ts[min_trades - 1]) + 1
        taus = np.arange(-(-start // step) * step, close - 3600, step)
        if len(taus) > max_per_market:
            taus = np.sort(rng.choice(taus, max_per_market, replace=False))
        for tau in taus:
            i = int(np.searchsorted(tp.ts, tau))      # trades strictly before tau
            if i < min_trades:
                continue
            p_now = float(tp.yes_price[i - 1])
            j = int(np.searchsorted(tp.ts, tau + 86400)) - 1
            p_24 = float(tp.yes_price[j]) if j >= i else p_now
            rows.append((cid, int(tau), i, end, created, int(r["yes_won"]),
                         int(r["yes_won"] == 1 and close - tau <= SOON_H * 3600),
                         int(p_24 - p_now >= 0.10), p_now, close))
    return pd.DataFrame(rows, columns=["cid", "tau", "i", "end", "created", "y_yes", "y_soon",
                                       "y_up", "p_now", "close"])
