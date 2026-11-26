"""Manifold Markets (free public API): resolved binary geopolitics markets + bet tapes.

Used as an extra market source: same strategy/backtest as Polymarket, tapes are the
market probability after each bet (`probAfter`).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from ..config import DATA

MF = DATA / "manifold"
BETS = MF / "bets"
API = "https://api.manifold.markets/v0"
TOPICS = ["geopolitics", "world-default", "ukraine-russia", "israel-hamas", "middle-east", "iran",
          "russia", "china", "military", "wars", "politics-default", "africa", "latin-america",
          "us-politics", "taiwan", "north-korea", "india", "venezuela", "syria", "terrorism"]
TERMS = ["", "strike", "attack", "invade", "ceasefire", "missile", "war", "coup", "killed",
         "troops", "drone", "assassinated", "arrested", "resign", "military", "sanctions"]
S = requests.Session()
S.headers["User-Agent"] = "event-prediction-research/0.1"


def _get(path, params):
    for i in range(4):
        try:
            r = S.get(f"{API}/{path}", params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** i)
    raise RuntimeError(f"manifold {path} {params}")


def fetch_markets(since="2023-01-01", min_bettors=15) -> pd.DataFrame:
    rows, seen = [], set()
    since_ms = pd.Timestamp(since, tz="UTC").timestamp() * 1000
    for topic in TOPICS:
        for term in TERMS:
            off = 0
            while off < 2000:
                try:
                    d = _get("search-markets", {"term": term, "filter": "resolved", "contractType": "BINARY",
                                            "topicSlug": topic, "sort": "newest", "limit": 1000, "offset": off})
                except RuntimeError:
                    break
                if not d:
                    break
                for m in d:
                    if m["id"] in seen or m.get("resolution") not in ("YES", "NO"):
                        continue
                    if (m.get("createdTime") or 0) < since_ms or (m.get("uniqueBettorCount") or 0) < min_bettors:
                        continue
                    seen.add(m["id"])
                    rows.append({"condition_id": m["id"], "question": m["question"], "event_slug": m.get("slug"),
                                 "created": m.get("createdTime"), "end": m.get("closeTime"),
                                 "closed_time": m.get("resolutionTime") or m.get("closeTime"),
                                 "volume": m.get("volume"), "bettors": m.get("uniqueBettorCount"),
                                 "yes_won": int(m["resolution"] == "YES"), "topic": topic})
                if len(d) < 1000:
                    break
                off += 1000
        print(f"{topic}: total {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    for c in ["created", "end", "closed_time"]:
        df[c] = pd.to_datetime(df[c], unit="ms", utc=True, errors="coerce")
    return df


def fetch_bets(mid: str) -> pd.DataFrame:
    rows, before = [], None
    for _ in range(100):
        p = {"contractId": mid, "limit": 1000}
        if before:
            p["before"] = before
        d = _get("bets", p)
        if not d:
            break
        rows += [(b["createdTime"] // 1000, b.get("probAfter"), b.get("amount"), b.get("userId"))
                 for b in d if b.get("probAfter") is not None and not b.get("isRedemption")]
        if len(d) < 1000:
            break
        before = d[-1]["id"]
    df = pd.DataFrame(rows, columns=["ts", "yes_price", "amount", "user"])
    return df.sort_values("ts").reset_index(drop=True)


def collect(workers=6):
    BETS.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(MF / "markets.parquet")
    todo = [c for c in m["condition_id"] if not (BETS / f"{c}.parquet").exists()]
    print(f"fetching bets for {len(todo)} markets")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_bets, c): c for c in todo}
        for f in tqdm(as_completed(futs), total=len(futs), mininterval=15):
            try:
                f.result().to_parquet(BETS / f"{futs[f]}.parquet", index=False)
            except Exception as e:  # noqa: BLE001
                print(e)


def tapes(cids) -> dict:
    out = {}
    for c in cids:
        f = BETS / f"{c}.parquet"
        if f.exists():
            t = pd.read_parquet(f)
            if len(t):
                out[c] = (t["ts"].to_numpy(), t["yes_price"].to_numpy())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-markets", action="store_true")
    a = ap.parse_args()
    MF.mkdir(parents=True, exist_ok=True)
    if not a.skip_markets:
        m = fetch_markets()
        m.to_parquet(MF / "markets.parquet", index=False)
        print(f"{len(m)} resolved markets, yes rate {m.yes_won.mean():.3f}")
    collect()


def live_signals(now, max_lead: int = 7, min_bettors: int = 15, pre_bar: float = 2.5) -> pd.DataFrame:
    """Open Manifold event markets scored exactly like the Polymarket candidates.
    Backtest (Oct 2025-Nov 2026, 22:00 issue, bar 3.5): Polymarket wide +18.6 per 14 days,
    Polymarket + Manifold (cross-site duplicates removed) +27.0."""
    import numpy as np
    from ..config import COUNTRIES, MIN_LEAD_DAYS
    from ..scoring import JUDGE_FACTOR
    from .live import _fallback_country, daily_hazard
    from .strategy import HIT, best_date, family, first_occurrence_probs, is_event_market, question_deadline
    tau = int(pd.Timestamp(now).timestamp())
    today = pd.Timestamp(now).normalize()
    seen, rows = set(), []
    for topic in TOPICS:
        for term in TERMS:
            try:
                d = _get("search-markets", {"term": term, "filter": "open", "contractType": "BINARY",
                                            "topicSlug": topic, "sort": "most-popular", "limit": 500})
            except RuntimeError:
                continue
            for m in d or []:
                if m["id"] in seen or (m.get("uniqueBettorCount") or 0) < min_bettors:
                    continue
                seen.add(m["id"])
                q = m["question"]
                if not is_event_market(q) or m.get("probability") is None:
                    continue
                end = pd.to_datetime(m.get("closeTime"), unit="ms", utc=True)
                if pd.isna(end) or end <= pd.Timestamp(now):
                    continue
                country = _fallback_country(q)
                if country not in COUNTRIES:
                    continue
                deadline, single = question_deadline(q, end)
                days_left = (deadline - today).days
                if days_left < MIN_LEAD_DAYS:
                    continue
                p = float(m["probability"])
                if single:
                    if days_left > max_lead:
                        continue
                    k, e = days_left, JUDGE_FACTOR * 10 * p
                else:
                    k, e = best_date(first_occurrence_probs(p, days_left), MIN_LEAD_DAYS, min(days_left, max_lead))
                if e < pre_bar:
                    continue
                rows.append({"condition_id": f"mf:{m['id']}", "mid": m["id"], "question": q,
                             "family": family(m.get("slug"), q), "country": country,
                             "country_name": COUNTRIES[country], "type": "market", "event_phrase": q,
                             "target": (today + pd.Timedelta(days=k)).date(), "p_now": p,
                             "days_left": days_left, "p_soon72": None, "p_jump24": None,
                             "lam_price": float(daily_hazard(np.array([p]), np.array([days_left]))[0]),
                             "exp_score": e, "volume": m.get("volume"), "source": "manifold"})
    out = []
    for r in rows:                      # same tape checks as the backtest: >=10 bets, fresh, never hit 90%
        try:
            b = fetch_bets(r.pop("mid"))
        except RuntimeError:
            continue
        b = b[b["ts"] < tau]
        if len(b) < 10 or tau - b["ts"].iloc[-1] > 3 * 86400 or (b["yes_price"] >= HIT).any():
            continue
        out.append(r)
    return pd.DataFrame(out)
