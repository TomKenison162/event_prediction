"""Polymarket data collection: resolved + active markets and their trade tapes.

Gamma API (markets/events):  https://gamma-api.polymarket.com/events/keyset
Data API v2 (trades):        https://data-api.polymarket.com/v2/trades?condition=<cid>
The trades endpoint serves a fixed ~3-year window per market, cursor-paged.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from ..config import DATA

POLY = DATA / "poly"
TRADES = POLY / "trades"
GAMMA = "https://gamma-api.polymarket.com/events/keyset"
TRADES_URL = "https://data-api.polymarket.com/v2/trades"
TAGS = ["geopolitics", "middle-east", "ukraine", "iran", "russia", "china", "israel",
        "military", "venezuela", "world-affairs", "world"]
EXTRA_TAGS = ["gaza", "lebanon", "syria", "yemen", "north-korea", "taiwan", "india", "pakistan",
              "india-pakistan", "africa", "south-america", "latin-america", "europe", "asia",
              "turkey", "mexico", "politics", "global-politics", "trump", "foreign-policy",
              "nato", "putin", "zelensky", "netanyahu", "hamas", "hezbollah", "houthis", "coups"]
SKIP = ("nba", "nfl", "nhl", "mlb", "ufc", "premier league", "champions league", "bitcoin",
        "ethereum", "solana", "crypto", " fc ", "grand prix", "oscars", "grammy", "box office",
        "say ", "tweet", "temperature", "election", "primary", "nominee", "poll")
MIN_TRADE_USD = 100.0
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "event-prediction-research/0.1"


def _get(url: str, params: dict, tries: int = 6):
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code in (429, 503):
                time.sleep(float(r.headers.get("Retry-After", 2 ** i)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** i)
    raise RuntimeError(f"failed {url} {params}")


def _market_row(ev: dict, m: dict, tag: str) -> dict | None:
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
        tokens = json.loads(m.get("clobTokenIds") or "[]")
    except (json.JSONDecodeError, ValueError):
        return None
    if outcomes != ["Yes", "No"] or len(tokens) != 2:
        return None
    q = (m.get("question") or "")
    if any(s in f" {q.lower()} " for s in SKIP):
        return None
    return {
        "condition_id": m.get("conditionId"), "question": q, "event_title": ev.get("title"),
        "event_slug": ev.get("slug"), "description": (m.get("description") or "")[:1500],
        "tag": tag, "yes_token": tokens[0], "no_token": tokens[1],
        "created": m.get("createdAt") or m.get("startDate"), "end": m.get("endDate"),
        "closed_time": m.get("closedTime"), "closed": bool(m.get("closed")),
        "volume": float(m.get("volumeNum") or 0), "yes_price": prices[0] if prices else None,
        "neg_risk": bool(m.get("negRisk")),
    }


def fetch_markets(closed: bool, min_volume: float = 50_000, since: str = "2023-11-01",
                  tags: list[str] | None = None) -> pd.DataFrame:
    rows, seen = [], set()
    for tag in tags or TAGS:
        cur, n, pages, seen_ev = None, 0, 0, set()
        while True:
            p = {"closed": str(closed).lower(), "limit": 100, "tag_slug": tag}
            if closed:
                p["end_date_min"] = f"{since}T00:00:00Z"
            if cur:
                p["after_cursor"] = cur            # Gamma keyset paging parameter
            d = _get(GAMMA, p)
            evs = d.get("events", [])
            ids = {e.get("id") for e in evs}
            if ids and ids <= seen_ev:             # guard: API returned a page we already saw
                print(f"{tag}: repeated page, stopping")
                break
            seen_ev |= ids
            pages += 1
            if pages % 20 == 0:
                print(f"  {tag}: {pages} pages, {len(seen_ev)} events, {n} markets kept", flush=True)
            for ev in evs:
                for m in ev.get("markets", []):
                    r = _market_row(ev, m, tag)
                    if not r or r["condition_id"] in seen or r["volume"] < min_volume:
                        continue
                    if closed and (r["closed_time"] or "") < since:
                        continue
                    seen.add(r["condition_id"])
                    rows.append(r)
                    n += 1
            cur = d.get("next_cursor")
            if not cur or not evs:
                break
        print(f"{tag}: {n} new markets (total {len(rows)})")
    df = pd.DataFrame(rows)
    for c in ["created", "end", "closed_time"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="mixed")
    if closed:
        df = df[df["yes_price"].isin([0.0, 1.0])]          # cleanly resolved only
        df["yes_won"] = (df["yes_price"] == 1.0).astype(int)
    return df.reset_index(drop=True)


def fetch_trades(cid: str, min_usd: float = MIN_TRADE_USD, max_pages: int = 200) -> pd.DataFrame:
    rows, cur = [], None
    for _ in range(max_pages):
        p = {"condition": cid, "limit": 1000, "taker_only": "true",
             "filter_type": "CASH", "filter_amount": min_usd}
        if cur:
            p["cursor"] = cur
        d = _get(TRADES_URL, p)
        for t in d.get("data") or []:
            rows.append((t["timestamp"], t["proxy_wallet"], t["side"], t["outcome_index"],
                         float(t["price"]), float(t["size"])))
        pg = d.get("pagination") or {}
        cur = pg.get("next_cursor")
        if not pg.get("has_more") or not cur:
            break
    df = pd.DataFrame(rows, columns=["ts", "wallet", "side", "outcome_index", "price", "size"])
    return df.sort_values("ts").reset_index(drop=True)


def collect_trades(markets: pd.DataFrame, workers: int = 12, refresh: bool = False) -> None:
    TRADES.mkdir(parents=True, exist_ok=True)
    todo = [c for c in markets["condition_id"] if refresh or not (TRADES / f"{c}.parquet").exists()]
    print(f"fetching trades for {len(todo)} markets")

    def one(c):
        fetch_trades(c).to_parquet(TRADES / f"{c}.parquet", index=False)

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for f in tqdm(as_completed(futs), total=len(futs), mininterval=15):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(e)


def load_trades(cids) -> pd.DataFrame:
    frames = []
    for c in cids:
        f = TRADES / f"{c}.parquet"
        if f.exists():
            t = pd.read_parquet(f)
            if len(t):
                t.insert(0, "condition_id", c)
                frames.append(t)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-markets", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--wide", action="store_true", help="extra tags, min volume 10k -> markets_wide.parquet")
    a = ap.parse_args()
    POLY.mkdir(parents=True, exist_ok=True)
    out = POLY / ("markets_wide.parquet" if a.wide else "markets_resolved.parquet")
    if not a.skip_markets:
        m = (fetch_markets(closed=True, min_volume=10_000, tags=TAGS + EXTRA_TAGS) if a.wide
             else fetch_markets(closed=True))
        m.to_parquet(out, index=False)
        print(f"{len(m)} resolved markets, yes rate {m['yes_won'].mean():.3f}")
    m = pd.read_parquet(out)
    collect_trades(m, a.workers)


if __name__ == "__main__":
    main()
