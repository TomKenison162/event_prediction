"""Live context collectors: news RSS headlines + Polymarket markets.

Every call writes a timestamped JSON snapshot under data/snapshots/, which is
committed by the GitHub Action — proof that only pre-prediction data was used.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import feedparser
import requests

from .config import SNAPSHOTS

RSS_FEEDS = {
    "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "guardian_world": "https://www.theguardian.com/world/rss",
    "nyt_world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "npr_world": "https://feeds.npr.org/1004/rss.xml",
}
POLY_URL = "https://gamma-api.polymarket.com/events"
POLY_TAGS = ["geopolitics", "world", "politics", "middle-east", "ukraine"]
UA = {"User-Agent": "event-prediction-research/0.1"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _save(kind: str, payload) -> None:
    t = _now()
    d = SNAPSHOTS / t.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{t:%H%M%S}_{kind}.json").write_text(json.dumps(payload, indent=1, default=str))


def headlines(max_per_feed: int = 60) -> list[dict]:
    out = []
    for name, url in RSS_FEEDS.items():
        try:
            r = requests.get(url, headers=UA, timeout=30)
            feed = feedparser.parse(r.content)
        except Exception as e:  # noqa: BLE001 - one dead feed must not kill the run
            print(f"rss {name}: {e}")
            continue
        for e in feed.entries[:max_per_feed]:
            out.append({"src": name, "title": e.get("title", ""),
                        "summary": (e.get("summary", "") or "")[:300],
                        "published": e.get("published", ""), "link": e.get("link", "")})
    _save("rss", out)
    return out


NOISE = ("say ", "said ", "tweet", "post ", "mention", "temperature", "weather", "box office",
         "spotify", "youtube", "rotten tomatoes", " vs. ", "o/u", "up or down", "price of")


def polymarket(limit: int = 100, min_vol_24h: float = 2000.0) -> list[dict]:
    seen, out = set(), []
    for tag in POLY_TAGS:
        try:
            r = requests.get(POLY_URL, params={"closed": "false", "active": "true", "limit": limit,
                                               "order": "volume24hr", "ascending": "false",
                                               "tag_slug": tag}, headers=UA, timeout=30)
            events = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"polymarket {tag}: {e}")
            continue
        for ev in events:
            for m in ev.get("markets", []):
                q = (m.get("question") or "").lower()
                if m.get("id") in seen or m.get("closed") or any(n in q for n in NOISE):
                    continue
                if float(m.get("volume24hr") or 0) < min_vol_24h:
                    continue
                seen.add(m.get("id"))
                try:
                    prices = json.loads(m.get("outcomePrices") or "[]")
                    outcomes = json.loads(m.get("outcomes") or "[]")
                except json.JSONDecodeError:
                    continue
                out.append({
                    "event": ev.get("title"), "question": m.get("question"),
                    "outcomes": dict(zip(outcomes, prices)), "end": m.get("endDate"),
                    "chg_1d": m.get("oneDayPriceChange"), "chg_1w": m.get("oneWeekPriceChange"),
                    "vol_24h": m.get("volume24hr"), "tag": tag,
                })
    _save("polymarket", out)
    return out


def market_movers(markets: list[dict], n: int = 40) -> list[dict]:
    """Markets with the biggest recent price moves — the 'someone knows something' signal."""
    def mv(m):
        return abs(float(m.get("chg_1d") or 0)) + 0.5 * abs(float(m.get("chg_1w") or 0))
    return sorted(markets, key=mv, reverse=True)[:n]
