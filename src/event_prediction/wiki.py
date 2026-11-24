"""Daily English-Wikipedia pageviews for each country article (attention signal)."""
from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import quote

import time

import pandas as pd
import requests

from .config import HISTORY_START, WIKI, WIKI_ARTICLES

URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/user/{a}/daily/{s}/{e}")
HEADERS = {"User-Agent": "event-prediction-research/0.1 (python-requests)"}


def fetch(start: date = HISTORY_START, end: date | None = None) -> pd.DataFrame:
    end = end or date.today() - timedelta(days=1)
    out = []
    for code, art in WIKI_ARTICLES.items():
        url = URL.format(a=quote(art, safe=""), s=f"{start:%Y%m%d}00", e=f"{end:%Y%m%d}00")
        for attempt in range(6):
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(2 ** attempt * 2)
        time.sleep(0.5)
        if r.status_code != 200:
            print(f"wiki {code} {art}: HTTP {r.status_code}")
            continue
        for it in r.json().get("items", []):
            out.append((pd.Timestamp(it["timestamp"][:8]), code, it["views"]))
    df = pd.DataFrame(out, columns=["date", "country", "views"])
    return df


def update(end: date | None = None) -> pd.DataFrame:
    """Fetch full history (cheap: one request per country) and cache it."""
    WIKI.mkdir(parents=True, exist_ok=True)
    df = fetch(end=end)
    df.to_parquet(WIKI / "pageviews.parquet", index=False)
    return df


def load() -> pd.DataFrame:
    p = WIKI / "pageviews.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "country", "views"])


if __name__ == "__main__":
    d = update()
    print(d.groupby("country").size().describe())
