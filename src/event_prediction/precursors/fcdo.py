"""UK FCDO travel-advice change history: a public, timestamped early-warning signal.

Every country page on gov.uk carries `change_history`: one timestamped note per
update (e.g. 2026-02-27 14:27 "UK staff have been temporarily withdrawn from Iran"
— the day before the US/Israeli strikes). We fetch all ~225 pages, keep every note,
and tag each with precursor categories via patterns.
"""
from __future__ import annotations

import re
import time

import pandas as pd
import requests

from ..config import COUNTRIES, DATA

OUT = DATA / "precursors"
INDEX = "https://www.gov.uk/api/content/foreign-travel-advice"
PAGE = "https://www.gov.uk/api/content/foreign-travel-advice/{slug}"
UA = {"User-Agent": "event-prediction-research/0.1"}

# precursor categories (ordered roughly by severity)
CATEGORIES = {
    "staff_withdrawn": r"staff (have been |are being )?(temporarily )?(withdrawn|evacuated|relocated)|embassy .*(closed|suspended|operat\w* remotely)|ordered departure|drawdown",
    "leave_now": r"\bleave (now|immediately|by any means)|should leave|depart (now|immediately)",
    "airspace": r"airspace|flights? (have been )?(suspended|cancel)|airport (is )?closed",
    "travel_upgrade": r"advise against all (but essential )?travel|now advises? against",
    "military": r"military (action|activity|operation|strikes?)|missile|air ?strikes?|drone attack|armed conflict|hostilities|invasion",
    "tensions": r"regional (tensions|escalation|risks|security)|escalation|heightened tension|security situation",
    "unrest": r"protests?|demonstrations?|unrest|civil disorder|curfew|state of emergency|martial law|coup",
    "terror": r"terroris|attack(s)? (are|is) (very )?likely",
    "downgrade": r"removal of|removed (the )?advice|no longer advise|lifted|reopened|has resumed",
}
CAT_RE = {k: re.compile(v, re.I) for k, v in CATEGORIES.items()}

# gov.uk slug -> our FIPS code (only countries we model)
SLUG_TO_FIPS = {
    "usa": "US", "france": "FR", "germany": "GM", "italy": "IT", "spain": "SP",
    "netherlands": "NL", "belgium": "BE", "poland": "PL", "ukraine": "UP", "russia": "RS",
    "belarus": "BO", "turkey": "TU", "greece": "GR", "romania": "RO", "hungary": "HU",
    "serbia": "SR", "georgia": "GG", "armenia": "AM", "azerbaijan": "AJ", "israel": "IS",
    "the-occupied-palestinian-territories": "WE", "lebanon": "LE", "syria": "SY", "iraq": "IZ",
    "iran": "IR", "saudi-arabia": "SA", "yemen": "YM", "jordan": "JO", "egypt": "EG", "libya": "LY",
    "tunisia": "TS", "sudan": "SU", "south-sudan": "OD", "ethiopia": "ET", "somalia": "SO",
    "kenya": "KE", "nigeria": "NI", "mali": "ML", "burkina-faso": "UV", "niger": "NG",
    "democratic-republic-of-the-congo": "CG", "south-africa": "SF", "zimbabwe": "ZI",
    "mozambique": "MZ", "afghanistan": "AF", "pakistan": "PK", "india": "IN", "bangladesh": "BG",
    "sri-lanka": "CE", "nepal": "NP", "myanmar": "BM", "thailand": "TH", "philippines": "RP",
    "indonesia": "ID", "malaysia": "MY", "china": "CH", "taiwan": "TW", "south-korea": "KS",
    "north-korea": "KN", "japan": "JA", "australia": "AS", "canada": "CA", "mexico": "MX",
    "brazil": "BR", "argentina": "AR", "colombia": "CO", "venezuela": "VE", "peru": "PE",
    "ecuador": "EC", "chile": "CI", "haiti": "HA", "cuba": "CU",
}


def _get(url):
    for i in range(5):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** i)
    r.raise_for_status()


def slugs() -> list[str]:
    d = _get(INDEX)
    kids = d.get("links", {}).get("children", [])
    return sorted({k["base_path"].rsplit("/", 1)[-1] for k in kids if "base_path" in k})


def tag(note: str) -> dict[str, int]:
    return {k: int(bool(rx.search(note or ""))) for k, rx in CAT_RE.items()}


def fetch_all(sleep: float = 0.15) -> pd.DataFrame:
    rows = []
    for s in slugs():
        try:
            d = _get(PAGE.format(slug=s))
        except Exception as e:  # noqa: BLE001
            print(f"fcdo {s}: {e}")
            continue
        det = d.get("details", {})
        alert = ",".join(det.get("alert_status") or [])
        for ch in det.get("change_history") or []:
            note = (ch.get("note") or "").strip()
            rows.append({"slug": s, "country": SLUG_TO_FIPS.get(s), "ts": ch.get("public_timestamp"),
                         "note": note, "current_alert": alert, **tag(note)})
        time.sleep(sleep)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="mixed")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def update() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df = fetch_all()
    df.to_parquet(OUT / "fcdo_changes.parquet", index=False)
    return df


def load() -> pd.DataFrame:
    f = OUT / "fcdo_changes.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def daily_features(df: pd.DataFrame, days: pd.DatetimeIndex, countries: list[str]) -> dict[str, "np.ndarray"]:
    """[D, C] count of notes per category published on each day (UTC). A note
    published on day d is usable for a cutoff at the end of day d."""
    import numpy as np
    df = df[df["country"].isin(countries)].copy()
    df["day"] = df["ts"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    out = {}
    ci = {c: i for i, c in enumerate(countries)}
    di = {d: i for i, d in enumerate(days)}
    for cat in ["any", *CATEGORIES]:
        a = np.zeros((len(days), len(countries)), np.float32)
        sub = df if cat == "any" else df[df[cat] == 1]
        for r in sub.itertuples():
            if r.day in di:
                a[di[r.day], ci[r.country]] += 1
        out[cat] = a
    return out


if __name__ == "__main__":
    d = update()
    print(f"{len(d)} notes, {d.slug.nunique()} countries, {d.ts.min()} -> {d.ts.max()}")
    print(d[[c for c in CATEGORIES]].sum().to_string())
