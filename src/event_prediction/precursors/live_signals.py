"""Live-only early-warning signals (no usable history, so they go to the LLM as
context, never into model training). All public data. Each call is snapshotted
with a timestamp; hourly runs build up our own history for trend comparisons.

  * US State Dept travel advisories (level + last update per country)
  * US military aircraft airborne over hotspot regions (OpenSky, ICAO block AE0000-AFFFFF)
  * recent escalation notes from the UK FCDO change history
  * GDELT threat / ultimatum spikes vs each country's 28-day baseline
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import feedparser
import numpy as np
import pandas as pd
import requests

from ..config import COUNTRIES, SNAPSHOTS

UA = {"User-Agent": "event-prediction-research/0.1"}
STATE_RSS = "https://travel.state.gov/_res/rss/TAsTWs.xml"
OPENSKY = "https://opensky-network.org/api/states/all"
REGIONS = {   # lamin, lomin, lamax, lomax
    "middle_east": (12, 30, 40, 63), "eastern_europe": (44, 20, 60, 42),
    "taiwan_strait": (18, 112, 30, 128), "korea": (33, 124, 43, 132),
    "caribbean_venezuela": (5, -80, 20, -58), "red_sea_horn": (5, 32, 22, 52),
}


def _snap(kind: str, payload) -> None:
    t = datetime.now(timezone.utc)
    d = SNAPSHOTS / t.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{t:%H%M%S}_{kind}.json").write_text(json.dumps(payload, indent=1, default=str))


def state_dept() -> list[dict]:
    r = requests.get(STATE_RSS, headers=UA, timeout=60)
    feed = feedparser.parse(r.content)
    out = []
    for e in feed.entries:
        m = re.match(r"(.+?)\s*-\s*Level\s*(\d)", e.get("title", ""))
        if not m:
            continue
        out.append({"country": m.group(1).strip(), "level": int(m.group(2)),
                    "updated": e.get("published", ""), "title": e.get("title")})
    _snap("state_dept", out)
    return out


def recent_state_changes(items: list[dict], days: int = 7) -> list[dict]:
    cut = datetime.now(timezone.utc) - timedelta(days=days)
    keep = []
    for it in items:
        try:
            t = pd.Timestamp(it["updated"], tz="UTC")
        except (ValueError, TypeError):
            continue
        if t >= cut and it["level"] >= 3:
            keep.append(it)
    return keep


def military_air() -> dict[str, dict]:
    """Count US-military-registered (ICAO AE0000-AFFFFF) aircraft airborne per region."""
    out = {}
    for name, (a, b, c, d) in REGIONS.items():
        try:
            r = requests.get(OPENSKY, params={"lamin": a, "lomin": b, "lamax": c, "lomax": d},
                             headers=UA, timeout=60)
            st = (r.json() or {}).get("states") or []
        except Exception as e:  # noqa: BLE001 - anonymous quota is small; don't fail the run
            out[name] = {"error": repr(e)}
            continue
        mil = [s for s in st if s[0] and "ae0000" <= s[0].lower() <= "afffff" and not s[8]]
        out[name] = {"total_airborne": sum(1 for s in st if not s[8]), "us_military": len(mil),
                     "us_military_callsigns": sorted({(s[1] or "").strip() for s in mil if s[1]})[:25]}
    _snap("military_air", out)
    return out


def military_air_trend(now: datetime, days: int = 7) -> dict[str, float]:
    """Mean us_military count per region over our own past hourly snapshots."""
    counts: dict[str, list] = {k: [] for k in REGIONS}
    for i in range(1, days + 1):
        d = SNAPSHOTS / (now - timedelta(days=i)).strftime("%Y-%m-%d")
        for f in d.glob("*_military_air.json") if d.exists() else []:
            try:
                for k, v in json.loads(f.read_text()).items():
                    if k in counts and "us_military" in v:
                        counts[k].append(v["us_military"])
            except (json.JSONDecodeError, OSError):
                continue
    return {k: float(np.mean(v)) for k, v in counts.items() if v}


def fcdo_recent(days: int = 7) -> list[dict]:
    from . import fcdo
    df = fcdo.load()
    if df.empty:
        return []
    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    keys = ["staff_withdrawn", "leave_now", "airspace", "travel_upgrade", "tensions", "military", "unrest"]
    sub = df[(df["ts"] >= cut) & (df[keys].sum(1) > 0)]
    return [{"country": r.slug, "ts": str(r.ts), "note": r.note[:220],
             "flags": [k for k in keys if getattr(r, k)]} for r in sub.itertuples()]


def threat_spikes(cutoff, min_ratio: float = 3.0) -> list[dict]:
    from . import threats
    th = threats.load(cutoff - timedelta(days=35), cutoff)
    if th.empty:
        return []
    out = []
    last = pd.Timestamp(cutoff)
    for (c, k), g in th[th["kind"].isin(["threat_force", "threat_attack", "ultimatum", "mobilise"])].groupby(["country", "kind"]):
        if c not in COUNTRIES:
            continue
        g = g.set_index("date")["articles"]
        recent = g[g.index > last - pd.Timedelta(days=3)].sum() / 3
        base = g[g.index <= last - pd.Timedelta(days=3)].sum() / 28 + 1
        if recent / base >= min_ratio and recent >= 50:
            out.append({"country": COUNTRIES[c], "kind": k, "articles_per_day_3d": round(recent),
                        "x_baseline": round(recent / base, 1)})
    return sorted(out, key=lambda x: -x["x_baseline"])[:20]


def collect(now: datetime, cutoff) -> dict:
    sig = {}
    for name, fn in [("state_dept_changes", lambda: recent_state_changes(state_dept())),
                     ("military_air", military_air),
                     ("military_air_7d_mean", lambda: military_air_trend(now)),
                     ("fcdo_recent", fcdo_recent),
                     ("threat_spikes", lambda: threat_spikes(cutoff))]:
        try:
            sig[name] = fn()
        except Exception as e:  # noqa: BLE001 - every signal optional
            sig[name] = {"error": repr(e)}
    return sig
