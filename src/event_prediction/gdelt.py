"""Download and aggregate GDELT event data.

History comes from GDELT 1.0 daily export files (one per day, from 2013).
Live data comes from GDELT 2.0 15-minute export files, aggregated into the
exact same per-day schema, so training and live features match.

Per day we write two small parquet files:
  gdelt_agg/YYYYMMDD.parquet : one row per (country, type) with counts
  gdelt_top/YYYYMMDD.parquet : top-5 most-covered events per (country, type)
"""
from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .config import EVENT_TYPES, GDELT_AGG, GDELT_TOP, HISTORY_START

V1_URL = "https://data.gdeltproject.org/events/{d}.export.CSV.zip"
V2_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.export.CSV.zip"

# column indices -> names. v1 daily files have 58 cols, v2 15-min have 61.
_COMMON = {6: "actor1", 16: "actor2", 25: "is_root", 26: "code", 29: "quad",
           30: "goldstein", 31: "mentions", 32: "sources", 33: "articles", 34: "tone"}
V1_COLS = {**_COMMON, 51: "country", 56: "added", 57: "url"}
V2_COLS = {**_COMMON, 53: "country", 59: "added", 60: "url"}

SESSION = requests.Session()


def _read(raw: bytes, cols: dict[int, str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(
                f, sep="\t", header=None, usecols=list(cols), dtype=str,
                quoting=3, on_bad_lines="skip", encoding_errors="replace",
            )
    df = df.rename(columns=cols)
    for c in ["is_root", "quad", "goldstein", "mentions", "sources", "articles", "tone"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["country", "code"])


def aggregate(df: pd.DataFrame, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn raw events for one day into (agg, top) frames."""
    rows, tops = [], []
    code = df["code"].astype(str)
    for t, spec in EVENT_TYPES.items():
        mask = np.zeros(len(df), dtype=bool)
        for p in spec["cameo"]:
            mask |= code.str.startswith(p).to_numpy()
        sub = df[mask]
        if sub.empty:
            continue
        root = sub[sub["is_root"] == 1]
        g = sub.groupby("country")
        gr = root.groupby("country")
        agg = pd.DataFrame({
            "n_events": g.size(),
            "n_root": gr.size(),
            "articles": g["articles"].sum(),
            "root_articles": gr["articles"].sum(),
            "sources": g["sources"].sum(),
            "max_sources": g["sources"].max(),
            "tone": g["tone"].mean(),
            "goldstein": g["goldstein"].mean(),
        }).fillna(0).reset_index()
        agg["type"] = t
        rows.append(agg)
        top = (sub.sort_values("articles", ascending=False)
                  .groupby("country").head(5)
                  [["country", "actor1", "actor2", "code", "articles", "url"]])
        top["type"] = t
        tops.append(top)

    # country-level totals (all events), by quad class
    g = df.groupby("country")
    tot = pd.DataFrame({
        "n_events": g.size(),
        "n_root": df[df["is_root"] == 1].groupby("country").size(),
        "articles": g["articles"].sum(),
        "root_articles": df[df["is_root"] == 1].groupby("country")["articles"].sum(),
        "sources": g["sources"].sum(),
        "max_sources": g["sources"].max(),
        "tone": g["tone"].mean(),
        "goldstein": g["goldstein"].mean(),
    }).fillna(0).reset_index()
    tot["type"] = "ALL"
    rows.append(tot)
    for q in (1, 2, 3, 4):
        s = df[df["quad"] == q].groupby("country")
        qa = pd.DataFrame({"n_events": s.size(), "articles": s["articles"].sum(),
                           "tone": s["tone"].mean(), "goldstein": s["goldstein"].mean()}
                          ).fillna(0).reset_index()
        qa["type"] = f"Q{q}"
        rows.append(qa)

    agg = pd.concat(rows, ignore_index=True)
    agg.insert(0, "date", pd.Timestamp(day))
    top = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()
    if not top.empty:
        top.insert(0, "date", pd.Timestamp(day))
    return agg, top


def _write(day: date, agg: pd.DataFrame, top: pd.DataFrame) -> None:
    GDELT_AGG.mkdir(parents=True, exist_ok=True)
    GDELT_TOP.mkdir(parents=True, exist_ok=True)
    key = day.strftime("%Y%m%d")
    agg.to_parquet(GDELT_AGG / f"{key}.parquet", index=False)
    if not top.empty:
        top.to_parquet(GDELT_TOP / f"{key}.parquet", index=False)


def fetch_v1_day(day: date, force: bool = False) -> bool:
    key = day.strftime("%Y%m%d")
    if (GDELT_AGG / f"{key}.parquet").exists() and not force:
        return True
    r = SESSION.get(V1_URL.format(d=key), timeout=120)
    if r.status_code != 200:
        return False
    df = _read(r.content, V1_COLS)
    agg, top = aggregate(df, day)
    _write(day, agg, top)
    return True


def fetch_v2_day(day: date, force: bool = False) -> bool:
    """Aggregate a day from the 96 fifteen-minute v2 files (used for live days,
    before/instead of the v1 daily file). Only uses files published by now."""
    key = day.strftime("%Y%m%d")
    if (GDELT_AGG / f"{key}.parquet").exists() and not force:
        return True
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    stamps = [(start + timedelta(minutes=15 * i)).strftime("%Y%m%d%H%M%S") for i in range(96)]
    frames = []

    def get(ts):
        r = SESSION.get(V2_URL.format(ts=ts), timeout=60)
        return _read(r.content, V2_COLS) if r.status_code == 200 else None

    with ThreadPoolExecutor(8) as ex:
        for f in ex.map(get, stamps):
            if f is not None:
                frames.append(f)
    if len(frames) < 90:          # incomplete day: don't write a partial aggregate
        return False
    agg, top = aggregate(pd.concat(frames, ignore_index=True), day)
    _write(day, agg, top)
    return True


def backfill(start: date, end: date, workers: int = 8) -> None:
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    todo = [d for d in days if not (GDELT_AGG / f"{d:%Y%m%d}.parquet").exists()]
    print(f"{len(days)} days in range, {len(todo)} to fetch")
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_v1_day, d): d for d in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), mininterval=10):
            d = futs[fut]
            try:
                if not fut.result():
                    failed.append(d)
            except Exception as e:  # noqa: BLE001 - keep going, report at end
                print(f"{d}: {e}")
                failed.append(d)
    # v1 sometimes misses days; fall back to v2 15-min files (available from 2015-02-19)
    for d in failed:
        try:
            ok = fetch_v2_day(d)
            print(f"{d}: v2 fallback {'ok' if ok else 'FAILED'}")
        except Exception as e:  # noqa: BLE001
            print(f"{d}: v2 fallback error {e}")


def load_agg(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    files = sorted(GDELT_AGG.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= f"{start:%Y%m%d}"]
    if end:
        files = [f for f in files if f.stem <= f"{end:%Y%m%d}"]
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill GDELT daily aggregates")
    ap.add_argument("--start", default=str(HISTORY_START))
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    backfill(date.fromisoformat(a.start), date.fromisoformat(a.end), a.workers)


if __name__ == "__main__":
    main()
