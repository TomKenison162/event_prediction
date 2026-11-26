"""GDELT precursor codes (threats, ultimatums, mobilisation, evacuation-like moves)
aggregated per (day, country). Features only — never targets. Same daily v1 files
as gdelt.py, so features are as-of end of day."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import gdelt
from ..config import DATA, HISTORY_START

OUT = DATA / "gdelt_precursor"
CODES = {
    "threaten": ["13"],                  # any threat
    "threat_force": ["138"],             # threaten with military force (blockade, occupation, attack, WMD)
    "threat_attack": ["1384", "1385"],   # threaten conventional attack / WMD
    "ultimatum": ["139"],                # give ultimatum
    "mobilise": ["152", "154"],          # increase police / military alert, mobilise armed forces
    "demand": ["10"],                    # demands
    "reject": ["12"],                    # reject / refuse
    "reduce_relations": ["16"],          # reduce relations (incl. aid cuts, expelling diplomats)
    "coerce": ["17"],                    # coerce
}


def aggregate(df: pd.DataFrame, day: date) -> pd.DataFrame:
    code = df["code"].astype(str)
    rows = []
    for name, prefixes in CODES.items():
        m = np.zeros(len(df), bool)
        for p in prefixes:
            m |= code.str.startswith(p).to_numpy()
        g = df[m].groupby("country")
        a = pd.DataFrame({"n": g.size(), "articles": g["articles"].sum(), "tone": g["tone"].mean()}).reset_index()
        a["kind"] = name
        rows.append(a)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "date", pd.Timestamp(day))
    return out


def fetch_day(day: date) -> bool:
    f = OUT / f"{day:%Y%m%d}.parquet"
    if f.exists():
        return True
    r = gdelt.SESSION.get(gdelt.V1_URL.format(d=f"{day:%Y%m%d}"), timeout=120)
    if r.status_code != 200:
        return False
    aggregate(gdelt._read(r.content, gdelt.V1_COLS), day).to_parquet(f, index=False)
    return True


def backfill(start: date, end: date, workers: int = 8) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    todo = [d for d in days if not (OUT / f"{d:%Y%m%d}.parquet").exists()]
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_day, d) for d in todo]
        for f in tqdm(as_completed(futs), total=len(futs), mininterval=30):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(e)


def load(start=None, end=None) -> pd.DataFrame:
    files = sorted(OUT.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= f"{start:%Y%m%d}"]
    if end:
        files = [f for f in files if f.stem <= f"{end:%Y%m%d}"]
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True) if files else pd.DataFrame()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=str(HISTORY_START))
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)))
    a = ap.parse_args()
    backfill(date.fromisoformat(a.start), date.fromisoformat(a.end))
