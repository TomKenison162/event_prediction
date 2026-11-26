"""Central configuration: paths, event taxonomy, competition constants."""
from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
GDELT_AGG = DATA / "gdelt_agg"          # one parquet per day: aggregated counts
GDELT_TOP = DATA / "gdelt_top"          # one parquet per day: top events (for LLM context)
WIKI = DATA / "wiki"
SNAPSHOTS = DATA / "snapshots"          # timestamped raw live inputs (proof of no look-ahead)
MODELS = ROOT / "models"
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"

HISTORY_START = date(2016, 1, 1)

# ---------------------------------------------------------------------------
# Event taxonomy, built from CAMEO codes. An event matches a type if its
# EventCode starts with any of the listed prefixes.
# Each type has a template the LLM uses to phrase a checkable prediction.
# ---------------------------------------------------------------------------
EVENT_TYPES: dict[str, dict] = {
    "protest":            {"cameo": ["14"],        "phrase": "Protesters hold demonstrations in {place}"},
    "strike":             {"cameo": ["143"],       "phrase": "Workers or groups stage a strike/boycott in {place}"},
    "riot":               {"cameo": ["145"],       "phrase": "Violent protests or riots break out in {place}"},
    "arrest":             {"cameo": ["173"],       "phrase": "Authorities in {place} arrest a prominent figure / group"},
    "deport_expel":       {"cameo": ["174"],       "phrase": "{place} expels or deports people"},
    "repression":         {"cameo": ["175"],       "phrase": "Security forces in {place} violently suppress protesters"},
    "sanctions":          {"cameo": ["163"],       "phrase": "{place} imposes or is hit by new sanctions"},
    "diplomatic_break":   {"cameo": ["161", "164"],"phrase": "{place} cuts or downgrades diplomatic ties"},
    "force_posture":      {"cameo": ["15"],        "phrase": "{place} mobilises or deploys military forces"},
    "armed_clash":        {"cameo": ["190", "193", "194"], "phrase": "Armed clashes / ground fighting in {place}"},
    "airstrike":          {"cameo": ["195"],       "phrase": "Air or drone strikes hit {place}"},
    "ceasefire_violation":{"cameo": ["196"],       "phrase": "Ceasefire violated in {place}"},
    "abduction":          {"cameo": ["181"],       "phrase": "Kidnapping / hostage-taking in {place}"},
    "bombing":            {"cameo": ["183"],       "phrase": "Bombing attack in {place}"},
    "assassination":      {"cameo": ["186"],       "phrase": "Assassination or attempted assassination in {place}"},
    "mass_violence":      {"cameo": ["20"],        "phrase": "Mass-casualty violence in {place}"},
    "deescalation":       {"cameo": ["087"],       "phrase": "Truce / ceasefire / troop pullback announced in {place}"},
    "agreement":          {"cameo": ["057"],       "phrase": "{place} signs a formal agreement"},
}
TYPE_NAMES = list(EVENT_TYPES)

# ---------------------------------------------------------------------------
# Countries modelled (GDELT uses FIPS 10-4 codes). Chosen for heavy
# English-language coverage; extend freely.
# ---------------------------------------------------------------------------
COUNTRIES: dict[str, str] = {
    "US": "United States", "UK": "United Kingdom", "FR": "France", "GM": "Germany",
    "IT": "Italy", "SP": "Spain", "NL": "Netherlands", "BE": "Belgium", "PL": "Poland",
    "UP": "Ukraine", "RS": "Russia", "BO": "Belarus", "TU": "Turkey", "GR": "Greece",
    "RO": "Romania", "HU": "Hungary", "SR": "Serbia", "GG": "Georgia", "AM": "Armenia",
    "AJ": "Azerbaijan", "IS": "Israel", "WE": "West Bank", "GZ": "Gaza Strip",
    "LE": "Lebanon", "SY": "Syria", "IZ": "Iraq", "IR": "Iran", "SA": "Saudi Arabia",
    "YM": "Yemen", "JO": "Jordan", "EG": "Egypt", "LY": "Libya", "TS": "Tunisia",
    "SU": "Sudan", "OD": "South Sudan", "ET": "Ethiopia", "SO": "Somalia", "KE": "Kenya",
    "NI": "Nigeria", "ML": "Mali", "UV": "Burkina Faso", "NG": "Niger", "CG": "DR Congo",
    "SF": "South Africa", "ZI": "Zimbabwe", "MZ": "Mozambique", "AF": "Afghanistan",
    "PK": "Pakistan", "IN": "India", "BG": "Bangladesh", "CE": "Sri Lanka", "NP": "Nepal",
    "BM": "Myanmar", "TH": "Thailand", "RP": "Philippines", "ID": "Indonesia",
    "MY": "Malaysia", "CH": "China", "TW": "Taiwan", "KS": "South Korea",
    "KN": "North Korea", "JA": "Japan", "AS": "Australia", "CA": "Canada",
    "MX": "Mexico", "BR": "Brazil", "AR": "Argentina", "CO": "Colombia",
    "VE": "Venezuela", "PE": "Peru", "EC": "Ecuador", "CI": "Chile", "HA": "Haiti",
    "CU": "Cuba",
}
# English Wikipedia article per country, for attention (pageview) features.
WIKI_ARTICLES = {c: n.replace(" ", "_") for c, n in COUNTRIES.items()}
WIKI_ARTICLES.update({"CG": "Democratic_Republic_of_the_Congo", "GZ": "Gaza_Strip",
                      "WE": "West_Bank", "GG": "Georgia_(country)"})

# ---------------------------------------------------------------------------
# Model / competition settings
# ---------------------------------------------------------------------------
HORIZONS = list(range(1, 9))   # days between last complete data day and target day
MIN_HORIZON = 2                # issue on day C+1 using data to end of C, target >= C+2
                               # (guarantees the target is >= 1 calendar day after issue)
SEQ_LEN = 56                   # days of history fed to the sequence model
MAX_PER_DAY = 5
LIVE_START = date(2026, 12, 8)
LIVE_END = date(2026, 12, 21)
LAST_TARGET = date(2026, 12, 28)
# Rule "predicted date at least 1 day after the issued time": 1 = next calendar day allowed
# (backtest +29.2 / 14 days); set 2 if the group reads it as a full 24h (+24.4).
MIN_LEAD_DAYS = 1
ISSUE_HOUR_UTC = 22            # issue late: backtest 14-day net +17.5 at 22:00 vs +13.5 at 07:00
SCORE_THRESHOLD = 5.0          # only predict when expected judge score exceeds this

# Label: a (country, type, day) is "notable" if that day's matching events were
# covered by at least this many distinct articles (GDELT NumArticles summed over
# root events). Tuned in dataset.py to keep base rates sensible.
NOTABLE_ARTICLES = 40

LLM_MODEL = "claude-opus-5-5"
