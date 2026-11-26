"""Download GDELT 2013-04-01..2015-12-31 into *_early folders (merged after final training)."""
from datetime import date
from event_prediction import gdelt
from event_prediction.config import DATA
from event_prediction.precursors import threats

gdelt.GDELT_AGG = DATA / "gdelt_agg_early"
gdelt.GDELT_TOP = DATA / "gdelt_top_early"
threats.OUT = DATA / "gdelt_precursor_early"
S, E = date(2013, 4, 1), date(2015, 12, 31)
gdelt.backfill(S, E, workers=6)
threats.backfill(S, E, workers=6) if "workers" in threats.backfill.__code__.co_varnames else threats.backfill(S, E)
print("early backfill done")
