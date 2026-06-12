"""
Fetch real short-term (3-month interbank) interest rates from the OECD SDMX API
(free, no key) for the 5 currencies behind our pairs. Used to (a) model swap/carry
income and (b) drive carry strategies.

Lookahead safety: a month's rate is stamped available from the 1st of the NEXT
month. (3-month interbank rates are observable in real time, so this is
conservative.)

Output: tools/rates_data.parquet  (available_date, ccy, rate)
"""
from __future__ import annotations
import csv, io, sys, urllib.request
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "rates_data.parquet"
BASE = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,/"
AREA = {"USD": "USA", "EUR": "EA20", "JPY": "JPN", "GBP": "GBR", "AUD": "AUS"}


def _fetch(area: str) -> pd.DataFrame:
    url = BASE + f"{area}.M.IR3TIB......?startPeriod=2023-06&format=csvfilewithlabels"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", "replace"))))
    df = pd.DataFrame({"period": [x["TIME_PERIOD"] for x in rows],
                       "rate": [float(x["OBS_VALUE"]) for x in rows]})
    return df.dropna().drop_duplicates("period").sort_values("period")


def main():
    frames = []
    for ccy, area in AREA.items():
        df = _fetch(area)
        # month YYYY-MM -> available on the 1st of the next month
        dt = pd.to_datetime(df["period"] + "-01") + pd.offsets.MonthBegin(1)
        frames.append(pd.DataFrame({"available_date": dt, "ccy": ccy,
                                    "rate": df["rate"].values}))
        print(f"  {ccy} ({area}): {len(df)} months "
              f"{df.period.min()}..{df.period.max()} | "
              f"rate {df.rate.min():.2f}..{df.rate.max():.2f}%")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
