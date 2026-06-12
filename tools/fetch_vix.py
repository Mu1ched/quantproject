"""
Fetch CBOE VIX (risk sentiment) daily history from Yahoo Finance — free, no key.
VIX is the cleanest free spot-directional macro signal for FX: risk-off (VIX up)
drives safe-haven JPY bid and sells risk currencies (AUD); risk-on the reverse.

Lookahead safety: VIX close for day D finalises at US close (~21:00 UTC), so for
FX bars we use the PRIOR day's VIX (available_date = vix_date + 1 day).

Output: tools/vix_data.parquet (available_date, vix_close, vix_chg, vix_z)
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "vix_data.parquet"
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
       "?period1=1680000000&period2=2000000000&interval=1d")


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        js = json.load(r)
    res = js["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    close = res["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"vix_date": ts.tz_localize(None).normalize(),
                       "vix_close": close}).dropna()
    df = df.drop_duplicates("vix_date").sort_values("vix_date").reset_index(drop=True)
    df["vix_chg"] = df["vix_close"].pct_change()
    m = df["vix_close"].rolling(20).mean()
    s = df["vix_close"].rolling(20).std()
    df["vix_z"] = ((df["vix_close"] - m) / s.replace(0, pd.NA)).astype(float)
    # lookahead-safe: prior day's VIX available next calendar day
    df["available_date"] = df["vix_date"] + pd.Timedelta(days=1)
    out = df[["available_date", "vix_close", "vix_chg", "vix_z"]]
    out.to_parquet(OUT, index=False)
    print(f"Saved {len(out)} VIX days {out.available_date.min().date()}.."
          f"{out.available_date.max().date()} | "
          f"VIX range {df.vix_close.min():.1f}..{df.vix_close.max():.1f} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
