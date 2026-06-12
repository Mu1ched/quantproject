"""
Fetch CFTC Commitments of Traders (legacy futures-only) for the FX futures that
map to our spot pairs, and build a daily, lookahead-safe positioning series.

Signal: net non-commercial (speculator) position as % of open interest, plus a
26-week COT index (0-100). Speculators are trend-followers who cluster at
extremes — a classic contrarian setup at the tails.

Lookahead safety: the COT report is dated Tuesday but RELEASED ~Friday 15:30 ET.
We stamp each report with an 'available_date' = report Friday + 1 day, and the
resampler must only use a value once available_date <= bar date.

Output: tools/cot_data.parquet  (columns: date, pair, net_spec_pct, cot_index)

Currency-futures sign note: JPY futures are JPY/USD, so net-long-JPY = bearish
USD/JPY -> sign flipped for USD_JPY. EUR_JPY proxied from EUR FX (sparse XRATE
contract skipped).
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "tools" / "cot_data.parquet"
BASE = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# CFTC contract name -> (pair, sign). sign=-1 means net-long-future is bearish-pair.
CONTRACTS = {
    "EURO FX - CHICAGO MERCANTILE EXCHANGE":                 ("EUR_USD", +1),
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE":           ("GBP_USD", +1),
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE":            ("USD_JPY", -1),
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE":       ("AUD_USD", +1),
}


def _get(where: str) -> list:
    params = {
        "$select": ("market_and_exchange_names,report_date_as_yyyy_mm_dd,"
                    "noncomm_positions_long_all,noncomm_positions_short_all,"
                    "open_interest_all"),
        "$where": where,
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": "5000",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "edge-hunt/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    frames = []
    for name, (pair, sign) in CONTRACTS.items():
        where = (f"market_and_exchange_names='{name}' "
                 f"and report_date_as_yyyy_mm_dd >= '2023-06-01T00:00:00'")
        rows = _get(where)
        if not rows:
            print(f"  {pair}: no rows"); continue
        df = pd.DataFrame(rows)
        for c in ("noncomm_positions_long_all", "noncomm_positions_short_all",
                  "open_interest_all"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        rep = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
        net = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        net_pct = sign * net / df["open_interest_all"].replace(0, pd.NA)
        out = pd.DataFrame({"report_date": rep, "pair": pair,
                            "net_spec_pct": net_pct.astype(float)})
        # 26-week COT index (0-100) within pair
        roll = out["net_spec_pct"]
        lo = roll.rolling(26, min_periods=8).min()
        hi = roll.rolling(26, min_periods=8).max()
        out["cot_index"] = ((roll - lo) / (hi - lo).replace(0, pd.NA) * 100).astype(float)
        # Lookahead-safe availability: report Tue -> released that week's Friday
        # (report_date is Tuesday; +3 days = Fri; +1 more day to be safe = Sat,
        # so first usable bar is the following Monday).
        out["available_date"] = (out["report_date"] + pd.Timedelta(days=4)).dt.normalize()
        frames.append(out)
        print(f"  {pair}: {len(out)} weekly reports "
              f"{out.report_date.min().date()}..{out.report_date.max().date()} | "
              f"net_spec_pct range {out.net_spec_pct.min():.2f}..{out.net_spec_pct.max():.2f}")

    if not frames:
        print("No COT data fetched."); return 1
    cot = pd.concat(frames, ignore_index=True)
    cot.to_parquet(OUT, index=False)
    print(f"\nSaved {len(cot)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
