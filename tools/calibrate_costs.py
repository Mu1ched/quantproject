"""
Calibrate backtest cost model to live MT5 execution data.

Reads `execution_quality.csv` at project root, computes per-pair live spread
and slippage stats, and writes `live_measured_spreads.json` so the engine's
`load_all_data` can override the (much looser) Dukascopy median spreads.

Lowers the default min_fills threshold from 20 (in agent/tca.py) to 3 so
sparse early-stage data still produces an anchor. Prints a comparison vs the
existing edge_measured_spreads.json.

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/calibrate_costs.py
    python tools/calibrate_costs.py --min-fills 5
    python tools/calibrate_costs.py --ny-only false   # include all hours
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Reuse pip table from agent/tca.py
from agent.tca import _PIP_SIZE

BROKER_TO_PAIR = {
    "EURUSD": "EUR_USD",
    "USDJPY": "USD_JPY",
    "XAUUSD": "XAU_USD",
    "GBPUSD": "GBP_USD",
    "AUDUSD": "AUD_USD",
    "EURJPY": "EUR_JPY",
    "GBPJPY": "GBP_JPY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--csv", default=str(PROJECT_ROOT / "execution_quality.csv"),
                   help="Path to live execution CSV")
    p.add_argument("--out", default=str(PROJECT_ROOT / "live_measured_spreads.json"))
    p.add_argument("--min-fills", type=int, default=3,
                   help="Minimum fills per pair to publish a live spread (default 3)")
    p.add_argument("--ny-only", default="true",
                   help="Filter to NY-range hours 13-22 UTC (true/false)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[calibrate] ERROR: {csv_path} not found.")
        return 2

    df = pd.read_csv(csv_path)
    df["pair_canonical"] = df["pair"].map(BROKER_TO_PAIR).fillna(df["pair"])
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour

    if args.ny_only.lower() == "true":
        before = len(df)
        df = df[df["hour"].between(13, 22)]
        print(f"[calibrate] NY-hour filter: {before} -> {len(df)} fills")

    # Load existing Dukascopy medians for comparison.
    dukascopy_path = PROJECT_ROOT / "edge_measured_spreads.json"
    dukascopy: dict = {}
    if dukascopy_path.exists():
        with open(dukascopy_path) as f:
            dukascopy = json.load(f).get("spreads", {})

    rows = []
    out_spreads: dict = {}
    for pair_canon, grp in df.groupby("pair_canonical"):
        n = len(grp)
        med_spread_pips = float(grp["spread_pips"].median())
        med_abs_slip_pips = float(grp["slippage_pips"].abs().median())
        pip = _PIP_SIZE.get(pair_canon, 1e-4)
        live_spread_price = med_spread_pips * pip
        duka_price = dukascopy.get(pair_canon)
        duka_pips = (duka_price / pip) if duka_price else None
        ratio = (live_spread_price / duka_price) if (duka_price and duka_price > 0) else None
        rows.append({
            "pair": pair_canon,
            "n_fills": n,
            "live_spread_pips": round(med_spread_pips, 3),
            "median_abs_slip_pips": round(med_abs_slip_pips, 3),
            "dukascopy_spread_pips": round(duka_pips, 3) if duka_pips is not None else None,
            "live/dukascopy_ratio": round(ratio, 3) if ratio is not None else None,
            "published": n >= args.min_fills,
        })
        if n >= args.min_fills:
            out_spreads[pair_canon] = live_spread_price

    # Print summary
    print("\nPer-pair calibration:")
    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False))
    sparse = summ[summ["n_fills"] < 10]
    if not sparse.empty:
        print(f"\n[calibrate] WARNING: {len(sparse)} pair(s) have <10 fills — "
              "calibration anchor is noisy. Treat as v1; refresh once "
              "MT5live accumulates more data.")

    if not out_spreads:
        print("\n[calibrate] No pairs met min_fills threshold; nothing written.")
        return 1

    payload = {
        "_meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": f"local tools/calibrate_costs.py from {csv_path.name}",
            "min_fills": args.min_fills,
            "ny_only": args.ny_only,
        },
        "spreads": out_spreads,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[calibrate] Wrote {len(out_spreads)} pair(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
