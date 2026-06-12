"""
Cost-sensitivity sweep — diagnose how much the engine's cost stack drives results.

Re-runs SWEEP_ORB_NY at five cost_mult levels and tabulates mean/best/positive
fraction of test Sharpe per level. Use the slope to judge how much of the
0/152-survivors result is "no edge" vs "costs too aggressive".

What cost_mult scales:
    Only the static measured spread and its derived slippage (slip = 20% of
    spread). See edge_engine.run_sweep lines 1514-1517.

What cost_mult does NOT scale (small but non-zero floor even at cost_mult=0):
    - NEWS_SLIPPAGE_MULT (3.0x on news-bar fills)
    - SWAP_PIPS_PER_NIGHT (overnight swap, rarely paid by intraday ORB)
    - The spread gate (blocks elevated-spread bars regardless of cost_mult)

For an intraday NY ORB that exits same-day at 21:00 UTC with a spread gate
already filtering high-spread bars, the news/swap floor is small. The slope
of mean Sharpe vs cost_mult is still the right diagnostic.

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/cost_sensitivity.py
    # or with custom levels and worker count:
    python tools/cost_sensitivity.py --cost-mults 0,0.5,1,2 --workers 4

Output:
    - Markdown table printed to stdout.
    - tools/cost_sensitivity_report.md (overwrites on each run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_engine import load_all_data, run_sweep, load_sweep_results
from edge_hypotheses import SWEEP_ORB_NY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument(
        "--cost-mults", default="0.0,0.25,0.5,1.0,1.5",
        help="Comma-separated cost_mult levels (default: 0.0,0.25,0.5,1.0,1.5)",
    )
    p.add_argument(
        "--workers", type=int, default=2,
        help="run_sweep n_workers (default: 2)",
    )
    p.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "tools" / "cost_sensitivity_report.md"),
        help="Path to write the markdown report",
    )
    return p.parse_args()


def summarise(df) -> dict:
    """Compute the metrics we care about for one cost_mult level."""
    if df.empty:
        return {
            "n": 0, "mean_sharpe": float("nan"), "median_sharpe": float("nan"),
            "best_sharpe": float("nan"), "frac_positive": float("nan"),
            "mean_trades": float("nan"),
        }
    s = df["test_sharpe"]
    return {
        "n": len(df),
        "mean_sharpe":   round(float(s.mean()), 3),
        "median_sharpe": round(float(s.median()), 3),
        "best_sharpe":   round(float(s.max()), 3),
        "frac_positive": round(float((s > 0).mean()), 3),
        "mean_trades":   round(float(df.get("test_n", df.get("train_n", 0)).mean()), 1),
    }


def format_table(rows: list[dict]) -> str:
    header = "| cost_mult | n | mean_sharpe | median_sharpe | best_sharpe | frac_positive | mean_trades |"
    divider = "|---|---|---|---|---|---|---|"
    body = "\n".join(
        f"| {r['cost_mult']} | {r['n']} | {r['mean_sharpe']} | {r['median_sharpe']} | "
        f"{r['best_sharpe']} | {r['frac_positive']} | {r['mean_trades']} |"
        for r in rows
    )
    return "\n".join([header, divider, body])


def main() -> int:
    args = parse_args()
    cost_mults = [float(x) for x in args.cost_mults.split(",")]

    print(f"[cost-sensitivity] Loading data (cached after first call)...")
    train_dfs, test_dfs, measured_spreads = load_all_data()
    print(f"[cost-sensitivity] Data loaded: {len(train_dfs)} pairs.")

    rows = []
    for cm in cost_mults:
        sweep_name = f"cost_sens_orb_ny_cm{cm:g}"
        print(f"\n[cost-sensitivity] Running cost_mult={cm} ...")
        sweep_id = run_sweep(
            sweep_name=sweep_name,
            entry_fn=SWEEP_ORB_NY["entry_fn"],
            manager_fn=SWEEP_ORB_NY["manager_fn"],
            grid=SWEEP_ORB_NY["grid"],
            pairs=SWEEP_ORB_NY["pairs"],
            session=SWEEP_ORB_NY["session"],
            regime_mult=SWEEP_ORB_NY["regime_mult"],
            train_dfs=train_dfs,
            test_dfs=test_dfs,
            measured_spreads=measured_spreads,
            cost_mult=cm,
            n_workers=args.workers,
        )
        df = load_sweep_results(sweep_id)
        summary = summarise(df)
        summary["cost_mult"] = cm
        summary["sweep_id"] = sweep_id
        rows.append(summary)
        print(f"  -> n={summary['n']}, mean_sharpe={summary['mean_sharpe']}, "
              f"best={summary['best_sharpe']}, frac_pos={summary['frac_positive']}")

    table = format_table(rows)
    report = (
        "# Cost-sensitivity sweep — SWEEP_ORB_NY\n\n"
        "Higher cost_mult should produce lower mean test Sharpe (monotonic).\n"
        "Non-monotonic = engine bug.\n\n"
        "If `mean_sharpe` is strongly positive at `cost_mult=0` and clearly\n"
        "negative at `cost_mult=1.0`, the cost stack is the dominant driver\n"
        "of strategy outcomes — recalibrate against live TCA before drawing\n"
        "edge conclusions.\n\n"
        f"{table}\n\n"
        "Sweep IDs (for `load_sweep_results` in a notebook):\n\n"
        + "\n".join(f"- `cost_mult={r['cost_mult']}` -> `{r['sweep_id']}`" for r in rows)
        + "\n"
    )
    print("\n" + table + "\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[cost-sensitivity] Report written: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
