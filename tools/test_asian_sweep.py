"""
Test the hand-crafted Asian-range sweep-and-reverse strategy.

This is a one-off baseline run before the agent starts generating alongside
it. Runs the 36-combo grid on all cached London-session pairs, writes a
clean summary, and compares against the null distribution.

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/test_asian_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_engine import (
    load_all_data, run_sweep, load_sweep_results, LONDON_PAIRS,
)
from edge_hypotheses import SWEEP_ASIAN_SWEEP_REVERSE


def cached_pairs() -> list[str]:
    cache = PROJECT_ROOT / "edge_prepared_cache"
    return sorted({p.stem.replace("_m1", "") for p in cache.glob("*_m1.parquet")})


def main() -> int:
    print("=" * 70)
    print(" Asian-range sweep-and-reverse — baseline test")
    print("=" * 70)

    avail = set(cached_pairs())
    london_avail = [p for p in LONDON_PAIRS if p in avail]
    print(f"\nCached London pairs to test: {london_avail}")
    if not london_avail:
        print("ERROR: no London pairs in cache.")
        return 2

    print("\nLoading data...")
    t0 = time.time()
    train_dfs, test_dfs, measured_spreads = load_all_data(pairs=london_avail)
    print(f"  load_all_data took {time.time()-t0:.1f}s")

    grid = SWEEP_ASIAN_SWEEP_REVERSE["grid"]
    n_combos = len(list(grid))
    print(f"\nRunning sweep: {n_combos} combos × {len(london_avail)} pairs")
    print("  cost_mult=0.5, dynamic spread, exit_hour=13 (London close)")

    t0 = time.time()
    sweep_id = run_sweep(
        sweep_name="asian_sweep_reverse_baseline",
        entry_fn=SWEEP_ASIAN_SWEEP_REVERSE["entry_fn"],
        manager_fn=SWEEP_ASIAN_SWEEP_REVERSE["manager_fn"],
        grid=SWEEP_ASIAN_SWEEP_REVERSE["grid"],
        pairs=london_avail,
        session=SWEEP_ASIAN_SWEEP_REVERSE["session"],
        regime_mult=SWEEP_ASIAN_SWEEP_REVERSE["regime_mult"],
        train_dfs=train_dfs, test_dfs=test_dfs,
        measured_spreads=measured_spreads,
        cost_mult=0.5, n_workers=2,
        use_dynamic_spread=True,
    )
    elapsed = time.time() - t0
    print(f"\nSweep complete in {elapsed/60:.1f} min  sweep_id={sweep_id}")

    df = load_sweep_results(sweep_id)
    if df.empty:
        print("\n  NO hypothesis results returned.")
        return 1

    print(f"\n  Hypotheses: {len(df)}")
    print(f"  Test Sharpe: mean={df['test_sharpe'].mean():.3f}  "
          f"median={df['test_sharpe'].median():.3f}  "
          f"max={df['test_sharpe'].max():.3f}  "
          f"min={df['test_sharpe'].min():.3f}")
    print(f"  Test trades: mean={df['test_n'].mean():.1f}  "
          f"median={df['test_n'].median():.1f}  max={df['test_n'].max()}")
    print(f"  Test DD: mean={df['test_max_dd'].mean():.4f}")
    print(f"  Frac with test_sharpe > 0: {(df['test_sharpe']>0).mean():.1%}")
    print(f"  Frac with test_sharpe > 0.3: {(df['test_sharpe']>0.3).mean():.1%}")
    print(f"  Frac with test_sharpe > 0.5: {(df['test_sharpe']>0.5).mean():.1%}")

    # Compare against null (rough reference: −0.7 dynamic null Sharpe)
    NULL_BASELINE = -0.7
    delta = df['test_sharpe'].mean() - NULL_BASELINE
    print(f"\n  vs null baseline ({NULL_BASELINE}): mean shift = {delta:+.2f}")

    # Top 10 by test_sharpe
    print("\n  Top 10 by test_sharpe:")
    top = df.nlargest(10, 'test_sharpe')[
        ['test_sharpe', 'test_n', 'test_wr', 'test_max_dd', 'params_json']
    ]
    for _, r in top.iterrows():
        print(f"    sharpe={r['test_sharpe']:+.3f}  trades={int(r['test_n'])}  "
              f"wr={r['test_wr']:.1%}  dd={r['test_max_dd']:+.3f}  "
              f"params={r['params_json']}")

    # Write report
    report = []
    report.append("# Asian-range sweep-and-reverse — baseline result\n")
    report.append(f"- Sweep ID: `{sweep_id}`")
    report.append(f"- Pairs: `{london_avail}`")
    report.append(f"- Combos: {len(df)}")
    report.append(f"- Wall time: {elapsed/60:.1f} min\n")
    report.append("## Summary statistics\n")
    report.append("| metric | value |")
    report.append("|---|---|")
    report.append(f"| Mean test Sharpe | {df['test_sharpe'].mean():.3f} |")
    report.append(f"| Median test Sharpe | {df['test_sharpe'].median():.3f} |")
    report.append(f"| Best variant test Sharpe | {df['test_sharpe'].max():.3f} |")
    report.append(f"| Mean test trades | {df['test_n'].mean():.1f} |")
    report.append(f"| Frac variants > 0 | {(df['test_sharpe']>0).mean():.1%} |")
    report.append(f"| Frac variants > 0.3 | {(df['test_sharpe']>0.3).mean():.1%} |")
    report.append(f"| Frac variants > 0.5 | {(df['test_sharpe']>0.5).mean():.1%} |")
    report.append(f"| Delta vs null baseline (−0.7) | {delta:+.2f} |\n")
    report.append("## Top 10 variants\n")
    report.append("| sharpe | trades | win_rate | max_dd | params |")
    report.append("|---|---|---|---|---|")
    for _, r in top.iterrows():
        report.append(f"| {r['test_sharpe']:+.3f} | {int(r['test_n'])} | "
                      f"{r['test_wr']:.1%} | {r['test_max_dd']:+.3f} | "
                      f"`{r['params_json']}` |")
    out = PROJECT_ROOT / "tools" / "asian_sweep_baseline.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n  Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
