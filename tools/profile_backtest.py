"""
Profile a single run_backtest call to see where the time actually goes.

Answers two questions:
  1. Where does the per-bar runtime get spent? (itertuples / entry_fn / engine
     fill logic / spread-slip computation / state mutation)
  2. How long does one full sweep (8 combos × 1 pair, dynamic mode, session
     filter) take after optimisations #1–#4?

Uses entry_orb_ny from edge_hypotheses (known to produce trades) on EUR_USD
train data with dynamic spread + cost_mult=0.5 + session-hour filter.

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/profile_backtest.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_engine import (
    load_all_data, run_backtest, run_sweep, load_sweep_results,
    make_manager,
)
from edge_hypotheses import SWEEP_ORB_NY


def pick_pair(train_dfs: dict) -> str:
    """Prefer EUR_USD; fall back to first cached pair."""
    return "EUR_USD" if "EUR_USD" in train_dfs else next(iter(train_dfs))


def build_single_combo_registry(entry_fn, manager_fn, pair: str,
                                 session: str, params: dict) -> tuple:
    slot_class = "profile_orb_ny"
    registry = [{
        "id":               "profile_hyp_0",
        "family":           "session_based",
        "slot_class":       slot_class,
        "pairs":            [pair],
        "session":          session,
        "allow_concurrent": False,
        "regime_mult":      SWEEP_ORB_NY["regime_mult"],
        "params":           params,
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}
    return registry, slot_managers, slot_entries


def main() -> int:
    print("=" * 70)
    print(" Profiling run_backtest under post-optimisation cost stack")
    print("=" * 70)

    print("\n[profile] Loading data (cached parquets)...")
    t0 = time.time()
    train_dfs, test_dfs, measured_spreads = load_all_data(
        pairs=["EUR_USD", "USD_JPY", "GBP_USD"],
    )
    print(f"  load_all_data took {time.time()-t0:.1f}s")

    pair = pick_pair(train_dfs)
    train_subset = {pair: train_dfs[pair]}
    print(f"  Profiling on pair: {pair}")
    print(f"  Train bars: {len(train_dfs[pair]):,} (post common-range trim)")

    # ── Part 1: cProfile a single run_backtest call ─────────────────────────
    print("\n" + "=" * 70)
    print(" Part 1: cProfile of one run_backtest call")
    print("=" * 70)
    entry_fn = SWEEP_ORB_NY["entry_fn"]
    manager_fn = make_manager(exit_hour=21, use_profit_lock=False)
    params = next(iter(SWEEP_ORB_NY["grid"]))   # first combo
    registry, slot_managers, slot_entries = build_single_combo_registry(
        entry_fn, manager_fn, pair, "ny", params,
    )

    print(f"  Strategy: entry_orb_ny  params={params}")
    print(f"  Mode: dynamic spread, cost_mult=0.5, session_hours=(13,21)")
    print(f"  Profiling...")

    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    trades, _bal, _ = run_backtest(
        train_subset, None, None,
        registry, slot_managers, slot_entries,
        cost_mult=0.5,
        session_hours=(13, 21),
    )
    prof.disable()
    elapsed = time.time() - t0

    print(f"\n  run_backtest wall-time: {elapsed:.2f}s")
    print(f"  Trades produced: {len(trades) if trades is not None else 0}")

    # Top-30 by cumulative time
    print("\n  Top 30 functions by cumulative time:")
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    # Top-15 by tottime (self time only, excludes children)
    print("\n  Top 15 functions by tottime (self time only):")
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("tottime")
    ps.print_stats(15)
    print(s.getvalue())

    # Bucket-style summary: entry_fn vs engine
    print("\n  Time bucket summary:")
    stats = pstats.Stats(prof)
    entry_fn_time = 0.0
    fill_helper_time = 0.0
    itertuples_time = 0.0
    spread_slip_time = 0.0
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        name = func[2]  # function name
        if "entry_orb_ny" in name or "entry_" in str(func[0]):
            entry_fn_time += ct
        if "_compute_stop_fill_price" in name or "_session_slip_mult" in name:
            fill_helper_time += ct
            if "_session_slip_mult" in name:
                spread_slip_time += ct
        if "itertuples" in name:
            itertuples_time += ct
    print(f"    entry_fn (user-supplied strategy):  ~{entry_fn_time:.2f}s "
          f"({entry_fn_time/elapsed*100:.1f}%)")
    print(f"    _compute_stop_fill_price + slip:    ~{fill_helper_time:.2f}s "
          f"({fill_helper_time/elapsed*100:.1f}%)")
    print(f"    itertuples iteration overhead:      ~{itertuples_time:.2f}s "
          f"({itertuples_time/elapsed*100:.1f}%)")
    print(f"    Everything else:                    ~{elapsed - entry_fn_time - fill_helper_time - itertuples_time:.2f}s")

    # ── Part 2: full sweep timing on 1 pair, 8 combos ───────────────────────
    print("\n" + "=" * 70)
    print(" Part 2: full run_sweep timing (1 pair, 8 combos = stage1 cost)")
    print("=" * 70)
    print("  Running full sweep on EUR_USD only (matches stage-1 in _sweep_one)...")

    t0 = time.time()
    sweep_id = run_sweep(
        sweep_name="profile_orb_ny",
        entry_fn=SWEEP_ORB_NY["entry_fn"],
        manager_fn=SWEEP_ORB_NY["manager_fn"],
        grid=SWEEP_ORB_NY["grid"],
        pairs=[pair],
        session=SWEEP_ORB_NY["session"],
        regime_mult=SWEEP_ORB_NY["regime_mult"],
        train_dfs=train_dfs, test_dfs=test_dfs,
        measured_spreads=measured_spreads,
        cost_mult=0.5, n_workers=2,
        use_dynamic_spread=True,
    )
    stage1_wall = time.time() - t0
    df = load_sweep_results(sweep_id)
    print(f"\n  Stage-1 sweep wall-time: {stage1_wall:.1f}s ({stage1_wall/60:.1f} min)")
    print(f"  Hypotheses produced: {len(df)}")
    if not df.empty:
        print(f"  Mean test_sharpe: {df['test_sharpe'].mean():.3f}, "
              f"best: {df['test_sharpe'].max():.3f}")
        print(f"  Mean test_n trades: {df['test_n'].mean():.1f}")

    print("\n" + "=" * 70)
    print(" Sweep-time projections")
    print("=" * 70)
    # Multi-pair stage-2 estimate: roughly linear with pair count
    print(f"  Stage-1 (1 pair, 8 combos):                   {stage1_wall:>6.0f}s")
    print(f"  Stage-2 (3 pairs, 8 combos, est.):            {stage1_wall*3:>6.0f}s  "
          f"({stage1_wall*3/60:.1f} min)")
    print(f"  Pre-screen reject (1 pair, 1 combo, ~18s):       18s")
    print()
    print("  Expected wall-time per round (2 strategies generated):")
    print(f"    Both fail pre-screen: ~{2*18 + 40}s  ({(2*18 + 40)/60:.1f} min)")
    print(f"    1 fails pre-screen, 1 fails stage-1: ~{18 + stage1_wall + 40:.0f}s "
          f"({(18 + stage1_wall + 40)/60:.1f} min)")
    print(f"    Both pass to full sweep: ~{2*(18 + 4*stage1_wall) + 40:.0f}s "
          f"({(2*(18 + 4*stage1_wall) + 40)/60:.1f} min)")
    print("    (×4 for stage1+stage2 = 4 pairs of work in sequence)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
