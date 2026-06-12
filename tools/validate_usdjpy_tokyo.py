"""
Validation pass on `tokyo_london_breakout` on USD_JPY.

Runs the strategy from the battery, then breaks down trades by month to see
whether the +1.79 OOS Sharpe is steady or concentrated in one window. Also
computes DSR and max_dd for the agent's promotion gauntlet.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import edge_engine as eng

from tools.test_strategy_battery import entry_tokyo_london_breakout


def _dsr(sharpe: float, n: int, skew: float = 0.0, kurt: float = 3.0,
          sr_threshold: float = 0.0) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado).
    Computes a one-sided CDF that the observed Sharpe exceeds the threshold
    after accounting for the variance of the Sharpe estimate under non-normal
    returns.  Returns a probability in [0, 1].
    """
    if n < 2 or sharpe == 0:
        return 0.0
    # Variance of the Sharpe estimator under non-normality (Mertens 2002)
    var_sr = (1 + 0.5 * sharpe ** 2 - skew * sharpe + (kurt - 3) / 4 * sharpe ** 2) / (n - 1)
    if var_sr <= 0:
        return 0.0
    z = (sharpe - sr_threshold) / math.sqrt(var_sr)
    # Normal CDF
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2)))


def run_and_breakdown():
    print("Loading cached data (need >=2 pairs for cross-pair feature builder)...")
    train_dfs, test_dfs, _ = eng.load_all_data(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"],
    )
    train_df, test_df = train_dfs["USD_JPY"], test_dfs["USD_JPY"]
    print(f"  train: {len(train_df):,} bars · test: {len(test_df):,} bars\n")

    manager_fn = eng.make_manager(exit_hour=13, use_breakeven=False)
    slot_class = "bat_tokyo_london_b"
    registry = [{
        "id":               "validate_usdjpy_tokyo",
        "family":           "battery",
        "slot_class":       slot_class,
        "pairs":            ["USD_JPY"],
        "session":          "ny",
        "allow_concurrent": False,
        "regime_mult": {"TRENDING": 1.0, "TRANSITIONING": 1.0, "RANGING": 1.0,
                        "VOLATILE": 0.5, "UNDEFINED": 0.5},
        "params":           {},
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_tokyo_london_breakout}

    out = {}
    for split, df in (("train", train_df), ("test", test_df)):
        print(f"=== Running {split} ({len(df):,} bars) ===")
        trades, _, _ = eng.run_backtest(
            {"USD_JPY": df}, None, None,
            registry, slot_managers, slot_entries,
            cost_mult=1.0,
        )
        if trades is None or trades.empty:
            print(f"  no trades on {split}")
            out[split] = trades
            continue
        trades = trades.copy()
        # Find the exit-time column (varies by engine version)
        ts_col = next((c for c in ('exit_ts', 'exit_time', 'close_ts', 'ts_close')
                       if c in trades.columns), None)
        if ts_col is None:
            print(f"  cols: {list(trades.columns)}")
            raise RuntimeError("no exit-time column found in trades DataFrame")
        trades[ts_col] = pd.to_datetime(trades[ts_col])
        trades['month'] = trades[ts_col].dt.to_period('M')
        out[split] = trades

    # ── Per-month breakdown ─────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PER-MONTH BREAKDOWN")
    print("=" * 80)
    for split in ("train", "test"):
        t = out.get(split)
        if t is None or t.empty:
            continue
        print(f"\n{split.upper()}:")
        print(f"{'month':10s} {'n':>4s} {'win_rate':>9s} {'pnl':>10s} "
              f"{'cum_pnl':>10s} {'sharpe':>8s}")
        grouped = t.groupby('month')
        cum = 0.0
        for month, g in grouped:
            n = len(g)
            wr = (g['pnl'] > 0).mean()
            pnl = float(g['pnl'].sum())
            cum += pnl
            # Per-month Sharpe (annualized assuming 21 trading days/month)
            if n > 1 and g['pnl'].std() > 0:
                sh = float(g['pnl'].mean() / g['pnl'].std() * math.sqrt(n))
            else:
                sh = 0.0
            print(f"{str(month):10s} {n:>4d} {wr*100:>7.1f}%  ${pnl:>+7,.0f}  "
                  f"${cum:>+7,.0f} {sh:>+8.2f}")
        # Aggregate stats
        n_total = len(t)
        pnl_total = float(t['pnl'].sum())
        wr_total = (t['pnl'] > 0).mean()
        stats = eng.calc_stats(t) or {}
        sh = float(stats.get("sharpe", 0) or 0)
        max_dd = float(stats.get("max_dd", 0) or 0)
        skew = float(t['pnl'].skew() or 0)
        kurt = float(t['pnl'].kurt() or 3) + 3.0  # pandas reports excess kurt
        dsr  = _dsr(sh, n_total, skew, kurt, sr_threshold=0.0)
        print(f"\n  Aggregate: n={n_total} sharpe={sh:+.3f} pnl=${pnl_total:+,.0f}"
              f" win_rate={wr_total*100:.1f}% max_dd=${max_dd:+,.0f}"
              f" skew={skew:+.2f} kurt={kurt:.1f} DSR={dsr:.3f}")

    # ── Steady or concentrated? ─────────────────────────────────────────────
    print()
    print("=" * 80)
    print("STEADINESS CHECK (test window)")
    print("=" * 80)
    test = out.get("test")
    if test is not None and not test.empty:
        monthly = test.groupby('month')['pnl'].sum()
        n_months = len(monthly)
        n_positive = (monthly > 0).sum()
        biggest_month_pnl = monthly.abs().max()
        total_pnl = monthly.sum()
        contribution_pct = (biggest_month_pnl / abs(total_pnl) * 100) if total_pnl != 0 else 0
        print(f"  Test months: {n_months}")
        print(f"  Positive months: {n_positive}/{n_months} "
              f"({n_positive/n_months*100:.0f}%)")
        print(f"  Largest month PnL: ${biggest_month_pnl:+,.0f}")
        print(f"  Largest month contribution to total PnL: {contribution_pct:.0f}%")
        if n_positive / n_months >= 0.6 and contribution_pct < 60:
            verdict = "STEADY — passes the steadiness check"
        elif n_positive / n_months < 0.5:
            verdict = "CONCENTRATED — most months are LOSING; the +1.79 Sharpe is driven by one or two months"
        else:
            verdict = "MIXED — borderline; verify by extending test window if possible"
        print(f"\n  VERDICT: {verdict}")
    return out


if __name__ == "__main__":
    run_and_breakdown()
