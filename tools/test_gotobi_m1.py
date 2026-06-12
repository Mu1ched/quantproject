"""
Gotobi Tokyo-fix strategy on USD/JPY — M1 (correct granularity).

The H1 test washed out the effect: an H1 bar forces holding the post-fix
reversal. The gotobi drift lives in the tight 00:00->00:55 UTC window (Tokyo
open -> 09:55 JST fix). On M1 we can hold exactly that window.

Cost logic: M1 is too expensive for MARGINAL edges, but the gotobi drift (if
real) is a several-pip directional move in <1h, so the ~0.7-pip USD/JPY cost
is a small fraction of it. M1 is the RIGHT tool for a fix-window strategy.

DESIGN:
  - On a gotobi day (date divisible by 5, or month-end), enter LONG USD/JPY at
    the Tokyo open (00:00 UTC bar -> fill 00:01) and exit at the fix (~00:55).
  - Sweep the exit minute to map the drift/reversal profile.
  - PLACEBO on non-gotobi days, same timing, to confirm the premium is
    gotobi-specific (not just "USD/JPY drifts up in the Tokyo morning").
  - Train/test split + per-trade stats.

Usage:
    python tools/test_gotobi_m1.py
"""
from __future__ import annotations

import sys
import math
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import edge_engine as eng

PAIR = "USD_JPY"
GOTOBI_DAYS = {5, 10, 15, 20, 25, 30}


def _is_month_end(d) -> bool:
    return (d + timedelta(days=1)).month != d.month


def _is_gotobi(d) -> bool:
    return (d.day in GOTOBI_DAYS) or _is_month_end(d)


def _utc_hours(m1: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the M1 df with hour/minute rebuilt as UTC (timing-
    critical). All other columns (spread_adj w/ override, atr, ma_trend...)
    are preserved."""
    df = m1.copy()
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/London')
    utc = df['timestamp'].dt.tz_convert('UTC')
    df['hour']   = utc.dt.hour.astype('int32')
    df['minute'] = utc.dt.minute.astype('int32')
    return df


def make_gotobi_long(trigger_hour=0, trigger_min=0, gotobi_only=True):
    """Fire LONG at the Tokyo-open bar on gotobi (or non-gotobi for placebo)."""
    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        ts_utc = ts.tz_convert('UTC')
        if ts_utc.hour != trigger_hour or ts_utc.minute != trigger_min:
            return False
        if ts_utc.weekday() >= 5:
            return False

        is_g = _is_gotobi(ts_utc)
        if gotobi_only and not is_g:
            return False
        if (not gotobi_only) and is_g:
            return False

        atr   = float(getattr(row, 'atr', 0) or 0)
        close = float(getattr(row, 'close', 0))
        if atr <= 0 or close <= 0:
            return False

        # LONG into the fix. Wide ATR backstops; the time-exit is primary.
        entry = close
        sl    = close - 2.0 * atr
        tp    = close + 3.0 * atr
        sl_dist = abs(entry - sl)
        risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
        size = eng.rv_size(pair, bst.balance, risk, entry, sl_dist, row)
        if not size or size <= 0:
            return False
        eng.place_pending(sc, ts, direction='long', entry=entry, sl=sl, tp=tp,
                          size=size, dist=sl_dist, mode='market_next_open')
        return False
    return entry_fn


def run(name, entry_fn, exit_h, exit_m, train_df, test_df, cost_mult=1.0):
    manager_fn = eng.make_manager(exit_hour=exit_h, exit_min=exit_m,
                                  use_breakeven=False)
    slot_class = f"gm1_{name[:10]}".replace("-", "_").lower()
    registry = [{
        "id": f"gm1_{name}", "family": "battery", "slot_class": slot_class,
        "pairs": [PAIR], "session": "ny", "allow_concurrent": False,
        "regime_mult": {"TRENDING": 1.0, "TRANSITIONING": 1.0, "RANGING": 1.0,
                        "VOLATILE": 1.0, "UNDEFINED": 1.0},
        "params": {},
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}
    out = {}
    for split, df in (("train", train_df), ("test", test_df)):
        trades, _, _ = eng.run_backtest(
            {PAIR: df}, None, None,
            registry, slot_managers, slot_entries, cost_mult=cost_mult)
        nt = 0 if trades is None or trades.empty else len(trades)
        stats = eng.calc_stats(trades) if nt > 0 else {}
        pnl = float(trades['pnl'].sum()) if nt > 0 else 0.0
        wr  = float((trades['pnl'] > 0).mean()) if nt > 0 else 0.0
        avg = float(trades['pnl'].mean()) if nt > 0 else 0.0
        out[split] = {"n": nt, "sharpe": float(stats.get("sharpe", 0) or 0),
                      "pnl": pnl, "wr": wr, "avg": avg, "trades": trades}
    return out


def main():
    print("Loading M1 cache...")
    train_dfs, test_dfs, _ = eng.load_all_data(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"])
    train = _utc_hours(train_dfs[PAIR])
    test  = _utc_hours(test_dfs[PAIR])
    print(f"  {PAIR}: train {len(train):,} M1 bars, test {len(test):,}\n")

    # Exit-minute sweep — map the drift/reversal profile.
    # (exit_hour, exit_min): 00:50, 00:55, 01:00, 01:30
    exit_grid = [(0, 50), (0, 55), (1, 0), (1, 30)]

    print("=" * 96)
    print(f"{'VARIANT (enter 00:00 UTC)':28s} {'TR_N':>5s} {'TR_SH':>7s} "
          f"{'TR_PNL':>9s} {'TR_AVG':>7s} {'TE_N':>5s} {'TE_SH':>7s} "
          f"{'TE_PNL':>9s} {'CAND':>5s}")
    print("-" * 96)
    results = []
    for eh, em in exit_grid:
        name = f"exit_{eh:02d}{em:02d}"
        fn = make_gotobi_long(0, 0, gotobi_only=True)
        r = run(name, fn, eh, em, train, test)
        t, te = r['train'], r['test']
        is_cand = (te['sharpe'] >= 0.5 and te['n'] >= 30
                   and t['pnl'] > 0 and te['pnl'] > 0)
        print(f"{name:28s} {t['n']:>5d} {t['sharpe']:>+7.2f} ${t['pnl']:>+7,.0f} "
              f"{t['avg']:>+7.1f} {te['n']:>5d} {te['sharpe']:>+7.2f} "
              f"${te['pnl']:>+7,.0f} {'✓' if is_cand else '—':>5s}")
        results.append((name, eh, em, r, is_cand))
    print("=" * 96)

    # ── PLACEBO at best-train-Sharpe exit ────────────────────────────────────
    best = max(results, key=lambda x: x[3]['train']['sharpe'])
    _, beh, bem, _, _ = best
    print(f"\nPLACEBO (NON-gotobi days, enter 00:00 exit {beh:02d}:{bem:02d}):")
    pr = run("placebo", make_gotobi_long(0, 0, gotobi_only=False),
             beh, bem, train, test)
    pt, pte = pr['train'], pr['test']
    print(f"  train: n={pt['n']:>4d} sh={pt['sharpe']:+.2f} wr={pt['wr']*100:.1f}% "
          f"avg=${pt['avg']:+.1f} ${pt['pnl']:+,.0f}")
    print(f"  test:  n={pte['n']:>4d} sh={pte['sharpe']:+.2f} wr={pte['wr']*100:.1f}% "
          f"avg=${pte['avg']:+.1f} ${pte['pnl']:+,.0f}")
    gsh = best[3]['train']['sharpe']
    gavg = best[3]['train']['avg']
    print(f"\n  Gotobi train: Sharpe {gsh:+.2f}, avg ${gavg:+.1f}/trade")
    print(f"  Placebo train: Sharpe {pt['sharpe']:+.2f}, avg ${pt['avg']:+.1f}/trade")
    premium = gavg - pt['avg']
    print(f"  Gotobi premium (avg PnL/trade): ${premium:+.1f}  "
          f"→ {'REAL EDGE SIGNAL' if (gsh - pt['sharpe'] > 0.3 and premium > 0) else 'NO CLEAR PREMIUM'}")

    # ── Per-month on best variant (test) ─────────────────────────────────────
    print(f"\nPer-month (best exit {beh:02d}:{bem:02d}, TEST):")
    te_tr = best[3]['test']['trades']
    if te_tr is not None and not te_tr.empty:
        tt = te_tr.copy()
        tc = next((c for c in ('exit_ts','exit_time','close_ts') if c in tt.columns), None)
        if tc:
            tt[tc] = pd.to_datetime(tt[tc])
            tt['m'] = tt[tc].dt.to_period('M')
            for m, g in tt.groupby('m'):
                print(f"  {str(m)}: n={len(g):>3d} wr={(g['pnl']>0).mean()*100:>4.0f}% "
                      f"${g['pnl'].sum():>+7,.0f}")

    print("\n" + "=" * 96)
    cands = [r for r in results if r[4]]
    if cands:
        print(f"PROMOTION CANDIDATE(S): {len(cands)}")
        for name, eh, em, r, _ in cands:
            print(f"  ✓ {name}: test_sharpe=+{r['test']['sharpe']:.2f} "
                  f"n={r['test']['n']} pnl=${r['test']['pnl']:+,.0f}")
    else:
        print("No gotobi M1 exit-variant cleared the bar "
              "(test_sharpe ≥ 0.5, n ≥ 30, both windows +).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
