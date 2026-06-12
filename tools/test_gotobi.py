"""
Gotobi Tokyo-fix strategy on USD/JPY (H1).

THESIS (participant flow): On "gotobi" days — dates divisible by 5
(5,10,15,20,25,30) plus month-end — Japanese corporates settle USD-denominated
import invoices and BUY USD into the 09:55 JST Tokyo fixing (= 00:55 UTC). This
is price-INSENSITIVE settlement demand: the corporate treasurer must convert by
the fix regardless of level, so the flow isn't arbitraged away. Documented in
FX microstructure literature (the "gotobi anomaly"). USD/JPY tends to drift UP
into the Tokyo fix on gotobi days. We go LONG USD/JPY ahead of the fix.

Why H1: the drift accumulates over the Tokyo morning; H1 bars capture the
00:00-01:00 UTC fix window cleanly. (Finer than H1 would capture it better but
the user asked for H1.)

DESIGN:
  - Trigger in the evening (UTC) when the NEXT Tokyo fix (tomorrow UTC) is a
    gotobi date AND a weekday. Fill LONG at the next bar's open.
  - Exit at a fixed UTC hour just after the 00:55 fix (manager exit_hour, here
    interpreted in UTC because we rebuild the H1 `hour` column as UTC).
  - SL/TP are wide ATR backstops; the time-exit is the primary exit.

VALIDATION built in:
  - Timing grid: several (trigger_hour, exit_hour) windows to find the best
    fix-capture window without cherry-picking after the fact (all reported).
  - PLACEBO: identical strategy on NON-gotobi evenings. If gotobi days have
    edge and non-gotobi days don't, the mechanism is real (not just "USD/JPY
    drifts up overnight").
  - Train/test split + per-month steadiness on the best variant.

Usage:
    python tools/test_gotobi.py
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
from tools.test_h1_poc import resample_h1   # reuse the validated resampler

PAIR = "USD_JPY"
GOTOBI_DAYS = {5, 10, 15, 20, 25, 30}


def _is_month_end(d) -> bool:
    return (d + timedelta(days=1)).month != d.month


def _is_gotobi(d) -> bool:
    return (d.day in GOTOBI_DAYS) or _is_month_end(d)


def _to_utc_h1(m1: pd.DataFrame) -> pd.DataFrame:
    """Resample to H1 and rebuild `hour` as UTC hour (timing-critical strategy)."""
    h1 = resample_h1(m1)
    h1['hour'] = h1['timestamp'].dt.tz_convert('UTC').dt.hour
    return h1


def make_gotobi_long(trigger_hour_utc: int, gotobi_only: bool = True):
    """Factory. gotobi_only=False → placebo (fire on NON-gotobi evenings)."""
    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        ts_utc = ts.tz_convert('UTC')
        if ts_utc.hour != trigger_hour_utc:
            return False

        # The upcoming Tokyo fix (00:55 UTC) is on tomorrow's UTC date
        fix_day = ts_utc + timedelta(days=1)
        if fix_day.weekday() >= 5:        # fix lands on a weekend → no bar, skip
            return False

        is_g = _is_gotobi(fix_day)
        if gotobi_only and not is_g:
            return False
        if (not gotobi_only) and is_g:     # placebo: only NON-gotobi evenings
            return False

        atr   = float(getattr(row, 'atr', 0) or 0)
        close = float(getattr(row, 'close', 0))
        if atr <= 0 or close <= 0:
            return False

        # LONG USD/JPY into the fix. Wide ATR backstops; time-exit is primary.
        entry = close
        sl    = close - 1.5 * atr
        tp    = close + 1.5 * atr
        sl_dist = abs(entry - sl)
        risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
        size = eng.rv_size(pair, bst.balance, risk, entry, sl_dist, row)
        if not size or size <= 0:
            return False
        eng.place_pending(sc, ts, direction='long', entry=entry, sl=sl, tp=tp,
                          size=size, dist=sl_dist, mode='market_next_open')
        return False
    return entry_fn


def run(name, entry_fn, exit_hour_utc, train_h1, test_h1, cost_mult=1.0):
    manager_fn = eng.make_manager(exit_hour=exit_hour_utc, use_breakeven=False)
    slot_class = f"gotobi_{name[:10]}".replace("-", "_").lower()
    registry = [{
        "id": f"gotobi_{name}", "family": "battery", "slot_class": slot_class,
        "pairs": [PAIR], "session": "ny", "allow_concurrent": False,
        "regime_mult": {"TRENDING": 1.0, "TRANSITIONING": 1.0, "RANGING": 1.0,
                        "VOLATILE": 1.0, "UNDEFINED": 1.0},
        "params": {},
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}
    out = {}
    for split, df in (("train", train_h1), ("test", test_h1)):
        trades, _, _ = eng.run_backtest(
            {PAIR: df}, None, None,
            registry, slot_managers, slot_entries, cost_mult=cost_mult)
        nt = 0 if trades is None or trades.empty else len(trades)
        stats = eng.calc_stats(trades) if nt > 0 else {}
        pnl = float(trades['pnl'].sum()) if nt > 0 else 0.0
        wr  = float((trades['pnl'] > 0).mean()) if nt > 0 else 0.0
        out[split] = {"n": nt, "sharpe": float(stats.get("sharpe", 0) or 0),
                      "pnl": pnl, "wr": wr, "trades": trades}
    return out


def main():
    print("Loading M1 cache...")
    train_dfs, test_dfs, _ = eng.load_all_data(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"])
    print(f"Resampling {PAIR} -> H1 (UTC hours)...")
    train_h1 = _to_utc_h1(train_dfs[PAIR])
    test_h1  = _to_utc_h1(test_dfs[PAIR])
    print(f"  train {len(train_h1):,} H1 bars, test {len(test_h1):,}\n")

    # ── Timing grid: (trigger_hour_utc, exit_hour_utc) ──────────────────────
    # trigger in the evening UTC; exit just after the 00:55 UTC fix.
    grid = [
        (23, 1),   # fill 00:00 UTC (Tokyo open), exit 01:00 (just past fix)
        (22, 1),   # fill 23:00, exit 01:00
        (23, 2),   # fill 00:00, exit 02:00 (hold 1h past fix)
        (21, 1),   # fill 22:00, exit 01:00 (wide pre-Tokyo)
        (0,  2),   # fill 01:00, exit 02:00 (post-fix continuation — control)
    ]

    print("=" * 92)
    print(f"{'VARIANT':22s} {'TR_N':>5s} {'TR_SH':>7s} {'TR_PNL':>9s} "
          f"{'TE_N':>5s} {'TE_SH':>7s} {'TE_PNL':>9s} {'CAND':>5s}")
    print("-" * 92)
    results = []
    for trig, ex in grid:
        name = f"gotobi_t{trig}_x{ex}"
        fn = make_gotobi_long(trig, gotobi_only=True)
        r = run(name, fn, ex, train_h1, test_h1)
        t, te = r['train'], r['test']
        is_cand = (te['sharpe'] >= 0.5 and te['n'] >= 30
                   and t['pnl'] > 0 and te['pnl'] > 0)
        print(f"{name:22s} {t['n']:>5d} {t['sharpe']:>+7.2f} ${t['pnl']:>+7,.0f} "
              f"{te['n']:>5d} {te['sharpe']:>+7.2f} ${te['pnl']:>+7,.0f} "
              f"{'✓' if is_cand else '—':>5s}")
        results.append((name, trig, ex, r, is_cand))
    print("=" * 92)

    # ── PLACEBO: best-by-train-Sharpe timing, but on NON-gotobi evenings ─────
    best = max(results, key=lambda x: x[3]['train']['sharpe'])
    _, btrig, bex, _, _ = best
    print(f"\nPLACEBO (non-gotobi evenings, same timing t{btrig}_x{bex}):")
    placebo_fn = make_gotobi_long(btrig, gotobi_only=False)
    pr = run(f"placebo_t{btrig}_x{bex}", placebo_fn, bex, train_h1, test_h1)
    pt, pte = pr['train'], pr['test']
    print(f"  train: n={pt['n']:>4d} sh={pt['sharpe']:+.2f} wr={pt['wr']*100:.1f}% ${pt['pnl']:+,.0f}")
    print(f"  test:  n={pte['n']:>4d} sh={pte['sharpe']:+.2f} wr={pte['wr']*100:.1f}% ${pte['pnl']:+,.0f}")
    gotobi_train_sh = best[3]['train']['sharpe']
    print(f"\n  Gotobi train Sharpe {gotobi_train_sh:+.2f} vs placebo {pt['sharpe']:+.2f} "
          f"→ {'MECHANISM LOOKS REAL' if gotobi_train_sh - pt['sharpe'] > 0.3 else 'NO CLEAR GOTOBI PREMIUM'}")

    # ── Per-month steadiness on the best variant ────────────────────────────
    print(f"\nPer-month (best variant t{btrig}_x{bex}, TEST window):")
    te_trades = best[3]['test']['trades']
    if te_trades is not None and not te_trades.empty:
        tt = te_trades.copy()
        ts_col = next((c for c in ('exit_ts','exit_time','close_ts') if c in tt.columns), None)
        if ts_col:
            tt[ts_col] = pd.to_datetime(tt[ts_col])
            tt['month'] = tt[ts_col].dt.to_period('M')
            for m, g in tt.groupby('month'):
                print(f"  {str(m)}: n={len(g):>3d} wr={(g['pnl']>0).mean()*100:>4.0f}% "
                      f"pnl=${g['pnl'].sum():>+7,.0f}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    cands = [r for r in results if r[4]]
    print("\n" + "=" * 92)
    if cands:
        print(f"PROMOTION CANDIDATE(S): {len(cands)}")
        for name, trig, ex, r, _ in cands:
            print(f"  ✓ {name}: test_sharpe=+{r['test']['sharpe']:.2f} "
                  f"n={r['test']['n']} pnl=${r['test']['pnl']:+,.0f}")
    else:
        print("No gotobi timing variant cleared the bar (test_sharpe ≥ 0.5, "
              "n ≥ 30, both windows +).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
