"""
Combo strategies — pair a structural pattern with a discovery-validated
statistical filter. Discovery found 8 bar-level edges (NY hurst > p90,
NY ma_dist < p10, London ma_dist > p90, etc.) that died at trade level
in isolation. Hypothesis: using them as REGIME FILTERS inside a known
structural pattern compounds two real signals into one survivable strategy.

Three combos:

  1. tokyo_london_orb_hurstfilter  — V2's tokyo_london_breakout on USD/JPY
     PLUS: only fire if hurst > p90 at London open (Tokyo persistence regime).
     Goal: improve the +1.79 OOS Sharpe by filtering out non-trending Tokyo
     sessions; reduce concentration risk on USD/JPY.

  2. ny_orb_madist_long           — NY 14:30 London opening-range breakout
     on EUR_USD, ONE-sided LONG only when ma_dist < p10 (deeply below MA =
     mean-reversion setup matching discovery's NY long-drift signal).

  3. london_orb_madist_short      — London 08:00 opening-range breakout on
     GBP_USD, ONE-sided SHORT only when ma_dist > p90 (deeply above MA =
     mean-reversion setup matching discovery's London short-drift signal).

Percentiles computed from TRAIN data only per pair (no look-ahead).
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

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"]


def _pip(pair):
    return 0.01 if "JPY" in pair else 0.0001


def _utc_hour(row) -> int:
    ts = getattr(row, 'timestamp', None)
    if ts is None:
        return int(row.hour)
    try:
        return int(ts.tz_convert('UTC').hour)
    except (AttributeError, TypeError):
        return int(getattr(row, 'hour', 0))


def _bar_idx(sc):
    sc['_bar_n'] = sc.get('_bar_n', 0) + 1
    return sc['_bar_n']


def _day_rollover(sc, row):
    cur = getattr(row, 'date', None)
    if sc.get('_day') != cur:
        sc['_day'] = cur
        return True
    return False


def _size_for(bst, regime_mult, pair, balance, entry, sl_dist, row):
    risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
    return eng.rv_size(pair, balance, risk, entry, sl_dist, row)


# ─────────────────────────────────────────────────────────────────────────────
# Combo 1: USD/JPY tokyo_london_breakout + hurst > p90 filter
# ─────────────────────────────────────────────────────────────────────────────
def make_combo_tokyo_h1trend(h1_strength_p75: float):
    """Factory: tokyo_london_breakout that only fires when |h1_trend_strength|
    >= p75. (USD/JPY doesn't have `hurst` in the cache — `h1_trend_strength`
    captures the equivalent "strong trend regime" filter using the precomputed
    60-EMA vs 240-EMA cross magnitude scaled by ATR.)
    """
    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        _bar_idx(sc)
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        if _day_rollover(sc, row):
            sc['tokyo_hi'] = None
            sc['tokyo_lo'] = None
            sc['placed']   = False

        uh = _utc_hour(row)
        h = float(getattr(row, 'high', float('nan')))
        l = float(getattr(row, 'low',  float('nan')))
        if not (math.isnan(h) or math.isnan(l)) and (uh >= 22 or uh < 7):
            sc['tokyo_hi'] = h if sc['tokyo_hi'] is None else max(sc['tokyo_hi'], h)
            sc['tokyo_lo'] = l if sc['tokyo_lo'] is None else min(sc['tokyo_lo'], l)

        if sc.get('placed') or (uh, row.minute) != (7, 0):
            return False
        if sc['tokyo_hi'] is None or sc['tokyo_lo'] is None:
            return False

        # ── NEW: strong H1 trend regime filter ──
        h1ts = getattr(row, 'h1_trend_strength', float('nan'))
        if math.isnan(h1ts) or abs(h1ts) < h1_strength_p75:
            return False

        rng = sc['tokyo_hi'] - sc['tokyo_lo']
        if rng <= 0:
            return False

        ma = getattr(row, 'ma_trend', float('nan'))
        if math.isnan(ma) or ma == 0:
            return False

        pip = _pip(pair)
        direction = 'long' if ma > 0 else 'short'
        if direction == 'long':
            level = sc['tokyo_hi'] + pip
            sl    = sc['tokyo_lo'] - pip
            tp    = level + 2.0 * rng
        else:
            level = sc['tokyo_lo'] - pip
            sl    = sc['tokyo_hi'] + pip
            tp    = level - 2.0 * rng

        sl_dist = abs(level - sl)
        if sl_dist <= 0:
            return False
        size = _size_for(bst, regime_mult, pair, bst.balance, level, sl_dist, row)
        if not size or size <= 0:
            return False

        eng.place_pending(sc, ts, direction=direction,
                           entry=level, sl=sl, tp=tp, size=size, dist=sl_dist,
                           level=level, mode='stop_at_level')
        sc['placed'] = True
        return False
    return entry_fn


# ─────────────────────────────────────────────────────────────────────────────
# Combo 2: NY 14:30 London-time opening-range breakout LONG when ma_dist < p10
# ─────────────────────────────────────────────────────────────────────────────
def make_combo_ny_madist_long(madist_p10: float):
    """Factory: NY ORB long-only, only fires when ma_dist < p10."""
    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        _bar_idx(sc)
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        if _day_rollover(sc, row):
            sc['ny_hi'] = None
            sc['ny_lo'] = None
            sc['placed'] = False

        # Range = 14:30-14:59 London local
        if row.hour == 14 and 30 <= row.minute <= 59:
            h = float(getattr(row, 'high', float('nan')))
            l = float(getattr(row, 'low',  float('nan')))
            if not (math.isnan(h) or math.isnan(l)):
                sc['ny_hi'] = h if sc['ny_hi'] is None else max(sc['ny_hi'], h)
                sc['ny_lo'] = l if sc['ny_lo'] is None else min(sc['ny_lo'], l)

        if sc.get('placed') or (row.hour, row.minute) != (15, 0):
            return False
        if sc['ny_hi'] is None or sc['ny_lo'] is None:
            return False

        # ── NEW: ma_dist < p10 filter (deeply below MA = long-drift setup) ──
        madist = getattr(row, 'ma_dist', float('nan'))
        if math.isnan(madist) or madist >= madist_p10:
            return False

        rng = sc['ny_hi'] - sc['ny_lo']
        if rng <= 0:
            return False

        # One-sided LONG entry at upper extreme
        pip = _pip(pair)
        level = sc['ny_hi'] + pip
        sl    = sc['ny_lo'] - pip
        tp    = level + 2.5 * rng

        sl_dist = abs(level - sl)
        if sl_dist <= 0:
            return False
        size = _size_for(bst, regime_mult, pair, bst.balance, level, sl_dist, row)
        if not size or size <= 0:
            return False

        eng.place_pending(sc, ts, direction='long',
                           entry=level, sl=sl, tp=tp, size=size, dist=sl_dist,
                           level=level, mode='stop_at_level')
        sc['placed'] = True
        return False
    return entry_fn


# ─────────────────────────────────────────────────────────────────────────────
# Combo 3: London 08:00 opening-range breakout SHORT when ma_dist > p90
# ─────────────────────────────────────────────────────────────────────────────
def make_combo_london_madist_short(madist_p90: float):
    """Factory: London ORB short-only, only fires when ma_dist > p90."""
    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        _bar_idx(sc)
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        if _day_rollover(sc, row):
            sc['ln_hi'] = None
            sc['ln_lo'] = None
            sc['placed'] = False

        # Range = 08:00-08:29 London local
        if row.hour == 8 and row.minute < 30:
            h = float(getattr(row, 'high', float('nan')))
            l = float(getattr(row, 'low',  float('nan')))
            if not (math.isnan(h) or math.isnan(l)):
                sc['ln_hi'] = h if sc['ln_hi'] is None else max(sc['ln_hi'], h)
                sc['ln_lo'] = l if sc['ln_lo'] is None else min(sc['ln_lo'], l)

        if sc.get('placed') or (row.hour, row.minute) != (8, 30):
            return False
        if sc['ln_hi'] is None or sc['ln_lo'] is None:
            return False

        # ── NEW: ma_dist > p90 filter (deeply above MA = short-drift setup) ──
        madist = getattr(row, 'ma_dist', float('nan'))
        if math.isnan(madist) or madist <= madist_p90:
            return False

        rng = sc['ln_hi'] - sc['ln_lo']
        if rng <= 0:
            return False

        pip = _pip(pair)
        level = sc['ln_lo'] - pip
        sl    = sc['ln_hi'] + pip
        tp    = level - 2.0 * rng

        sl_dist = abs(level - sl)
        if sl_dist <= 0:
            return False
        size = _size_for(bst, regime_mult, pair, bst.balance, level, sl_dist, row)
        if not size or size <= 0:
            return False

        eng.place_pending(sc, ts, direction='short',
                           entry=level, sl=sl, tp=tp, size=size, dist=sl_dist,
                           level=level, mode='stop_at_level')
        sc['placed'] = True
        return False
    return entry_fn


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def percentile(df: pd.DataFrame, col: str, p: float) -> float:
    s = df[col].dropna()
    if s.empty:
        return float('nan')
    return float(np.percentile(s, p))


def run_strategy(name: str, entry_fn, exit_hour: int, pair: str,
                  train_df, test_df, cost_mult: float = 1.0) -> dict:
    manager_fn = eng.make_manager(exit_hour=exit_hour, use_breakeven=False)
    slot_class = f"combo_{name[:10]}".replace("-", "_").lower()
    registry = [{
        "id": f"combo_{name}_{pair}", "family": "battery",
        "slot_class": slot_class,
        "pairs": [pair], "session": "ny",
        "allow_concurrent": False,
        "regime_mult": {"TRENDING": 1.0, "TRANSITIONING": 1.0, "RANGING": 1.0,
                        "VOLATILE": 0.5, "UNDEFINED": 0.5},
        "params": {},
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}

    out = {}
    for split, df in (("train", train_df), ("test", test_df)):
        trades, _, _ = eng.run_backtest(
            {pair: df}, None, None,
            registry, slot_managers, slot_entries,
            cost_mult=cost_mult,
        )
        n = 0 if trades is None or trades.empty else len(trades)
        stats = eng.calc_stats(trades) if n > 0 else {}
        pnl   = float(trades['pnl'].sum()) if n > 0 else 0.0
        wr    = float((trades['pnl'] > 0).mean()) if n > 0 else 0.0
        out[split] = {"n": n, "sharpe": float(stats.get("sharpe", 0) or 0),
                      "pnl": pnl, "wr": wr,
                      "max_dd": float(stats.get("max_dd", 0) or 0)}
    return out


def main():
    print("Loading cached data...")
    train_dfs, test_dfs, _ = eng.load_all_data(pairs=PAIRS)
    print()

    combos = []

    # Combo 1: USD/JPY tokyo_london + |h1_trend_strength| > p75 (USD/JPY has no hurst)
    train_h1ts_abs = train_dfs["USD_JPY"]["h1_trend_strength"].dropna().abs()
    p75_usdjpy_h1ts = float(np.percentile(train_h1ts_abs, 75)) if not train_h1ts_abs.empty else float('nan')
    print(f"USD/JPY train |h1_trend_strength| p75 = {p75_usdjpy_h1ts:.4f}")
    combos.append(("tokyo_h1trend_p75",
                   make_combo_tokyo_h1trend(p75_usdjpy_h1ts), 13, "USD_JPY"))

    # Combo 2: EUR/USD NY ORB long + ma_dist < p10
    p10_eurusd_madist = percentile(train_dfs["EUR_USD"], "ma_dist", 10)
    print(f"EUR/USD train ma_dist p10 = {p10_eurusd_madist:.6f}")
    combos.append(("ny_madist_lt_p10_long",
                   make_combo_ny_madist_long(p10_eurusd_madist), 21, "EUR_USD"))

    # Combo 3: GBP/USD London ORB short + ma_dist > p90
    p90_gbpusd_madist = percentile(train_dfs["GBP_USD"], "ma_dist", 90)
    print(f"GBP/USD train ma_dist p90 = {p90_gbpusd_madist:.6f}")
    combos.append(("london_madist_gt_p90_short",
                   make_combo_london_madist_short(p90_gbpusd_madist), 13, "GBP_USD"))

    print()
    print("=" * 80)
    rows = []
    for name, fn, exit_h, pair in combos:
        print(f"\n=== {name} ({pair}, exit={exit_h}:00) ===")
        r = run_strategy(name, fn, exit_h, pair, train_dfs[pair], test_dfs[pair])
        rows.append((name, pair, r))
        t, te = r["train"], r["test"]
        print(f"  train: n={t['n']:>4d} sh={t['sharpe']:+7.2f} wr={t['wr']*100:>4.1f}% "
              f"pnl=${t['pnl']:+8,.0f}")
        print(f"  test:  n={te['n']:>4d} sh={te['sharpe']:+7.2f} wr={te['wr']*100:>4.1f}% "
              f"pnl=${te['pnl']:+8,.0f}")
        is_cand = (te['sharpe'] >= 0.5 and te['n'] >= 30
                   and t['pnl'] > 0 and te['pnl'] > 0)
        print(f"  CANDIDATE: {'✓ YES' if is_cand else '— no'}")

    print()
    print("=" * 80)
    candidates = [(n, p, r) for n, p, r in rows
                  if r['test']['sharpe'] >= 0.5 and r['test']['n'] >= 30
                  and r['train']['pnl'] > 0 and r['test']['pnl'] > 0]
    if candidates:
        print(f"PROMOTION CANDIDATES ({len(candidates)}):")
        for n, p, r in candidates:
            print(f"  ✓ {n} on {p}: test_sharpe=+{r['test']['sharpe']:.2f} "
                  f"n={r['test']['n']}")
    else:
        print("No combo passed the bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
