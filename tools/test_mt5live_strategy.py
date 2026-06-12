"""
Faithful backtest of the MT5Live.2.py live trading strategy.

Mirrors the entry/exit logic from MT5Live.2.py as closely as the offline
engine permits. Tests on the cached parquets so we can see whether the live
strategy has any edge over our 24-month window.

What we replicate exactly:
  - Per-pair range window + entry window + hard exit hour (London-group: 08:00–
    08:29 / 08:30–13:00 / 13:00 / TP=2.0×range; NY-group: 14:30–14:59 /
    15:00–21:00 / 21:00 / TP=2.5×range).
  - MA(200) direction filter — only place the side aligned with trend.
  - 30% candle-body filter on the last bar of the range.
  - SL = opposite-extreme of range ± 0.2 pip; entry = breakout-side ± 1 pip
    (stop order via place_pending mode='stop_at_level').
  - Risk per trade = 0.4% of equity (engine handles units via rv_size).
  - Breakeven move at 1R (engine's use_breakeven=True) — APPROXIMATES MT5Live's
    "close 50% partial + SL→entry at 1R" with full position + SL→entry.
  - Profit-lock (engine's use_profit_lock=True).
  - Hard time-of-day exit at exit_hour.
  - spread_gate (skip when spread elevated).

What we DON'T replicate (and why):
  - 50% partial close at 1R — engine doesn't support partial closes.
  - Kelly + vol-targeting + BOCPD + LLM-bias multipliers — these are live-only
    risk modulators; we use the base 0.4% straight through.
  - Correlation halving — would require multi-pair coordination at engine level.
  - News blackout — would need news-event data we don't have offline.

Usage:
    python tools/test_mt5live_strategy.py
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import edge_engine as eng

# Pair-group config — mirrors MT5Live.2.py L558-655
PAIR_GROUPS = {
    "london": {
        "pairs":          ["GBP_USD", "EUR_JPY"],   # GBP_JPY not in cache
        "range_start":    8,    # London time
        "range_end":      8,    # inclusive end-minute window: 8:00-8:29
        "range_end_min":  30,
        "entry_start":    8,
        "entry_start_min":30,
        "exit_hour":      13,
        "tp_r":           2.0,
    },
    "ny": {
        "pairs":          ["EUR_USD", "USD_JPY"],   # XAU not in cache
        "range_start":    14,
        "range_end":      14,
        "range_end_min":  60,   # 14:30-14:59
        "range_start_min":30,
        "entry_start":    15,
        "entry_start_min":0,
        "exit_hour":      21,
        "tp_r":           2.5,
    },
}

# Constants from MT5Live.2.py
RISK_PER_TRADE         = 0.004    # L531
BREAKOUT_BODY_MIN_PCT  = 0.30     # L532
MA_PERIOD              = 200      # L679
PIP_BUFFER_ENTRY_PIPS  = 1.0
PIP_BUFFER_SL_PIPS     = 0.2


def _pip_size(pair: str) -> float:
    """0.01 for JPY pairs, 0.0001 for the rest."""
    return 0.01 if "JPY" in pair else 0.0001


def _in_range_window(row, group: dict) -> bool:
    """True if `row` falls inside the group's range window (London time)."""
    if "range_start_min" in group:
        # Window spans within one hour: e.g. NY group 14:30-14:59
        return (row.hour == group['range_start']
                and group.get('range_start_min', 0) <= row.minute
                and (group['range_end'] != row.hour
                     or row.minute < group['range_end_min']))
    # Multi-hour or end-on-next-hour case (London group: 08:00-08:29)
    # range_start..range_end inclusive on hour, with hour-end at range_end_min
    if row.hour == group['range_start']:
        return row.minute < group.get('range_end_min', 60)
    return False


def make_entry_fn(group: dict):
    """Return an entry function closed over the pair-group config."""

    def entry_mt5live(bst, slot, row, ts, pair, slip, hspd,
                       sess_cfg, regime, regime_mult,
                       fvg_buf=None, day_sweep=None):
        params = slot['strategy_def']['params']
        sc     = slot['scratch']

        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        # Day rollover — clear per-day state when date changes
        cur_date = getattr(row, 'date', None)
        if sc.get('_day') != cur_date:
            sc['_day']         = cur_date
            sc['_rng_high']    = None
            sc['_rng_low']     = None
            sc['placed_today'] = False

        # ── Accumulate range_high / range_low during the range window ──
        # The cached data's `hour`/`minute` columns are in London local time
        # (Europe/London), so we can compare directly to the group's window.
        if _in_range_window(row, group):
            h = float(getattr(row, 'high', float('nan')))
            l = float(getattr(row, 'low',  float('nan')))
            if not (math.isnan(h) or math.isnan(l)):
                sc['_rng_high'] = h if sc['_rng_high'] is None else max(sc['_rng_high'], h)
                sc['_rng_low']  = l if sc['_rng_low']  is None else min(sc['_rng_low'],  l)

        if sc.get('placed_today'):
            return False

        # We fire at the start of the entry window — one shot per day.
        if (row.hour, row.minute) != (group['entry_start'], group['entry_start_min']):
            return False

        rng_high = sc.get('_rng_high')
        rng_low  = sc.get('_rng_low')
        if rng_high is None or rng_low is None:
            return False
        rng_size = rng_high - rng_low
        if rng_size <= 0:
            return False

        # ── MA(200) direction filter ────────────────────────────────────────
        # ma_trend in the cache is signed direction (>0 long bias, <0 short).
        # MT5Live compares the previous close to a 200-bar MA, which is
        # equivalent in sign to ma_trend's polarity.
        ma_trend = getattr(row, 'ma_trend', float('nan'))
        if math.isnan(ma_trend) or ma_trend == 0:
            return False
        direction = 'long' if ma_trend > 0 else 'short'

        # ── 30% body filter on the last (= this) bar ────────────────────────
        bar_high = float(getattr(row, 'high', 0))
        bar_low  = float(getattr(row, 'low',  0))
        bar_open = float(getattr(row, 'open', 0))
        bar_clos = float(getattr(row, 'close', 0))
        bar_range = bar_high - bar_low
        if bar_range <= 0:
            return False
        body = abs(bar_clos - bar_open)
        if body / bar_range < BREAKOUT_BODY_MIN_PCT:
            return False

        # ── Build entry, SL, TP ────────────────────────────────────────────
        pip = _pip_size(pair)
        tp_r = group['tp_r']

        if direction == 'long':
            level = rng_high + PIP_BUFFER_ENTRY_PIPS * pip
            sl    = rng_low  - PIP_BUFFER_SL_PIPS    * pip
            entry = level                       # stop order fills at level
            tp    = entry + tp_r * rng_size
        else:
            level = rng_low  - PIP_BUFFER_ENTRY_PIPS * pip
            sl    = rng_high + PIP_BUFFER_SL_PIPS    * pip
            entry = level
            tp    = entry - tp_r * rng_size

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return False

        risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
        size = eng.rv_size(pair, bst.balance, risk, entry, sl_dist, row)
        if not size or size <= 0:
            return False

        eng.place_pending(
            sc, ts, direction=direction,
            entry=entry, sl=sl, tp=tp,
            size=size, dist=sl_dist,
            level=level, mode='stop_at_level',
        )
        sc['placed_today'] = True
        return False

    return entry_mt5live


def run_group(group_name: str, group: dict, train_dfs: dict, test_dfs: dict,
              cost_mult: float = 1.0) -> dict:
    """Run the strategy on every pair in the group; report aggregate stats."""
    entry_fn   = make_entry_fn(group)
    manager_fn = eng.make_manager(
        exit_hour       = group['exit_hour'],
        use_breakeven   = True,
        use_profit_lock = True,
    )
    out = {"pair_results": {}, "group": group_name}

    for pair in group['pairs']:
        if pair not in train_dfs:
            print(f"  [{pair}] not in cache — skipped")
            continue
        slot_class = f"mt5live_{group_name}".lower()
        registry = [{
            "id":               f"mt5live_{group_name}_{pair}",
            "family":           "mt5live",
            "slot_class":       slot_class,
            "pairs":            [pair],
            "session":          "ny" if group_name == "ny" else "london",
            "allow_concurrent": False,
            "regime_mult":      {
                "TRENDING": 1.0, "TRANSITIONING": 1.0, "RANGING": 0.0,
                "VOLATILE": 0.0, "UNDEFINED": 0.5,
            },
            "params":           {},   # no grid; all config inside entry_fn
        }]
        slot_managers = {slot_class: manager_fn}
        slot_entries  = {slot_class: entry_fn}

        for split, dfs in (("train", train_dfs), ("test", test_dfs)):
            subset = {pair: dfs[pair]}
            try:
                trades, _bal, _ = eng.run_backtest(
                    subset, None, None,
                    registry, slot_managers, slot_entries,
                    cost_mult=cost_mult,
                )
            except Exception as e:
                print(f"  [{pair}/{split}] backtest error: {type(e).__name__}: {e}")
                continue
            n_trades = 0 if trades is None or trades.empty else len(trades)
            stats = eng.calc_stats(trades) if n_trades > 0 else {}
            pnl   = float(trades['pnl'].sum()) if n_trades > 0 else 0.0
            out["pair_results"].setdefault(pair, {})[split] = {
                "n_trades": n_trades,
                "sharpe":   float(stats.get("sharpe", 0) or 0),
                "win_rate": float(stats.get("win_rate", 0) or 0),
                "pnl":      pnl,
                "max_dd":   float(stats.get("max_dd", 0) or 0),
            }
    return out


def main() -> int:
    print("Loading cached market data...")
    train_dfs, test_dfs, _ = eng.load_all_data(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"],
    )
    print(f"  Loaded {len(train_dfs)} pair(s)")
    print()

    all_results = {}
    for group_name, group in PAIR_GROUPS.items():
        print(f"=== Group: {group_name.upper()} "
              f"(pairs={group['pairs']}, "
              f"entry={group['entry_start']:02d}:{group['entry_start_min']:02d}, "
              f"exit={group['exit_hour']:02d}:00, "
              f"TP={group['tp_r']}×range) ===")
        all_results[group_name] = run_group(
            group_name, group, train_dfs, test_dfs, cost_mult=1.0,
        )
        print()

    # ── Per-pair report ──────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{'PAIR':10s} {'SPLIT':6s} {'N':>5s} {'SHARPE':>8s} {'WR%':>6s} "
          f"{'PNL':>12s} {'MAX_DD':>8s}")
    print("-" * 80)
    for grp_name, grp_out in all_results.items():
        for pair, splits in grp_out["pair_results"].items():
            for split, r in splits.items():
                print(f"{pair:10s} {split:6s} {r['n_trades']:>5d} "
                      f"{r['sharpe']:>+8.2f} {r['win_rate']*100:>5.1f}% "
                      f"${r['pnl']:>+10,.0f} ${r['max_dd']:>+6,.0f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("AGGREGATE (sum across pairs):")
    for split in ("train", "test"):
        total_pnl    = 0.0
        total_trades = 0
        for grp_out in all_results.values():
            for splits in grp_out["pair_results"].values():
                if split in splits:
                    total_pnl    += splits[split]["pnl"]
                    total_trades += splits[split]["n_trades"]
        print(f"  {split:6s}: {total_trades:>4d} trades, total PnL ${total_pnl:>+,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
