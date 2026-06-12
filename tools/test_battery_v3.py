"""
Strategy Battery V3 — 5 more strategies on angles V2 didn't cover.

Each targets a specific failure mode we've now seen repeatedly:
  - 1:1 R:R loses to costs → use asymmetric R:R (TP=2-3x SL)
  - Triggers fire too often → use rare conditions
  - Filters compound multiplicatively too narrow → use OR-of-pairs

Strategies:

  1. asian_range_fade_london    — V2's USD/JPY breakout WORKS; for the other
     3 pairs it didn't. Hypothesis: those pairs fade the Asian breakout
     (London open is the fakeout direction, not the continuation).

  2. cumdelta_divergence_revert — uses cumulative_delta (order-flow column).
     When price went up but cum_delta went down over 30 bars (or vice versa),
     the move is unsustainable. Enter against price.

  3. tight_spread_tick_scalp    — when spread is in lowest decile AND
     tick_imbalance is in top/bottom decile, enter in tick direction.
     Bet that liquid + strong-flow moments produce continuation.

  4. five_bar_momentum_cont     — 5 consecutive same-direction body-dominant
     bars → enter on bar 6 same direction. Asymmetric R:R (TP=3 ATR, SL=1 ATR)
     to overcome the cost overhead that killed 1:1 strategies.

  5. vol_expansion_breakout     — yz_vol_ratio rises from <0.8 to >1.5 within
     30 bars + price breaks 60-bar high/low → enter in breakout direction.
     Catches the START of an expansion move (not the late part).
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
# S1: Asian Range Fade at London Open
# ─────────────────────────────────────────────────────────────────────────────
def entry_asian_range_fade_london(bst, slot, row, ts, pair, slip, hspd,
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

    rng = sc['tokyo_hi'] - sc['tokyo_lo']
    if rng <= 0:
        return False

    # FADE — opposite direction to MT5Live's MA200-following breakout
    ma = getattr(row, 'ma_trend', float('nan'))
    if math.isnan(ma) or ma == 0:
        return False

    pip = _pip(pair)
    # If MA says long, FADE by going short on a breakout BELOW Asian low
    direction = 'short' if ma > 0 else 'long'
    if direction == 'short':
        level = sc['tokyo_lo'] - pip
        sl    = sc['tokyo_hi'] + pip
        tp    = level - 2.0 * rng
    else:
        level = sc['tokyo_hi'] + pip
        sl    = sc['tokyo_lo'] - pip
        tp    = level + 2.0 * rng

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


# ─────────────────────────────────────────────────────────────────────────────
# S2: Cumulative Delta Divergence Reversal
# ─────────────────────────────────────────────────────────────────────────────
# Track last 30 bars of price and cumulative_delta. When they diverge in
# direction (price up, delta down, or vice versa), enter against price.
def entry_cumdelta_divergence_revert(bst, slot, row, ts, pair, slip, hspd,
                                       sess_cfg, regime, regime_mult,
                                       fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    cur_n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    if cur_n - sc.get('last_trade_bar', -10_000) < 60:
        return False

    close = float(getattr(row, 'close', 0))
    cd    = getattr(row, 'cumulative_delta', float('nan'))
    atr   = float(getattr(row, 'atr', 0) or 0)
    if close <= 0 or math.isnan(cd) or atr <= 0:
        return False

    # Ring buffer of last 30 (close, cum_delta) pairs
    ring = sc.setdefault('cd_ring', [])
    ring.append((close, float(cd)))
    if len(ring) > 30:
        ring.pop(0)
    if len(ring) < 30:
        return False

    old_close, old_cd = ring[0]
    cur_close, cur_cd = ring[-1]
    price_change = cur_close - old_close
    delta_change = cur_cd - old_cd

    if abs(price_change) < 0.5 * atr:
        return False    # too small a move to act on
    # Divergence: sign(price_change) != sign(delta_change)
    if (price_change > 0) == (delta_change > 0):
        return False
    # Also require delta_change to be meaningful (not just noise)
    if abs(delta_change) < 1e-6:
        return False

    direction = 'short' if price_change > 0 else 'long'
    if direction == 'long':
        entry = close
        sl    = close - 1.0 * atr
        tp    = close + 2.0 * atr
    else:
        entry = close
        sl    = close + 1.0 * atr
        tp    = close - 2.0 * atr
    sl_dist = abs(entry - sl)
    size = _size_for(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
    if not size or size <= 0:
        return False

    eng.place_pending(sc, ts, direction=direction,
                       entry=entry, sl=sl, tp=tp,
                       size=size, dist=sl_dist,
                       mode='market_next_open')
    sc['last_trade_bar'] = cur_n
    return False


# ─────────────────────────────────────────────────────────────────────────────
# S3: Tight-Spread Tick-Imbalance Scalp
# ─────────────────────────────────────────────────────────────────────────────
# Factory pattern: takes precomputed percentile thresholds from train data.
def make_tight_spread_tick_scalp(spread_p10: float, tick_p90: float):
    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        cur_n = _bar_idx(sc)
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        if cur_n - sc.get('last_trade_bar', -10_000) < 30:
            return False

        spread = float(getattr(row, 'spread_mean', float('nan')))
        tick   = float(getattr(row, 'tick_imbalance', float('nan')))
        atr    = float(getattr(row, 'atr', 0) or 0)
        close  = float(getattr(row, 'close', 0))
        if math.isnan(spread) or math.isnan(tick) or atr <= 0 or close <= 0:
            return False

        if spread > spread_p10:
            return False    # spread not in tightest decile
        if abs(tick) < tick_p90:
            return False    # tick imbalance not at extreme

        direction = 'long' if tick > 0 else 'short'
        if direction == 'long':
            entry = close
            sl    = close - 0.8 * atr
            tp    = close + 1.6 * atr
        else:
            entry = close
            sl    = close + 0.8 * atr
            tp    = close - 1.6 * atr
        sl_dist = abs(entry - sl)
        size = _size_for(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
        if not size or size <= 0:
            return False

        eng.place_pending(sc, ts, direction=direction,
                           entry=entry, sl=sl, tp=tp,
                           size=size, dist=sl_dist,
                           mode='market_next_open')
        sc['last_trade_bar'] = cur_n
        return False
    return entry_fn


# ─────────────────────────────────────────────────────────────────────────────
# S4: 5-Bar Momentum Continuation (asymmetric R:R)
# ─────────────────────────────────────────────────────────────────────────────
def entry_five_bar_momentum_cont(bst, slot, row, ts, pair, slip, hspd,
                                   sess_cfg, regime, regime_mult,
                                   fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    cur_n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    if cur_n - sc.get('last_trade_bar', -10_000) < 60:
        return False

    open_ = float(getattr(row, 'open', 0))
    close = float(getattr(row, 'close', 0))
    high  = float(getattr(row, 'high', 0))
    low   = float(getattr(row, 'low', 0))
    atr   = float(getattr(row, 'atr', 0) or 0)
    if close <= 0 or open_ <= 0 or atr <= 0:
        return False

    body  = close - open_
    body_abs = abs(body)
    bar_range = high - low

    # Need bar to be body-dominant (body >= 50% of range)
    body_dominant = (bar_range > 0 and body_abs / bar_range >= 0.5
                     and body_abs >= 0.4 * atr)

    sign = 1 if body > 0 else (-1 if body < 0 else 0)
    ring = sc.setdefault('mom_ring', [])
    ring.append((sign, body_dominant))
    if len(ring) > 5:
        ring.pop(0)
    if len(ring) < 5:
        return False

    # Need all 5 same-sign AND all body-dominant
    signs = [s for s, _ in ring]
    doms  = [d for _, d in ring]
    if 0 in signs or len(set(signs)) > 1 or not all(doms):
        return False

    direction = 'long' if signs[0] > 0 else 'short'
    # Asymmetric R:R — TP=3 ATR, SL=1 ATR, 3:1
    if direction == 'long':
        entry = close
        sl    = close - 1.0 * atr
        tp    = close + 3.0 * atr
    else:
        entry = close
        sl    = close + 1.0 * atr
        tp    = close - 3.0 * atr
    sl_dist = abs(entry - sl)
    size = _size_for(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
    if not size or size <= 0:
        return False

    eng.place_pending(sc, ts, direction=direction,
                       entry=entry, sl=sl, tp=tp,
                       size=size, dist=sl_dist,
                       mode='market_next_open')
    sc['last_trade_bar'] = cur_n
    sc['mom_ring'] = []
    return False


# ─────────────────────────────────────────────────────────────────────────────
# S5: Volatility Expansion Breakout
# ─────────────────────────────────────────────────────────────────────────────
def entry_vol_expansion_breakout(bst, slot, row, ts, pair, slip, hspd,
                                   sess_cfg, regime, regime_mult,
                                   fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    cur_n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    if cur_n - sc.get('last_trade_bar', -10_000) < 60:
        return False

    yzr   = float(getattr(row, 'yz_vol_ratio', float('nan')))
    atr   = float(getattr(row, 'atr', 0) or 0)
    close = float(getattr(row, 'close', 0))
    high  = float(getattr(row, 'high', 0))
    low   = float(getattr(row, 'low', 0))
    if math.isnan(yzr) or atr <= 0 or close <= 0:
        return False

    # Ring of last 30 yz_vol_ratios
    yzring = sc.setdefault('yz_ring', [])
    yzring.append(yzr)
    if len(yzring) > 30:
        yzring.pop(0)
    if len(yzring) < 30:
        return False

    # Need: was compressed (min < 0.8) AND now expanded (current >= 1.5)
    if min(yzring) >= 0.8 or yzr < 1.5:
        return False

    # Ring of last 60 highs/lows for breakout level
    hring = sc.setdefault('h60', [])
    lring = sc.setdefault('l60', [])
    hring.append(high)
    lring.append(low)
    if len(hring) > 60:
        hring.pop(0)
        lring.pop(0)
    if len(hring) < 60:
        return False

    hi60 = max(hring[:-1])
    lo60 = min(lring[:-1])

    direction = None
    if close > hi60:
        direction = 'long'
    elif close < lo60:
        direction = 'short'
    if direction is None:
        return False

    if direction == 'long':
        entry = close
        sl    = close - 1.0 * atr
        tp    = close + 2.0 * atr
    else:
        entry = close
        sl    = close + 1.0 * atr
        tp    = close - 2.0 * atr
    sl_dist = abs(entry - sl)
    size = _size_for(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
    if not size or size <= 0:
        return False

    eng.place_pending(sc, ts, direction=direction,
                       entry=entry, sl=sl, tp=tp,
                       size=size, dist=sl_dist,
                       mode='market_next_open')
    sc['last_trade_bar'] = cur_n
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(name: str, entry_fn, exit_hour: int, pair: str, train_df, test_df,
            cost_mult: float = 1.0) -> dict:
    manager_fn = eng.make_manager(exit_hour=exit_hour, use_breakeven=False)
    slot_class = f"v3_{name[:14]}".replace("-", "_").lower()
    registry = [{
        "id": f"v3_{name}_{pair}", "family": "battery",
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
        try:
            trades, _, _ = eng.run_backtest(
                {pair: df}, None, None,
                registry, slot_managers, slot_entries,
                cost_mult=cost_mult,
            )
        except Exception as e:
            out[split] = {"n": 0, "sharpe": 0, "pnl": 0, "wr": 0,
                          "error": f"{type(e).__name__}: {e}"}
            continue
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

    # Per-pair percentiles for S3 (computed on TRAIN only)
    spread_p10 = {p: float(np.percentile(train_dfs[p]['spread_mean'].dropna(), 10))
                  for p in PAIRS}
    tick_p90   = {p: float(np.percentile(train_dfs[p]['tick_imbalance'].dropna().abs(), 90))
                  for p in PAIRS}

    # Strategy registry — (name, factory_or_fn, exit_hour, is_factory)
    strategies = [
        ("asian_range_fade_london",    entry_asian_range_fade_london,     13, False),
        ("cumdelta_divergence_revert", entry_cumdelta_divergence_revert,  21, False),
        ("tight_spread_tick_scalp",    make_tight_spread_tick_scalp,      21, True),
        ("five_bar_momentum_cont",     entry_five_bar_momentum_cont,      21, False),
        ("vol_expansion_breakout",     entry_vol_expansion_breakout,      21, False),
    ]

    rows = []
    for name, fn_or_fac, exit_h, is_fac in strategies:
        print(f"=== {name} ===")
        for pair in PAIRS:
            if is_fac:
                fn = fn_or_fac(spread_p10[pair], tick_p90[pair])
            else:
                fn = fn_or_fac
            r = run_one(name, fn, exit_h, pair, train_dfs[pair], test_dfs[pair])
            rows.append((name, pair, r))
            t, te = r['train'], r['test']
            print(f"  {pair}: train n={t['n']:>4d} sh={t['sharpe']:+6.2f} "
                  f"${t['pnl']:>+8,.0f}  |  test n={te['n']:>4d} "
                  f"sh={te['sharpe']:+6.2f} ${te['pnl']:>+8,.0f}")
        print()

    # Summary
    print("=" * 100)
    print(f"{'STRATEGY':30s} {'PAIR':9s} {'TR_N':>5s} {'TR_SH':>7s} "
          f"{'TR_PNL':>10s} {'TE_N':>5s} {'TE_SH':>7s} {'TE_PNL':>10s} "
          f"{'CAND':>5s}")
    print("-" * 100)
    candidates = []
    for name, pair, r in rows:
        t, te = r['train'], r['test']
        is_cand = (te['sharpe'] >= 0.5 and te['n'] >= 30
                   and t['pnl'] > 0 and te['pnl'] > 0)
        mark = "✓" if is_cand else "—"
        print(f"{name:30s} {pair:9s} {t['n']:>5d} {t['sharpe']:>+7.2f} "
              f"${t['pnl']:>+8,.0f} {te['n']:>5d} {te['sharpe']:>+7.2f} "
              f"${te['pnl']:>+8,.0f} {mark:>5s}")
        if is_cand:
            candidates.append((name, pair, te['sharpe'], te['n'], te['pnl']))
    print("=" * 100)
    print()
    if candidates:
        print(f"PROMOTION CANDIDATES ({len(candidates)}):")
        for n, p, sh, nn, pnl in sorted(candidates, key=lambda x: -x[2]):
            print(f"  ✓ {n} on {p}: test_sharpe=+{sh:.2f} n={nn} pnl=${pnl:+,.0f}")
    else:
        print("No V3 strategy/pair combo cleared the bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
