# -*- coding: utf-8 -*-
"""
Edge Hypotheses — User Workspace

This is the ONLY file you edit during edge research.

How it works:
  1. Write an entry function (entry_*).
  2. Add it to SWEEPS dict with a ParameterGrid.
  3. The GUI picks it up automatically — no other files change.

Building blocks available from edge_engine:
  spread_gate(row)              — True = spread elevated, skip entry
  rv_size(pair, bal, risk, entry, dist, row)  — position size with RV scaling
  check_and_fill(sc, row, slot, ts, regime, hspd, slip) — pending fill check
  place_pending(sc, ts, dir, entry, sl, tp, size, dist, level=None,
                mode='stop_at_level' | 'market_next_open')
  place_oco_pending(sc, ts,  long_level, long_sl, long_tp, long_size, long_dist,
                              short_level, short_sl, short_tp, short_size, short_dist)
  has_pending(sc)               — True if any pending order is staged
  cancel_pending(sc)            — clear staged pending (e.g. at session exit)
  make_manager(exit_hour, ...)  — factory for the position manager
  resolve_risk(bst, mult, mode) — dynamic risk per trade
  PAIR_PIP_SIZE, NY_PAIRS, ASIAN_PAIRS, SESSION_CONFIG

Look-ahead bias rules (do not violate):
  • Pending orders never fill on the bar they were placed — the engine enforces
    this via sc['pending_placed_ts'].
  • For BREAKOUT strategies (where direction is unknown until price moves),
    use place_oco_pending at the moment the range is fully known. Both legs
    persist; whichever triggers first wins.
  • For CONFIRMATION strategies (where a bar-close-time signal determines
    direction), use place_pending(..., mode='market_next_open'). The order
    fills at the next bar's open with slippage.
"""

import math
import numpy as np

from edge_engine import (
    ParameterGrid,
    make_manager,
    spread_gate,
    rv_size,
    check_and_fill,
    place_pending,
    place_oco_pending,
    has_pending,
    cancel_pending,
    resolve_risk,
    PAIR_PIP_SIZE,
    NY_PAIRS,
    ASIAN_PAIRS,
    LONDON_PAIRS,
    SESSION_CONFIG,
    INITIAL_BALANCE,
)

# =============================================================================
# H0a: Opening Range Breakout (NY) — pre-stage OCO bracket
# =============================================================================
# At 15:00 UTC (NY pre-market range complete) place a long-stop at range_high
# and a short-stop at range_low. Whichever side breaks first fills; the other
# is cancelled. No bar-close-time gate — this is pure ORB. Matches the live
# MT5 implementation which uses real persistent stop orders.
# =============================================================================

def entry_orb_ny(bst, slot, row, ts, pair, slip, hspd,
                 sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('oco_placed_today'):
        return False
    if row.hour != 15 or row.minute != 0:
        return False

    rng_high = getattr(row, 'range_high', float('nan'))
    rng_low  = getattr(row, 'range_low',  float('nan'))
    if math.isnan(rng_high) or math.isnan(rng_low):
        return False
    rng_size = rng_high - rng_low
    if rng_size <= 0:
        return False

    if params.get('ma_req', False):
        ma = getattr(row, 'ma_trend', float('nan'))
        if math.isnan(ma):
            return False

    tp_r    = params['tp_r']
    sl_dist = rng_size
    tp_dist = rng_size * tp_r
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    long_size  = rv_size(pair, bst.balance, risk, rng_high, sl_dist, row)
    short_size = rv_size(pair, bst.balance, risk, rng_low,  sl_dist, row)

    place_oco_pending(
        sc, ts,
        long_level=rng_high,
        long_sl=rng_high - sl_dist,
        long_tp=rng_high + tp_dist,
        long_size=long_size,
        long_dist=sl_dist,
        short_level=rng_low,
        short_sl=rng_low + sl_dist,
        short_tp=rng_low - tp_dist,
        short_size=short_size,
        short_dist=sl_dist,
    )
    sc['oco_placed_today'] = True
    return False


SWEEP_ORB_NY = {
    'entry_fn':   entry_orb_ny,
    'manager_fn': make_manager(exit_hour=21, use_profit_lock=False),
    'pairs':      NY_PAIRS,
    'session':    'ny',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       0.5,
        'TRANSITIONING': 0.5,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.3,
    },
    'grid': ParameterGrid({
        'tp_r':   [1.5, 2.0, 2.5, 3.0],
        'ma_req': [True, False],
    }),
}


# =============================================================================
# NULL: ORB-shaped, random direction (diagnostic baseline)
# =============================================================================
# Same entry time, SL/TP scheme, sizing, regime gating, manager, and costs as
# entry_orb_ny — only the directional signal is removed. A seeded coin flip
# per (seed, pair, date) picks ONE leg to stage instead of an OCO bracket.
#
# Purpose: under no edge, the distribution of test Sharpes across seeds should
# be roughly bell-shaped around zero. If the mean is materially negative, the
# engine's cost stack is over-modeled (every "strategy" loses to costs alone),
# which would explain the 0/152-survivors result without any real-edge claim.
# =============================================================================

def entry_random_ny(bst, slot, row, ts, pair, slip, hspd,
                    sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('oco_placed_today'):
        return False
    if row.hour != 15 or row.minute != 0:
        return False

    rng_high = getattr(row, 'range_high', float('nan'))
    rng_low  = getattr(row, 'range_low',  float('nan'))
    if math.isnan(rng_high) or math.isnan(rng_low):
        return False
    rng_size = rng_high - rng_low
    if rng_size <= 0:
        return False

    tp_r    = params['tp_r']
    sl_dist = rng_size
    tp_dist = rng_size * tp_r
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    seed_base = params['seed']
    rng = np.random.RandomState(hash((seed_base, pair, ts.date())) & 0x7FFFFFFF)
    go_long = rng.random() < 0.5

    if go_long:
        place_pending(
            sc, ts, 'long',
            entry=rng_high,
            sl=rng_high - sl_dist,
            tp=rng_high + tp_dist,
            size=rv_size(pair, bst.balance, risk, rng_high, sl_dist, row),
            dist=sl_dist,
            level=rng_high,
            mode='stop_at_level',
        )
    else:
        place_pending(
            sc, ts, 'short',
            entry=rng_low,
            sl=rng_low + sl_dist,
            tp=rng_low - tp_dist,
            size=rv_size(pair, bst.balance, risk, rng_low, sl_dist, row),
            dist=sl_dist,
            level=rng_low,
            mode='stop_at_level',
        )
    sc['oco_placed_today'] = True
    return False


SWEEP_NULL_NY = {
    'entry_fn':   entry_random_ny,
    'manager_fn': make_manager(exit_hour=21, use_profit_lock=False),
    'pairs':      NY_PAIRS,
    'session':    'ny',
    'regime_mult': SWEEP_ORB_NY['regime_mult'],
    'grid': ParameterGrid({
        'seed':   list(range(1, 201)),
        'tp_r':   [2.0],
    }),
}


SWEEP_NULL_NY_SMOKE = {
    'entry_fn':   entry_random_ny,
    'manager_fn': make_manager(exit_hour=21, use_profit_lock=False),
    'pairs':      NY_PAIRS,
    'session':    'ny',
    'regime_mult': SWEEP_ORB_NY['regime_mult'],
    'grid': ParameterGrid({
        'seed':   [1, 2, 3],
        'tp_r':   [2.0],
    }),
}


# =============================================================================
# H0b: Asian Range Breakout — pre-stage OCO at London open
# =============================================================================
# At 08:00 UTC place an OCO bracket at asian_high / asian_low. Whichever side
# breaks during the London session fills.
# =============================================================================

def entry_asian_breakout(bst, slot, row, ts, pair, slip, hspd,
                         sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('oco_placed_today'):
        return False
    if row.hour != 8 or row.minute != 0:
        return False

    asian_high = getattr(row, 'asian_high', float('nan'))
    asian_low  = getattr(row, 'asian_low',  float('nan'))
    if math.isnan(asian_high) or math.isnan(asian_low):
        return False
    rng_size = asian_high - asian_low
    if rng_size <= 0:
        return False

    tp_r    = params['tp_r']
    sl_mult = params.get('sl_mult', 1.0)
    sl_dist = rng_size * sl_mult
    tp_dist = rng_size * tp_r
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    long_size  = rv_size(pair, bst.balance, risk, asian_high, sl_dist, row)
    short_size = rv_size(pair, bst.balance, risk, asian_low,  sl_dist, row)

    place_oco_pending(
        sc, ts,
        long_level=asian_high,
        long_sl=asian_high - sl_dist,
        long_tp=asian_high + tp_dist,
        long_size=long_size,
        long_dist=sl_dist,
        short_level=asian_low,
        short_sl=asian_low + sl_dist,
        short_tp=asian_low - tp_dist,
        short_size=short_size,
        short_dist=sl_dist,
    )
    sc['oco_placed_today'] = True
    return False


SWEEP_ASIAN_BREAKOUT = {
    'entry_fn':   entry_asian_breakout,
    'manager_fn': make_manager(exit_hour=13, use_profit_lock=True,
                               lock_trigger=2.0, lock_sl=1.5),
    'pairs':      ASIAN_PAIRS,
    'session':    'asian',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       0.5,
        'TRANSITIONING': 0.5,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.3,
    },
    'grid': ParameterGrid({
        'tp_r':       [1.5, 2.0, 2.5, 3.0],
        'sl_mult':    [0.8, 1.0, 1.2],
    }),
}


# =============================================================================
# H0d: Asian-range Sweep-and-Reverse at London open  (hand-crafted, 2026-06-04)
# =============================================================================
# Mechanical thesis (passes the four-question test):
#
# Who's on the other side?
#   Retail traders who place stops just above the Asian high / below the Asian
#   low. Convention bias is strong: stops cluster at obvious range extremes.
#
# Why do they take it?
#   Standard retail stop-placement; once stopped, retail rarely re-enters in
#   the same direction within the same session.
#
# Why does the edge persist despite competition?
#   Retail behaviour is sticky at session boundaries; the sweep extent is
#   usually too small (sub-pip on majors) for HFT arbs to bother monetising
#   directly via the sweep itself; the inefficiency is calendar-anchored —
#   it recurs every London open as long as the Asian/London session
#   structure exists.
#
# Smoking gun in the data:
#   A single M1 bar in the 07:00-10:00 UTC window whose HIGH exceeds
#   asian_high by ≥ min_sweep * rng_size AND whose CLOSE is back inside
#   the Asian range by at least max_close_dist * rng_size. Symmetric for
#   the LOW case. The strategy fires a market-next-open order in the
#   reversal direction.
# =============================================================================

def entry_asian_sweep_reverse_london(bst, slot, row, ts, pair, slip, hspd,
                                       sess_cfg, regime, regime_mult,
                                       fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('fired_today'):
        return False

    # London sweep-detection window: 07:00-09:59 UTC. After 10:00 the day's
    # directional flow is too established for the sweep-and-revert mechanic.
    if row.hour < 7 or row.hour >= 10:
        return False

    asian_high = getattr(row, 'asian_high', float('nan'))
    asian_low  = getattr(row, 'asian_low',  float('nan'))
    if math.isnan(asian_high) or math.isnan(asian_low):
        return False
    rng_size = asian_high - asian_low
    if rng_size <= 0:
        return False

    tp_r           = params.get('tp_r',           1.0)
    sl_buffer      = params.get('sl_buffer',      0.3)
    min_sweep      = params.get('min_sweep',      0.05)
    max_close_dist = params.get('max_close_dist', 0.3)

    high  = getattr(row, 'high',  float('nan'))
    low   = getattr(row, 'low',   float('nan'))
    close = getattr(row, 'close', float('nan'))
    if math.isnan(high) or math.isnan(low) or math.isnan(close):
        return False

    swept_up   = (high  > asian_high + min_sweep      * rng_size
                  and close < asian_high - max_close_dist * rng_size)
    swept_down = (low   < asian_low  - min_sweep      * rng_size
                  and close > asian_low  + max_close_dist * rng_size)
    if not (swept_up or swept_down):
        return False

    risk = resolve_risk(bst, regime_mult, 'dynamic')

    if swept_up:
        # Sweep above Asian high then rejected back inside → SHORT
        direction = 'short'
        sl        = high + sl_buffer * rng_size
        tp        = close - tp_r * rng_size
        sl_dist   = sl - close
    else:
        # Sweep below Asian low then rejected back inside → LONG
        direction = 'long'
        sl        = low - sl_buffer * rng_size
        tp        = close + tp_r * rng_size
        sl_dist   = close - sl

    if sl_dist <= 0:
        return False

    size = rv_size(pair, bst.balance, risk, close, sl_dist, row)

    place_pending(
        sc, ts, direction,
        entry=close, sl=sl, tp=tp,
        size=size, dist=sl_dist,
        mode='market_next_open',
    )
    sc['fired_today'] = True
    return False


SWEEP_ASIAN_SWEEP_REVERSE = {
    'entry_fn':   entry_asian_sweep_reverse_london,
    'manager_fn': make_manager(exit_hour=13, use_breakeven=True),
    'pairs':      LONDON_PAIRS,
    'session':    'london',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       1.0,
        'TRANSITIONING': 0.7,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.5,
    },
    'grid': ParameterGrid({
        'tp_r':           [0.7, 1.0, 1.5],
        'sl_buffer':      [0.2, 0.3, 0.5],
        'min_sweep':      [0.03, 0.07],
        'max_close_dist': [0.2, 0.4],
    }),
}


# =============================================================================
# H0c: Swing Fade — confirmation entry, fills at next bar open
# =============================================================================

def entry_swing_fade(bst, slot, row, ts, pair, slip, hspd,
                     sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if regime not in ('RANGING', 'TRANSITIONING'):
        return False
    active_s = sess_cfg['active_start']
    active_e = sess_cfg['active_end']
    if row.hour < active_s or row.hour >= active_e:
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    swing_high = getattr(row, 'swing_high5', float('nan'))
    swing_low  = getattr(row, 'swing_low5',  float('nan'))
    if math.isnan(swing_high) or math.isnan(swing_low):
        return False

    tp_r    = params['tp_r']
    sl_r    = params['sl_r']
    atr     = getattr(row, 'atr', float('nan'))
    if math.isnan(atr) or atr <= 0:
        return False

    sl_dist = atr * sl_r
    tp_dist = atr * tp_r
    risk    = resolve_risk(bst, regime_mult, 'fixed')

    if row.high >= swing_high:
        # Wicked above swing high — fade the move (short)
        size  = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    elif row.low <= swing_low:
        size  = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    return False


SWEEP_SWING_FADE = {
    'entry_fn':   entry_swing_fade,
    'manager_fn': make_manager(exit_hour=21, use_breakeven=False),
    'pairs':      ['GBP_USD', 'EUR_USD', 'USD_JPY'],
    'session':    'ny',
    'regime_mult': {
        'TRENDING':      0.0,
        'RANGING':       1.0,
        'TRANSITIONING': 0.5,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.0,
    },
    'grid': ParameterGrid({
        'tp_r': [0.8, 1.0, 1.2, 1.5],
        'sl_r': [0.5, 0.8, 1.0],
    }),
}


SWEEPS = {
    'orb_ny':                     SWEEP_ORB_NY,
    'null_ny':                    SWEEP_NULL_NY,
    'null_ny_smoke':              SWEEP_NULL_NY_SMOKE,
    'asian_breakout':             SWEEP_ASIAN_BREAKOUT,
    'asian_sweep_reverse_london': SWEEP_ASIAN_SWEEP_REVERSE,
    'swing_fade':                 SWEEP_SWING_FADE,
}


# =============================================================================
# H1: Liquidity Grab Fade — confirmation entry
# =============================================================================

def entry_liq_grab_fade(bst, slot, row, ts, pair, slip, hspd,
                        sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params    = slot['strategy_def']['params']
    sc        = slot['scratch']
    pip       = PAIR_PIP_SIZE[pair]
    day_sweep = day_sweep or {}

    if spread_gate(row):
        return False
    if row.hour < 8 or row.hour >= 21:
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    rh = getattr(row, 'range_high', float('nan'))
    rl = getattr(row, 'range_low',  float('nan'))
    if math.isnan(rh) or math.isnan(rl):
        return False

    tp_r           = params['tp_r']
    sl_buffer_pips = params['sl_buffer_pips']
    imb_thresh     = params.get('imb_thresh', 0.0)
    tick_imb       = getattr(row, 'tick_imbalance', 0.0)
    risk           = resolve_risk(bst, regime_mult, 'dynamic')

    # Short fade after high wick
    if day_sweep.get(pair, {}).get('high') and not sc.get('faded_high'):
        if tick_imb <= -imb_thresh:
            sl_price = rh + sl_buffer_pips * pip
            sl_dist  = sl_price - row.close
            if sl_dist <= 0:
                return False
            tp_dist = sl_dist * tp_r
            size    = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
            sc['faded_high'] = True
            place_pending(sc, ts, 'short',
                          entry=row.close,
                          sl=sl_price,
                          tp=row.close - tp_dist,
                          size=size, dist=sl_dist,
                          mode='market_next_open')

    # Long fade after low wick
    elif day_sweep.get(pair, {}).get('low') and not sc.get('faded_low'):
        if tick_imb >= imb_thresh:
            sl_price = rl - sl_buffer_pips * pip
            sl_dist  = row.close - sl_price
            if sl_dist <= 0:
                return False
            tp_dist = sl_dist * tp_r
            size    = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
            sc['faded_low'] = True
            place_pending(sc, ts, 'long',
                          entry=row.close,
                          sl=sl_price,
                          tp=row.close + tp_dist,
                          size=size, dist=sl_dist,
                          mode='market_next_open')

    return False


SWEEP_LIQ_GRAB_LONDON = {
    'entry_fn':    entry_liq_grab_fade,
    'manager_fn':  make_manager(exit_hour=13, use_breakeven=True),
    'pairs':       LONDON_PAIRS,
    'session':     'london',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       0.8,
        'TRANSITIONING': 0.8,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.3,
    },
    'grid': ParameterGrid({
        'tp_r':          [0.8, 1.0, 1.2, 1.5],
        'imb_thresh':    [0.0, 0.10, 0.15],
        'sl_buffer_pips':[2, 3, 5],
    }),
}

SWEEP_LIQ_GRAB_NY = {
    'entry_fn':    entry_liq_grab_fade,
    'manager_fn':  make_manager(exit_hour=21, use_breakeven=True),
    'pairs':       NY_PAIRS,
    'session':     'ny',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       0.8,
        'TRANSITIONING': 0.8,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.3,
    },
    'grid': ParameterGrid({
        'tp_r':          [0.8, 1.0, 1.2, 1.5],
        'imb_thresh':    [0.0, 0.10, 0.15],
        'sl_buffer_pips':[2, 3, 5],
    }),
}


# =============================================================================
# H2: Aligned Trending Breakout — pre-stage OCO with regime gate at placement
# =============================================================================
# At 15:00 UTC, if the regime at range close is TRENDING and ADX clears the
# threshold, place an OCO bracket at rng_high/rng_low. The regime/ADX check is
# applied at PLACEMENT (when we have all bar-close info to make the call), not
# at fill time. The orders persist through the session; whichever fires first
# wins.
# =============================================================================

def entry_trend_breakout(bst, slot, row, ts, pair, slip, hspd,
                         sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('oco_placed_today'):
        return False
    if row.hour != 15 or row.minute != 0:
        return False
    if regime != 'TRENDING':
        return False

    adx = getattr(row, 'adx', float('nan'))
    if math.isnan(adx) or adx < params['adx_min']:
        return False

    rh = getattr(row, 'range_high', float('nan'))
    rl = getattr(row, 'range_low',  float('nan'))
    if math.isnan(rh) or math.isnan(rl):
        return False
    rng_size = rh - rl
    if rng_size <= 0:
        return False

    tp_r    = params['tp_r']
    sl_dist = rng_size
    tp_dist = rng_size * tp_r
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    long_size  = rv_size(pair, bst.balance, risk, rh, sl_dist, row)
    short_size = rv_size(pair, bst.balance, risk, rl, sl_dist, row)

    place_oco_pending(
        sc, ts,
        long_level=rh,  long_sl=rh - sl_dist,
        long_tp=rh + tp_dist, long_size=long_size, long_dist=sl_dist,
        short_level=rl, short_sl=rl + sl_dist,
        short_tp=rl - tp_dist, short_size=short_size, short_dist=sl_dist,
    )
    sc['oco_placed_today'] = True
    return False


SWEEP_TREND_BREAKOUT = {
    'entry_fn':    entry_trend_breakout,
    'manager_fn':  make_manager(exit_hour=21, use_profit_lock=True,
                                lock_trigger=2.0, lock_sl=1.5),
    'pairs':       NY_PAIRS,
    'session':     'ny',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       0.0,
        'TRANSITIONING': 0.0,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.0,
    },
    'grid': ParameterGrid({
        'tp_r':      [2.0, 2.5, 3.0, 3.5],
        'adx_min':   [20, 25, 30],
    }),
}


# =============================================================================
# H3: Spike Fade — confirmation entry
# =============================================================================

def entry_spike_fade(bst, slot, row, ts, pair, slip, hspd,
                     sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if regime not in ('RANGING', 'TRANSITIONING'):
        return False
    active_s = sess_cfg['active_start']
    active_e = sess_cfg['active_end']
    if row.hour < active_s or row.hour >= active_e:
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    rv     = getattr(row, 'realized_vol', float('nan'))
    rv_med = getattr(row, 'rv_median',    float('nan'))
    if math.isnan(rv) or math.isnan(rv_med) or rv_med <= 0:
        return False
    if rv < params['rv_thresh'] * rv_med:
        return False

    swing_high = getattr(row, 'swing_high5', float('nan'))
    swing_low  = getattr(row, 'swing_low5',  float('nan'))
    atr        = getattr(row, 'atr',         float('nan'))
    if math.isnan(swing_high) or math.isnan(swing_low) or math.isnan(atr) or atr <= 0:
        return False

    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'fixed')

    if row.close > swing_high:
        size  = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    elif row.close < swing_low:
        size  = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    return False


SWEEP_SPIKE_FADE = {
    'entry_fn':    entry_spike_fade,
    'manager_fn':  make_manager(exit_hour=21, use_breakeven=False),
    'pairs':       ['GBP_USD', 'EUR_USD', 'USD_JPY', 'EUR_GBP', 'GBP_JPY', 'EUR_JPY'],
    'session':     'ny',
    'regime_mult': {
        'TRENDING':      0.0,
        'RANGING':       1.0,
        'TRANSITIONING': 0.75,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.0,
    },
    'grid': ParameterGrid({
        'rv_thresh': [1.3, 1.5, 2.0],
        'tp_r':      [0.75, 1.0, 1.5],
        'sl_r':      [0.5, 0.75, 1.0],
    }),
}


# =============================================================================
# H4: Time of Day — confirmation entry
# =============================================================================

def entry_time_of_day(bst, slot, row, ts, pair, slip, hspd,
                      sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('fired_today'):
        return False
    if row.hour != params['entry_hour'] or row.minute >= 5:
        return False

    atr = getattr(row, 'atr', float('nan'))
    if math.isnan(atr) or atr <= 0:
        return False

    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    direction_mode = params.get('direction_mode', 'trend')
    if direction_mode == 'trend':
        ma = getattr(row, 'ma_trend', float('nan'))
        if math.isnan(ma):
            return False
        go_long  = row.close > ma
        go_short = row.close < ma
    else:
        go_long  = row.close > row.open
        go_short = row.close < row.open

    if go_long:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    elif go_short:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    return False


SWEEP_TOD_LONDON = {
    'entry_fn':    entry_time_of_day,
    'manager_fn':  make_manager(exit_hour=13, use_breakeven=True),
    'pairs':       LONDON_PAIRS,
    'session':     'london',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       0.5,
        'TRANSITIONING': 0.5,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.3,
    },
    'grid': ParameterGrid({
        'entry_hour':     [8, 9, 10, 11],
        'tp_r':           [1.0, 1.5, 2.0],
        'sl_r':           [0.75, 1.0],
        'direction_mode': ['trend', 'momentum'],
    }),
}

SWEEP_TOD_NY = {
    'entry_fn':    entry_time_of_day,
    'manager_fn':  make_manager(exit_hour=21, use_breakeven=True),
    'pairs':       NY_PAIRS,
    'session':     'ny',
    'regime_mult': {
        'TRENDING':      1.0,
        'RANGING':       0.5,
        'TRANSITIONING': 0.5,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.3,
    },
    'grid': ParameterGrid({
        'entry_hour':     [15, 16, 17, 18, 19, 20],
        'tp_r':           [1.0, 1.5, 2.0],
        'sl_r':           [0.75, 1.0],
        'direction_mode': ['trend', 'momentum'],
    }),
}


SWEEPS['liq_grab_london'] = SWEEP_LIQ_GRAB_LONDON
SWEEPS['liq_grab_ny']     = SWEEP_LIQ_GRAB_NY
SWEEPS['trend_breakout']  = SWEEP_TREND_BREAKOUT
SWEEPS['spike_fade']      = SWEEP_SPIKE_FADE
SWEEPS['tod_london']      = SWEEP_TOD_LONDON
SWEEPS['tod_ny']          = SWEEP_TOD_NY


# =============================================================================
# H5: Cumulative Delta Divergence Fade — confirmation entry
# =============================================================================

def entry_cum_delta_div(bst, slot, row, ts, pair, slip, hspd,
                        sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if row.hour < 8 or row.hour >= 21:
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('fired_today'):
        return False

    delta_mom = getattr(row, 'delta_momentum', float('nan'))
    atr       = getattr(row, 'atr',            float('nan'))
    if math.isnan(delta_mom) or math.isnan(atr) or atr <= 0:
        return False

    threshold = params['delta_thresh']
    sl_dist   = atr * params['sl_r']
    tp_dist   = atr * params['tp_r']
    risk      = resolve_risk(bst, regime_mult, 'dynamic')

    if row.close > row.open and delta_mom < -threshold:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    elif row.close < row.open and delta_mom > threshold:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    return False


SWEEP_CUM_DELTA_DIV = {
    'entry_fn':    entry_cum_delta_div,
    'manager_fn':  make_manager(exit_hour=21, use_breakeven=True),
    'pairs':       NY_PAIRS,
    'session':     'ny',
    'regime_mult': {'TRENDING':0.5,'RANGING':1.0,'TRANSITIONING':1.0,
                    'VOLATILE':0.0,'UNDEFINED':0.3},
    'grid': ParameterGrid({
        'tp_r':         [1.0, 1.5, 2.0],
        'sl_r':         [0.75, 1.0],
        'delta_thresh': [0.05, 0.10, 0.15],
    }),
}


# =============================================================================
# H6: Stop Run Fade — confirmation entry
# =============================================================================

def entry_stop_run_fade2(bst, slot, row, ts, pair, slip, hspd,
                         sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if row.hour < 8 or row.hour >= 21:
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    score = getattr(row, 'stop_run_score',  float('nan'))
    cl    = getattr(row, 'close_location',  float('nan'))
    atr   = getattr(row, 'atr',             float('nan'))
    if math.isnan(score) or math.isnan(cl) or math.isnan(atr) or atr <= 0:
        return False
    if score < params['score_thresh']:
        return False

    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    if cl > 0.7:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    elif cl < 0.3:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    return False


SWEEP_STOP_RUN_LONDON = {
    'entry_fn':    entry_stop_run_fade2,
    'manager_fn':  make_manager(exit_hour=13, use_breakeven=True),
    'pairs':       LONDON_PAIRS,
    'session':     'london',
    'regime_mult': {'TRENDING':0.5,'RANGING':1.0,'TRANSITIONING':1.0,
                    'VOLATILE':0.0,'UNDEFINED':0.3},
    'grid': ParameterGrid({
        'tp_r':        [0.8, 1.0, 1.5],
        'sl_r':        [0.75, 1.0],
        'score_thresh':[0.5, 1.0, 1.5],
    }),
}

SWEEP_STOP_RUN_NY = {
    'entry_fn':    entry_stop_run_fade2,
    'manager_fn':  make_manager(exit_hour=21, use_breakeven=True),
    'pairs':       NY_PAIRS,
    'session':     'ny',
    'regime_mult': {'TRENDING':0.5,'RANGING':1.0,'TRANSITIONING':1.0,
                    'VOLATILE':0.0,'UNDEFINED':0.3},
    'grid': ParameterGrid({
        'tp_r':        [0.8, 1.0, 1.5],
        'sl_r':        [0.75, 1.0],
        'score_thresh':[0.5, 1.0, 1.5],
    }),
}


# =============================================================================
# H7: Session Open Order Flow — confirmation entry
# =============================================================================

def entry_open_flow(bst, slot, row, ts, pair, slip, hspd,
                    sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if not getattr(row, 'active_session', False):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('fired_today'):
        return False

    bars_since = getattr(row, 'bars_since_open',      float('nan'))
    pers_imb   = getattr(row, 'persistent_imbalance', float('nan'))
    atr        = getattr(row, 'atr',                  float('nan'))
    if math.isnan(bars_since) or math.isnan(pers_imb) or math.isnan(atr) or atr <= 0:
        return False
    if bars_since > params['open_bars']:
        return False

    imb_min = params['imb_min']
    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    if pers_imb >= imb_min:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    elif pers_imb <= -imb_min:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    return False


SWEEP_OPEN_FLOW_LONDON = {
    'entry_fn':    entry_open_flow,
    'manager_fn':  make_manager(exit_hour=13, use_breakeven=True),
    'pairs':       LONDON_PAIRS,
    'session':     'london',
    'regime_mult': {'TRENDING':1.0,'RANGING':0.75,'TRANSITIONING':0.75,
                    'VOLATILE':0.0,'UNDEFINED':0.3},
    'grid': ParameterGrid({
        'tp_r':      [1.0, 1.5, 2.0],
        'sl_r':      [0.75, 1.0],
        'open_bars': [3, 5, 8],
        'imb_min':   [3, 4],
    }),
}

SWEEP_OPEN_FLOW_NY = {
    'entry_fn':    entry_open_flow,
    'manager_fn':  make_manager(exit_hour=21, use_breakeven=True),
    'pairs':       NY_PAIRS,
    'session':     'ny',
    'regime_mult': {'TRENDING':1.0,'RANGING':0.75,'TRANSITIONING':0.75,
                    'VOLATILE':0.0,'UNDEFINED':0.3},
    'grid': ParameterGrid({
        'tp_r':      [1.0, 1.5, 2.0],
        'sl_r':      [0.75, 1.0],
        'open_bars': [3, 5, 8],
        'imb_min':   [3, 4],
    }),
}


SWEEPS['cum_delta_div']      = SWEEP_CUM_DELTA_DIV
SWEEPS['stop_run_london']    = SWEEP_STOP_RUN_LONDON
SWEEPS['stop_run_ny']        = SWEEP_STOP_RUN_NY
SWEEPS['open_flow_london']   = SWEEP_OPEN_FLOW_LONDON
SWEEPS['open_flow_ny']       = SWEEP_OPEN_FLOW_NY


# =============================================================================
# H8: HMM Regime Transition — confirmation entry
# =============================================================================

def entry_hmm_transition(bst, slot, row, ts, pair, slip, hspd,
                          sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if row.hour < 8 or row.hour >= 21:
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('fired_today'):
        return False

    hmm_trans = getattr(row, 'hmm_transition',  float('nan'))
    delta_mom = getattr(row, 'delta_momentum',   float('nan'))
    atr       = getattr(row, 'atr',              float('nan'))
    hmm_state = getattr(row, 'hmm_state',        float('nan'))
    if math.isnan(hmm_trans) or math.isnan(delta_mom) or math.isnan(atr) or atr <= 0:
        return False

    if hmm_trans != 1:
        return False

    prob_col  = f'hmm_prob_{int(hmm_state)}' if not math.isnan(hmm_state) else None
    if prob_col is not None:
        state_prob = getattr(row, prob_col, float('nan'))
        if math.isnan(state_prob) or state_prob < params['prob_thresh']:
            return False

    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    if delta_mom > params['delta_thresh']:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    elif delta_mom < -params['delta_thresh']:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')

    return False


SWEEP_HMM_TRANSITION = {
    'entry_fn':    entry_hmm_transition,
    'manager_fn':  make_manager(exit_hour=21, use_breakeven=True),
    'pairs':       NY_PAIRS + LONDON_PAIRS,
    'session':     'ny',
    'regime_mult': {'TRENDING':1.0,'RANGING':1.0,'TRANSITIONING':1.0,
                    'VOLATILE':0.0,'UNDEFINED':0.5},
    'grid': ParameterGrid({
        'tp_r':        [1.5, 2.0, 2.5],
        'sl_r':        [0.75, 1.0],
        'prob_thresh': [0.55, 0.65, 0.75],
        'delta_thresh':[0.05, 0.10],
    }),
}

SWEEPS['hmm_transition'] = SWEEP_HMM_TRANSITION


# =============================================================================
# TEMPLATE — copy this to create a new hypothesis
# =============================================================================
#
# def entry_my_idea(bst, slot, row, ts, pair, slip, hspd,
#                   sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
#     params = slot['strategy_def']['params']
#     sc     = slot['scratch']
#
#     # 1. Skip conditions
#     if spread_gate(row):
#         return False
#     if has_pending(sc):
#         return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
#
#     # 2. Decide direction at bar close (CONFIRMATION entry)
#     #    OR pre-stage OCO orders at session-range close (BREAKOUT entry).
#
#     # Confirmation example:
#     # place_pending(sc, ts, 'long',
#     #               entry=row.close,
#     #               sl=row.close - sl_dist,
#     #               tp=row.close + tp_dist,
#     #               size=size, dist=sl_dist,
#     #               mode='market_next_open')
#
#     # Breakout OCO example:
#     # place_oco_pending(sc, ts,
#     #                   long_level=rh,  long_sl=...,
#     #                   short_level=rl, short_sl=...)
#
#     return False
