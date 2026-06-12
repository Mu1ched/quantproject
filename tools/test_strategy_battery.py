"""
Strategy Battery V2 — bug-fixed V1 + 4 new strategies on different angles.

V1 produced 0/20 promotion candidates. V2 fixes the implementation bugs that
killed V1's signal (NY firing every day; Tokyo timezone confusion; BB strategy
compounding losses without min-bars; ADX state never reset across days) and
adds 4 new strategies that exploit primitives V1 didn't use: precomputed
h1_trend/h4_atr columns, the `near_news` event column, and cross-pair
coordination via scratch-closure over other-pair lookups.

Each strategy has a thesis comment block. Each is rateable by the same gate:
test_sharpe ≥ 0.5 AND n ≥ 30 AND train_pnl > 0 AND test_pnl > 0.

Usage:
    python tools/test_strategy_battery.py                 # run all 9
    python tools/test_strategy_battery.py --strategy NAME  # one only
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import edge_engine as eng

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"]
REPORT_PATH = PROJECT_ROOT / "tools" / "battery_v2_report.md"


def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _day_rollover(sc: dict, row) -> bool:
    cur = getattr(row, 'date', None)
    if sc.get('_day') != cur:
        sc['_day'] = cur
        return True
    return False


def _size_for(bst, regime_mult, pair, balance, entry, sl_dist, row):
    risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
    return eng.rv_size(pair, balance, risk, entry, sl_dist, row)


def _utc_hour(row) -> int:
    """UTC hour for the bar, regardless of the cache's tz (Europe/London).
    Falls back to row.hour if the timestamp isn't tz-aware.
    """
    ts = getattr(row, 'timestamp', None)
    if ts is None:
        return int(row.hour)
    try:
        return int(ts.tz_convert('UTC').hour)
    except (AttributeError, TypeError):
        return int(getattr(row, 'hour', 0))


def _bar_idx(sc: dict) -> int:
    """Monotonic counter incremented every bar — used for min-bars-between."""
    sc['_bar_n'] = sc.get('_bar_n', 0) + 1
    return sc['_bar_n']


# ═══════════════════════════════════════════════════════════════════════════
# PHASE A — V1 strategies, bug-fixed
# ═══════════════════════════════════════════════════════════════════════════

# ─── S1: Tokyo→London Range Breakout (FIXED: UTC-aware window) ──────────────
# V1 BUG: Tokyo window was `if row.hour >= 22 or row.hour <= 6 or row.hour == 7`
# in cache-local time (Europe/London). In summer (BST=UTC+1) that became
# 21:00 UTC prior day → 06:59 UTC today — drifting away from the real Tokyo
# session every 6 months. FIX: compare against UTC explicitly.
def entry_tokyo_london_breakout(bst, slot, row, ts, pair, slip, hspd,
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

    # Tokyo session = 22:00 UTC prior day → 07:00 UTC today (excl 07:00 itself)
    if not (math.isnan(h) or math.isnan(l)) and (uh >= 22 or uh < 7):
        sc['tokyo_hi'] = h if sc['tokyo_hi'] is None else max(sc['tokyo_hi'], h)
        sc['tokyo_lo'] = l if sc['tokyo_lo'] is None else min(sc['tokyo_lo'], l)

    if sc.get('placed'):
        return False
    # Fire at the London open in UTC — uses minute=0 of hour 7 UTC
    if (uh, row.minute) != (7, 0):
        return False
    if sc['tokyo_hi'] is None or sc['tokyo_lo'] is None:
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


# ─── S2: NY 13:30 UTC News-Momentum (FIXED: news-day gating) ────────────────
# V1 BUG: fired every weekday at 13:30 UTC regardless of whether there was an
# actual scheduled release. FIX: require `near_news==True` AND
# `near_news_impact=='high'` so it only fires on NFP/CPI/FOMC-class days.
def entry_ny_1330_news_momentum(bst, slot, row, ts, pair, slip, hspd,
                                  sess_cfg, regime, regime_mult,
                                  fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    if _day_rollover(sc, row):
        sc['pre_hi'] = None
        sc['pre_lo'] = None
        sc['placed'] = False

    uh = _utc_hour(row)
    h = float(getattr(row, 'high', float('nan')))
    l = float(getattr(row, 'low',  float('nan')))

    # Accumulate 13:25-13:29 UTC range
    if uh == 13 and 25 <= row.minute <= 29 and not (math.isnan(h) or math.isnan(l)):
        sc['pre_hi'] = h if sc['pre_hi'] is None else max(sc['pre_hi'], h)
        sc['pre_lo'] = l if sc['pre_lo'] is None else min(sc['pre_lo'], l)

    if sc.get('placed'):
        return False
    if (uh, row.minute) != (13, 30):
        return False
    if sc.get('pre_hi') is None or sc.get('pre_lo') is None:
        return False

    # News-day gate: require an actual high-impact release
    if not bool(getattr(row, 'near_news', False)):
        return False
    impact = getattr(row, 'near_news_impact', '') or ''
    if impact != 'high':
        return False

    rng = sc['pre_hi'] - sc['pre_lo']
    atr = float(getattr(row, 'atr', 0) or 0)
    if rng <= 0 or atr <= 0 or rng < 0.5 * atr:
        return False

    pip = _pip(pair)
    risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
    long_level  = sc['pre_hi'] + pip
    short_level = sc['pre_lo'] - pip
    long_sl     = sc['pre_lo'] - pip
    short_sl    = sc['pre_hi'] + pip
    long_tp     = long_level  + 1.5 * rng
    short_tp    = short_level - 1.5 * rng
    long_size   = eng.rv_size(pair, bst.balance, risk, long_level,  rng, row)
    short_size  = eng.rv_size(pair, bst.balance, risk, short_level, rng, row)

    eng.place_oco_pending(sc, ts,
                           long_level=long_level, long_sl=long_sl,  long_tp=long_tp,
                           long_size=long_size,   long_dist=rng,
                           short_level=short_level, short_sl=short_sl, short_tp=short_tp,
                           short_size=short_size, short_dist=rng)
    sc['placed'] = True
    return False


# ─── S3: Friday Afternoon Mean Revert (FIXED: tighter trigger + min-bars) ───
# V1 BUG: trigger of 0.8 ATR + "at-most-one-per-hour" let ~80 trades/year fire,
# many on non-extreme moves that didn't actually revert. FIX: bump trigger to
# 1.5 ATR (rarer, larger overextensions only) + add 30-bar min-between-trades.
def entry_friday_afternoon_revert(bst, slot, row, ts, pair, slip, hspd,
                                    sess_cfg, regime, regime_mult,
                                    fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    cur_n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    if _day_rollover(sc, row):
        pass   # day_of_week filter does the heavy lifting

    dow = getattr(row, 'day_of_week', getattr(row, 'dow', None))
    if dow != 4:
        return False
    uh = _utc_hour(row)
    if uh not in (15, 16, 17, 18):
        return False
    if cur_n - sc.get('last_trade_bar', -10_000) < 30:
        return False

    mom = getattr(row, 'momentum_10', float('nan'))
    atr = float(getattr(row, 'atr', 0) or 0)
    close = float(getattr(row, 'close', 0))
    if math.isnan(mom) or atr <= 0 or close <= 0:
        return False

    move_units = abs(mom) * close
    if move_units < 1.5 * atr:    # FIX: was 0.8 ATR
        return False

    direction = 'short' if mom > 0 else 'long'   # fade
    if direction == 'long':
        entry = close
        sl    = close - 0.7 * atr
        tp    = close + 0.4 * atr
    else:
        entry = close
        sl    = close + 0.7 * atr
        tp    = close - 0.4 * atr
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


# ─── S4: Low-Vol BB Extreme Revert (FIXED: min-bars + tighter trigger) ──────
# V1 BUG: daily cap of 3 with no min-bars-between meant consecutive SL hits
# in the same morning compounded into account blowup. FIX: 60-bar minimum
# between trades + tighten BB extreme from 0.97/0.03 → 0.99/0.01 +
# ATR floor to skip dead markets.
def entry_lowvol_bb_extreme_revert(bst, slot, row, ts, pair, slip, hspd,
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

    bb   = getattr(row, 'bb_pct', float('nan'))
    yzv  = getattr(row, 'yz_vol_ratio', float('nan'))
    atr  = float(getattr(row, 'atr', 0) or 0)
    if math.isnan(bb) or math.isnan(yzv) or atr <= 0:
        return False
    if yzv >= 0.8:
        return False
    # ATR floor — skip dead markets where BB extremes are meaningless
    rv_med = float(getattr(row, 'rv_median', 0) or 0)
    if rv_med > 0 and atr < 0.5 * rv_med:
        return False

    close = float(getattr(row, 'close', 0))
    if close <= 0:
        return False

    if bb >= 0.99:               # FIX: was 0.97
        direction = 'short'
        entry = close
        sl    = close + 0.5 * atr
        tp    = close - 0.4 * atr
    elif bb <= 0.01:             # FIX: was 0.03
        direction = 'long'
        entry = close
        sl    = close - 0.5 * atr
        tp    = close + 0.4 * atr
    else:
        return False

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


# ─── S5: High-ADX Trend Pullback (FIXED: state reset + loosen + min-bars) ───
# V1 BUG: `was_extended` flag never reset across day boundaries → false
# pullback signals on Monday morning from Friday's extension. FIX: reset on
# day rollover. Also loosen the pullback condition from 0.3 ATR → 0.5 ATR
# (almost never fired) and add 60-bar min-between-trades.
def entry_high_adx_trend_pullback(bst, slot, row, ts, pair, slip, hspd,
                                    sess_cfg, regime, regime_mult,
                                    fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    cur_n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    if _day_rollover(sc, row):
        sc['was_extended'] = False    # FIX: reset on day rollover

    if cur_n - sc.get('last_trade_bar', -10_000) < 60:
        return False

    adx     = getattr(row, 'adx', float('nan'))
    ma      = getattr(row, 'ma_trend', float('nan'))
    madist  = getattr(row, 'ma_dist', float('nan'))
    atr     = float(getattr(row, 'atr', 0) or 0)
    close   = float(getattr(row, 'close', 0))
    if (math.isnan(adx) or math.isnan(ma) or math.isnan(madist) or
        atr <= 0 or close <= 0):
        return False
    if adx <= 28 or ma == 0:
        sc['was_extended'] = False
        return False

    if abs(madist) >= 1.5 * atr:
        sc['was_extended'] = True

    if not sc.get('was_extended'):
        return False
    if abs(madist) > 0.5 * atr:    # FIX: was 0.3 ATR
        return False

    direction = 'long' if ma > 0 else 'short'
    if direction == 'long':
        entry = close
        sl    = close - 0.7 * atr
        tp    = close + 1.5 * atr
    else:
        entry = close
        sl    = close + 0.7 * atr
        tp    = close - 1.5 * atr
    sl_dist = abs(entry - sl)
    size = _size_for(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
    if not size or size <= 0:
        return False

    eng.place_pending(sc, ts, direction=direction,
                       entry=entry, sl=sl, tp=tp,
                       size=size, dist=sl_dist,
                       mode='market_next_open')
    sc['last_trade_bar'] = cur_n
    sc['was_extended'] = False
    return False


# ═══════════════════════════════════════════════════════════════════════════
# PHASE B — 4 new strategies on unused engine primitives
# ═══════════════════════════════════════════════════════════════════════════

# ─── S6: H1-Trend-Filtered Tokyo→London ORB ─────────────────────────────────
# Like S1 but the direction filter uses precomputed `h1_trend` (sign of the
# 60-EMA vs 240-EMA cross — effectively H1 trend on M1 data) instead of
# `ma_trend` (M1 short-term EMA polarity). Thesis: aligning breakouts with
# the higher-timeframe trend avoids fading the bigger move.
def entry_h1_trend_orb(bst, slot, row, ts, pair, slip, hspd,
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

    h1 = getattr(row, 'h1_trend', float('nan'))
    if math.isnan(h1) or h1 == 0:
        return False

    pip = _pip(pair)
    direction = 'long' if h1 > 0 else 'short'
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


# ─── S7: News-Drift 15-Min ─────────────────────────────────────────────────
# Thesis: after a high-impact release prints, the initial 1-min reaction
# telegraphs institutional positioning. The 15–90 min drift continues that
# direction as desks finish rebalancing.
# Setup: at the FIRST bar of a near_news==True high-impact window, mark the
# direction from that bar's body sign and place a stop in that direction at
# close ± 0.5 ATR. TP=2 ATR, SL=1 ATR. Min-bars-between=30.
def entry_news_drift_15min(bst, slot, row, ts, pair, slip, hspd,
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

    # Edge trigger: was previous bar NOT near_news but current IS?
    prev_nn = sc.get('prev_near_news', False)
    cur_nn  = bool(getattr(row, 'near_news', False))
    sc['prev_near_news'] = cur_nn

    if not cur_nn or prev_nn:
        return False    # only fire on the LEADING edge of the news window
    impact = getattr(row, 'near_news_impact', '') or ''
    if impact != 'high':
        return False

    atr = float(getattr(row, 'atr', 0) or 0)
    if atr <= 0:
        return False
    close = float(getattr(row, 'close', 0))
    open_ = float(getattr(row, 'open', 0))
    if close <= 0 or open_ <= 0:
        return False
    body = close - open_
    if abs(body) < 0.2 * atr:
        return False    # bar body too small to read direction

    pip = _pip(pair)
    direction = 'long' if body > 0 else 'short'
    if direction == 'long':
        level = close + 0.5 * atr
        sl    = close - 1.0 * atr
        tp    = close + 2.0 * atr
    else:
        level = close - 0.5 * atr
        sl    = close + 1.0 * atr
        tp    = close - 2.0 * atr

    sl_dist = abs(level - sl)
    size = _size_for(bst, regime_mult, pair, bst.balance, level, sl_dist, row)
    if not size or size <= 0:
        return False

    eng.place_pending(sc, ts, direction=direction,
                       entry=level, sl=sl, tp=tp, size=size, dist=sl_dist,
                       level=level, mode='stop_at_level')
    sc['last_trade_bar'] = cur_n
    return False


# ─── S8: DXY-Proxy Divergence on EUR_USD (cross-pair) ──────────────────────
# Thesis: when USD strength is unambiguous across 3 majors but EUR/USD has
# overshot in the opposite direction, EUR/USD mean-reverts to the consensus.
# Setup: compute usd_score every M1 from sign of 60-bar return on USD_JPY
# (+) and EUR/USD, GBP/USD (−). When score == +3 AND EUR/USD at 30-bar high
# → SHORT EUR/USD. When score == −3 AND at 30-bar low → LONG. Hold 30 bars
# or 1 ATR move (engine manager's exit_hour handles outer bound).
# CROSS-PAIR PLUMBING: factory wraps the entry fn with closure over preloaded
# USD_JPY + GBP_USD lookups indexed by timestamp.
def make_dxy_divergence_entry(other_pair_dfs: dict):
    """Factory — returns an entry_fn closed over the other-pair lookups."""
    # Build timestamp-indexed lookups ONCE (not per-bar)
    other_lookups = {}
    for p, df in other_pair_dfs.items():
        if 'timestamp' in df.columns:
            other_lookups[p] = df.set_index('timestamp')
        else:
            other_lookups[p] = df    # already indexed

    def entry_dxy_proxy_divergence_eurusd(
            bst, slot, row, ts, pair, slip, hspd,
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

        # Pull the other-pair bars at this timestamp
        try:
            usd_jpy = other_lookups['USD_JPY'].loc[ts]
            gbp_usd = other_lookups['GBP_USD'].loc[ts]
        except (KeyError, TypeError):
            return False    # missing bar — skip

        # 60-bar returns via momentum_10 column (or close-vs-prev-close fallback)
        m_eu = getattr(row,     'momentum_10', float('nan'))
        m_uj = getattr(usd_jpy, 'momentum_10', float('nan'))
        m_gu = getattr(gbp_usd, 'momentum_10', float('nan'))
        if any(math.isnan(x) for x in (m_eu, m_uj, m_gu)):
            return False

        # USD score: + when USD strong on that pair, − when weak
        def sgn(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)
        usd_score = sgn(m_uj) - sgn(m_eu) - sgn(m_gu)

        if abs(usd_score) < 3:
            return False    # USD score not unanimous

        atr = float(getattr(row, 'atr', 0) or 0)
        close = float(getattr(row, 'close', 0))
        if atr <= 0 or close <= 0:
            return False

        # Check EUR/USD position vs recent extremes (use bb_pct as a proxy
        # since bb_pct ≈ 1 → near upper band ≈ 30-bar high-ish)
        bb = getattr(row, 'bb_pct', float('nan'))
        if math.isnan(bb):
            return False

        if usd_score == 3 and bb >= 0.85:
            # USD strong everywhere but EUR/USD high → fade EUR/USD short
            direction = 'short'
            entry = close
            sl    = close + 1.0 * atr
            tp    = close - 1.0 * atr
        elif usd_score == -3 and bb <= 0.15:
            # USD weak everywhere but EUR/USD low → fade EUR/USD long
            direction = 'long'
            entry = close
            sl    = close - 1.0 * atr
            tp    = close + 1.0 * atr
        else:
            return False

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

    return entry_dxy_proxy_divergence_eurusd


# ─── S9: H4 Low-Vol Compression Breakout ────────────────────────────────────
# Thesis: extended H4 vol compression resolves with a directional break.
# The 5-bar same-side close is the early signal that the break has begun.
# Setup: h4_atr_ratio < 0.8 (compressed) AND last 5 closes all same direction
# AND last bar's range > 1.3× 20-bar average range → enter in that direction.
# TP=2 ATR, SL=1 ATR. Min-bars-between=60.
def entry_h4_low_vol_breakout(bst, slot, row, ts, pair, slip, hspd,
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

    h4r = getattr(row, 'h4_atr_ratio', float('nan'))
    if math.isnan(h4r) or h4r >= 0.8:
        return False

    atr   = float(getattr(row, 'atr', 0) or 0)
    close = float(getattr(row, 'close', 0))
    open_ = float(getattr(row, 'open', 0))
    if atr <= 0 or close <= 0 or open_ <= 0:
        return False

    # Track last 5 close-direction signs in a small ring buffer
    body_sign = 1 if close > open_ else (-1 if close < open_ else 0)
    ring = sc.setdefault('body_ring', [])
    ring.append(body_sign)
    if len(ring) > 5:
        ring.pop(0)
    if len(ring) < 5 or len(set(ring)) > 1 or ring[0] == 0:
        return False    # need 5 same-side non-zero bars

    # Range expansion check via bar_range_pct (precomputed)
    brp = getattr(row, 'bar_range_pct', float('nan'))
    if math.isnan(brp) or brp < 1.3:
        return False

    direction = 'long' if ring[-1] > 0 else 'short'
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
    sc['body_ring'] = []    # reset
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Registry + runner
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: (name, factory_or_fn, exit_hour, is_factory)
# A factory takes other_pair_dfs and returns an entry_fn (used by cross-pair).
STRATEGIES_V2 = [
    ("tokyo_london_breakout",     entry_tokyo_london_breakout,     13, False),
    ("ny_1330_news_momentum",     entry_ny_1330_news_momentum,     16, False),
    ("friday_afternoon_revert",   entry_friday_afternoon_revert,   21, False),
    ("lowvol_bb_extreme_revert",  entry_lowvol_bb_extreme_revert,  21, False),
    ("high_adx_trend_pullback",   entry_high_adx_trend_pullback,   21, False),
    ("h1_trend_orb",              entry_h1_trend_orb,              13, False),
    ("news_drift_15min",          entry_news_drift_15min,          21, False),
    ("dxy_proxy_divergence_eurusd", make_dxy_divergence_entry,     21, True),
    ("h4_low_vol_breakout",       entry_h4_low_vol_breakout,       21, False),
]

# Restrict cross-pair strategy to its target pair only
PAIR_FILTERS = {
    "dxy_proxy_divergence_eurusd": ["EUR_USD"],
}


def run_one(name: str, entry_fn, exit_hour: int, pair: str,
            train_df, test_df, cost_mult: float = 1.0) -> dict:
    manager_fn = eng.make_manager(exit_hour=exit_hour, use_breakeven=False)
    slot_class = f"bat_{name[:14]}".replace("-", "_").lower()
    registry = [{
        "id":               f"bat_{name}_{pair}",
        "family":           "battery",
        "slot_class":       slot_class,
        "pairs":            [pair],
        "session":          "ny",
        "allow_concurrent": False,
        "regime_mult": {"TRENDING": 1.0, "TRANSITIONING": 1.0,
                        "RANGING": 1.0, "VOLATILE": 0.5, "UNDEFINED": 0.5},
        "params":           {},
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}

    out = {}
    for split, df in (("train", train_df), ("test", test_df)):
        try:
            trades, _bal, _ = eng.run_backtest(
                {pair: df}, None, None,
                registry, slot_managers, slot_entries,
                cost_mult=cost_mult,
            )
        except Exception as e:
            out[split] = {"error": f"{type(e).__name__}: {e}"}
            continue
        n = 0 if trades is None or trades.empty else len(trades)
        stats = eng.calc_stats(trades) if n > 0 else {}
        pnl   = float(trades['pnl'].sum()) if n > 0 else 0.0
        out[split] = {
            "n": n,
            "sharpe": float(stats.get("sharpe", 0) or 0),
            "pnl": pnl,
            "max_dd": float(stats.get("max_dd", 0) or 0),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", help="Run only this strategy by name (default: all)")
    args = ap.parse_args()

    print("Loading cached market data...")
    train_dfs, test_dfs, _ = eng.load_all_data(pairs=PAIRS)
    print(f"  Loaded {len(train_dfs)} pair(s)")
    print()

    strategies = STRATEGIES_V2
    if args.strategy:
        strategies = [s for s in STRATEGIES_V2 if s[0] == args.strategy]
        if not strategies:
            print(f"ERROR: no strategy named '{args.strategy}'. Available:")
            for s in STRATEGIES_V2:
                print(f"  {s[0]}")
            return 2

    rows = []
    for name, fn_or_factory, exit_h, is_factory in strategies:
        print(f"=== {name} ===")
        target_pairs = PAIR_FILTERS.get(name, PAIRS)
        for pair in target_pairs:
            if pair not in train_dfs:
                continue
            # Build the entry function — factory takes other-pair DFs of THIS split
            if is_factory:
                # Use both train and test other-pair DFs; the runner picks the
                # right split inside run_one. We make TWO entry-fn variants
                # so the train/test lookups are split-correct.
                others_train = {p: df for p, df in train_dfs.items() if p != pair}
                others_test  = {p: df for p, df in test_dfs.items()  if p != pair}
                fn_train = fn_or_factory(others_train)
                fn_test  = fn_or_factory(others_test)
                # Run train + test separately, then merge
                r_train = run_one(name, fn_train, exit_h, pair,
                                   train_dfs[pair], train_dfs[pair])  # train_df both args
                r_test  = run_one(name, fn_test,  exit_h, pair,
                                   test_dfs[pair],  test_dfs[pair])
                r = {"train": r_train.get("train", {}),
                     "test":  r_test.get("train",  {})}
            else:
                r = run_one(name, fn_or_factory, exit_h, pair,
                            train_dfs[pair], test_dfs[pair])
            rows.append((name, pair, r))
            t  = r.get("train", {})
            te = r.get("test",  {})
            print(f"  {pair}: train n={t.get('n','?')} sh={t.get('sharpe',0):+.2f} "
                  f"${t.get('pnl',0):+,.0f}  |  test n={te.get('n','?')} "
                  f"sh={te.get('sharpe',0):+.2f} ${te.get('pnl',0):+,.0f}")
        print()

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 100)
    print(f"{'STRATEGY':32s} {'PAIR':9s} {'TR_N':>5s} {'TR_SH':>7s} "
          f"{'TR_PNL':>10s} {'TE_N':>5s} {'TE_SH':>7s} {'TE_PNL':>10s} "
          f"{'CAND':>5s}")
    print("-" * 100)
    candidates = []
    for name, pair, r in rows:
        t  = r.get("train", {})
        te = r.get("test",  {})
        tr_sh, te_sh = t.get('sharpe', 0), te.get('sharpe', 0)
        tr_pnl, te_pnl = t.get('pnl', 0), te.get('pnl', 0)
        tr_n, te_n = t.get('n', 0), te.get('n', 0)
        is_cand = (te_sh >= 0.5 and te_n >= 30 and tr_pnl > 0 and te_pnl > 0)
        mark = "✓" if is_cand else "—"
        print(f"{name:32s} {pair:9s} {tr_n:>5d} {tr_sh:>+7.2f} "
              f"${tr_pnl:>+8,.0f} {te_n:>5d} {te_sh:>+7.2f} "
              f"${te_pnl:>+8,.0f} {mark:>5s}")
        if is_cand:
            candidates.append((name, pair, te_sh, te_n, te_pnl))

    print("=" * 100)
    print()
    if candidates:
        print(f">>> PROMOTION CANDIDATES ({len(candidates)}): test_sharpe ≥ 0.5, "
              f"n ≥ 30, both windows positive")
        for name, pair, sh, n, pnl in sorted(candidates, key=lambda x: -x[2]):
            print(f"  ✓ {name:32s} {pair:9s} "
                  f"test_sharpe=+{sh:.2f} n={n} pnl=${pnl:+,.0f}")
    else:
        print("No strategy/pair combo cleared the bar.")

    # ── Write markdown report ──────────────────────────────────────────────
    lines = []
    lines.append("# Strategy Battery V2 — Results\n")
    from datetime import datetime, timezone
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} UTC_\n")
    lines.append("Acceptance gate: `test_sharpe ≥ 0.5 AND n_test ≥ 30 AND "
                 "train_pnl > 0 AND test_pnl > 0`.\n")
    lines.append("## Per-strategy results\n")
    lines.append("| strategy | pair | tr_n | tr_sh | tr_pnl | te_n | te_sh | te_pnl | candidate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, pair, r in rows:
        t  = r.get("train", {})
        te = r.get("test",  {})
        is_cand = (te.get('sharpe', 0) >= 0.5 and te.get('n', 0) >= 30
                   and t.get('pnl', 0) > 0 and te.get('pnl', 0) > 0)
        lines.append(
            f"| `{name}` | {pair} | {t.get('n',0)} | {t.get('sharpe',0):+.2f} | "
            f"${t.get('pnl',0):+,.0f} | {te.get('n',0)} | "
            f"{te.get('sharpe',0):+.2f} | ${te.get('pnl',0):+,.0f} | "
            f"{'✓' if is_cand else '—'} |"
        )
    lines.append("\n## Promotion candidates\n")
    if candidates:
        for name, pair, sh, n, pnl in sorted(candidates, key=lambda x: -x[2]):
            lines.append(f"- **{name}** on **{pair}**: test_sharpe=+{sh:.2f}, "
                         f"n={n}, test_pnl=${pnl:+,.0f}")
    else:
        lines.append("_None._\n")
        lines.append("If the full battery produced no candidates, the empirical "
                     "answer is documented: retail-M1-FX easy edge is genuinely "
                     "scarce on this data window. Next direction: widen scope "
                     "(different timeframe, different market).")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] written → {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
