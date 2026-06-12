"""
Edge-hunt strategy library (H1 FX).

Module-level entry functions so they are importable / picklable by run_sweep's
spawned worker processes (the classic Windows-spawn requirement — closures and
__main__-defined functions can't be unpickled in workers).

Every strategy here is mean-reversion / trend-pullback / session-flow — NOT a
raw price breakout. The prior agent run produced 0 survivors from ~half-breakout
hypotheses (breakout_continuation alone: 49 catastrophic-Sharpe rejects), so the
breakout family is deliberately excluded.

Contract (matches edge_hypotheses.py + agent/generated/*.py):
    entry_fn(bst, slot, row, ts, pair, slip, hspd, sess_cfg, regime,
             regime_mult, fvg_buf=None, day_sweep=None) -> bool
Params arrive via slot['strategy_def']['params'] with at least tp_r, sl_r.
Features are recomputed on H1 bars by tools/edge_hunt.py::resample_h1 (UTC).
"""
from __future__ import annotations

import math

import edge_engine as eng


def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _bar_idx(sc: dict) -> int:
    sc['_n'] = sc.get('_n', 0) + 1
    return sc['_n']


def _common_guard(slot, row, min_gap_bars: int = 3, gap_safe: bool = False):
    """Shared preamble. Returns (sc, n, atr, close) or None to skip this bar.

    gap_safe=True (batch 2+): skip entries on news bars and during the
    21:00-22:00 UTC rollover window, where batch-1 data showed gap-stops
    cluster and cost ~30%+ of total PnL. Strictly removes adverse bars.
    """
    sc = slot['scratch']
    n = _bar_idx(sc)
    params = slot.get('strategy_def', {}).get('params', {})
    if eng.spread_gate(row):
        return None
    if eng.has_pending(sc):
        # Let the engine fill/cancel the resting order; signal handled there.
        return ('pending', sc, n)
    if n - sc.get('last', -10_000) < min_gap_bars:
        return None
    if gap_safe or params.get('gap_safe'):
        if bool(getattr(row, 'near_news', False)):
            return None
        hr = int(getattr(row, 'hour', -1))
        if hr in (21, 22):            # rollover gap window (UTC)
            return None
    # Batch 3: intraday liquid-hours only. Enter 8-16 UTC so the position is
    # held entirely within London/NY liquidity and flattened before rollover
    # (exit_hour=20) — removes overnight/weekend/illiquid gaps, which batch-1/2
    # data showed account for ~84% of all PnL lost.
    if params.get('intraday'):
        hr = int(getattr(row, 'hour', -1))
        if hr < 8 or hr > 16:
            return None
    atr = float(getattr(row, 'atr', 0) or 0)
    close = float(getattr(row, 'close', 0) or 0)
    if atr <= 0 or close <= 0:
        return None
    return (sc, n, atr, close)


def _place(sc, slot, row, ts, pair, bst, regime_mult, direction, entry, sl, tp, n):
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return False
    risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
    size = eng.rv_size(pair, bst.balance, risk, entry, sl_dist, row)
    if not size or size <= 0:
        return False
    eng.place_pending(sc, ts, direction=direction, entry=entry, sl=sl, tp=tp,
                      size=size, dist=sl_dist, mode='market_next_open')
    sc['last'] = n
    return False


# ════════════════════════════════════════════════════════════════════════════
# S1 — RSI(14) mean reversion in a NON-trending regime
# Rationale: in low-trend conditions, short-term overextension (RSI extreme)
# mean-reverts. The edge exists because liquidity providers fade stretched
# moves when there is no trend to sustain them. ADX gate avoids fading real
# trends (the classic way RSI reversion dies).
# ════════════════════════════════════════════════════════════════════════════
def entry_rsi_range_reversion(bst, slot, row, ts, pair, slip, hspd,
                              sess_cfg, regime, regime_mult,
                              fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    rsi = float(getattr(row, 'rsi_14', float('nan')))
    adx = float(getattr(row, 'adx', float('nan')))
    if math.isnan(rsi) or math.isnan(adx):
        return False
    if adx > params.get('adx_max', 22.0):
        return False  # trending — don't fade

    lo_th = params.get('rsi_lo', 30.0)
    hi_th = params.get('rsi_hi', 70.0)
    if rsi <= lo_th:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if rsi >= hi_th:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# S2 — Bollinger-band fade in a NON-trending regime
# Rationale: a close beyond the 2σ band in a range is a statistical
# overextension; reversion to the mean is the edge. ADX gate again excludes
# genuine trends (where bands ride and fading bleeds).
# ════════════════════════════════════════════════════════════════════════════
def entry_bb_fade_range(bst, slot, row, ts, pair, slip, hspd,
                        sess_cfg, regime, regime_mult,
                        fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    bb_up = float(getattr(row, 'bb_up', float('nan')))
    bb_lo = float(getattr(row, 'bb_lo', float('nan')))
    adx = float(getattr(row, 'adx', float('nan')))
    if math.isnan(bb_up) or math.isnan(bb_lo) or math.isnan(adx):
        return False
    if adx > params.get('adx_max', 22.0):
        return False

    if close > bb_up:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    if close < bb_lo:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# S3 — Trend pullback to EMA20 in a TRENDING regime (continuation, not breakout)
# Rationale: established trends persist; entering on a pullback to the fast EMA
# gives a better entry price (smaller stop, larger RR vs cost) than chasing the
# breakout — which is exactly why breakouts die on cost and pullbacks may not.
# ════════════════════════════════════════════════════════════════════════════
def entry_trend_pullback_ema(bst, slot, row, ts, pair, slip, hspd,
                             sess_cfg, regime, regime_mult,
                             fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    ema_f = float(getattr(row, 'ema_fast', float('nan')))
    ema_s = float(getattr(row, 'ema_slow', float('nan')))
    adx = float(getattr(row, 'adx', float('nan')))
    if math.isnan(ema_f) or math.isnan(ema_s) or math.isnan(adx):
        return False
    if adx < params.get('adx_min', 25.0):
        return False  # not trending

    pull = params.get('pullback_atr', 0.3)
    if abs(close - ema_f) > pull * atr:
        return False  # not pulled back to the EMA

    if ema_f > ema_s:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if ema_f < ema_s:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# S4 — Asian-range liquidity sweep fade at the London open
# Rationale: thin Asian session builds a range; the first London hours often
# spike beyond it to run stops, then revert as real two-way liquidity arrives.
# Fade the failed breakout (poke beyond range, close back inside). Classic
# stop-hunt reversion, conditioned on a real session-handoff window.
# ════════════════════════════════════════════════════════════════════════════
def entry_asian_sweep_fade(bst, slot, row, ts, pair, slip, hspd,
                           sess_cfg, regime, regime_mult,
                           fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    hour = int(getattr(row, 'hour', -1))
    if hour < 7 or hour > 10:           # London open window (UTC)
        return False
    ah = float(getattr(row, 'asian_high', float('nan')))
    al = float(getattr(row, 'asian_low', float('nan')))
    high = float(getattr(row, 'high', float('nan')))
    low = float(getattr(row, 'low', float('nan')))
    if any(math.isnan(x) for x in (ah, al, high, low)) or ah <= al:
        return False

    buf = params.get('close_buffer', 0.1) * atr
    # Poked above Asian high but closed back inside → short the failure
    if high > ah and close < ah - buf:
        entry = close
        sl = high + buf
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    # Poked below Asian low but closed back inside → long the failure
    if low < al and close > al + buf:
        entry = close
        sl = low - buf
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# S5 — Prior-day high/low rejection fade
# Rationale: prior-day extremes are reference levels where resting liquidity
# and take-profit orders cluster; first touch often rejects. Fade a wick
# through the level that closes back inside. Mean reversion at a structural level.
# ════════════════════════════════════════════════════════════════════════════
def entry_prevday_level_fade(bst, slot, row, ts, pair, slip, hspd,
                             sess_cfg, regime, regime_mult,
                             fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    pdh = float(getattr(row, 'prev_day_high', float('nan')))
    pdl = float(getattr(row, 'prev_day_low', float('nan')))
    high = float(getattr(row, 'high', float('nan')))
    low = float(getattr(row, 'low', float('nan')))
    if any(math.isnan(x) for x in (pdh, pdl, high, low)):
        return False

    buf = params.get('close_buffer', 0.1) * atr
    if high > pdh and close < pdh - buf:
        entry = close
        sl = high + buf
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    if low < pdl and close > pdl + buf:
        entry = close
        sl = low - buf
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# BATCH 2 — trade WITH momentum (batch-1 data: fades get gap-stopped; the only
# positive-test strategy traded with trend). All gap_safe (skip news+rollover),
# wider stops to survive next-bar gaps. The bar to clear is POSITIVE TRAIN Sharpe.
# ════════════════════════════════════════════════════════════════════════════

# S6 — Trend pullback WITH confirmation candle (continuation)
# Rationale: as S3 but require the entry bar to close back in the trend's
# direction (a resumption signal), not just proximity to the EMA. Filters the
# pullbacks that are actually reversals — the ones that gap-stopped S3.
def entry_trend_pullback_confirmed(bst, slot, row, ts, pair, slip, hspd,
                                   sess_cfg, regime, regime_mult,
                                   fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    ema_f = float(getattr(row, 'ema_fast', float('nan')))
    ema_s = float(getattr(row, 'ema_slow', float('nan')))
    adx = float(getattr(row, 'adx', float('nan')))
    o = float(getattr(row, 'open', float('nan')))
    if any(math.isnan(x) for x in (ema_f, ema_s, adx, o)):
        return False
    if adx < params.get('adx_min', 22.0):
        return False
    pull = params.get('pullback_atr', 0.5)

    if ema_f > ema_s and abs(close - ema_f) <= pull * atr and close > o:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if ema_f < ema_s and abs(close - ema_f) <= pull * atr and close < o:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# S7 — Higher-timeframe (EMA200) trend + RSI pullback entry
# Rationale: a pullback (RSI dip) in the direction of the dominant trend is a
# discount entry that resumes WITH momentum — the opposite of S1's counter-trend
# fade. Aligns the mean-reversion entry with the larger trend.
def entry_htf_rsi_pullback(bst, slot, row, ts, pair, slip, hspd,
                           sess_cfg, regime, regime_mult,
                           fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    ema200 = float(getattr(row, 'ema_200', float('nan')))
    rsi = float(getattr(row, 'rsi_14', float('nan')))
    if math.isnan(ema200) or math.isnan(rsi):
        return False
    dip = params.get('rsi_dip', 40.0)

    if close > ema200 and rsi < dip:                 # uptrend pullback → long
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if close < ema200 and rsi > (100 - dip):         # downtrend pullback → short
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# S8 — Donchian breakout, trend-filtered by EMA200 (continuation, not raw breakout)
# Rationale: a 20-bar breakout taken ONLY in the direction of the dominant
# (EMA200) trend is momentum continuation, not a naive two-sided breakout.
# Wide ATR stop to survive the post-breakout retest gap.
def entry_donchian_trend_follow(bst, slot, row, ts, pair, slip, hspd,
                                sess_cfg, regime, regime_mult,
                                fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    dhi = float(getattr(row, 'donchian_hi', float('nan')))
    dlo = float(getattr(row, 'donchian_lo', float('nan')))
    ema200 = float(getattr(row, 'ema_200', float('nan')))
    if math.isnan(dhi) or math.isnan(dlo) or math.isnan(ema200):
        return False

    if close > dhi and close > ema200:               # up-breakout in uptrend
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if close < dlo and close < ema200:               # down-breakout in downtrend
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# S9 — Momentum persistence (consecutive directional closes + ADX)
# Rationale: short-run autocorrelation — runs of same-direction H1 closes with
# rising trend strength tend to persist one more leg. Pure momentum continuation.
def entry_momentum_persistence(bst, slot, row, ts, pair, slip, hspd,
                               sess_cfg, regime, regime_mult,
                               fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    adx = float(getattr(row, 'adx', float('nan')))
    o = float(getattr(row, 'open', float('nan')))
    if math.isnan(adx) or math.isnan(o):
        return False
    if adx < params.get('adx_min', 20.0):
        return False

    # Track consecutive directional closes in scratch
    up = close > o
    run = sc.get('run', 0)
    sc['run'] = (run + 1) if (up and run >= 0) else (run - 1) if (not up and run <= 0) else (1 if up else -1)
    need = int(params.get('run_len', 3))

    if sc['run'] >= need:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if sc['run'] <= -need:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# BATCH 4 — order-flow & positioning (NOT price patterns). 13 price-only
# strategies were 0/13 train-positive; pivot to the feature class price can't
# see. All intraday + gap_safe (the proven gap fix).
# ════════════════════════════════════════════════════════════════════════════

# S — Order-flow momentum: trade WITH a strong aggressive-volume imbalance
# Rationale: a large hourly net delta (aggressive buyers >> sellers) reflects
# informed/persistent flow that tends to continue for another leg.
def entry_order_flow_momentum(bst, slot, row, ts, pair, slip, hspd,
                              sess_cfg, regime, regime_mult,
                              fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    dz = float(getattr(row, 'delta_z', float('nan')))
    if math.isnan(dz):
        return False
    th = params.get('dz_th', 1.5)
    if dz >= th:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if dz <= -th:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# S — Delta divergence reversal: price extends but flow disagrees
# Rationale: a new local high made on NEGATIVE net delta (selling into strength)
# signals exhaustion; the move is unsupported by flow and tends to reverse.
def entry_delta_divergence(bst, slot, row, ts, pair, slip, hspd,
                           sess_cfg, regime, regime_mult,
                           fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    dz = float(getattr(row, 'delta_z', float('nan')))
    dhi = float(getattr(row, 'donchian_hi', float('nan')))
    dlo = float(getattr(row, 'donchian_lo', float('nan')))
    high = float(getattr(row, 'high', float('nan')))
    low = float(getattr(row, 'low', float('nan')))
    if any(math.isnan(x) for x in (dz, dhi, dlo, high, low)):
        return False
    th = params.get('dz_th', 1.0)
    # New high but net selling → short (exhaustion)
    if high >= dhi and dz <= -th:
        entry = close
        sl = high + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    # New low but net buying → long
    if low <= dlo and dz >= th:
        entry = close
        sl = low - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    return False


# S — Retail contrarian: fade extreme retail positioning (EUR_USD/GBP_USD only)
# Rationale: retail FX traders are a well-documented contrarian indicator —
# when the crowd is heavily long, price tends to fall, and vice versa.
def entry_retail_contrarian(bst, slot, row, ts, pair, slip, hspd,
                            sess_cfg, regime, regime_mult,
                            fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    rl = float(getattr(row, 'retail_long_pct', float('nan')))
    if math.isnan(rl):
        return False
    hi_th = params.get('retail_hi', 0.65)
    lo_th = params.get('retail_lo', 0.35)
    if rl >= hi_th:                  # crowd very long → fade short
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    if rl <= lo_th:                  # crowd very short → fade long
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# BATCH 5 — REFINED order-flow: require flow AND price to AGREE.
# Batch-4 naive flow lost (flow alone insufficient); order_flow was still the
# least-bad family, so the hypothesis is "flow confirms price, not replaces it".
# ════════════════════════════════════════════════════════════════════════════

# S — Flow + dual-trend alignment: delta and BOTH trend filters must agree
def entry_flow_trend_align(bst, slot, row, ts, pair, slip, hspd,
                           sess_cfg, regime, regime_mult,
                           fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    dz = float(getattr(row, 'delta_z', float('nan')))
    ema_f = float(getattr(row, 'ema_fast', float('nan')))
    ema_s = float(getattr(row, 'ema_slow', float('nan')))
    ema200 = float(getattr(row, 'ema_200', float('nan')))
    if any(math.isnan(x) for x in (dz, ema_f, ema_s, ema200)):
        return False
    th = params.get('dz_th', 1.0)

    if dz >= th and ema_f > ema_s and close > ema200:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if dz <= -th and ema_f < ema_s and close < ema200:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# S — Flow persistence: N consecutive same-sign delta bars + trend filter
def entry_flow_persistence(bst, slot, row, ts, pair, slip, hspd,
                           sess_cfg, regime, regime_mult,
                           fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    delta = float(getattr(row, 'delta', float('nan')))
    ema200 = float(getattr(row, 'ema_200', float('nan')))
    if math.isnan(delta) or math.isnan(ema200):
        return False
    pos = delta > 0
    run = sc.get('frun', 0)
    sc['frun'] = (run + 1) if (pos and run >= 0) else (run - 1) if (not pos and run <= 0) else (1 if pos else -1)
    need = int(params.get('run_len', 3))

    if sc['frun'] >= need and close > ema200:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if sc['frun'] <= -need and close < ema200:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# S — Cumulative-delta momentum: 6h net flow slope agrees with EMA200 trend
def entry_cum_delta_momentum(bst, slot, row, ts, pair, slip, hspd,
                             sess_cfg, regime, regime_mult,
                             fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, gap_safe=True)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    cds = float(getattr(row, 'cd_slope6', float('nan')))
    ema200 = float(getattr(row, 'ema_200', float('nan')))
    if math.isnan(cds) or math.isnan(ema200):
        return False

    if cds > 0 and close > ema200:
        entry = close
        sl = close - params['sl_r'] * atr
        tp = close + params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'long', entry, sl, tp, n)
    if cds < 0 and close < ema200:
        entry = close
        sl = close + params['sl_r'] * atr
        tp = close - params['tp_r'] * atr
        return _place(sc, slot, row, ts, pair, bst, regime_mult, 'short', entry, sl, tp, n)
    return False


# ════════════════════════════════════════════════════════════════════════════
# BATCH 6 — COT positioning (REAL data now joined, lookahead-safe).
# One decision/day at 08:00 UTC (COT is a weekly signal, not intraday). Tests
# whether speculator-positioning extremes predict daily direction.
# ════════════════════════════════════════════════════════════════════════════
def entry_cot_extreme(bst, slot, row, ts, pair, slip, hspd,
                      sess_cfg, regime, regime_mult,
                      fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    daily = bool(params.get('daily'))
    # H1: ~1 trade/day (min_gap 20 bars) + single 08:00 decision.
    # D1: every bar is a day; min_gap 1, no hour gate, hold runs to SL/TP.
    g = _common_guard(slot, row, min_gap_bars=1 if daily else 20)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    if not daily and int(getattr(row, 'hour', -1)) != 8:   # H1: one decision/day
        return False
    ci = float(getattr(row, 'cot_index', float('nan')))
    if math.isnan(ci):
        return False
    hi = params.get('ci_hi', 80.0)
    lo = params.get('ci_lo', 20.0)
    # contrarian=+1: fade crowded specs (extreme long -> short). =-1: follow.
    sign = params.get('contrarian', 1.0)

    if ci >= hi:
        direction = 'short' if sign > 0 else 'long'
    elif ci <= lo:
        direction = 'long' if sign > 0 else 'short'
    else:
        return False

    if direction == 'long':
        entry = close; sl = close - params['sl_r'] * atr; tp = close + params['tp_r'] * atr
    else:
        entry = close; sl = close + params['sl_r'] * atr; tp = close - params['tp_r'] * atr
    return _place(sc, slot, row, ts, pair, bst, regime_mult, direction, entry, sl, tp, n)


# ════════════════════════════════════════════════════════════════════════════
# BATCH 8 — risk sentiment (VIX). Free Yahoo data. Spot-directional for
# risk-sensitive FX: risk-off (VIX up) bids safe-haven JPY and sells risk
# currencies (AUD); risk-on the reverse. One decision/day, multi-day hold.
# RISK_BETA[pair] = sign of the pair's move when risk is ON (VIX falling).
# ════════════════════════════════════════════════════════════════════════════
RISK_BETA = {"AUD_USD": +1, "USD_JPY": +1, "EUR_JPY": +1,
             "GBP_USD": +1, "EUR_USD": +1}


def entry_vix_risk(bst, slot, row, ts, pair, slip, hspd,
                   sess_cfg, regime, regime_mult,
                   fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, min_gap_bars=20)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    if int(getattr(row, 'hour', -1)) != 8:          # one decision/day
        return False
    vz = float(getattr(row, 'vix_z', float('nan')))
    if math.isnan(vz):
        return False
    beta = RISK_BETA.get(pair, 0)
    if beta == 0:
        return False
    # sign=+1: contrarian (high VIX -> bet on risk recovery -> long risk pair).
    # sign=-1: momentum (high VIX -> risk-off continues -> short risk pair).
    sign = params.get('sign', 1.0)
    hi = params.get('vz_hi', 1.0)
    lo = params.get('vz_lo', -1.0)
    if vz >= hi:
        bias = +1 * sign
    elif vz <= lo:
        bias = -1 * sign
    else:
        return False

    direction = 'long' if bias * beta > 0 else 'short'
    if direction == 'long':
        entry = close; sl = close - params['sl_r'] * atr; tp = close + params['tp_r'] * atr
    else:
        entry = close; sl = close + params['sl_r'] * atr; tp = close - params['tp_r'] * atr
    return _place(sc, slot, row, ts, pair, bst, regime_mult, direction, entry, sl, tp, n)


# ════════════════════════════════════════════════════════════════════════════
# BATCH 9 — CARRY (real OECD rates). Hold the positive-carry direction to earn
# the rate differential (swap), exposed to spot. Evaluated with swap income
# modelled (the backtest's blind spot). One entry/day, multi-day hold.
# ════════════════════════════════════════════════════════════════════════════
def entry_carry(bst, slot, row, ts, pair, slip, hspd,
                sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    g = _common_guard(slot, row, min_gap_bars=20)
    if g is None:
        return False
    if g[0] == 'pending':
        _, sc, n = g
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    sc, n, atr, close = g

    if int(getattr(row, 'hour', -1)) != 8:
        return False
    cd = float(getattr(row, 'carry_diff', float('nan')))
    if math.isnan(cd):
        return False
    th = params.get('carry_th', 1.0)          # min annual % differential
    if abs(cd) < th:
        return False
    # sign=+1: carry (long the higher-yielder). =-1: anti-carry (test reversal).
    sign = params.get('sign', 1.0)
    direction = 'long' if cd * sign > 0 else 'short'
    if direction == 'long':
        entry = close; sl = close - params['sl_r'] * atr; tp = close + params['tp_r'] * atr
    else:
        entry = close; sl = close + params['sl_r'] * atr; tp = close - params['tp_r'] * atr
    return _place(sc, slot, row, ts, pair, bst, regime_mult, direction, entry, sl, tp, n)


# Registry consumed by tools/edge_hunt.py. Each entry: economic rationale +
# session + which extra grid params to sweep beyond tp_r/sl_r.
_WIDE_SL = [1.5, 2.5, 3.5]   # batch-2 stops: wide enough to survive next-bar gaps

STRATEGIES = [
    {
        "name": "rsi_range_reversion", "family": "mean_reversion", "batch": 1,
        "fn": entry_rsi_range_reversion, "session": "all", "exit_hour": 23,
        "extra_grid": {"adx_max": [18.0, 25.0]},
        "rationale": "Fade RSI extremes only when ADX shows no trend; overextension reverts in ranges.",
    },
    {
        "name": "bb_fade_range", "family": "mean_reversion", "batch": 1,
        "fn": entry_bb_fade_range, "session": "all", "exit_hour": 23,
        "extra_grid": {"adx_max": [18.0, 25.0]},
        "rationale": "Fade 2-sigma band breaks in non-trending regimes; statistical reversion to mean.",
    },
    {
        "name": "trend_pullback_ema", "family": "trend_pullback", "batch": 1,
        "fn": entry_trend_pullback_ema, "session": "all", "exit_hour": 23,
        "extra_grid": {"adx_min": [22.0, 30.0]},
        "rationale": "Buy pullbacks to EMA20 in strong trends; better RR-vs-cost than chasing breakouts.",
    },
    {
        "name": "asian_sweep_fade", "family": "liquidity_sweep", "batch": 1,
        "fn": entry_asian_sweep_fade, "session": "london", "exit_hour": 16,
        "extra_grid": {"close_buffer": [0.05, 0.15]},
        "rationale": "Fade London-open stop-runs beyond the Asian range that close back inside.",
    },
    {
        "name": "prevday_level_fade", "family": "level_reversion", "batch": 1,
        "fn": entry_prevday_level_fade, "session": "all", "exit_hour": 23,
        "extra_grid": {"close_buffer": [0.05, 0.15]},
        "rationale": "Fade first-touch rejections of prior-day high/low where liquidity clusters.",
    },
    # ── Batch 2: momentum-aligned, gap-gated, wide stops ────────────────────
    {
        "name": "trend_pullback_confirmed", "family": "trend_pullback", "batch": 2,
        "fn": entry_trend_pullback_confirmed, "session": "all", "exit_hour": 23,
        "extra_grid": {"sl_r": _WIDE_SL, "adx_min": [22.0, 30.0]},
        "rationale": "Pullback to EMA in trend WITH a confirmation close in trend direction; filters reversals that gap-stopped batch 1.",
    },
    {
        "name": "htf_rsi_pullback", "family": "trend_pullback", "batch": 2,
        "fn": entry_htf_rsi_pullback, "session": "all", "exit_hour": 23,
        "extra_grid": {"sl_r": _WIDE_SL, "rsi_dip": [35.0, 45.0]},
        "rationale": "RSI dip in the direction of the EMA200 trend — discount entry that resumes WITH momentum (opposite of the fade that failed).",
    },
    {
        "name": "donchian_trend_follow", "family": "trend_continuation", "batch": 2,
        "fn": entry_donchian_trend_follow, "session": "all", "exit_hour": 23,
        "extra_grid": {"sl_r": _WIDE_SL},
        "rationale": "20-bar breakout taken only in the EMA200 trend direction; momentum continuation, not naive two-sided breakout.",
    },
    {
        "name": "momentum_persistence", "family": "trend_continuation", "batch": 2,
        "fn": entry_momentum_persistence, "session": "all", "exit_hour": 23,
        "extra_grid": {"sl_r": _WIDE_SL, "run_len": [3.0, 4.0]},
        "rationale": "Runs of same-direction H1 closes with rising ADX tend to extend one more leg (short-run autocorrelation).",
    },
    # ── Batch 3: same momentum logic, INTRADAY liquid-hours only (8-16 UTC),
    #    flat by 20:00 — directly attacks the 84%-of-losses gap problem ───────
    {
        "name": "momentum_persistence_intraday", "family": "trend_continuation", "batch": 3,
        "fn": entry_momentum_persistence, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "run_len": [3.0, 4.0], "intraday": [1.0]},
        "rationale": "Momentum persistence, but enter 8-16 UTC and flat by 20:00 — removes overnight/illiquid gaps (84% of batch-1/2 losses).",
    },
    {
        "name": "donchian_trend_intraday", "family": "trend_continuation", "batch": 3,
        "fn": entry_donchian_trend_follow, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "intraday": [1.0]},
        "rationale": "EMA200-filtered Donchian continuation, intraday liquid-hours only, flat before rollover.",
    },
    {
        "name": "htf_rsi_pullback_intraday", "family": "trend_pullback", "batch": 3,
        "fn": entry_htf_rsi_pullback, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "rsi_dip": [35.0, 45.0], "intraday": [1.0]},
        "rationale": "EMA200-trend RSI-pullback, intraday liquid-hours only — momentum-aligned entry with gaps removed.",
    },
    {
        "name": "trend_pullback_confirmed_intraday", "family": "trend_pullback", "batch": 3,
        "fn": entry_trend_pullback_confirmed, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "adx_min": [22.0, 30.0], "intraday": [1.0]},
        "rationale": "Confirmed trend-pullback, intraday liquid-hours only, flat before rollover.",
    },
    # ── Batch 4: order-flow & positioning (non-price feature class) ──────────
    {
        "name": "order_flow_momentum", "family": "order_flow", "batch": 4,
        "fn": entry_order_flow_momentum, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "dz_th": [1.5, 2.5], "intraday": [1.0]},
        "rationale": "Trade WITH a strong hourly aggressive-volume imbalance (delta z-score); informed flow continues.",
    },
    {
        "name": "delta_divergence", "family": "order_flow", "batch": 4,
        "fn": entry_delta_divergence, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "dz_th": [1.0, 1.5], "intraday": [1.0]},
        "rationale": "New price extreme made on opposing net delta = exhaustion; fade the unsupported move.",
    },
    {
        "name": "retail_contrarian", "family": "positioning", "batch": 4,
        "fn": entry_retail_contrarian, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "retail_hi": [0.60, 0.70], "intraday": [1.0]},
        "rationale": "Fade extreme retail long/short positioning (EUR_USD/GBP_USD) — crowd is contrarian.",
    },
    # ── Batch 5: refined order-flow — flow AND price must agree ──────────────
    {
        "name": "flow_trend_align", "family": "order_flow", "batch": 5,
        "fn": entry_flow_trend_align, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "dz_th": [0.8, 1.5], "intraday": [1.0]},
        "rationale": "Enter only when hourly delta AND EMA20>EMA50 AND price>EMA200 all agree — flow confirms trend.",
    },
    {
        "name": "flow_persistence", "family": "order_flow", "batch": 5,
        "fn": entry_flow_persistence, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "run_len": [3.0, 4.0], "intraday": [1.0]},
        "rationale": "N consecutive same-sign delta bars in the EMA200 trend direction — sustained informed flow.",
    },
    {
        "name": "cum_delta_momentum", "family": "order_flow", "batch": 5,
        "fn": entry_cum_delta_momentum, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": _WIDE_SL, "intraday": [1.0]},
        "rationale": "6-hour cumulative-delta slope agreeing with the EMA200 trend — sustained directional flow.",
    },
    # ── Batch 6: COT positioning (real CFTC data) ───────────────────────────
    {
        "name": "cot_contrarian", "family": "positioning", "batch": 6,
        "fn": entry_cot_extreme, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": [2.0, 3.5], "tp_r": [2.5, 4.0],
                       "ci_hi": [80.0], "ci_lo": [20.0], "contrarian": [1.0]},
        "rationale": "Fade speculator-positioning extremes (COT index >=80 short / <=20 long) — crowd caught at extremes.",
    },
    {
        "name": "cot_follow", "family": "positioning", "batch": 6,
        "fn": entry_cot_extreme, "session": "all", "exit_hour": 20,
        "extra_grid": {"sl_r": [2.0, 3.5], "tp_r": [2.5, 4.0],
                       "ci_hi": [80.0], "ci_lo": [20.0], "contrarian": [-1.0]},
        "rationale": "Follow speculator positioning (COT index >=80 long / <=20 short) — test trend-persistence direction.",
    },
    # ── Batch 7: COT contrarian, H1 bars, NO daily force-exit (exit_hour=99)
    #    so positions hold multi-day to SL/TP (weekend-flatten caps ~1 week) —
    #    matches the weekly signal horizon while avoiding the D1 reset bug ────
    {
        "name": "cot_swing_contrarian", "family": "positioning", "batch": 7,
        "fn": entry_cot_extreme, "session": "all", "exit_hour": 99,
        "extra_grid": {"sl_r": [2.0, 3.5], "tp_r": [3.0, 5.0],
                       "ci_hi": [70.0, 80.0], "ci_lo": [30.0, 20.0],
                       "contrarian": [1.0]},
        "rationale": "Fade COT speculator extremes (one entry/day at 08:00), holding multi-day to SL/TP with no intraday force-exit — matches the weekly COT horizon.",
    },
    # ── Batch 8: VIX risk sentiment (free Yahoo data), multi-day holds ───────
    {
        "name": "vix_risk_contrarian", "family": "risk_sentiment", "batch": 8,
        "fn": entry_vix_risk, "session": "all", "exit_hour": 99,
        "extra_grid": {"sl_r": [2.0, 3.5], "tp_r": [3.0, 5.0],
                       "vz_hi": [1.0, 1.5], "vz_lo": [-1.0], "sign": [1.0]},
        "rationale": "Fade VIX extremes — high VIX = fear overshoot, bet on risk recovery (long risk pairs); low VIX = complacency (short).",
    },
    {
        "name": "vix_risk_momentum", "family": "risk_sentiment", "batch": 8,
        "fn": entry_vix_risk, "session": "all", "exit_hour": 99,
        "extra_grid": {"sl_r": [2.0, 3.5], "tp_r": [3.0, 5.0],
                       "vz_hi": [1.0, 1.5], "vz_lo": [-1.0], "sign": [-1.0]},
        "rationale": "Follow VIX — rising fear (high VIX) = risk-off continues (short risk pairs); falling VIX = risk-on (long).",
    },
]
