# -*- coding: utf-8 -*-
"""
Edge Discovery Engine — Dukascopy Tick Data

Infrastructure layer: data fetching, tick processing, risk management,
backtest loop, reporting, hypothesis testing, Monte Carlo.

No strategies are defined here. Strategies are defined in edge_hypotheses.py
and registered via register_strategy() / run_sweep().

Usage:
    from edge_engine import (
        load_all_data, run_sweep, ParameterGrid,
        make_manager, spread_gate, rv_size, check_and_fill,
        load_sweep_results, load_hypothesis_trades,
        NY_PAIRS, ASIAN_PAIRS, SESSION_CONFIG, PAIR_PIP_SIZE,
    )
"""

import io
import itertools
import json
import lzma
import math
import pickle
import sqlite3
import struct
import tempfile
import time
import warnings
import concurrent.futures
import multiprocessing as mp
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# 1. DUKASCOPY CONFIGURATION
# =============================================================================

DUKA_BASE = "https://datafeed.dukascopy.com/datafeed"

DUKA_INST = {
    'GBP_USD': 'GBPUSD', 'GBP_JPY': 'GBPJPY', 'EUR_USD': 'EURUSD',
    'USD_JPY': 'USDJPY', 'EUR_JPY': 'EURJPY', 'AUD_USD': 'AUDUSD',
    'XAU_USD': 'XAUUSD',
    'USD_CAD': 'USDCAD', 'NZD_USD': 'NZDUSD', 'AUD_JPY': 'AUDJPY',
    'EUR_GBP': 'EURGBP', 'USD_CHF': 'USDCHF', 'CHF_JPY': 'CHFJPY',
}
DUKA_POINT = {
    'GBPUSD': 1e5, 'GBPJPY': 1e3, 'EURUSD': 1e5,
    'USDJPY': 1e3, 'EURJPY': 1e3, 'AUDUSD': 1e5, 'XAUUSD': 1e5,
    'USDCAD': 1e5, 'NZDUSD': 1e5, 'AUDJPY': 1e3,
    'EURGBP': 1e5, 'USDCHF': 1e5, 'CHFJPY': 1e3,
}

TICK_FMT   = '>IIIff'
TICK_BYTES = struct.calcsize(TICK_FMT)

CACHE_DIR          = Path("duka_cache")
PREPARED_CACHE_DIR = Path("edge_prepared_cache")
DB_PATH            = Path("edge_results.db")
SPREADS_JSON       = Path("edge_measured_spreads.json")

CACHE_DIR.mkdir(exist_ok=True)
PREPARED_CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# 2. CONSTANTS
# =============================================================================

INITIAL_BALANCE  = 100_000
RISK_BASE        = 0.004
RISK_MAX         = 0.006
RISK_MIN         = 0.002
RISK_WIN_STEP    = 0.001
RISK_LOSS_STEP   = 0.001

REGIME_ADX_TREND    = 25
REGIME_ADX_RANGE    = 20
REGIME_ATR_VOLATILE = 1.5
REGIME_ADX_PERIOD   = 14
REGIME_ATR_PERIOD   = 14
REGIME_ATR_REF_BARS = 30

NEWS_BUFFER_MINS    = 15

# Phase 6.3 — impact-aware buffer windows. ForexFactory tags each event as
# High / Medium / Low; these dictate how wide a window we sit out around it.
# Only 'high' is used by default in the lookups below; medium/low fold in if
# the calendar source supplies them.
NEWS_BUFFER_BY_IMPACT = {
    'high':   30,
    'medium': 10,
    'low':     0,
}
ASIAN_HOUR_START    = 0
ASIAN_HOUR_END      = 2

PROP_DAILY_LOSS_LIMIT   = 0.05
PROP_MAX_DRAWDOWN_LIMIT = 0.06
PROP_PROFIT_TARGET      = 0.06

TRAIN_DAYS = 548
TEST_DAYS  = 182
TOTAL_DAYS = TRAIN_DAYS + TEST_DAYS

ALL_PAIRS    = [
    'GBP_USD', 'GBP_JPY', 'EUR_USD', 'USD_JPY', 'EUR_JPY', 'AUD_USD',
    'XAU_USD', 'USD_CAD', 'NZD_USD', 'AUD_JPY', 'EUR_GBP', 'USD_CHF', 'CHF_JPY',
]
LONDON_PAIRS = ['GBP_USD', 'GBP_JPY', 'EUR_JPY', 'AUD_USD',
                'EUR_GBP', 'USD_CHF', 'CHF_JPY']
NY_PAIRS     = ['EUR_USD', 'USD_JPY', 'USD_CAD', 'XAU_USD']
ASIAN_PAIRS  = ['USD_JPY', 'EUR_JPY', 'AUD_USD', 'AUD_JPY', 'NZD_USD']

PAIR_SESSION = {
    'GBP_USD': 'london', 'GBP_JPY': 'london', 'EUR_USD': 'ny',
    'USD_JPY': 'ny',     'EUR_JPY': 'london',  'AUD_USD': 'london',
    'XAU_USD': 'ny',
    'USD_CAD': 'ny',     'NZD_USD': 'asian',   'AUD_JPY': 'asian',
    'EUR_GBP': 'london', 'USD_CHF': 'london',  'CHF_JPY': 'london',
}
PAIR_LABELS = {
    'GBP_USD': 'GBP/USD', 'GBP_JPY': 'GBP/JPY', 'EUR_USD': 'EUR/USD',
    'USD_JPY': 'USD/JPY', 'EUR_JPY': 'EUR/JPY',  'AUD_USD': 'AUD/USD',
    'XAU_USD': 'XAU/USD',
    'USD_CAD': 'USD/CAD', 'NZD_USD': 'NZD/USD', 'AUD_JPY': 'AUD/JPY',
    'EUR_GBP': 'EUR/GBP', 'USD_CHF': 'USD/CHF', 'CHF_JPY': 'CHF/JPY',
}
PAIR_PIP_SIZE = {
    'GBP_USD': 0.0001, 'GBP_JPY': 0.010, 'EUR_USD': 0.0001,
    'USD_JPY': 0.010,  'EUR_JPY': 0.010,  'AUD_USD': 0.0001,
    'XAU_USD': 1.00,
    'USD_CAD': 0.0001, 'NZD_USD': 0.0001, 'AUD_JPY': 0.010,
    'EUR_GBP': 0.0001, 'USD_CHF': 0.0001, 'CHF_JPY': 0.010,
}
PAIR_MIN_RANGE_PIPS = {
    'GBP_USD': 3, 'GBP_JPY': 3, 'EUR_USD': 3,
    'USD_JPY': 3, 'EUR_JPY': 3, 'AUD_USD': 3, 'XAU_USD': 5,
    'USD_CAD': 3, 'NZD_USD': 3, 'AUD_JPY': 3,
    'EUR_GBP': 3, 'USD_CHF': 3, 'CHF_JPY': 3,
}
PAIR_MAX_RANGE_PIPS = {
    'GBP_USD': 30, 'GBP_JPY': 30, 'EUR_USD': 30,
    'USD_JPY': 30, 'EUR_JPY': 30, 'AUD_USD': 30, 'XAU_USD': 50,
    'USD_CAD': 30, 'NZD_USD': 30, 'AUD_JPY': 30,
    'EUR_GBP': 25, 'USD_CHF': 30, 'CHF_JPY': 30,
}
MIN_RANGE_BARS = 20

SESSION_CONFIG = {
    'london': {
        'active_start': 8, 'active_end': 13,
        'range_hour': 8, 'range_min_start': 0, 'range_min_end': 29,
        'entry_after': (8, 30), 'entry_until': (11, 0), 'exit_time': (13, 0),
        'high_impact_times': [(7, 0), (9, 0), (9, 30), (10, 0)],
    },
    'ny': {
        'active_start': 14, 'active_end': 21,
        'range_hour': 14, 'range_min_start': 30, 'range_min_end': 59,
        'entry_after': (15, 0), 'entry_until': (21, 0), 'exit_time': (21, 0),
        'high_impact_times': [(13, 30), (15, 0), (15, 30)],
    },
    'asian': {
        'active_start': 0, 'active_end': 8,
        'range_hour': 0, 'range_min_start': 0, 'range_min_end': 59,
        'entry_after': (8, 0), 'entry_until': (10, 0), 'exit_time': (13, 0),
        'high_impact_times': [],
    },
}

SPREAD_GATE_MULT     = 2.0
TICK_IMB_THRESHOLD   = 0.15
RV_SCALE_WINDOW      = 60
RV_SCALE_CAP         = 2.0
SPREAD_MEDIAN_WINDOW = 120

# ── Realistic fill model (Phase 2) ───────────────────────────────────────────
# review#P2#9 — REALISTIC_FILLS flag was previously settable but only gated
# 2 of ~8 fill paths (entry/exit fills, stop-out fills always applied
# realism regardless). The flag was a footgun: setting it False didn't
# actually disable realism, just inconsistently halved it. Always-on now.
# Kept as a True alias so any external imports (mission_control, tests)
# keep working — but flipping to False has no effect.
REALISTIC_FILLS         = True
BROKER_STOPS_LEVEL_PIPS = 2.0     # MT5 trade_stops_level — pending levels closer than this to bar-open are rejected
WEEKEND_FLATTEN         = True    # close any position that survives into the weekend gap (priced at Sunday open)
SWAP_PIPS_PER_NIGHT     = {       # broker overnight swap, pips per side (negative = cost)
    'GBP_USD': {'long': -0.40, 'short':  0.20},
    'GBP_JPY': {'long':  0.30, 'short': -0.50},
    'EUR_USD': {'long': -0.35, 'short':  0.15},
    'USD_JPY': {'long':  0.25, 'short': -0.45},
    'EUR_JPY': {'long': -0.10, 'short': -0.10},
    'AUD_USD': {'long': -0.30, 'short':  0.10},
    'XAU_USD': {'long': -0.50, 'short': -0.50},
    'USD_CAD': {'long': -0.20, 'short':  0.10},
    'NZD_USD': {'long':  0.10, 'short': -0.30},
    'AUD_JPY': {'long':  0.40, 'short': -0.60},
    'EUR_GBP': {'long': -0.05, 'short': -0.05},
    'USD_CHF': {'long':  0.20, 'short': -0.40},
    'CHF_JPY': {'long':  0.05, 'short': -0.15},
}
SLIPPAGE_PROFILE = {              # multiplier on half-spread to derive slip, by UTC-hour bucket
    'rollover':    1.50,          # 21–23 UTC: brokers silently widen
    'asian':       0.80,          # 0–7 UTC: thin liquidity
    'london_open': 0.05,          # 7–9 UTC: deepest liquidity
    'london_ny':   0.15,          # 9–13 UTC: overlap
    'ny_open':     0.10,          # 13–15 UTC
    'ny_late':     0.30,          # 15–21 UTC
    'default':     0.20,
}
# review#12 — extra slippage multiplier on news bars, applied on TOP of the
# session-hour multiplier. NEWS_SPREAD_MULT (in agent.config) widens the
# spread; this widens the slippage independently. Lowered 3.0 → 2.0 after
# the 2026-06-02 cost-stack recalibration: 3.0× was a worst-case anchor that
# made every news-adjacent strategy reject. 2.0× still pessimistic relative
# to a well-timed retail entry but less brutal across the agent's grid.
NEWS_SLIPPAGE_MULT = 2.0

# Static-spread-mode slippage as fraction of (scaled) spread. Was hardcoded
# at 0.20 in run_sweep; extracted here on 2026-06-02 so callers can override.
# 0.10 is conservative for stop orders at non-news hours; raise toward 0.20
# once live TCA data shows actual fill quality.
SLIP_RATIO_STATIC = 0.10

HMM_N_STATES = 4
HMM_FEATURES = [
    'tick_imbalance', 'vol_imbalance', 'realized_vol', 'atr_ratio',
    'adx', 'delta_momentum', 'persistent_imbalance', 'bar_range_pct',
]

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def dynamic_risk(consecutive_wins, consecutive_losses):
    if consecutive_wins > 0:
        return min(RISK_MAX, RISK_BASE + consecutive_wins * RISK_WIN_STEP)
    if consecutive_losses > 0:
        return max(RISK_MIN, RISK_BASE - consecutive_losses * RISK_LOSS_STEP)
    return RISK_BASE


def compute_hurst(series: pd.Series, min_window: int = 10, max_window: int = 100) -> float:
    """
    Rescaled-range (R/S) Hurst exponent estimate over `series`.

    Returns:
        H ≈ 0.5 → random walk (no edge in trend or fade)
        H > 0.5 → trending (persistent — breakout strategies favoured)
        H < 0.5 → mean-reverting (anti-persistent — fade strategies favoured)

    For a 200-bar window this is fast (~0.5ms). NaN if series has zero variance
    or insufficient length.
    """
    s = pd.Series(series).dropna().values
    n = len(s)
    if n < max_window * 2:
        return float('nan')
    rets = np.diff(np.log(np.clip(s, 1e-12, None)))
    if rets.size < max_window:
        return float('nan')
    lags = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 10).astype(int))
    rs   = []
    for lag in lags:
        if lag < 4 or lag > rets.size:
            continue
        chunks = rets.size // lag
        if chunks < 2:
            continue
        cuts = rets[: chunks * lag].reshape(chunks, lag)
        # rescaled range per chunk
        mean = cuts.mean(axis=1, keepdims=True)
        dev  = (cuts - mean).cumsum(axis=1)
        rng  = dev.max(axis=1) - dev.min(axis=1)
        sd   = cuts.std(axis=1)
        valid = sd > 0
        if not valid.any():
            continue
        rs.append((lag, np.mean(rng[valid] / sd[valid])))
    if len(rs) < 4:
        return float('nan')
    log_lag = np.log([r[0] for r in rs])
    log_rs  = np.log([r[1] for r in rs])
    slope   = np.polyfit(log_lag, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(close: pd.Series, window: int = 200) -> pd.Series:
    """Rolling-window Hurst exponent. NaN until enough bars have accumulated."""
    return close.rolling(window).apply(
        lambda x: compute_hurst(x, min_window=8, max_window=window // 2),
        raw=False,
    )


def compute_permutation_entropy(
    series, order: int = 4, normalize: bool = True
) -> float:
    """
    Bandt-Pompe permutation entropy of a 1-D sequence.

    Strictly orthogonal to Hurst:
      * Hurst measures self-similarity / persistence of magnitudes.
      * Permutation entropy measures complexity of the *ordinal pattern* sequence —
        whether successive returns follow recognisable shapes (low entropy) or
        look like noise (high entropy).
    Low PE   → exploitable structure (predictable ordinal patterns).
    High PE  → near-random; avoid.

    Returns normalised entropy ∈ [0, 1].  NaN on insufficient data.
    """
    s = np.asarray(pd.Series(series).dropna().values, dtype=float)
    n = len(s)
    if n < order + 1:
        return float('nan')

    perms = {}
    for i in range(n - order + 1):
        window = s[i:i + order]
        # ordinal pattern = argsort permutation, encoded as a tuple of ranks
        pat = tuple(np.argsort(window, kind='mergesort'))
        perms[pat] = perms.get(pat, 0) + 1

    counts = np.array(list(perms.values()), dtype=float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    if normalize:
        H_max = np.log(math.factorial(order))
        if H_max <= 0:
            return float('nan')
        return float(H / H_max)
    return float(H)


def rolling_perm_entropy(
    close: pd.Series, window: int = 100, order: int = 4
) -> pd.Series:
    """Rolling permutation entropy over log-returns of `close`."""
    rets = np.log(close.clip(lower=1e-12)).diff()
    return rets.rolling(window).apply(
        lambda x: compute_permutation_entropy(x, order=order, normalize=True),
        raw=False,
    )


def compute_hawkes_intensity(
    events: pd.Series, decay: float = 0.05, baseline_window: int = 240
) -> pd.Series:
    """
    Self-exciting Hawkes process intensity over a binary event series.

    Model:
        λ(t) = μ + Σ_{t_i < t} α · exp(-decay · (t - t_i))

    Implemented online with a single exponential moving sum so it's O(N).
    The output is the *ratio* of current intensity over a rolling baseline
    intensity, yielding a unit-free feature where:
      ratio > 2 → flow events are clustering (2× baseline rate)
      ratio < 0.5 → unusually quiet (drought of events)

    `events` must be a binary Series indexed like the source df. A typical
    use is `events = (tick_imbalance.abs() > 0.5).astype(float)` — strong
    directional ticks. Decay is per-bar (0.05 = half-life ≈ 14 bars).
    """
    e = events.astype(float).fillna(0.0).values
    n = len(e)
    if n == 0:
        return pd.Series(dtype=float, index=events.index)

    # Online exponential intensity
    intensity = np.zeros(n)
    decay_factor = math.exp(-decay)
    accum = 0.0
    for i in range(n):
        accum = accum * decay_factor + e[i]
        intensity[i] = accum

    out = pd.Series(intensity, index=events.index)
    base = out.rolling(baseline_window, min_periods=baseline_window // 4).mean()
    ratio = (out / base.replace(0, np.nan)).clip(0, 20).fillna(1.0)
    return ratio


def compute_yang_zhang_vol(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Yang-Zhang realized volatility — captures overnight gaps + intraday range.
    More efficient (lower estimator variance) than close-to-close stdev for OHLC bars.

    Returns annualised vol per bar (so a 1-min bar gives a per-minute vol; multiply
    by sqrt(bars_per_year) externally if you want an annualised number).
    """
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_co = np.log(c / o)
    log_oc_prev = np.log(o / c.shift(1))

    # Overnight (close-to-open) variance
    sigma_o2 = log_oc_prev.rolling(window).var()
    # Open-to-close variance
    sigma_c2 = log_co.rolling(window).var()
    # Rogers-Satchell intraday variance
    sigma_rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs
    return np.sqrt(yz.clip(lower=0))


def compute_adx_atr(df, period=14):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr  = pd.concat([(hi - lo), (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()],
                    axis=1).max(axis=1)
    up  = hi.diff(); dn = lo.diff().mul(-1)
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    alpha = 1.0 / period
    atr   = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    safe  = atr.replace(0, np.nan)
    pdi   = 100 * pdm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / safe
    ndi   = 100 * ndm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / safe
    dx    = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100).fillna(0)
    adx   = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx.round(2), atr


def classify_regime(adx_val, atr_ratio_val):
    if pd.isna(adx_val) or pd.isna(atr_ratio_val):
        return 'UNDEFINED'
    if atr_ratio_val > REGIME_ATR_VOLATILE:
        return 'VOLATILE'
    if adx_val > REGIME_ADX_TREND:
        return 'TRENDING'
    if adx_val < REGIME_ADX_RANGE:
        return 'RANGING'
    return 'TRANSITIONING'


# review#17 — sizing parity. Backtest must match MT5Live.compute_units so live
# trades aren't over- or under-sized vs the validation backtest. The set below
# now includes EVERY pair where USD is the quote currency (XXX/USD), where the
# loss-per-unit is exactly sl_dist USD and `units = balance*risk/dist` is the
# correct formula. Missing entries previously caused GBP_USD, NZD_USD etc. to
# size at `entry*` the live amount (~25% too large for GBP_USD).
# Cross-rates (EUR/JPY, EUR/GBP) and USD/XXX pairs (USD_JPY, USD_CAD,
# USD_CHF) take the entry-multiplied branch so the formula correctly converts
# the quote-currency loss into USD via the entry price.
_USD_QUOTED = {
    'EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD',  # XXX/USD majors and minors
    'XAU_USD',                                    # gold quoted in USD
}

def pnl_fx(pair, raw_pnl, ep):
    return raw_pnl if pair in _USD_QUOTED else raw_pnl / ep


def calc_size(pair, balance, risk_pct, entry, sl_dist):
    """Position size in *units*. Mirrors MT5Live.compute_units (review#17)."""
    # review#17 follow-up — zero-distance guard. MT5Live.compute_units returns 0
    # when entry==sl (degenerate). Backtest must do the same or sizing parity
    # breaks and `test_zero_distance_does_not_raise` blows up with ZeroDivisionError.
    if sl_dist is None or sl_dist <= 0:
        return 0.0
    if pair in _USD_QUOTED:
        return (balance * risk_pct) / sl_dist
    return (balance * risk_pct * entry) / sl_dist


def calc_size_with_rv(pair, balance, risk_pct, entry, sl_dist, rv_now, rv_median):
    base = calc_size(pair, balance, risk_pct, entry, sl_dist)
    if rv_median is None or rv_median <= 0 or np.isnan(rv_median) or np.isnan(rv_now):
        return base
    # methodology log entry #5 (2026-05-11): rv_now == 0 produces ratio == 0
    # and a ZeroDivisionError. Rare in FX tick aggregation (always some
    # microstructure activity) but reachable on degenerate flat-price bars.
    # Back-ported from the crypto fork where it surfaces routinely.
    if rv_now == 0:
        return base
    ratio = min(rv_now / rv_median, RV_SCALE_CAP)
    # Floor ratio symmetrically — caps both shrinkage and amplification.
    ratio = max(ratio, 1.0 / RV_SCALE_CAP)
    return base / ratio


def resolve_risk(bst, risk_mult, risk_mode):
    base = dynamic_risk(bst.consecutive_wins, bst.consecutive_losses)
    if risk_mode == 'fixed':
        base = RISK_BASE
    elif risk_mode == 'half':
        base = RISK_BASE * 0.5
    return base * risk_mult


def _session_slip_mult(hour, weekday=None) -> float:
    """Slippage multiplier on half-spread for the given UTC hour.

    review#P2#8 — weekday-aware. Sunday open (21–22 UTC) is illiquid for
    ~1 hour after the weekly open; Friday late (21–22 UTC) sees dealers
    hedge before the weekend gap. Both events were previously bucketed
    into 'rollover' (1.5×) but they're materially worse — bumping to 2.0×
    Friday and 1.2× Sunday matches observed retail slippage on those bars.
    Weekday convention: Mon=0, Tue=1, ..., Sat=5, Sun=6.
    """
    if hour is None or (isinstance(hour, float) and np.isnan(hour)):
        return SLIPPAGE_PROFILE['default']
    h = int(hour)
    if weekday is not None:
        try:
            wd = int(weekday)
            if wd == 4 and 21 <= h <= 22:   # Friday pre-weekend hedging
                return 2.0
            if wd == 6 and 21 <= h <= 22:   # Sunday open (week start)
                return 1.2
        except (TypeError, ValueError):
            pass
    if 21 <= h <= 23:
        return SLIPPAGE_PROFILE['rollover']
    if 0 <= h < 7:
        return SLIPPAGE_PROFILE['asian']
    if 7 <= h < 9:
        return SLIPPAGE_PROFILE['london_open']
    if 9 <= h < 13:
        return SLIPPAGE_PROFILE['london_ny']
    if 13 <= h < 15:
        return SLIPPAGE_PROFILE['ny_open']
    if 15 <= h < 21:
        return SLIPPAGE_PROFILE['ny_late']
    return SLIPPAGE_PROFILE['default']


# =============================================================================
# 4. PIPELINE API — BUILDING BLOCKS FOR STRATEGY CONSTRUCTION
# =============================================================================

class ParameterGrid:
    """Expand parameter value lists into every combination.

    Example::

        grid = ParameterGrid({'tp_r': [1.5, 2.0, 2.5], 'ma_req': [True, False]})
        len(grid)   # 6
        list(grid)  # [{'tp_r': 1.5, 'ma_req': True}, {'tp_r': 1.5, 'ma_req': False}, ...]
    """
    def __init__(self, param_ranges: dict):
        self.param_ranges = param_ranges
        self._ranges      = param_ranges   # alias for OptunaGrid conversion

    def __len__(self):
        result = 1
        for v in self.param_ranges.values():
            result *= len(v)
        return result

    def __iter__(self):
        keys   = list(self.param_ranges.keys())
        values = [self.param_ranges[k] for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))


# ── Microstructure building blocks ─────────────────────────────────────────

def spread_gate(row, mult: float = SPREAD_GATE_MULT) -> bool:
    """Return True if the current spread is elevated — entry should be skipped.

    Fast path (2026-06-04 optimisation B): read the precomputed
    `qp_spread_blocked` column populated by run_backtest. Falls back to
    per-call computation for code paths that don't go through run_backtest
    (e.g., live trading)."""
    pre = getattr(row, 'qp_spread_blocked', None)
    if pre is not None:
        return bool(pre)
    sn = getattr(row, 'spread_mean',   float('nan'))
    sm = getattr(row, 'spread_median', float('nan'))
    if np.isnan(sn) or np.isnan(sm) or sm <= 0:
        return False
    return sn > mult * sm


def rv_size(pair: str, balance: float, risk: float,
            entry: float, dist: float, row) -> float:
    """Position size scaled down when realized vol is elevated."""
    return calc_size_with_rv(
        pair, balance, risk, entry, dist,
        getattr(row, 'realized_vol', float('nan')),
        getattr(row, 'rv_median',    float('nan')),
    )


def place_pending(sc: dict, ts, direction: str,
                  entry: float, sl: float, tp: float,
                  size: float, dist: float,
                  level: float = None,
                  mode: str = 'stop_at_level') -> None:
    """Stage a single-direction pending order on bar `ts`.

    Use for confirmation-signal strategies (mode='market_next_open') or
    one-sided stop-order strategies (mode='stop_at_level'). For two-sided
    breakout brackets, use place_oco_pending() instead.

    The pending will not fill on the placement bar — earliest fill is the
    next bar. See check_and_fill() for fill semantics by mode.
    """
    sc['pending_dir']        = direction
    sc['pending_entry']      = entry
    sc['pending_sl']         = sl
    sc['pending_tp']         = tp
    sc['pending_size']       = size
    sc['pending_dist']       = dist
    sc['pending_mode']       = mode
    sc['pending_placed_ts']  = ts
    if mode == 'stop_at_level':
        if level is None:
            raise ValueError("level required for stop_at_level pending")
        sc['pending_level'] = level


def place_oco_pending(sc: dict, ts,
                      long_level: float, long_sl: float, long_tp: float,
                      long_size: float, long_dist: float,
                      short_level: float, short_sl: float, short_tp: float,
                      short_size: float, short_dist: float) -> None:
    """Stage a two-sided OCO bracket on bar `ts`.

    Whichever side triggers first on a subsequent bar fills; the other is
    cancelled (one-cancels-other). Used by pre-stage breakout strategies
    that don't yet know which direction will break.
    """
    sc['pending_long'] = {
        'level': long_level, 'sl': long_sl, 'tp': long_tp,
        'size': long_size,   'dist': long_dist,
    }
    sc['pending_short'] = {
        'level': short_level, 'sl': short_sl, 'tp': short_tp,
        'size': short_size,   'dist': short_dist,
    }
    sc['pending_mode']      = 'stop_at_level'
    sc['pending_placed_ts'] = ts


def cancel_pending(sc: dict) -> None:
    """Clear any pending order from scratch. Strategies call this at session
    exit time to retire orders that never triggered intraday."""
    for k in ('pending_dir', 'pending_level', 'pending_entry',
              'pending_sl', 'pending_tp', 'pending_size',
              'pending_dist', 'pending_mode', 'pending_placed_ts',
              'pending_long', 'pending_short'):
        sc.pop(k, None)


def has_pending(sc: dict) -> bool:
    """True if any pending order (single or OCO) is staged in scratch."""
    return sc.get('pending_placed_ts') is not None


def _compute_stop_fill_price(direction: str, level: float, bar_open: float,
                             row, hspd: float, slip: float):
    """Return fill price for a stop_at_level pending, or None if not triggered.

    Handles gap-through-level fills (bar opens past level) at the adverse open
    price; otherwise fills at level price with slippage.
    """
    if direction == 'long':
        if bar_open >= level:
            return bar_open + hspd + slip
        if row.high >= level:
            return level + hspd + slip
        return None
    if bar_open <= level:
        return bar_open - hspd - slip
    if row.low <= level:
        return level - hspd - slip
    return None


def _open_position(slot, sc, ts, regime, direction, fill_price,
                   sl, tp, size, dist):
    """Apply a fill to a slot. Shared by single-pending and OCO paths."""
    slot['position']       = direction
    slot['entry_price']    = fill_price
    slot['stop_loss']      = sl
    slot['take_profit']    = tp
    slot['pos_size']       = size
    slot['sl_ref_dist']    = dist
    slot['partial_size']   = size * 0.5
    slot['remainder_size'] = size * 0.5
    slot['entry_time']     = ts
    slot['opened_today']   = True
    slot['regime']         = regime
    # Discard all placement-time fields (covers single + OCO pending sets).
    for k in ('pending_dir', 'pending_level', 'pending_entry',
              'pending_sl', 'pending_tp', 'pending_size',
              'pending_dist', 'pending_mode', 'pending_placed_ts',
              'pending_long', 'pending_short'):
        sc.pop(k, None)


def check_and_fill(sc: dict, row, slot: dict, ts, regime: str,
                   hspd: float = 0.0, slip: float = 0.0) -> bool:
    """Check whether a previously-placed pending order should fill on this bar.

    Look-ahead-bias prevention: a pending order placed on bar t cannot fill on
    bar t. It only becomes eligible on bar t+1 onward. Three pending shapes
    are supported:

      1) Single-direction stop_at_level — sc['pending_dir'] is 'long' or
         'short' and sc['pending_level'] is set. Stop-buy/stop-sell fill on
         the first subsequent bar where the level is touched. Gap-through is
         handled by filling at the bar's open with adverse slippage.

      2) Single-direction market_next_open — sc['pending_mode'] ==
         'market_next_open'. Fills at row.open ± slip on the bar after
         placement. Used for confirmation-signal strategies.

      3) OCO bracket — sc['pending_long'] and sc['pending_short'] are dicts
         with keys {level, sl, tp, size, dist}. Whichever side triggers first
         fills; the other is cancelled. stop_at_level semantics only.

    Returns True if a position was opened on this call.
    """
    placed_ts = sc.get('pending_placed_ts')
    if placed_ts is None:
        return False
    # Same-bar fills disallowed — pending must rest at least one bar.
    if placed_ts == ts:
        return False

    bar_open = getattr(row, 'open', row.close)

    # OCO bracket: two stop orders, first to trigger wins.
    if 'pending_long' in sc or 'pending_short' in sc:
        long_p  = sc.get('pending_long')
        short_p = sc.get('pending_short')
        long_fp  = (_compute_stop_fill_price('long',  long_p['level'],
                                              bar_open, row, hspd, slip)
                    if long_p else None)
        short_fp = (_compute_stop_fill_price('short', short_p['level'],
                                              bar_open, row, hspd, slip)
                    if short_p else None)
        if long_fp is None and short_fp is None:
            return False
        # If both sides triggered on the same bar, prefer the side closer to
        # the bar's open (more likely to have triggered first intra-bar).
        if long_fp is not None and short_fp is not None:
            if abs(long_p['level'] - bar_open) <= abs(short_p['level'] - bar_open):
                pick, p = 'long',  long_p
                fp = long_fp
            else:
                pick, p = 'short', short_p
                fp = short_fp
        elif long_fp is not None:
            pick, p, fp = 'long', long_p, long_fp
        else:
            pick, p, fp = 'short', short_p, short_fp
        _open_position(slot, sc, ts, regime, pick, fp,
                       p['sl'], p['tp'], p['size'], p['dist'])
        return True

    # Single-direction pending.
    if sc.get('pending_dir') is None:
        return False
    mode = sc.get('pending_mode', 'stop_at_level')
    direction = sc['pending_dir']

    if mode == 'market_next_open':
        fill_price = (bar_open + hspd + slip) if direction == 'long' \
                     else (bar_open - hspd - slip)
    else:
        level = sc['pending_level']
        fill_price = _compute_stop_fill_price(direction, level, bar_open,
                                              row, hspd, slip)
        if fill_price is None:
            return False

    _open_position(slot, sc, ts, regime, direction, fill_price,
                   sc['pending_sl'], sc['pending_tp'],
                   sc['pending_size'], sc['pending_dist'])
    return True


# ── Manager factory ─────────────────────────────────────────────────────────

class _Manager:
    """Picklable position manager.  Created via make_manager()."""

    def __init__(self, exit_hour: int, exit_min: int,
                 use_breakeven: bool, use_profit_lock: bool,
                 lock_trigger: float, lock_sl: float):
        self.exit_hour      = exit_hour
        self.exit_min       = exit_min
        self.use_breakeven  = use_breakeven
        self.use_profit_lock = use_profit_lock
        self.lock_trigger   = lock_trigger
        self.lock_sl        = lock_sl

    def __call__(self, bst, slot, row, ts, pair, slip, hspd, sess_cfg,
                 fvg_buf=None, day_sweep=None):
        if slot['position'] is None:
            return False

        rng_s    = slot['sl_ref_dist']
        price    = row.close
        exit_h   = self.exit_hour
        exit_m   = self.exit_min

        past_exit = row.hour > exit_h or (row.hour == exit_h and row.minute >= exit_m)
        if past_exit and not slot['session_exited']:
            ep = price - hspd - slip if slot['position'] == 'long' else price + hspd + slip
            _log_exit(bst, pair, slot, ts, ep, 'session_exit')
            slot['session_exited'] = True
            return True

        sl_bar = slot['stop_loss']

        if self.use_breakeven and not slot['breakeven_set']:
            if slot['position'] == 'long'  and row.high >= slot['entry_price'] + rng_s:
                slot['stop_loss']     = slot['entry_price']
                slot['breakeven_set'] = True
            elif slot['position'] == 'short' and row.low <= slot['entry_price'] - rng_s:
                slot['stop_loss']     = slot['entry_price']
                slot['breakeven_set'] = True

        if self.use_profit_lock and slot['breakeven_set'] and not slot['profit_lock_set']:
            trig = self.lock_trigger * rng_s
            lsl  = self.lock_sl * rng_s
            if slot['position'] == 'long'  and row.high >= slot['entry_price'] + trig:
                slot['stop_loss']       = slot['entry_price'] + lsl
                slot['profit_lock_set'] = True
            elif slot['position'] == 'short' and row.low <= slot['entry_price'] - trig:
                slot['stop_loss']       = slot['entry_price'] - lsl
                slot['profit_lock_set'] = True

        ep = None; reason = None
        if slot['position'] == 'long':
            if row.low  <= sl_bar:
                ep, reason = sl_bar - hspd - slip,              'stop_loss'
            elif slot['take_profit'] < 1e30 and row.high >= slot['take_profit']:
                ep, reason = slot['take_profit'] - hspd - slip, 'take_profit'
        else:
            if row.high >= sl_bar:
                ep, reason = sl_bar + hspd + slip,              'stop_loss'
            elif slot['take_profit'] < 1e30 and row.low <= slot['take_profit']:
                ep, reason = slot['take_profit'] + hspd + slip, 'take_profit'

        if ep is not None:
            _log_exit(bst, pair, slot, ts, ep, reason)
            return True
        return False


def make_manager(exit_hour: int, exit_min: int = 0,
                 use_breakeven: bool = True,
                 use_profit_lock: bool = False,
                 lock_trigger: float = 2.0,
                 lock_sl: float = 1.5) -> _Manager:
    """Return a picklable manager function for use in register_strategy / run_sweep.

    Args:
        exit_hour: UTC hour to force-exit all positions (session close).
        exit_min: Minute within exit_hour (default 0).
        use_breakeven: Move SL to entry after 1R profit (default True).
        use_profit_lock: Lock in partial profit at lock_trigger × 1R (default False).
        lock_trigger: R-multiple at which profit lock triggers.
        lock_sl: R-multiple at which locked SL is placed.

    Examples::

        mgr_basic  = make_manager(exit_hour=21)
        mgr_locked = make_manager(exit_hour=21, use_profit_lock=True,
                                  lock_trigger=2.0, lock_sl=1.5)
        mgr_london = make_manager(exit_hour=13, use_breakeven=True)
    """
    return _Manager(exit_hour, exit_min, use_breakeven,
                    use_profit_lock, lock_trigger, lock_sl)


# ── SQLite result storage ───────────────────────────────────────────────────

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sweeps (
            sweep_id    TEXT PRIMARY KEY,
            sweep_name  TEXT,
            created_at  TEXT,
            n_total     INTEGER,
            cost_mult   REAL,
            params_json TEXT
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id  TEXT PRIMARY KEY,
            sweep_id       TEXT,
            params_json    TEXT,
            train_n        INTEGER,  train_wr REAL, train_pnl REAL,
            train_sharpe   REAL,     train_max_dd REAL,
            test_n         INTEGER,  test_wr  REAL, test_pnl  REAL,
            test_sharpe    REAL,     test_max_dd  REAL,
            test_sortino   REAL,     test_calmar  REAL,
            p_raw          REAL,     p_adj   REAL,  bh_sig INTEGER,
            verdict        TEXT,
            created_at     TEXT,
            dsr            REAL,
            sharpe_ci_low  REAL,
            sharpe_ci_high REAL,
            regime_stable  INTEGER DEFAULT 0
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id  TEXT,
            split          TEXT,
            instrument     TEXT, strategy TEXT,  family   TEXT,
            regime         TEXT, entry_time TEXT, exit_time TEXT,
            entry          REAL, exit      REAL,  position  TEXT,
            exit_reason    TEXT, pnl       REAL,  balance   REAL,
            partial        INTEGER
        )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_hyp_sweep ON hypotheses(sweep_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trades_hyp ON trades(hypothesis_id, split)")

    # Migrate existing DBs — add columns if they don't exist yet
    for col_sql in [
        "ALTER TABLE hypotheses ADD COLUMN dsr REAL",
        "ALTER TABLE hypotheses ADD COLUMN sharpe_ci_low REAL",
        "ALTER TABLE hypotheses ADD COLUMN sharpe_ci_high REAL",
        "ALTER TABLE hypotheses ADD COLUMN regime_stable INTEGER DEFAULT 0",
        "ALTER TABLE hypotheses ADD COLUMN psr REAL",            # Phase 7
        "ALTER TABLE hypotheses ADD COLUMN outlier_ratio REAL",  # Phase 7
        "ALTER TABLE hypotheses ADD COLUMN pbo_score REAL",      # Phase 7 (sweep-level)
        "ALTER TABLE hypotheses ADD COLUMN wf_sharpe_mean REAL", # review#18
        "ALTER TABLE hypotheses ADD COLUMN wf_sharpe_min REAL",  # review#18
        "ALTER TABLE hypotheses ADD COLUMN wf_n_folds INTEGER",  # review#18
        "ALTER TABLE hypotheses ADD COLUMN mc_eval_pass_pct REAL",   # §1 Tier 1
        "ALTER TABLE hypotheses ADD COLUMN mc_blown_pct REAL",       # §1 Tier 1
        "ALTER TABLE hypotheses ADD COLUMN mc_n_sims INTEGER",       # §1 Tier 1
        "ALTER TABLE sweeps ADD COLUMN params_json TEXT",
    ]:
        try:
            con.execute(col_sql)
        except Exception:
            pass   # column already exists

    con.commit()
    con.close()


def _save_result(sweep_id: str, hyp_id: str, params: dict,
                 train_s, test_s, ext_s,
                 train_trades: pd.DataFrame, test_trades: pd.DataFrame,
                 dsr: float = None, sharpe_ci_low: float = None,
                 sharpe_ci_high: float = None, regime_stable: int = 0,
                 psr: float = None, outlier_ratio: float = None,
                 wf_sharpe_mean: float = None, wf_sharpe_min: float = None,
                 wf_n_folds: int = None,
                 mc_eval_pass_pct: float = None,   # §1 Tier 1
                 mc_blown_pct: float = None,       # §1 Tier 1
                 mc_n_sims: int = None):           # §1 Tier 1
    con = sqlite3.connect(DB_PATH)
    now = datetime.utcnow().isoformat()

    def _g(d, k, default=0.0):
        return d[k] if d and k in d else default

    con.execute("""
        INSERT OR REPLACE INTO hypotheses (
            hypothesis_id, sweep_id, params_json,
            train_n, train_wr, train_pnl, train_sharpe, train_max_dd,
            test_n, test_wr, test_pnl, test_sharpe, test_max_dd,
            test_sortino, test_calmar,
            p_raw, p_adj, bh_sig, verdict, created_at,
            dsr, sharpe_ci_low, sharpe_ci_high, regime_stable,
            psr, outlier_ratio,
            wf_sharpe_mean, wf_sharpe_min, wf_n_folds,
            mc_eval_pass_pct, mc_blown_pct, mc_n_sims
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        hyp_id, sweep_id, json.dumps(params),
        _g(train_s,'n',0),  _g(train_s,'wr'),   _g(train_s,'pnl'),
        _g(train_s,'sharpe'), _g(train_s,'max_dd'),
        _g(test_s,'n',0),   _g(test_s,'wr'),    _g(test_s,'pnl'),
        _g(test_s,'sharpe'),  _g(test_s,'max_dd'),
        _g(ext_s,'sortino'), _g(ext_s,'calmar'),
        None, None, 0,
        'pending', now,
        dsr, sharpe_ci_low, sharpe_ci_high, regime_stable,
        psr, outlier_ratio,
        wf_sharpe_mean, wf_sharpe_min, wf_n_folds,  # review#18
        mc_eval_pass_pct, mc_blown_pct, mc_n_sims,  # §1 Tier 1
    ))

    # Save trades (both splits)
    for split, df in [('train', train_trades), ('test', test_trades)]:
        if df is not None and not df.empty:
            rows = []
            for _, r in df.iterrows():
                rows.append((
                    hyp_id, split,
                    r.get('instrument',''), r.get('strategy',''), r.get('family',''),
                    r.get('regime',''), str(r.get('entry_time','')), str(r.get('exit_time','')),
                    float(r.get('entry',0)), float(r.get('exit',0)),
                    r.get('position',''), r.get('exit_reason',''),
                    float(r.get('pnl',0)), float(r.get('balance',0)),
                    int(r.get('partial',0)),
                ))
            con.executemany(
                "INSERT INTO trades (hypothesis_id,split,instrument,strategy,family,"
                "regime,entry_time,exit_time,entry,exit,position,exit_reason,pnl,balance,partial)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    con.commit()
    con.close()


def _save_sweep(sweep_id: str, sweep_name: str, n_total: int, cost_mult: float):
    _init_db()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO sweeps "
        "(sweep_id, sweep_name, created_at, n_total, cost_mult) "
        "VALUES (?,?,?,?,?)",
        (sweep_id, sweep_name, datetime.utcnow().isoformat(), n_total, cost_mult))
    con.commit()
    con.close()


def load_sweep_list() -> list[dict]:
    """Return all sweep records ordered by creation time."""
    _init_db()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT sweep_id, sweep_name, created_at, n_total, cost_mult "
        "FROM sweeps ORDER BY created_at DESC").fetchall()
    con.close()
    return [{'sweep_id': r[0], 'sweep_name': r[1], 'created_at': r[2],
             'n_total': r[3], 'cost_mult': r[4]} for r in rows]


def load_sweep_results(sweep_id: str) -> pd.DataFrame:
    """Return all hypotheses for a sweep as a DataFrame, sorted by test Sharpe desc."""
    _init_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM hypotheses WHERE sweep_id=? ORDER BY test_sharpe DESC",
        con, params=(sweep_id,))
    con.close()
    if df.empty:
        return df
    df['params'] = df['params_json'].apply(json.loads)
    return df


def load_hypothesis_trades(hyp_id: str, split: str = 'test') -> pd.DataFrame:
    """Return trade log for one hypothesis (test or train split)."""
    _init_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM trades WHERE hypothesis_id=? AND split=?",
        con, params=(hyp_id, split))
    con.close()
    return df


def apply_bh_correction(sweep_id: str):
    """Apply FAMILY-WIDE FDR correction across ALL hypotheses in the DB.

    review#6 — was previously per-sweep (~16 hypotheses per family) which is
    locally correct but globally meaningless: the actual research family is
    every hypothesis the agent has ever tested (~1,000+). Per-sweep BH
    over-discovers because the family is artificially small.

    sweep_id is now effectively a trigger (called after each sweep) — the
    correction itself is computed across every row in `hypotheses` that has
    a non-null p_raw, then writes p_adj/bh_sig/p_adj_by/by_sig back across
    the whole family. Older rows' significance gets re-evaluated as the
    population grows, which is the correct multiple-testing behavior.

    Stores BOTH BH (legacy bh_sig) and BY (by_sig) adjusted values. Strategies
    must pass the more conservative BY check to be considered survivors —
    BY handles correlated hypotheses (which is the actual reality of grid
    sweeps over related parameter combos) where BH assumes independence and
    over-discovers.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return

    con = sqlite3.connect(DB_PATH)
    # review#6 — pull EVERY hypothesis with p_raw (full family), not just the
    # sweep that just finished.
    rows = con.execute(
        "SELECT hypothesis_id, p_raw FROM hypotheses WHERE p_raw IS NOT NULL"
    ).fetchall()
    con.close()
    if not rows:
        return
    if len(rows) < 2:
        # multipletests degenerates on N=1; nothing to correct against.
        return

    ids   = [r[0] for r in rows]
    pvals = [r[1] for r in rows]
    _, p_bh, _, _ = multipletests(pvals, method='fdr_bh')
    _, p_by, _, _ = multipletests(pvals, method='fdr_by')

    con = sqlite3.connect(DB_PATH)
    # bh_sig retained for back-compat; new by_sig column is the gate that
    # downstream survivor filters should consult.
    try:
        con.execute("ALTER TABLE hypotheses ADD COLUMN p_adj_by REAL")
    except Exception:
        pass
    try:
        con.execute("ALTER TABLE hypotheses ADD COLUMN by_sig INTEGER DEFAULT 0")
    except Exception:
        pass
    # executemany for the cross-family update — typically ~1k rows.
    payload = [
        (float(p_bh_v), 1 if p_bh_v < 0.05 else 0,
         float(p_by_v), 1 if p_by_v < 0.05 else 0,
         hid)
        for hid, p_bh_v, p_by_v in zip(ids, p_bh, p_by)
    ]
    con.executemany("""
        UPDATE hypotheses
        SET p_adj    = ?, bh_sig = ?,
            p_adj_by = ?, by_sig = ?
        WHERE hypothesis_id = ?
    """, payload)
    con.commit()
    con.close()


def apply_pbo_correction(sweep_id: str):
    """Compute sweep-level PBO and stamp it on every row in the sweep.

    PBO is a family-level statistic — it tells you whether the *best* strategy
    out of N candidates is likely overfit. Stamping the same value on every
    row lets the per-row survivor filter reject any candidate from an
    overfit family without re-deriving the score downstream.
    """
    try:
        con = sqlite3.connect(DB_PATH)
        ids = [r[0] for r in con.execute(
            "SELECT hypothesis_id FROM hypotheses WHERE sweep_id=?",
            (sweep_id,)).fetchall()]
        if len(ids) < 2:
            con.close()
            return
        # Build trade matrix: rows = unique exit_date, cols = hypothesis_id, values = sum(pnl).
        trades = pd.read_sql(
            f"SELECT hypothesis_id, exit_time, pnl FROM trades "
            f"WHERE split='test' AND hypothesis_id IN ({','.join('?'*len(ids))})",
            con, params=ids)
        con.close()
        if trades.empty:
            return
        trades['day'] = pd.to_datetime(trades['exit_time']).dt.tz_localize(None).dt.date
        matrix = trades.pivot_table(
            index='day', columns='hypothesis_id', values='pnl',
            aggfunc='sum', fill_value=0.0)
        score = pbo_score(matrix)
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return
        con = sqlite3.connect(DB_PATH)
        con.execute("UPDATE hypotheses SET pbo_score=? WHERE sweep_id=?",
                    (float(score), sweep_id))
        con.commit()
        con.close()
    except Exception:
        pass


# ── Worker (must be top-level for multiprocessing pickling) ─────────────────
# review#13 — was: each worker pickle-loaded the full {pair: DataFrame} dict
# into _W_TRAIN/_W_TEST globals at boot. With 2 workers × 5 pairs × ~100k
# bars × ~200 columns, RSS hit ~5 GB per worker = OOM on the 8 GB VPS and
# ProcessPoolExecutor death mid-sweep. Now: workers receive a directory of
# per-pair parquet files; pairs are lazy-loaded on first access, kept in a
# small LRU cache, freed when evicted. Per-worker RSS is bounded to ~1-2
# pairs in memory at any time.

_W_TRAIN: dict = {}
_W_TEST:  dict = {}


class _MmapPairCache:
    """Dict-like lazy per-pair DataFrame loader for worker processes.

    Provides the `dict[pair] -> DataFrame` and `dict.get`/`pair in dict` API
    the existing call sites assume, but loads parquet files on demand via
    PyArrow memory-mapped IO and bounds the cache to MAX_CACHED pairs.
    """
    MAX_CACHED = 2  # tasks touch 1-2 pairs at a time; LRU keeps only those resident

    def __init__(self, parquet_dir, split: str):
        self._dir = Path(parquet_dir)
        self._split = split
        self._cache: dict = {}
        self._order: list = []  # LRU order, oldest first

    def __contains__(self, key) -> bool:
        if key in self._cache:
            return True
        return (self._dir / f"{self._split}_{key}.parquet").exists()

    def __getitem__(self, key):
        if key in self._cache:
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return self._cache[key]
        path = self._dir / f"{self._split}_{key}.parquet"
        if not path.exists():
            raise KeyError(key)
        import pyarrow.parquet as pq
        df = pq.read_table(str(path), memory_map=True).to_pandas()
        while len(self._cache) >= self.MAX_CACHED:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        self._cache[key] = df
        self._order.append(key)
        return df

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        # Enumerate parquet files on disk for callers that iterate.
        return [p.stem.split('_', 1)[1] for p in self._dir.glob(f"{self._split}_*.parquet")]


def _worker_init(parquet_dir: str):
    """review#13 — initargs is now a directory path; lazy-load per-pair."""
    global _W_TRAIN, _W_TEST
    _W_TRAIN = _MmapPairCache(parquet_dir, 'train')
    _W_TEST  = _MmapPairCache(parquet_dir, 'test')


def _run_single_hypothesis(task: dict) -> dict:
    """Execute one hypothesis. Called in a worker process (run_sweep) or
    in-process (run_sweep_optuna, where globals may not be populated)."""
    entry_fn         = task['entry_fn']
    manager_fn       = task['manager_fn']
    params           = task['params']
    pairs            = task['pairs']
    regime_mult      = task['regime_mult']
    family           = task['family']
    allow_concurrent = task.get('allow_concurrent', False)
    spread_ov        = task['spread_ov']
    slip_ov          = task['slip_ov']
    cost_mult        = task.get('cost_mult', 1.0)
    session_hours    = task.get('session_hours')
    hyp_id           = task['hyp_id']

    slot_class = hyp_id.split('_')[0]   # derived from sweep name prefix

    # Support both in-process (data in task) and worker-process (data in globals)
    train_data = task.get('_train_dfs') or _W_TRAIN
    test_data  = task.get('_test_dfs')  or _W_TEST

    registry = [{
        'id': hyp_id, 'family': family, 'slot_class': slot_class,
        'pairs': pairs, 'session': task['session'],
        'allow_concurrent': allow_concurrent,
        'regime_mult': regime_mult,
        'params': params,
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}

    train_subset = {p: train_data[p] for p in pairs if p in train_data}
    test_subset  = {p: test_data[p]  for p in pairs if p in test_data}

    try:
        tr, tr_bal, _ = run_backtest(train_subset, spread_ov, slip_ov,
                                      registry, slot_managers, slot_entries,
                                      cost_mult=cost_mult,
                                      session_hours=session_hours)
        te, te_bal, _ = run_backtest(test_subset,  spread_ov, slip_ov,
                                      registry, slot_managers, slot_entries,
                                      cost_mult=cost_mult,
                                      session_hours=session_hours)
    except Exception as exc:
        return {'hyp_id': hyp_id, 'params': params, 'error': str(exc)}

    tr_m = _merge_partials(tr) if not tr.empty else tr
    te_m = _merge_partials(te) if not te.empty else te

    tr_s  = calc_stats(tr_m)
    te_s  = calc_stats(te_m)
    ext_s = extended_risk_metrics(te_m)

    # review#8 — NULL p_raw policy: when the test set has <10 trades or the
    # variance is zero, p_raw stays None and is excluded from the family-wide
    # BH correction. These rows keep bh_sig=0 by default — i.e., NULL is
    # interpreted as "not eligible for significance, treated as non-significant".
    # Empirically ~50% of historical hypotheses fall into this bucket because
    # they fired too rarely; with the recent Phase 1 prompt change reducing
    # over-gating, the fraction should drop materially.
    # review#7 — raw p-value with autocorrelation-aware effective sample size.
    # Naive ttest_1samp treats trades as i.i.d., which is false for FX (same-
    # session clustering, vol regimes). We deflate N to the autocorrelation-
    # adjusted effective sample size n_eff = N * (1 - rho) / (1 + rho), where
    # rho is the lag-1 autocorrelation of trade PnL. This pulls overconfident
    # p-values back toward what an honest test would report. Skip the
    # adjustment when n < 10 (guarded already) or rho is unreliable.
    p_raw = None
    try:
        from scipy.stats import t as _t_dist
        if te_m is not None and not te_m.empty and len(te_m) >= 10:
            pnl = te_m['pnl'].to_numpy(dtype=float)
            n   = len(pnl)
            mean = float(pnl.mean())
            sd   = float(pnl.std(ddof=1))
            if sd > 0:
                # Lag-1 autocorrelation; clamp to [-0.99, 0.99] to keep the
                # effective-N transform finite.
                if n >= 3:
                    a = pnl[:-1] - pnl[:-1].mean()
                    b = pnl[1:]  - pnl[1:].mean()
                    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
                    rho = float((a * b).sum() / denom) if denom > 0 else 0.0
                    rho = max(-0.99, min(0.99, rho))
                else:
                    rho = 0.0
                # Effective sample size; floor at 5 (don't claim n_eff < 5).
                n_eff = max(5.0, n * (1.0 - rho) / (1.0 + rho))
                # t-statistic recomputed at n_eff
                t_eff = mean / (sd / np.sqrt(n_eff))
                df_eff = max(1.0, n_eff - 1.0)
                p_raw = float(_t_dist.sf(t_eff, df_eff))  # one-sided > 0
    except Exception:
        pass

    # ── Fake-edge Layer 3: Deflated Sharpe + Bootstrap CI ────────────────────
    n_trials_hint = task.get('n_trials_total', 1)   # set by sweep runner
    test_sharpe   = float(te_s.get('sharpe', 0)) if te_s else 0.0
    n_obs         = int(te_s.get('n', 1)) if te_s else 1
    dsr_val = deflated_sharpe_ratio(test_sharpe, n_trials_hint, max(n_obs, 2))

    ci_lo, ci_hi = bootstrap_sharpe_ci(te_m) if te_m is not None and not te_m.empty else (None, None)

    # ── Phase 7: PSR + outlier ratio. PSR uses skew/kurt of test PnL when
    # available; falls back to normal moments. Outlier ratio is the un-
    # winsorised / winsorised Sharpe — a strategy with ratio > 2 is leaning
    # on a few lucky days.
    psr_val      = float('nan')
    outlier_rat  = float('nan')
    if te_m is not None and not te_m.empty and len(te_m) >= 10:
        try:
            te_daily = (te_m.assign(d=pd.to_datetime(te_m['exit_time'])
                                       .dt.tz_localize(None).dt.date)
                            .groupby('d')['pnl'].sum().values)
            if len(te_daily) >= 5:
                from scipy.stats import skew as _sc_skew, kurtosis as _sc_kurt
                _sk = float(_sc_skew(te_daily))
                _ku = float(_sc_kurt(te_daily, fisher=False))
                psr_val = probabilistic_sharpe_ratio(test_sharpe, len(te_daily),
                                                     skew=_sk, kurt=_ku)
                ws = winsorize_sharpe(te_daily)
                outlier_rat = ws.get('outlier_ratio', float('nan'))
        except Exception:
            pass

    # ── Fake-edge Layer 4: Regime stability ──────────────────────────────────
    reg_stable = int(is_regime_stable(te_m)) if te_m is not None and not te_m.empty else 0

    # review#18 — walk-forward inside the gauntlet. Split test trades into
    # WF_N_FOLDS chronological folds and compute Sharpe per fold. Store mean
    # and min so the scorer (gated behind USE_WF_GATE) can reject strategies
    # whose worst fold is non-positive even when the aggregate Sharpe looks
    # fine. Was previously a post-hoc check in agent/robustness.py — selection
    # bias from the single train/test split was therefore uncorrected.
    wf_sharpe_mean = None
    wf_sharpe_min  = None
    wf_n_folds_used = None
    try:
        from agent.config import WF_N_FOLDS as _WF_N
        # methodology log entry #1 (2026-05-11): per-fold annualised Sharpe
        # on 5 trades produces nonsense values (median −20, min −1190 across
        # 168 populated rows). Raised the minimum from 5 → 10 trades per fold;
        # folds smaller than that contribute NaN to wf_sharpe_mean and are
        # excluded from wf_sharpe_min instead of dragging the metric to
        # mathematically-meaningless extremes.
        _MIN_TRADES_PER_FOLD = 10
        if (te_m is not None and not te_m.empty
                and 'pnl' in te_m.columns and 'exit_time' in te_m.columns
                and len(te_m) >= _WF_N * _MIN_TRADES_PER_FOLD):
            ordered = te_m.sort_values('exit_time').reset_index(drop=True)
            fold_size = len(ordered) // _WF_N
            fold_sharpes = []
            for i in range(_WF_N):
                lo = i * fold_size
                hi = (i + 1) * fold_size if i < _WF_N - 1 else len(ordered)
                sub = ordered.iloc[lo:hi]
                if len(sub) < _MIN_TRADES_PER_FOLD:
                    continue
                fs = calc_stats(sub)
                if fs:
                    fold_sharpes.append(float(fs.get('sharpe', 0.0)))
            if fold_sharpes:
                wf_sharpe_mean  = float(np.mean(fold_sharpes))
                wf_sharpe_min   = float(min(fold_sharpes))
                wf_n_folds_used = len(fold_sharpes)
    except Exception:
        pass

    # §1 Tier 1 back-port (TIER1_3_PLAN.md) — inline Monte-Carlo prop-firm
    # pass rate. n_sims=1000 for speed; the 10000-sim post-hoc check still
    # runs in robustness.py for survivors. On FX, USE_MC_GATE defaults to
    # False so the survivor population isn't retro-rejected; these numbers
    # populate the columns for future cross-market parity audits.
    mc_eval_pass_pct = None
    mc_blown_pct     = None
    mc_n_sims        = None
    try:
        from agent.config import MC_CHALLENGE_DAYS as _MC_DAYS
        if te_m is not None and not te_m.empty and len(te_m) >= 20:
            mc_result = run_monte_carlo(te_m, label=hyp_id,
                                         n_sims=1000,
                                         challenge_days=_MC_DAYS)
            mc_eval_pass_pct = mc_result.get('eval_pass_pct',
                                              mc_result.get('pass_pct'))
            mc_blown_pct     = mc_result.get('blown_pct')
            mc_n_sims        = mc_result.get('n_sims')
    except Exception:
        pass

    return {
        'hyp_id':         hyp_id,
        'params':         params,
        'train_s':        tr_s,
        'test_s':         te_s,
        'ext_s':          ext_s,
        'train_trades':   tr_m,
        'test_trades':    te_m,
        'p_raw':          p_raw,
        'dsr':            dsr_val,
        'psr':            psr_val,
        'outlier_ratio':  outlier_rat,
        'sharpe_ci_low':  ci_lo,
        'sharpe_ci_high': ci_hi,
        'regime_stable':  reg_stable,
        'wf_sharpe_mean': wf_sharpe_mean,
        'wf_sharpe_min':  wf_sharpe_min,
        'wf_n_folds':     wf_n_folds_used,
        'mc_eval_pass_pct': mc_eval_pass_pct,   # §1 Tier 1
        'mc_blown_pct':     mc_blown_pct,
        'mc_n_sims':        mc_n_sims,
    }


# ── Sweep orchestrator ──────────────────────────────────────────────────────

def _prepare_sweep_data_dir(train_dfs: dict, test_dfs: dict) -> Path:
    """Write per-pair train/test parquets into a temp dir for worker lazy-load.
    Returns the dir path. Caller is responsible for shutil.rmtree on cleanup
    (or persistent reuse across many sweeps — see optimisation #4)."""
    data_dir = Path(tempfile.mkdtemp(prefix='sweep_data_'))
    for _pair, _df in train_dfs.items():
        _df.to_parquet(data_dir / f"train_{_pair}.parquet", index=False)
    for _pair, _df in test_dfs.items():
        _df.to_parquet(data_dir / f"test_{_pair}.parquet", index=False)
    return data_dir


def run_sweep(
    sweep_name:   str,
    entry_fn,
    manager_fn:   _Manager,
    grid:         ParameterGrid,
    pairs:        list,
    session:      str,
    regime_mult:  dict,
    train_dfs:    dict,
    test_dfs:     dict,
    measured_spreads: dict,
    family:             str   = 'session_based',
    allow_concurrent:   bool  = False,
    cost_mult:          float = 1.0,
    n_workers:          int   = 4,
    progress_callback         = None,
    use_dynamic_spread: bool  = False,
    executor:           "concurrent.futures.ProcessPoolExecutor | None" = None,
    data_dir:           "Path | None" = None,
) -> str:
    """Run a full parameter grid sweep.

    Creates one isolated backtest per parameter combination, runs them in
    parallel, stores all results to SQLite, applies BH FDR correction, and
    returns the sweep_id.

    Args:
        sweep_name:        Human-readable name for this sweep.
        entry_fn:          Top-level entry function (picklable).
        manager_fn:        _Manager instance from make_manager() (picklable).
        grid:              ParameterGrid of parameter combinations to test.
        pairs:             List of instrument keys to trade.
        session:           Session key ('ny', 'london', 'asian').
        regime_mult:       Dict mapping regime names to size multipliers.
        train_dfs:         {pair: prepared M1 DataFrame} for training period.
        test_dfs:          {pair: prepared M1 DataFrame} for test period.
        measured_spreads:  {pair: median_spread_in_price} from load_all_data().
        family:            Strategy family label (default 'session_based').
        allow_concurrent:  Allow >1 trade per slot per day (default False).
        cost_mult:         Spread multiplier: 0.5=optimistic,1.0=realistic,1.5=pessimistic.
        n_workers:         Number of parallel worker processes.
        progress_callback: Optional callable(done:int, total:int) for progress reporting.

    Returns:
        sweep_id string (use with load_sweep_results).
    """
    _init_db()
    sweep_id = f"{sweep_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    combos   = list(grid)
    n_total  = len(combos)
    # review#P2#1 — DSR family is the entire research history, not just this
    # sweep's grid. Use cumulative hypothesis count across the DB so DSR
    # accounts for selection bias from every prior trial. Mirrors review#6
    # (BH family-wide). Falls back to n_total on read failure.
    try:
        _con = sqlite3.connect(DB_PATH)
        n_trials_family = int(_con.execute(
            "SELECT COUNT(*) FROM hypotheses").fetchone()[0]) + n_total
        _con.close()
    except Exception:
        n_trials_family = n_total

    # Build spread/slippage overrides from measured spreads
    # When use_dynamic_spread=True, tasks get None → run_backtest uses per-bar spread_adj
    spread_ov = (None if use_dynamic_spread
                 else {p: measured_spreads.get(p, PAIR_PIP_SIZE.get(p, 1e-4)) * cost_mult
                       for p in ALL_PAIRS})
    slip_ov   = (None if use_dynamic_spread
                 else {p: spread_ov[p] * SLIP_RATIO_STATIC for p in ALL_PAIRS})

    _save_sweep(sweep_id, sweep_name, n_total, cost_mult)
    spread_mode = "dynamic (spread_adj)" if use_dynamic_spread else f"static cost_mult={cost_mult}×"
    print(f"\n  Sweep: {sweep_name}  ({n_total} combinations, spread={spread_mode})")

    # Session-hour bar pre-filter: skip bars outside the strategy's session
    # window. ~40–60% wall-time cut on session-bounded strategies.
    _SESSION_HOURS = {'asian': (0, 7), 'london': (7, 13), 'ny': (13, 21)}
    session_hours = _SESSION_HOURS.get(session)
    if session_hours is not None:
        print(f"  Session hours filter: {session} = {session_hours[0]}-{session_hours[1]} UTC")

    # Slot class name derived from sweep name (sanitised)
    slot_cls = sweep_name.replace(' ', '_').replace('-', '_').lower()[:20]

    tasks = []
    for i, params in enumerate(combos):
        hyp_id = f"{slot_cls}_{i:05d}"
        tasks.append({
            'entry_fn':        entry_fn,
            'manager_fn':      manager_fn,
            'params':          params,
            'pairs':           pairs,
            'session':         session,
            'regime_mult':     regime_mult,
            'family':          family,
            'allow_concurrent': allow_concurrent,
            'spread_ov':       spread_ov,
            'slip_ov':         slip_ov,
            'cost_mult':       cost_mult,
            'session_hours':   session_hours,
            'hyp_id':          hyp_id,
            'n_trials_total':  n_trials_family,  # review#P2#1: family-wide DSR
        })

    # review#13 — write per-pair parquet files into a temp directory so
    # workers can lazy-load on demand instead of pickle-loading the full
    # multi-pair dict at boot. Each pair becomes ~50-300 MB on disk; the
    # parquet reader uses memory-mapped IO so OS file cache is shared
    # across worker processes.
    #
    # 2026-06-04 optimisation #4: when `data_dir` and `executor` are passed in
    # by the caller (the agent loop), reuse them across sweeps instead of
    # spawning a fresh pool + rewriting parquets every call. Saves ~10–30s of
    # worker-spawn overhead per sweep.
    import shutil
    own_data_dir = (data_dir is None)
    if own_data_dir:
        data_dir = _prepare_sweep_data_dir(train_dfs, test_dfs)
    own_executor = (executor is None)
    if own_executor:
        ctx = mp.get_context('spawn')
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(str(data_dir),),
        )

    done_count = 0
    try:
        futures = {executor.submit(_run_single_hypothesis, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            done_count += 1
            if 'error' not in result:
                _save_result(
                    sweep_id, result['hyp_id'], result['params'],
                    result['train_s'], result['test_s'], result['ext_s'],
                    result['train_trades'], result['test_trades'],
                    dsr            = result.get('dsr'),
                    sharpe_ci_low  = result.get('sharpe_ci_low'),
                    sharpe_ci_high = result.get('sharpe_ci_high'),
                    regime_stable  = result.get('regime_stable', 0),
                    psr            = result.get('psr'),
                    outlier_ratio  = result.get('outlier_ratio'),
                    wf_sharpe_mean = result.get('wf_sharpe_mean'),  # review#18
                    wf_sharpe_min  = result.get('wf_sharpe_min'),
                    wf_n_folds     = result.get('wf_n_folds'),
                    mc_eval_pass_pct = result.get('mc_eval_pass_pct'),  # §1 Tier 1
                    mc_blown_pct     = result.get('mc_blown_pct'),
                    mc_n_sims        = result.get('mc_n_sims'),
                )
                if result.get('p_raw') is not None:
                    con = sqlite3.connect(DB_PATH)
                    con.execute(
                        "UPDATE hypotheses SET p_raw=? WHERE hypothesis_id=?",
                        (result['p_raw'], result['hyp_id']))
                    con.commit()
                    con.close()
            else:
                print(f"  [WARN] {result['hyp_id']}: {result['error']}")

            if progress_callback:
                progress_callback(done_count, n_total)
    finally:
        if own_executor:
            executor.shutdown()
        if own_data_dir:
            shutil.rmtree(data_dir, ignore_errors=True)

    # Apply BH correction across the whole sweep
    apply_bh_correction(sweep_id)

    # Phase 7 — sweep-level PBO score stamped on every row
    apply_pbo_correction(sweep_id)

    # Compute verdicts
    _update_verdicts(sweep_id)
    print(f"  Sweep complete: {done_count}/{n_total} results saved → sweep_id={sweep_id}")
    return sweep_id


def _update_verdicts(sweep_id: str):
    """Update verdicts using all fake-edge filters.

    VIABLE:   BH-significant + positive test PnL + DSR > 0.5 + CI_low > 0
    MARGINAL: BH-significant + positive PnL but DSR/CI weak
    SUSPECT:  Positive PnL but fails DSR or bootstrap CI (likely fake edge)
    NO EDGE:  Negative PnL or not BH-significant
    """
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT hypothesis_id, test_pnl, bh_sig, dsr, sharpe_ci_low "
        "FROM hypotheses WHERE sweep_id=?",
        (sweep_id,)).fetchall()
    for hid, pnl, sig, dsr, ci_low in rows:
        pos_pnl    = pnl and pnl > 0
        bh_ok      = bool(sig)
        dsr_ok     = (dsr is not None and dsr > 0.5)
        ci_ok      = (ci_low is not None and ci_low > 0)

        if bh_ok and pos_pnl and dsr_ok and ci_ok:
            verdict = 'VIABLE'
        elif bh_ok and pos_pnl:
            verdict = 'MARGINAL'
        elif pos_pnl and not bh_ok:
            verdict = 'SUSPECT'
        else:
            verdict = 'NO EDGE'
        con.execute("UPDATE hypotheses SET verdict=? WHERE hypothesis_id=?", (verdict, hid))
    con.commit()
    con.close()


# ── Robustness tools ────────────────────────────────────────────────────────

def walk_forward_test(trades_df: pd.DataFrame, n_folds: int = 4) -> pd.DataFrame:
    """Split trade log into n_folds time periods and compute stats per fold.

    Returns a DataFrame with columns: fold, start, end, n, wr, sharpe, pnl.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df['_exit_dt'] = pd.to_datetime(df['exit_time']).dt.tz_localize(None)
    df = df.sort_values('_exit_dt')
    fold_size = max(1, len(df) // n_folds)
    records   = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx   = (i + 1) * fold_size if i < n_folds - 1 else len(df)
        fold_df   = df.iloc[start_idx:end_idx]
        s = calc_stats(fold_df)
        if s:
            records.append({
                'fold':   i + 1,
                'start':  str(fold_df['_exit_dt'].min().date()),
                'end':    str(fold_df['_exit_dt'].max().date()),
                'n':      s['n'], 'wr': round(s['wr'], 1),
                'sharpe': round(s['sharpe'], 2), 'pnl': round(s['pnl'], 0),
            })
    return pd.DataFrame(records)


def regime_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-regime stats as a DataFrame."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    records = []
    for regime in ['TRENDING', 'RANGING', 'TRANSITIONING', 'VOLATILE', 'UNDEFINED']:
        sub = trades_df[trades_df['regime'] == regime]
        s   = calc_stats(sub)
        if s and s['n'] > 0:
            records.append({'regime': regime, 'n': s['n'],
                            'wr': round(s['wr'], 1), 'rr': round(s['rr'], 2),
                            'sharpe': round(s['sharpe'], 2), 'pnl': round(s['pnl'], 0)})
    return pd.DataFrame(records)


def plot_equity_figure(trades_df: pd.DataFrame,
                       label: str = '') -> matplotlib.figure.Figure:
    """Return a Matplotlib Figure of the equity and drawdown curves."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={'height_ratios': [2, 1]})
    if trades_df is None or trades_df.empty:
        ax0.set_title(f"{label} — No trades")
        return fig

    df = trades_df.copy()
    df['equity']   = INITIAL_BALANCE + df['pnl'].cumsum()
    df['peak']     = df['equity'].cummax()
    df['drawdown'] = df['equity'] - df['peak']

    ax0.plot(range(len(df)), df['equity'].values, color='steelblue', linewidth=1.5)
    ax0.axhline(INITIAL_BALANCE, color='gray', linestyle='--', alpha=0.5)
    ax0.axhline(INITIAL_BALANCE * (1 + PROP_PROFIT_TARGET),
                color='green', linestyle=':', alpha=0.6, label='6% target')
    ax0.axhline(INITIAL_BALANCE * (1 - PROP_MAX_DRAWDOWN_LIMIT),
                color='red', linestyle=':', alpha=0.6, label='6% DD limit')
    ax0.set_title(f"{label} — Equity Curve")
    ax0.set_ylabel('Balance (£)')
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.fill_between(range(len(df)), df['drawdown'].values, 0, alpha=0.5, color='red')
    ax1.axhline(-(INITIAL_BALANCE * PROP_MAX_DRAWDOWN_LIMIT),
                color='darkred', linestyle='--', alpha=0.7)
    ax1.set_title('Drawdown')
    ax1.set_ylabel('Drawdown (£)')
    ax1.set_xlabel('Trade #')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mc_figure(trades_df: pd.DataFrame,
                   n_sims: int = 500,
                   challenge_days: int = 90,
                   seed: int = 42) -> matplotlib.figure.Figure:
    """Return a Matplotlib Figure of Monte Carlo equity paths."""
    fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(14, 5))
    if trades_df is None or trades_df.empty:
        ax_p.set_title('No trades')
        return fig

    df = trades_df.copy()
    df['exit_date'] = pd.to_datetime(df['exit_time']).dt.tz_localize(None).dt.date
    daily_pnl  = df.groupby('exit_date')['pnl'].sum()
    all_bdays  = pd.bdate_range(daily_pnl.index.min(), daily_pnl.index.max())
    daily_dist = daily_pnl.reindex([d.date() for d in all_bdays], fill_value=0).values

    pt    = INITIAL_BALANCE * PROP_PROFIT_TARGET
    mdd_l = INITIAL_BALANCE * PROP_MAX_DRAWDOWN_LIMIT
    dl    = INITIAL_BALANCE * PROP_DAILY_LOSS_LIMIT
    rng   = np.random.default_rng(seed)
    finals = []

    # review#9 — block bootstrap (mirrors run_monte_carlo). Plot fn must use
    # the same resampling as the headline-stat fn or the chart and the
    # pass/blow rates disagree.
    _MC_BLOCK_LEN_PLOT = 20
    n_dist_plot = len(daily_dist)
    for _ in range(n_sims):
        bal     = float(INITIAL_BALANCE)
        path    = [bal]
        if n_dist_plot >= 5:
            idx = _stationary_bootstrap_indices(challenge_days, _MC_BLOCK_LEN_PLOT, rng)
            sampled = daily_dist[idx % n_dist_plot]
        else:
            sampled = rng.choice(daily_dist, size=challenge_days, replace=True)
        blown   = False
        for dpnl in sampled:
            dpnl = max(dpnl, -dl)
            bal  = bal + dpnl
            path.append(bal)
            if bal - INITIAL_BALANCE >= pt or INITIAL_BALANCE - min(path) >= mdd_l:
                blown = (INITIAL_BALANCE - min(path) >= mdd_l)
                break
        finals.append(path[-1])
        ax_p.plot(np.arange(len(path)), path,
                  color='crimson' if blown else 'steelblue', alpha=0.04, linewidth=0.5)

    fe = np.array(finals)
    ax_p.axhline(INITIAL_BALANCE + pt,  color='green', linestyle=':', linewidth=1.2)
    ax_p.axhline(INITIAL_BALANCE - mdd_l, color='red', linestyle=':', linewidth=1.2)
    pct_pass = (fe >= INITIAL_BALANCE + pt).mean() * 100
    pct_blow = (fe <= INITIAL_BALANCE - mdd_l).mean() * 100
    ax_p.set_title(f"MC Paths — Pass {pct_pass:.0f}%  Blown {pct_blow:.0f}%")
    ax_p.set_xlabel('Days'); ax_p.set_ylabel('Balance (£)')
    ax_p.grid(True, alpha=0.25)

    ax_d.hist(fe, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax_d.axvline(INITIAL_BALANCE,        color='gray',  linestyle='--', linewidth=1)
    ax_d.axvline(INITIAL_BALANCE + pt,   color='green', linestyle=':', linewidth=1.5)
    ax_d.axvline(INITIAL_BALANCE - mdd_l, color='red',  linestyle=':', linewidth=1.5)
    ax_d.axvline(float(np.median(fe)),   color='black', linewidth=1.5,
                 label=f"Median £{np.median(fe):,.0f}")
    ax_d.set_title('Final Balance Distribution')
    ax_d.set_xlabel('Balance (£)'); ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


# =============================================================================
# 4B. BAYESIAN OPTIMISATION — Optuna-driven sweep (drop-in for run_sweep)
# =============================================================================

class OptunaGrid:
    """Bayesian parameter search using Optuna TPE.

    Drop-in replacement for ParameterGrid when param spaces are large.
    Use continuous ranges (tuple) for floats/ints, lists for categoricals.

    Example:
        grid = OptunaGrid({
            'tp_r':      (1.0, 4.0),       # float range, sampled continuously
            'adx_min':   (15, 35),          # int range  (both ints → int suggest)
            'ma_req':    [True, False],     # categorical
        }, n_trials=100)
    """

    def __init__(self, param_spaces: dict, n_trials: int = 100):
        self.param_spaces = param_spaces
        self.n_trials = n_trials

    def __len__(self):
        return self.n_trials

    def suggest(self, trial) -> dict:
        """Call inside an Optuna objective to get one param dict."""
        params = {}
        for name, space in self.param_spaces.items():
            if isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space, tuple) and len(space) == 2:
                lo, hi = space
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = trial.suggest_int(name, lo, hi)
                else:
                    params[name] = trial.suggest_float(name, float(lo), float(hi))
            else:
                raise ValueError(
                    f"OptunaGrid: param '{name}' space must be a list or 2-tuple, got {space!r}"
                )
        return params


def run_sweep_optuna(
    sweep_name:        str,
    entry_fn:          callable,
    manager_fn:        callable,
    grid:              'OptunaGrid',
    pairs:             list,
    session:           str,
    regime_mult:       dict,
    train_dfs:         dict,
    test_dfs:          dict,
    n_workers:         int   = 1,
    cost_mult:         float = 1.0,
    progress_callback: callable = None,
) -> str:
    """Optuna TPE-driven sweep. Objective = train-set Sharpe.

    After n_trials, BH correction is applied across all results.
    Returns sweep_id (same as run_sweep).

    Notes
    -----
    - n_workers > 1 uses Optuna's built-in parallel sampler via threading.
      Each trial still calls _run_single_hypothesis in a subprocess to keep
      the backtest isolated, but trial suggestion is sequential in the Optuna
      study (thread-safe).
    - Results are stored in the same SQLite schema as run_sweep().
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError("run_sweep_optuna requires optuna: pip install optuna") from e

    _init_db()
    sweep_id = f"{sweep_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_optuna"
    total     = grid.n_trials
    # review#P2#1 — family-wide DSR for the optuna path too.
    try:
        _con = sqlite3.connect(DB_PATH)
        _n_trials_family = int(_con.execute(
            "SELECT COUNT(*) FROM hypotheses").fetchone()[0]) + total
        _con.close()
    except Exception:
        _n_trials_family = total

    _save_sweep(sweep_id, sweep_name, total, cost_mult)

    # Shared spread/slip overrides (use median pip size as fallback)
    spread_ov = {p: PAIR_PIP_SIZE.get(p, 1e-4) * cost_mult for p in ALL_PAIRS}
    slip_ov   = {p: spread_ov[p] * 0.20 for p in ALL_PAIRS}
    slot_cls  = sweep_name.replace(' ', '_').replace('-', '_').lower()[:20]
    trial_ctr = [0]

    def objective(trial):
        params = grid.suggest(trial)
        trial_ctr[0] += 1
        hyp_id = f"{slot_cls}_opt{trial_ctr[0]:05d}"
        task = {
            'entry_fn':        entry_fn,
            'manager_fn':      manager_fn,
            'params':          params,
            'pairs':           pairs,
            'session':         session,
            'regime_mult':     regime_mult,
            'family':          'session_based',
            'allow_concurrent': False,
            'spread_ov':       spread_ov,
            'slip_ov':         slip_ov,
            'hyp_id':          hyp_id,
            'n_trials_total':  _n_trials_family,  # review#P2#1
            '_train_dfs':      train_dfs,   # inline data for in-process execution
            '_test_dfs':       test_dfs,
        }

        result = _run_single_hypothesis(task)

        if progress_callback:
            progress_callback(trial_ctr[0], total)

        if result and 'error' not in result:
            _save_result(
                sweep_id, result['hyp_id'], result['params'],
                result['train_s'], result['test_s'], result['ext_s'],
                result['train_trades'], result['test_trades'],
                dsr            = result.get('dsr'),
                sharpe_ci_low  = result.get('sharpe_ci_low'),
                sharpe_ci_high = result.get('sharpe_ci_high'),
                regime_stable  = result.get('regime_stable', 0),
                psr            = result.get('psr'),
                outlier_ratio  = result.get('outlier_ratio'),
                wf_sharpe_mean = result.get('wf_sharpe_mean'),  # review#18
                wf_sharpe_min  = result.get('wf_sharpe_min'),
                wf_n_folds     = result.get('wf_n_folds'),
                mc_eval_pass_pct = result.get('mc_eval_pass_pct'),  # §1 Tier 1
                mc_blown_pct     = result.get('mc_blown_pct'),
                mc_n_sims        = result.get('mc_n_sims'),
            )
            if result.get('p_raw') is not None:
                con = sqlite3.connect(DB_PATH)
                con.execute("UPDATE hypotheses SET p_raw=? WHERE hypothesis_id=?",
                            (result['p_raw'], result['hyp_id']))
                con.commit()
                con.close()

        train_sharpe = (result or {}).get('train_s', {})
        if isinstance(train_sharpe, dict):
            train_sharpe = train_sharpe.get('sharpe', 0.0) or 0.0
        else:
            train_sharpe = 0.0
        return float(train_sharpe)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Warm-start: enqueue previously successful param dicts as forced initial
    # trials so TPE begins from known-good regions rather than random.
    warm_params = getattr(grid, 'warm_params', [])
    for wp in warm_params:
        try:
            study.enqueue_trial(wp)
        except Exception:
            pass   # skip if params don't match the current search space

    if n_workers > 1:
        import threading
        threads = []
        trials_per_worker = max(1, total // n_workers)

        def _worker():
            study.optimize(objective, n_trials=trials_per_worker, show_progress_bar=False)

        for _ in range(n_workers):
            t = threading.Thread(target=_worker, daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        study.optimize(objective, n_trials=total, show_progress_bar=False)

    apply_bh_correction(sweep_id)
    apply_pbo_correction(sweep_id)
    return sweep_id


# =============================================================================
# 5. DUKASCOPY TICK DATA FETCHING
# =============================================================================

_SESSION = requests.Session()
_SESSION.headers.update({'User-Agent': 'Mozilla/5.0'})


def _fetch_hour_ticks(duka_inst: str, dt_utc, retries: int = 3) -> list:
    point = DUKA_POINT[duka_inst]
    year  = dt_utc.year
    month = dt_utc.month - 1
    day   = dt_utc.day
    hour  = dt_utc.hour
    url   = f"{DUKA_BASE}/{duka_inst}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=45)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            raw = lzma.decompress(resp.content)
            n   = len(raw) // TICK_BYTES
            out = []
            for i in range(n):
                chunk = raw[i * TICK_BYTES:(i + 1) * TICK_BYTES]
                ms, ask_r, bid_r, av, bv = struct.unpack(TICK_FMT, chunk)
                out.append((ms, ask_r / point, bid_r / point, float(av), float(bv)))
            return out
        except lzma.LZMAError:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                return []
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                print(f"    [{duka_inst}] fetch error {dt_utc} h{hour}: {e}")
    return []


def fetch_day_ticks(instrument: str, day_date) -> pd.DataFrame:
    """Fetch one day's ticks for `instrument`, with hour-level resumable cache.

    Two cache layers:
      1. Day-level: CACHE_DIR/{ticker}_{YYYY-MM-DD}.parquet — written once all
         24 hours have been seen. Subsequent calls return this directly.
      2. Hour-level: CACHE_DIR/{ticker}_hours/{YYYY-MM-DD}_{HH}.parquet —
         written the moment a single hour fetch succeeds. Lets a kill+restart
         lose < 1 hour of work instead of restarting the day.

    An hour with zero ticks (weekends, market-closed) is recorded as an
    empty marker file so we don't re-hit Dukascopy for known-empty hours.
    """
    duka_inst  = DUKA_INST[instrument]
    cache_path = CACHE_DIR / f"{duka_inst}_{day_date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    hour_dir = CACHE_DIR / f"{duka_inst}_hours"
    hour_dir.mkdir(exist_ok=True)

    base_dt = datetime(day_date.year, day_date.month, day_date.day, tzinfo=timezone.utc)
    hour_frames: list = []

    for hour in range(24):
        h_path  = hour_dir / f"{day_date}_{hour:02d}.parquet"
        h_empty = hour_dir / f"{day_date}_{hour:02d}.empty"

        if h_path.exists():
            try:
                hour_frames.append(pd.read_parquet(h_path))
                continue
            except Exception:
                h_path.unlink(missing_ok=True)
        if h_empty.exists():
            continue

        ticks = _fetch_hour_ticks(duka_inst, base_dt.replace(hour=hour))
        if ticks:
            rows = []
            for ms, ask, bid, av, bv in ticks:
                ts = base_dt + timedelta(hours=hour, milliseconds=ms)
                rows.append((ts, ask, bid, av, bv))
            hdf = pd.DataFrame(rows, columns=['timestamp', 'ask', 'bid', 'ask_vol', 'bid_vol'])
            hdf['timestamp'] = pd.to_datetime(hdf['timestamp'], utc=True)
            try:
                hdf.to_parquet(h_path, index=False)
            except Exception:
                pass
            hour_frames.append(hdf)
        else:
            try:
                h_empty.touch()
            except Exception:
                pass
        time.sleep(0.1)  # throttle: ~10 req/sec per pair to avoid Dukascopy rate limiting

    if not hour_frames:
        # All 24 hours empty (weekend / holiday). Mark the day as cached with
        # an empty parquet so we don't re-fetch next run.
        empty = pd.DataFrame(columns=['timestamp', 'ask', 'bid', 'ask_vol', 'bid_vol'])
        try:
            empty.to_parquet(cache_path, index=False)
        except Exception:
            pass
        return empty

    df = pd.concat(hour_frames, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    try:
        df.to_parquet(cache_path, index=False)
        # Day cache committed — hour fragments no longer needed.
        for hour in range(24):
            (hour_dir / f"{day_date}_{hour:02d}.parquet").unlink(missing_ok=True)
            (hour_dir / f"{day_date}_{hour:02d}.empty").unlink(missing_ok=True)
    except Exception:
        pass
    return df


def fetch_all_data(instrument: str, progress_callback=None) -> pd.DataFrame:
    """Download TOTAL_DAYS+5 days of tick data for one instrument.

    progress_callback, if given, is called after each day completes with a
    dict {pair, days_done, days_total, phase}. Used by the GUI Data tab to
    drive the live progress bar; CLI path ignores it.
    """
    end_date   = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=TOTAL_DAYS + 5)
    days       = [start_date + timedelta(days=i)
                  for i in range((end_date - start_date).days)]
    print(f"\n  [{instrument}] {len(days)} days ({start_date} → {end_date})...")

    results = {}
    def _worker(d):
        try:
            return d, fetch_day_ticks(instrument, d)
        except Exception:
            return d, pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        futures = {ex.submit(_worker, d): d for d in days}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            d, df = fut.result()
            results[d] = df
            done += 1
            if done % 60 == 0:
                print(f"    [{instrument}] {done}/{len(days)} days done")
            if progress_callback is not None:
                try:
                    progress_callback({
                        'pair':       instrument,
                        'days_done':  done,
                        'days_total': len(days),
                        'phase':      'downloading',
                    })
                except Exception:
                    pass

    frames = [results[d] for d in sorted(results) if not results[d].empty]
    if not frames:
        print(f"    [{instrument}] WARNING: no tick data")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"    [{instrument}] {len(combined):,} ticks")
    return combined


# =============================================================================
# 6. TICK PROCESSING
# =============================================================================

def process_ticks(tick_df: pd.DataFrame) -> pd.DataFrame:
    if tick_df.empty:
        return tick_df
    df = tick_df.copy()
    df['mid']    = (df['ask'] + df['bid']) / 2
    df['spread'] = df['ask'] - df['bid']
    prev_mid     = df['mid'].shift(1)
    df['direction'] = np.where(df['mid'] > prev_mid,  1,
                      np.where(df['mid'] < prev_mid, -1, np.nan))
    df['direction'] = df['direction'].ffill().fillna(0).astype(np.int8)
    return df


def ticks_to_m1(tick_df: pd.DataFrame, tz: str = 'Europe/London') -> pd.DataFrame:
    if tick_df.empty:
        return pd.DataFrame()
    df = process_ticks(tick_df)
    df['ts_local'] = df['timestamp'].dt.tz_convert(tz)
    df['bar']      = df['ts_local'].dt.floor('1min')

    def _rv(series):
        lr = np.diff(np.log(series.values))
        return float(np.sqrt((lr ** 2).sum())) if len(lr) > 0 else 0.0

    agg = df.groupby('bar').agg(
        open           = ('mid',       'first'),
        high           = ('mid',       'max'),
        low            = ('mid',       'min'),
        close          = ('mid',       'last'),
        ask_vol_sum    = ('ask_vol',   'sum'),
        bid_vol_sum    = ('bid_vol',   'sum'),
        tick_imbalance = ('direction', 'mean'),
        spread_mean    = ('spread',    'mean'),
        spread_max     = ('spread',    'max'),
        tick_count     = ('mid',       'count'),
    )
    agg['volume']       = agg['ask_vol_sum'] + agg['bid_vol_sum']
    total_safe          = agg['volume'].replace(0, np.nan)
    agg['vol_imbalance'] = (agg['ask_vol_sum'] - agg['bid_vol_sum']) / total_safe
    agg['vol_imbalance'] = agg['vol_imbalance'].fillna(0.0)
    agg['delta']         = agg['ask_vol_sum'] - agg['bid_vol_sum']

    # Vectorised realized vol — avoids slow groupby.apply over 350k groups
    log_mid             = np.log(df['mid'].replace(0, np.nan))
    log_ret             = log_mid.diff()
    log_ret[df['bar'] != df['bar'].shift(1)] = np.nan  # zero bar boundaries
    df['_lr2']          = log_ret ** 2
    rv_vals             = np.sqrt(df.groupby('bar')['_lr2'].sum().fillna(0.0))
    df.drop(columns=['_lr2'], inplace=True)
    agg['realized_vol'] = rv_vals

    # Vectorised up-tick count — avoids slow groupby.apply
    df['_up']           = (df['direction'] > 0).astype('int8')
    up_ticks            = df.groupby('bar')['_up'].sum()
    df.drop(columns=['_up'], inplace=True)
    agg['aggressive_buy_ratio'] = (up_ticks / agg['tick_count']).fillna(0.0)
    bar_range           = (agg['high'] - agg['low']).replace(0, np.nan)
    agg['bar_momentum']  = (agg['close'] - agg['open']) / bar_range
    agg['close_location'] = (agg['close'] - agg['low']) / bar_range
    agg['bar_momentum']   = agg['bar_momentum'].fillna(0.0)
    agg['close_location'] = agg['close_location'].fillna(0.5)
    agg = agg.drop(columns=['ask_vol_sum', 'bid_vol_sum'])
    agg.index.name = 'timestamp'
    return agg.reset_index()


# =============================================================================
# 7. DATA PREPARATION
# =============================================================================

class _GaussianHMM:
    """Vectorised Gaussian HMM — pure NumPy/SciPy, no external dependencies.

    Uses full covariance matrices.  Baum-Welch EM for training, Viterbi for
    decoding, forward-backward for posterior state probabilities.
    All inner loops are O(T) with numpy vectorisation; suitable for T~350k.
    """

    def __init__(self, n_components=4, n_iter=50, tol=1e-3, random_state=42):
        self.n_components = n_components
        self.n_iter       = n_iter
        self.tol          = tol
        self.rng          = np.random.RandomState(random_state)

    # ── Emission log-probabilities ────────────────────────────────────────────
    def _log_emit(self, X):
        from scipy.special import logsumexp as _lse   # noqa: F401 (used below)
        T, D   = X.shape
        N      = self.n_components
        log_b  = np.zeros((T, N))
        for k in range(N):
            diff    = X - self.means_[k]             # (T, D)
            cov_inv = np.linalg.inv(self.covs_[k])
            log_det = np.linalg.slogdet(self.covs_[k])[1]
            maha    = np.einsum('td,dd,td->t', diff, cov_inv, diff)
            log_b[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)
        return log_b

    # ── Forward pass (log-domain) ─────────────────────────────────────────────
    def _forward(self, log_b):
        from scipy.special import logsumexp
        T, N    = log_b.shape
        log_A   = np.log(self.transmat_  + 1e-300)
        log_a   = np.full((T, N), -np.inf)
        log_a[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        for t in range(1, T):
            log_a[t] = logsumexp(log_a[t-1, :, None] + log_A,
                                 axis=0) + log_b[t]
        return log_a

    # ── Backward pass (log-domain) ────────────────────────────────────────────
    def _backward(self, log_b):
        from scipy.special import logsumexp
        T, N   = log_b.shape
        log_A  = np.log(self.transmat_ + 1e-300)
        log_be = np.zeros((T, N))
        for t in range(T - 2, -1, -1):
            log_be[t] = logsumexp(log_A + log_b[t+1] + log_be[t+1],
                                  axis=1)
        return log_be

    # ── K-means initialisation ────────────────────────────────────────────────
    def _kmeans_init(self, X):
        from sklearn.cluster import KMeans
        N = self.n_components
        km     = KMeans(n_clusters=N, random_state=self.rng.randint(1000),
                        n_init=5, max_iter=100)
        labels = km.fit_predict(X)
        D      = X.shape[1]
        self.startprob_ = np.ones(N) / N
        self.transmat_  = np.ones((N, N)) / N
        self.means_     = km.cluster_centers_.copy()
        self.covs_      = np.array([
            np.cov(X[labels == k].T) + np.eye(D) * 1e-3
            if (labels == k).sum() > 1
            else np.eye(D)
            for k in range(N)
        ])

    # ── Baum-Welch EM ─────────────────────────────────────────────────────────
    def fit(self, X):
        from scipy.special import logsumexp
        T, D = X.shape
        N    = self.n_components
        self._kmeans_init(X)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            log_b  = self._log_emit(X)
            log_a  = self._forward(log_b)
            log_be = self._backward(log_b)
            ll     = logsumexp(log_a[-1])
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            log_g = log_a + log_be
            log_g -= logsumexp(log_g, axis=1, keepdims=True)
            g      = np.exp(log_g)             # (T, N)  gamma

            # Xi (joint state pairs) — vectorised over all T-1 steps
            log_A = np.log(self.transmat_ + 1e-300)
            log_xi = (log_a[:-1, :, None]      # (T-1, N, 1)
                      + log_A[None]             # (  1, N, N)
                      + log_b[1:, None, :]      # (T-1, 1, N)
                      + log_be[1:, None, :])    # (T-1, 1, N)
            log_xi -= ll
            xi = np.exp(logsumexp(log_xi, axis=0))  # (N, N)

            self.startprob_ = g[0] / (g[0].sum() + 1e-300)
            self.transmat_  = xi / (xi.sum(axis=1, keepdims=True) + 1e-300)
            for k in range(N):
                w  = g[:, k]
                ws = w.sum() + 1e-300
                self.means_[k] = (w @ X) / ws
                diff            = X - self.means_[k]
                self.covs_[k]   = (diff.T * w) @ diff / ws + np.eye(D) * 1e-4
        return self

    # ── Viterbi decoding ──────────────────────────────────────────────────────
    def predict(self, X):
        T, N  = X.shape[0], self.n_components
        log_b = self._log_emit(X)
        log_A = np.log(self.transmat_ + 1e-300)
        v     = np.full((T, N), -np.inf)
        bp    = np.zeros((T, N), dtype=int)
        v[0]  = np.log(self.startprob_ + 1e-300) + log_b[0]
        for t in range(1, T):
            trans   = v[t-1, :, None] + log_A   # (N, N)
            bp[t]   = trans.argmax(0)
            v[t]    = trans.max(0) + log_b[t]
        states      = np.zeros(T, dtype=int)
        states[-1]  = v[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = bp[t+1, states[t+1]]
        return states

    # ── Posterior state probabilities ─────────────────────────────────────────
    def predict_proba(self, X):
        from scipy.special import logsumexp
        log_b = self._log_emit(X)
        log_a = self._forward(log_b)
        log_be= self._backward(log_b)
        log_g = log_a + log_be
        log_g -= logsumexp(log_g, axis=1, keepdims=True)
        return np.exp(log_g)


def _fit_and_apply_hmm(df: pd.DataFrame,
                        n_states: int = HMM_N_STATES) -> pd.DataFrame:
    """Fit a Gaussian HMM on training-period bars, apply to the full dataset.

    Trained only on bars before the test split to prevent look-ahead leakage.
    Adds per-bar columns:
        hmm_state        — Viterbi most-likely hidden state (0..n_states-1)
        hmm_prob_0..N-1  — posterior probability of each state (forward-backward)
        hmm_transition   — 1 when state changed from previous bar
    """
    from sklearn.preprocessing import StandardScaler

    feat_cols = [c for c in HMM_FEATURES if c in df.columns]
    if len(feat_cols) < 3:
        return df

    split_date = (datetime.now(timezone.utc) - timedelta(days=TEST_DAYS)).date()
    train_mask = (df['date'] < split_date).values
    n_train    = int(train_mask.sum())
    if n_train < n_states * 50:
        print(f"    [HMM] insufficient training data ({n_train} bars) — skipping")
        return df

    # review#5 — fill NaNs with TRAIN-only median, not full-series median.
    # Test-period observations must not influence training fill values.
    train_median = df.loc[train_mask, feat_cols].median()
    X_raw        = df[feat_cols].fillna(train_median)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_raw.values[train_mask])
    X_all   = scaler.transform(X_raw.values)

    # Subsample training sequence to cap fitting time — temporal order preserved
    HMM_FIT_CAP  = 5_000
    HMM_CHUNK    = 5_000   # apply in chunks to avoid O(T) Python loop over full series
    if len(X_train) > HMM_FIT_CAP:
        step   = max(1, len(X_train) // HMM_FIT_CAP)
        X_fit  = X_train[::step]
    else:
        X_fit  = X_train

    try:
        model = _GaussianHMM(n_components=n_states, n_iter=15, random_state=42)
        model.fit(X_fit)

        # Apply to full dataset in fixed-size chunks (avoids single O(T) Python loop)
        states_parts = []
        probs_parts  = []
        for start in range(0, len(X_all), HMM_CHUNK):
            chunk = X_all[start:start + HMM_CHUNK]
            states_parts.append(model.predict(chunk))
            probs_parts.append(model.predict_proba(chunk))
        states = np.concatenate(states_parts)
        probs  = np.concatenate(probs_parts, axis=0)

        df['hmm_state'] = states
        for i in range(n_states):
            df[f'hmm_prob_{i}'] = probs[:, i]
        state_s = pd.Series(states, index=df.index)
        df['hmm_transition'] = (state_s.diff().fillna(0) != 0).astype(int)

        counts = dict(zip(*np.unique(states, return_counts=True)))
        print(f"    [HMM] fitted on {len(X_fit):,} bars (subsample)  "
              f"applied to {len(X_all):,}  state_counts={counts}")
    except Exception as e:
        print(f"    [HMM] fitting failed ({e}) — skipping")

    return df


def prepare_df(m1_df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    # review#5 — train/test lookahead bias audit (COMPLETE).
    # First pass fixed two confirmed leaks: HMM NaN-fill now uses train-only
    # median (see _fit_hmm) and measured spreads are computed over train
    # window only (see load_all_data).
    # Second pass (review#5 deep audit) confirmed leak-free for everything
    # else: every rolling feature uses .rolling(N) with default closed='right'
    # (past-only window), .shift() values are non-negative (always past),
    # cumulative ops (.cumsum, .cummax) are within-day or post-trade equity,
    # and .ffill is forward-fill (past propagation, not future). No
    # series-wide z-scoring, qcut bins, or expanding() statistics are used
    # for entry features. The only series-wide aggregates (e.g. spread_mean
    # printout, pair date trim) are reporting/coverage, not features.
    # Existing edge_results.db rows pre-date these fixes; treat as historical.
    session = PAIR_SESSION[instrument]
    cfg     = SESSION_CONFIG[session]
    pip     = PAIR_PIP_SIZE[instrument]

    if m1_df.empty:
        return m1_df

    df = m1_df.copy()
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/London')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/London')

    df['date']   = df['timestamp'].dt.date
    df['hour']   = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    df['active_session'] = (
        (df['hour'] >= cfg['active_start']) & (df['hour'] < cfg['active_end'])
    )

    # Phase 6.3 — impact-aware buffer. Accept both legacy 2-tuples
    # (h, m) and 3-tuples (h, m, impact) so older configs keep working
    # but new pipelines that ingest ForexFactory's impact tag get a
    # buffer width that matches the event severity.
    hit = cfg['high_impact_times']
    def _norm_event(e):
        if len(e) >= 3:
            return e[0], e[1], (str(e[2]) or 'high').lower()
        return e[0], e[1], 'high'
    norm_hit = [_norm_event(e) for e in hit] if hit else []

    def _near_news_impact(row):
        bm = row['hour'] * 60 + row['minute']
        worst = ''
        for h, m, imp in norm_hit:
            buf = NEWS_BUFFER_BY_IMPACT.get(imp, NEWS_BUFFER_MINS)
            if buf > 0 and abs(bm - (h * 60 + m)) <= buf:
                # rank high > medium > low
                if imp == 'high':
                    return imp
                if imp == 'medium' and worst != 'high':
                    worst = 'medium'
                elif imp == 'low' and worst not in ('high', 'medium'):
                    worst = 'low'
        return worst

    impact_series = df.apply(_near_news_impact, axis=1)
    df['near_news_impact'] = impact_series
    df['near_news'] = impact_series != ''

    df['ma_trend'] = df['close'].rolling(200).mean().shift(1)

    adx, atr = compute_adx_atr(df)
    df['adx']      = adx
    df['atr']      = atr
    atr_mean       = atr.rolling(REGIME_ATR_REF_BARS).mean().shift(1)
    df['atr_ratio'] = (atr / atr_mean).where(atr_mean > 0)
    df['regime']   = [classify_regime(a, r) for a, r in zip(df['adx'], df['atr_ratio'])]

    # Hurst exponent (rolling 200-bar) — H>0.55 trending-favourable, H<0.45 fade-favourable
    df['hurst'] = rolling_hurst(df['close'], window=200)

    # Yang-Zhang realized volatility (rolling 30-bar) and its rolling baseline.
    # yz_vol_ratio = current YZ vol / 60-bar median — > 1 means elevated vol.
    # Median is shifted so the denominator excludes the current bar — without
    # this, an entry function reading yz_vol_ratio peeks at its own bar's vol.
    df['yz_vol']        = compute_yang_zhang_vol(df, window=30)
    yz_med              = df['yz_vol'].rolling(60, min_periods=20).median().shift(1)
    df['yz_vol_ratio']  = (df['yz_vol'] / yz_med.replace(0, np.nan)).clip(0, 10)

    # Permutation entropy (Bandt-Pompe) — ordinal-pattern complexity of returns.
    # Orthogonal to Hurst: low PE = predictable shape, high PE = near-random.
    df['perm_entropy_100'] = rolling_perm_entropy(df['close'], window=100, order=4)

    # Hawkes self-exciting intensity ratio over strong tick-imbalance events.
    # Captures order-flow CLUSTERING that thresholded tick_imbalance can't see —
    # ratio > 2 means flow events arriving 2× faster than baseline (informed
    # flow regime), <0.5 means drought.
    if 'tick_imbalance' in df.columns:
        ev = (df['tick_imbalance'].abs() > 0.5).astype(float)
        df['hawkes_intensity'] = compute_hawkes_intensity(
            ev, decay=0.05, baseline_window=240,
        )
    else:
        df['hawkes_intensity'] = 1.0

    df['spread_median'] = df['spread_mean'].rolling(SPREAD_MEDIAN_WINDOW, min_periods=10).median().shift(1)
    df['rv_median']     = df['realized_vol'].rolling(RV_SCALE_WINDOW,    min_periods=10).median().shift(1)

    asian_bars = df[(df['hour'] >= ASIAN_HOUR_START) & (df['hour'] < ASIAN_HOUR_END)]
    if not asian_bars.empty:
        asian_agg = asian_bars.groupby('date').agg(
            asian_high=('high', 'max'), asian_low=('low', 'min'),
            asian_bars_n=('close', 'count'),
        ).reset_index()
        asian_agg['asian_valid'] = asian_agg['asian_bars_n'] >= 60
        df = df.merge(
            asian_agg[['date', 'asian_high', 'asian_low', 'asian_valid']],
            on='date', how='left')
        bad = ~df['asian_valid'].fillna(False).astype(bool)
        df.loc[bad, ['asian_high', 'asian_low']] = np.nan
    else:
        df['asian_high'] = df['asian_low'] = np.nan

    df['swing_high5'] = df['high'].rolling(5).max().shift(1)
    df['swing_low5']  = df['low'].rolling(5).min().shift(1)

    rh, rms, rme = cfg['range_hour'], cfg['range_min_start'], cfg['range_min_end']
    range_bars = df[(df['hour'] == rh) & (df['minute'] >= rms) & (df['minute'] <= rme)]
    range_agg  = range_bars.groupby('date').agg(
        range_high=('high', 'max'), range_low=('low', 'min'), range_bars=('close', 'count'),
    ).reset_index()
    range_agg['range_size'] = range_agg['range_high'] - range_agg['range_low']
    valid_range = range_agg[
        (range_agg['range_bars'] >= MIN_RANGE_BARS) &
        (range_agg['range_size'] >= PAIR_MIN_RANGE_PIPS[instrument] * pip) &
        (range_agg['range_size'] <= PAIR_MAX_RANGE_PIPS[instrument] * pip)
    ]
    df = df.merge(valid_range[['date', 'range_high', 'range_low', 'range_size']],
                  on='date', how='left')

    # ── Microstructure causality features ────────────────────────────────────
    if 'delta' in df.columns:
        df['cumulative_delta'] = df.groupby('date')['delta'].cumsum()
        df['delta_ma5']       = df['delta'].rolling(5, min_periods=1).mean()
        df['delta_momentum']  = df['delta'] - df['delta_ma5']
        price_dir             = np.sign(df['close'] - df['close'].shift(1))
        df['delta_divergence'] = ((price_dir * np.sign(df['delta_momentum'])) < 0).astype(float)

    if 'tick_imbalance' in df.columns:
        df['persistent_imbalance'] = (
            np.sign(df['tick_imbalance']).rolling(5, min_periods=1).sum()
        )

    if 'atr' in df.columns:
        atr_safe = df['atr'].replace(0, np.nan)
        df['bar_range_pct'] = (df['high'] - df['low']) / atr_safe
    if all(c in df.columns for c in ['bar_range_pct', 'spread_max', 'spread_median',
                                      'close_location']):
        spread_spike      = (df['spread_max']
                             / df['spread_median'].replace(0, np.nan)).clip(upper=10)
        reversal_strength = (1 - (df['close_location'] - 0.5).abs() * 2).clip(lower=0)
        df['stop_run_score'] = (df['bar_range_pct'] * spread_spike
                                * reversal_strength).fillna(0.0)

    if 'active_session' in df.columns:
        df['bars_since_open'] = (
            df.groupby('date')['active_session']
            .transform(lambda x: x.astype(int).cumsum())
            .where(df['active_session'], 0)
        )
        session_length = df.groupby('date')['active_session'].transform('sum').clip(lower=1)
        df['bar_phase'] = (
            (df['bars_since_open'] / session_length * 3).clip(0, 2.99).astype(int)
        )

    daily_last_close = df.groupby('date')['close'].last()
    prev_close       = daily_last_close.shift(1)
    first_open       = df.groupby('date')['open'].transform('first')
    df['daily_gap']  = (first_open - df['date'].map(prev_close)).abs().fillna(0.0)

    df = _fit_and_apply_hmm(df)

    # ── Extended features: fill gaps referenced in system prompt ──────────────
    atr_s = df['atr'].replace(0, np.nan)

    # Lag / derivative features for order-flow signals
    if 'tick_imbalance' in df.columns:
        df['tick_imb_lag1']   = df['tick_imbalance'].shift(1)
        df['tick_imb_roll5']  = df['tick_imbalance'].rolling(5, min_periods=1).mean().shift(1)
        df['tick_imb_delta']  = df['tick_imbalance'] - df['tick_imb_roll5']
    if 'vol_imbalance' in df.columns:
        df['vol_imb_lag1'] = df['vol_imbalance'].shift(1)

    # Structural distance features
    if 'ma_trend' in df.columns:
        df['ma_dist'] = (df['close'] - df['ma_trend']) / atr_s
    if 'swing_high5' in df.columns:
        df['swing_high_dist'] = (df['swing_high5'] - df['close']) / atr_s
        df['swing_low_dist']  = (df['close'] - df['swing_low5'])  / atr_s

    # Volatility momentum
    if 'realized_vol' in df.columns and 'rv_median' in df.columns:
        rv_med_s = df['rv_median'].replace(0, np.nan)
        df['rv_delta'] = df['realized_vol'] / rv_med_s

    # ATR percentile rank (0 = quietest, 1 = most volatile) over rolling 2-day window
    df['atr_rank'] = df['atr'].rolling(288, min_periods=60).rank(pct=True).fillna(0.5)

    # Previous day's high/low — structural support/resistance levels
    daily_high = df.groupby('date')['high'].max()
    daily_low  = df.groupby('date')['low'].min()
    df['prev_day_high'] = df['date'].map(daily_high.shift(1))
    df['prev_day_low']  = df['date'].map(daily_low.shift(1))
    df['dist_prev_high'] = (df['prev_day_high'] - df['close']) / atr_s
    df['dist_prev_low']  = (df['close'] - df['prev_day_low'])  / atr_s

    # Multi-bar price momentum in ATR units
    df['momentum_3']  = (df['close'] - df['close'].shift(3))  / atr_s
    df['momentum_10'] = (df['close'] - df['close'].shift(10)) / atr_s

    # Bollinger Band %B — position within ±2σ band (0 = lower band, 1 = upper band)
    bb_mid = df['close'].rolling(20, min_periods=10).mean()
    bb_std = df['close'].rolling(20, min_periods=10).std().replace(0, np.nan)
    df['bb_pct'] = ((df['close'] - (bb_mid - bb_std)) / (2 * bb_std)).clip(0, 1).fillna(0.5)

    # RSI-14
    delta_c = df['close'].diff()
    gain    = delta_c.clip(lower=0).rolling(14, min_periods=14).mean()
    loss    = (-delta_c.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs      = gain / loss.replace(0, np.nan)
    df['rsi_14'] = (100 - 100 / (1 + rs)).fillna(50.0)

    # Day of week (0=Monday … 4=Friday) — sessions behave differently Monday vs Friday
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # ── Higher-timeframe context (built from M1 — no extra data fetch) ──────────
    # H1 trend: sign of EMA(60) - EMA(240) on close. +1=up, -1=down, 0=flat.
    h1_fast = df['close'].ewm(span=60,  min_periods=60,  adjust=False).mean()
    h1_slow = df['close'].ewm(span=240, min_periods=240, adjust=False).mean()
    df['h1_trend'] = np.sign(h1_fast - h1_slow).fillna(0).astype(int)
    df['h1_trend_strength'] = ((h1_fast - h1_slow) / atr_s).fillna(0.0)

    # H4 ATR: 240-bar ATR (rolling 4-hour true-range mean).
    h4_tr = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['h4_atr']       = h4_tr.rolling(240, min_periods=60).mean()
    df['h4_atr_ratio'] = (df['atr'] / df['h4_atr'].replace(0, np.nan)).fillna(1.0)

    # Daily classic pivot from previous day (P, R1, S1) and position vs P.
    daily_close_last = df.groupby('date')['close'].last()
    daily_high_p     = df.groupby('date')['high'].max()
    daily_low_p      = df.groupby('date')['low'].min()
    pivot_p          = (daily_high_p.shift(1) + daily_low_p.shift(1) + daily_close_last.shift(1)) / 3.0
    pivot_r1         = (2 * pivot_p) - daily_low_p.shift(1)
    pivot_s1         = (2 * pivot_p) - daily_high_p.shift(1)
    df['daily_pivot']    = df['date'].map(pivot_p)
    df['daily_pivot_r1'] = df['date'].map(pivot_r1)
    df['daily_pivot_s1'] = df['date'].map(pivot_s1)
    df['daily_pivot_position'] = ((df['close'] - df['daily_pivot']) / atr_s).fillna(0.0)
    df['dist_pivot_r1'] = ((df['daily_pivot_r1'] - df['close']) / atr_s).fillna(0.0)
    df['dist_pivot_s1'] = ((df['close'] - df['daily_pivot_s1']) / atr_s).fillna(0.0)

    # ── Execution-realism: spread_adj includes news-time widening + commission ──
    # review#10 — commission convention. Backtest fills use hspd = spread_adj/2
    # for both entry and exit, so a value added once to spread_adj is charged
    # at half on each side (= once round-trip total). The COMMISSION_PIPS
    # constant is named per-side, which is the standard retail convention; to
    # match that, charge 2 * COMMISSION_PIPS in spread_adj so each round-trip
    # actually pays 2 * COMMISSION_PIPS. Was previously single-applied, which
    # under-modelled live cost by ~0.3 pips per round-trip on majors.
    from agent.config import NEWS_SPREAD_MULT, COMMISSION_PIPS
    pip          = PAIR_PIP_SIZE.get(instrument, 0.0001)
    commission_p = 2.0 * COMMISSION_PIPS * pip   # round-trip = 2 × per-side
    news_mult    = df['near_news'].map({True: NEWS_SPREAD_MULT, False: 1.0}).fillna(1.0)
    df['spread_adj'] = df['spread_mean'] * news_mult + commission_p

    # Phase 9 — best-effort macro / COT / retail-positioning feature merge.
    # All sources are network-cached; failures degrade silently and the rest
    # of the feature set continues to work.
    try:
        from agent import macro_data
        df = macro_data.merge_macro_features(df, instrument)
    except Exception:
        pass

    print(f"    {instrument}: {len(df):,} M1 bars  "
          f"median_spread={df['spread_mean'].median():.6f}  "
          f"adj_spread={df['spread_adj'].median():.6f}")
    return df


# =============================================================================
# 8. DATA LOADING — call once; results cached to prepared parquet
# =============================================================================

def _add_cross_pair_features(pair_dfs: dict) -> dict:
    """Add DXY proxy + cross-pair correlation/lag features in-place per pair.

    All pair DataFrames share a UTC-aligned 1-minute grid. We build a wide
    close-price matrix, derive cross signals on it, and broadcast back per-pair.
    Pure post-processing — no I/O, fast vectorised work.
    """
    if not pair_dfs:
        return pair_dfs

    closes = {}
    for pair, df in pair_dfs.items():
        if df.empty or 'timestamp' not in df.columns:
            continue
        s = df.set_index('timestamp')['close']
        closes[pair] = s[~s.index.duplicated(keep='last')]

    if not closes:
        return pair_dfs

    wide = pd.DataFrame(closes).sort_index()

    # ── DXY proxy: equal-weighted log-price of inverse-USD-quoted majors ─────
    # Components present in our universe: EUR/USD, GBP/USD, AUD/USD (all USD-quoted).
    # USD strengthens when these fall, so DXY proxy = -mean(log price of these).
    inv_components = [p for p in ('EUR_USD', 'GBP_USD', 'AUD_USD') if p in wide.columns]
    if inv_components:
        log_inv = np.log(wide[inv_components].replace(0, np.nan)).mean(axis=1)
        dxy_proxy        = (-log_inv).rename('dxy_proxy')
        dxy_change_5     = dxy_proxy.diff(5).rename('dxy_change_5')
        dxy_change_60    = dxy_proxy.diff(60).rename('dxy_change_60')
    else:
        dxy_proxy = dxy_change_5 = dxy_change_60 = pd.Series(dtype=float)

    # ── EU/GU 20-bar rolling correlation + 1-bar lead/lag ─────────────────────
    if 'EUR_USD' in wide.columns and 'GBP_USD' in wide.columns:
        eu_ret = wide['EUR_USD'].pct_change()
        gu_ret = wide['GBP_USD'].pct_change()
        eu_gu_corr20 = eu_ret.rolling(20, min_periods=10).corr(gu_ret).rename('eu_gu_corr20')
        eu_gu_lag1   = eu_ret.shift(1).rename('eu_gu_lag1')
        gu_eu_lag1   = gu_ret.shift(1).rename('gu_eu_lag1')
    else:
        eu_gu_corr20 = eu_gu_lag1 = gu_eu_lag1 = pd.Series(dtype=float)

    # ── Build cross-feature wide frame indexed by timestamp ───────────────────
    cross = pd.concat(
        [s for s in (dxy_proxy, dxy_change_5, dxy_change_60,
                     eu_gu_corr20, eu_gu_lag1, gu_eu_lag1) if not s.empty],
        axis=1,
    )
    if cross.empty:
        return pair_dfs

    # Broadcast back into each per-pair df (left-merge on timestamp)
    out = {}
    for pair, df in pair_dfs.items():
        if df.empty:
            out[pair] = df
            continue
        merged = df.merge(
            cross, left_on='timestamp', right_index=True, how='left',
        )
        # Fill forward small gaps so isnan checks downstream behave well
        for col in cross.columns:
            if col in merged.columns:
                merged[col] = merged[col].ffill(limit=5)
        out[pair] = merged

    return out


def load_all_data(force_refresh: bool = False, pairs: list | None = None,
                  progress_callback=None) -> tuple:
    """Download + prepare pairs. Returns (train_dfs, test_dfs, measured_spreads).

    Results are cached to edge_prepared_cache/ so subsequent calls are instant.
    Set force_refresh=True to re-download everything.

    Args:
      pairs: Subset of ALL_PAIRS to load. None = read from gui_config.json
             selected_pairs (falls back to ALL_PAIRS if not set). Pass an
             explicit list to override the user's GUI selection.
      progress_callback: Optional callable(dict) forwarded to fetch_all_data.
    """
    if pairs is None:
        try:
            from agent import gui_config as _gui_cfg
            pairs = _gui_cfg.selected_pairs() or ALL_PAIRS
        except Exception:
            pairs = ALL_PAIRS
    pairs = [p for p in pairs if p in ALL_PAIRS]
    if not pairs:
        raise RuntimeError("load_all_data: no valid pairs selected")

    print("=" * 65)
    print(f"  Edge Discovery Engine — loading data for {', '.join(pairs)}")
    print("=" * 65)

    # review#P3#4 — per-pair status. Pairs that fail to load get a status
    # recorded; written to runtime/load_status.json so loop.py can read it
    # without us having to change the return-signature contract. ok/failed/
    # skipped + last error message.
    _pair_status: dict = {p: {'status': 'pending', 'error': ''} for p in pairs}

    pair_dfs = {}
    for pair in pairs:
        cache_path = PREPARED_CACHE_DIR / f"{pair}_m1.parquet"
        if not force_refresh and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if 'date' in df.columns and not df.empty:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                if 'timestamp' in df.columns and not df.empty:
                    if df['timestamp'].dt.tz is None:
                        df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/London')
                pair_dfs[pair] = df
                print(f"  [{pair}] loaded from cache ({len(df):,} bars)")
                _pair_status[pair] = {'status': 'ok', 'error': '', 'source': 'cache'}
                continue
            except Exception as e:
                print(f"  [{pair}] Cache corrupt ({e}) — re-downloading")
                cache_path.unlink(missing_ok=True)
        if True:
            raw = pd.DataFrame()
            for _attempt in range(3):
                raw = fetch_all_data(pair, progress_callback=progress_callback)
                if not raw.empty:
                    break
                wait = 60 * (_attempt + 1)
                print(f"  [{pair}] WARNING: no data (attempt {_attempt + 1}/3) — "
                      f"rate-limited? Waiting {wait}s before retry...")
                time.sleep(wait)
            if raw.empty:
                print(f"  [{pair}] SKIPPING — all 3 download attempts returned empty.")
                _pair_status[pair] = {'status': 'failed',
                                      'error': 'all download attempts returned empty'}
                continue
            time.sleep(30)  # rest between pairs to let Dukascopy CDN rate limit recover
            m1       = ticks_to_m1(raw)
            df       = prepare_df(m1, pair)
            if df.empty:
                _pair_status[pair] = {'status': 'failed', 'error': 'prepare_df returned empty'}
                continue
            # Parquet doesn't support date objects — convert to string for storage
            save_df = df.copy()
            save_df['date'] = save_df['date'].astype(str)
            save_df.to_parquet(cache_path, index=False)
            pair_dfs[pair] = df
            _pair_status[pair] = {'status': 'ok', 'error': '', 'source': 'fresh'}

    # review#P3#4 — write per-pair status to runtime/load_status.json so
    # callers (loop.py, mission_control GUI) can detect partial-load
    # without us changing the return-signature contract.
    try:
        _runtime = Path(__file__).parent / 'runtime'
        _runtime.mkdir(exist_ok=True)
        with open(_runtime / 'load_status.json', 'w') as _f:
            json.dump({
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'pairs':       _pair_status,
            }, _f, indent=2)
    except Exception as _e:
        print(f"  [load_status] write failed: {_e}")

    if not pair_dfs:
        raise RuntimeError("No data loaded. Check your internet connection.")

    # review#19 — trim every pair to the common date range. Different pairs
    # have staggered Dukascopy start dates (~10-day spread observed); without
    # this, cross-pair regime/correlation features train on offset windows.
    common_min = max(df['date'].min() for df in pair_dfs.values())
    common_max = min(df['date'].max() for df in pair_dfs.values())
    print(f"\n  Cross-pair common date range: {common_min} → {common_max}")
    for pair in list(pair_dfs.keys()):
        df = pair_dfs[pair]
        before = len(df)
        df = df[(df['date'] >= common_min) & (df['date'] <= common_max)].reset_index(drop=True)
        after = len(df)
        if before != after:
            print(f"  [{pair}] trimmed {before-after} bars to common range "
                  f"({before} → {after})")
        pair_dfs[pair] = df

    # ── Cross-pair features: DXY proxy + correlation/lag signals ──────────────
    # All pair DFs share a 1-minute timestamp grid. Build a wide close matrix,
    # derive DXY proxy from inverse-USD-quoted majors, then write back per-pair
    # features (dxy_proxy, dxy_change_5, eu_gu_corr20, eu_gu_lag1, etc.).
    pair_dfs = _add_cross_pair_features(pair_dfs)

    # review#5 — measured spreads must reflect TRAIN-period costs only,
    # otherwise test-period spread observations leak into the cost baseline
    # used by training backtests.
    split_date     = (datetime.now(timezone.utc) - timedelta(days=TEST_DAYS)).date()
    measured_spreads = {
        pair: float(df.loc[df['date'] < split_date, 'spread_mean'].median())
        for pair, df in pair_dfs.items()
    }
    # review#11 — write the measured spreads with an updated_at timestamp so
    # load_measured_spreads can refuse stale data.
    payload = {
        '_meta': {
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'window':     'train_only',
        },
        'spreads': measured_spreads,
    }
    with open(SPREADS_JSON, 'w') as f:
        json.dump(payload, f)
    print("\n  Measured spreads (median, TRAIN window only):")
    for pair, spd in measured_spreads.items():
        pip = PAIR_PIP_SIZE[pair]
        print(f"    {PAIR_LABELS[pair]}: {spd:.6f}  ({spd/pip:.2f} pips)")

    # ── Live spread override (review#P2#4 — live evidence beats Dukascopy proxy)
    # If `live_measured_spreads.json` exists at the project root (produced by
    # tools/calibrate_costs.py or agent/tca.py), prefer per-pair live spreads
    # over the Dukascopy medians and rescale the per-bar spread_mean column so
    # dynamic-mode spread_adj reflects live too. Static-mode is affected via
    # the measured_spreads dict; dynamic-mode via the rescaled spread_adj.
    live_path = Path("live_measured_spreads.json")
    if live_path.exists():
        try:
            with open(live_path) as _f:
                _live_data = json.load(_f)
            _live_spreads = _live_data.get("spreads", {})
        except Exception as _e:
            print(f"  [live spreads] WARNING: failed to read {live_path}: {_e}")
            _live_spreads = {}
        if _live_spreads:
            try:
                from agent.config import NEWS_SPREAD_MULT, COMMISSION_PIPS
            except Exception:
                NEWS_SPREAD_MULT, COMMISSION_PIPS = 2.0, 0.3
            print("\n  Live spread overrides applied:")
            for _pair, _live_spd in _live_spreads.items():
                if _pair not in measured_spreads:
                    continue
                _old = measured_spreads[_pair]
                _new = float(_live_spd)
                _scale = (_new / _old) if _old > 0 else 1.0
                measured_spreads[_pair] = _new
                _pip = PAIR_PIP_SIZE[_pair]
                _commission_p = 2.0 * COMMISSION_PIPS * _pip
                _df = pair_dfs[_pair]
                _df["spread_mean"] = _df["spread_mean"] * _scale
                if "spread_median" in _df.columns:
                    _df["spread_median"] = _df["spread_median"] * _scale
                _news_mult = (_df["near_news"]
                              .map({True: NEWS_SPREAD_MULT, False: 1.0})
                              .fillna(1.0))
                _df["spread_adj"] = _df["spread_mean"] * _news_mult + _commission_p
                print(f"    {PAIR_LABELS[_pair]}: {_old/_pip:.2f} → {_new/_pip:.2f} pips  "
                      f"(scale={_scale:.3f}, spread_adj rebuilt)")

    # Train/test split (split_date defined above for spread computation)
    train_pair_dfs = {p: df[df['date'] < split_date].reset_index(drop=True)
                      for p, df in pair_dfs.items()}
    test_pair_dfs  = {p: df[df['date'] >= split_date].reset_index(drop=True)
                      for p, df in pair_dfs.items()}

    ref = next(iter(train_pair_dfs.values()))
    print(f"\n  Train: {ref['date'].min()} → {ref['date'].max()} "
          f"({ref['date'].nunique()} days)")
    ref = next(iter(test_pair_dfs.values()))
    print(f"  Test:  {ref['date'].min()} → {ref['date'].max()} "
          f"({ref['date'].nunique()} days)")

    return train_pair_dfs, test_pair_dfs, measured_spreads


def load_measured_spreads() -> dict:
    """Load previously computed median spreads from disk.

    review#11 — refuse loading if the JSON is older than 30 days OR if any
    measured spread is below 0.4 pips for majors (an unrealistically tight
    floor — typical retail-ECN reality is 0.4–0.8 pips). Returns empty dict
    on staleness or floor violation, which forces the caller to recompute or
    use a sensible default rather than silently using stale optimistic data.
    Backwards-compat: also accepts the legacy flat {pair: spread} format.

    review#P2#4 — also reads `live_measured_spreads.json` (written by
    `agent.tca.update_measured_spreads_from_live`). If the live file is
    fresher than 7 days, its per-pair spreads override the backtest medians —
    so the next sweep cycle uses costs that reflect what the broker is
    actually charging right now, not what the historical mid-prices implied.
    """
    if not SPREADS_JSON.exists():
        return {}
    try:
        with open(SPREADS_JSON) as f:
            raw = json.load(f)
    except Exception:
        return {}

    # New schema: {'_meta': {...}, 'spreads': {...}}
    if isinstance(raw, dict) and '_meta' in raw and 'spreads' in raw:
        meta = raw['_meta']
        spreads = raw['spreads']
        ts = meta.get('updated_at')
        if ts:
            try:
                from datetime import datetime as _dt
                age_days = (datetime.now(timezone.utc)
                            - _dt.fromisoformat(ts)).days
                if age_days > 30:
                    print(f"  [spreads] WARNING: edge_measured_spreads.json is "
                          f"{age_days} days old — ignoring (rebuild via load_all_data).")
                    return {}
            except Exception:
                pass
    else:
        # Legacy flat format — accept but warn it has no metadata.
        spreads = raw if isinstance(raw, dict) else {}
        print("  [spreads] legacy format (no _meta) — recommend rebuilding.")

    # Floor check: majors below 0.4 pips are unrealistically tight.
    MAJOR_FLOOR_PIPS = 0.4
    for pair, spd in list(spreads.items()):
        try:
            pip = PAIR_PIP_SIZE.get(pair, 0.0001)
            if (float(spd) / pip) < MAJOR_FLOOR_PIPS:
                print(f"  [spreads] WARNING: {pair} measured at "
                      f"{float(spd)/pip:.2f} pips < {MAJOR_FLOOR_PIPS} floor — "
                      f"ignoring (suggests off-peak sample or stale file).")
                return {}
        except Exception:
            pass

    # review#P2#4 — overlay live-measured spreads if fresh (<7 days old).
    live_path = SPREADS_JSON.parent / 'live_measured_spreads.json'
    if live_path.exists():
        try:
            with open(live_path) as f:
                live_raw = json.load(f)
            live_meta    = live_raw.get('_meta') or {}
            live_spreads = live_raw.get('spreads') or {}
            ts = live_meta.get('updated_at')
            if ts:
                from datetime import datetime as _dt
                age_days = (datetime.now(timezone.utc)
                            - _dt.fromisoformat(ts)).days
                if age_days <= 7 and live_spreads:
                    spreads = {**spreads, **live_spreads}  # live wins on overlap
                    print(f"  [spreads] overlaid {len(live_spreads)} live-measured "
                          f"spread(s) from live_measured_spreads.json "
                          f"(age={age_days}d).")
        except Exception as e:
            print(f"  [spreads] live overlay failed: {e}")

    return spreads


# =============================================================================
# 9. SLOT FACTORY
# =============================================================================

def fresh_slot(strat_def: dict) -> dict:
    return {
        'strategy_def':   strat_def,
        'slot_id':        strat_def['id'],
        'position':       None,
        'entry_price':    0.0, 'stop_loss':    0.0, 'take_profit':    0.0,
        'pos_size':       0.0, 'partial_size': 0.0, 'remainder_size': 0.0,
        'sl_ref_dist':    0.0, 'entry_time':   None, 'opened_today':  False,
        'session_exited': False, 'breakeven_set': False, 'profit_lock_set': False,
        'partial_tp_done': False, 'partial_pnl': 0.0,
        'regime': 'UNDEFINED', 'scratch': {},
    }


def daily_reset_slot(slot: dict):
    slot['opened_today'] = slot['session_exited'] = slot['breakeven_set'] = False
    slot['profit_lock_set'] = slot['partial_tp_done'] = False
    slot['partial_pnl'] = 0.0
    slot['scratch']     = {}


_ALL_FAMILIES = ['session_based']


class _BST:
    __slots__ = ['balance', 'consecutive_wins', 'consecutive_losses',
                 'trade_log', 'account_blown', 'day_start_bal', 'days_blocked',
                 'family_day_pnl', 'family_total_pnl', 'family_blown', 'family_day_blocked',
                 '_progress_callback']

    def __init__(self):
        self.balance            = float(INITIAL_BALANCE)
        self.consecutive_wins   = 0
        self.consecutive_losses = 0
        self.trade_log          = []
        self.account_blown      = False
        self.day_start_bal      = float(INITIAL_BALANCE)
        self.days_blocked       = []
        self.family_day_pnl     = {f: 0.0  for f in _ALL_FAMILIES}
        self.family_total_pnl   = {f: 0.0  for f in _ALL_FAMILIES}
        self.family_blown       = {f: False for f in _ALL_FAMILIES}
        self.family_day_blocked = {f: False for f in _ALL_FAMILIES}
        self._progress_callback = None


# =============================================================================
# 10. LOG FUNCTIONS
# =============================================================================

def _log_partial_exit(bst, pair, slot, ts, ep, size):
    raw = (ep - slot['entry_price']) * size
    if slot['position'] == 'short':
        raw *= -1
    pnl = pnl_fx(pair, raw, ep)
    bst.balance        += pnl
    slot['partial_pnl'] += pnl
    fam = slot['strategy_def']['family']
    if fam in bst.family_day_pnl:
        bst.family_day_pnl[fam]   += pnl
        bst.family_total_pnl[fam] += pnl
    bst.trade_log.append({
        'instrument': pair, 'strategy': slot['slot_id'],
        'family': slot['strategy_def']['family'], 'regime': slot['regime'],
        'entry_time': slot['entry_time'], 'exit_time': ts,
        'entry': slot['entry_price'], 'exit': ep,
        'position': slot['position'], 'exit_reason': 'partial_tp',
        'pnl': pnl, 'balance': bst.balance, 'partial': True,
    })


def _compute_swap_pips(pair, position, entry_time, exit_time):
    """Total broker swap in pips over the holding period.

    Counts each 22:00 UTC crossing between entry_time and exit_time.
    Wednesday rollover charges 3x to cover the weekend.
    """
    if not REALISTIC_FILLS:
        return 0.0
    swap_table = SWAP_PIPS_PER_NIGHT.get(pair)
    if not swap_table:
        return 0.0
    swap_per_night = swap_table.get(position, 0.0)
    if swap_per_night == 0.0 or entry_time is None or exit_time is None:
        return 0.0
    entry_ts = pd.Timestamp(entry_time)
    exit_ts  = pd.Timestamp(exit_time)
    rollover = entry_ts.replace(hour=22, minute=0, second=0, microsecond=0)
    if entry_ts >= rollover:
        rollover += pd.Timedelta(days=1)
    nights = 0.0
    while rollover <= exit_ts:
        nights += 3.0 if rollover.weekday() == 2 else 1.0
        rollover += pd.Timedelta(days=1)
    return swap_per_night * nights


def _log_exit(bst, pair, slot, ts, ep, exit_reason):
    raw = (ep - slot['entry_price']) * slot['pos_size']
    if slot['position'] == 'short':
        raw *= -1
    # Overnight swap (Phase 2): deducted as price-units * pos_size, then converted
    # to account currency by pnl_fx like any other PnL component.
    swap_pips  = _compute_swap_pips(pair, slot['position'], slot.get('entry_time'), ts)
    swap_quote = swap_pips * PAIR_PIP_SIZE.get(pair, 0.0001) * slot['pos_size']
    pnl = pnl_fx(pair, raw, ep) + pnl_fx(pair, swap_quote, ep)
    bst.balance += pnl
    fam = slot['strategy_def']['family']
    if fam in bst.family_day_pnl:
        bst.family_day_pnl[fam]   += pnl
        bst.family_total_pnl[fam] += pnl
    bst.trade_log.append({
        'instrument': pair, 'strategy': slot['slot_id'],
        'family': slot['strategy_def']['family'], 'regime': slot['regime'],
        'entry_time': slot['entry_time'], 'exit_time': ts,
        'entry': slot['entry_price'], 'exit': ep,
        'position': slot['position'], 'exit_reason': exit_reason,
        'pnl': pnl, 'balance': bst.balance, 'partial': False,
    })
    # 2026-06-05: emit a trade_closed event for the live-equity GUI.
    # Cheap when callback is None (the common case).
    cb = getattr(bst, '_progress_callback', None)
    if cb is not None:
        try:
            cb({
                "type": "trade_closed",
                "trade": {
                    "pair": pair,
                    "entry_ts": str(slot.get('entry_time')),
                    "exit_ts": str(ts),
                    "pnl": float(pnl),
                    "direction": slot.get('position'),
                    "exit_reason": exit_reason,
                    "entry_price": float(slot.get('entry_price', 0)),
                    "exit_price": float(ep),
                },
                "balance": float(bst.balance),
            })
        except Exception:
            pass
    if exit_reason == 'stop_loss' and not slot['partial_tp_done']:
        bst.consecutive_losses += 1; bst.consecutive_wins = 0
    else:
        bst.consecutive_wins += 1; bst.consecutive_losses = 0
    slot['position'] = None; slot['partial_pnl'] = 0.0


def _apply_post_fill_realism(bst, pair, slot, row, ts, hspd, slip):
    """Simulate broker rejection at placement and gap-through SL at fill.

    Called after entry_fn returns. Two distinct checks now run on different
    bars because pending orders no longer fill same-bar:

      1. Broker stops-level rejection — applies on the PLACEMENT bar.
         If a pending order was just staged at a level within
         BROKER_STOPS_LEVEL_PIPS of bar-open, MT5 would reject it. We clear
         the pending. Only relevant for 'stop_at_level' mode; market orders
         have no stops-level constraint.

      2. Gap-through SL — applies on the FILL bar. If the bar that opened the
         position also breached the stop, close immediately with 'gap_stop'.
    """
    if not REALISTIC_FILLS:
        return

    sc = slot.get('scratch') or {}
    pip = PAIR_PIP_SIZE.get(pair, 0.0001)
    bar_open = getattr(row, 'open', row.close)

    # 1. Placement-time broker stops-level rejection
    if (slot['position'] is None
            and sc.get('pending_placed_ts') == ts
            and sc.get('pending_mode', 'stop_at_level') == 'stop_at_level'):
        threshold = BROKER_STOPS_LEVEL_PIPS * pip
        # OCO: drop only the leg that's too close. If both legs survive nothing
        # changes; if both are too close everything is cleared.
        if 'pending_long' in sc or 'pending_short' in sc:
            long_p  = sc.get('pending_long')
            short_p = sc.get('pending_short')
            if long_p and abs(long_p['level'] - bar_open) < threshold:
                sc.pop('pending_long', None)
            if short_p and abs(short_p['level'] - bar_open) < threshold:
                sc.pop('pending_short', None)
            if 'pending_long' not in sc and 'pending_short' not in sc:
                for k in ('pending_mode', 'pending_placed_ts'):
                    sc.pop(k, None)
            return
        level = sc.get('pending_level')
        if level is not None and abs(level - bar_open) < threshold:
            for k in ('pending_dir', 'pending_level', 'pending_entry',
                      'pending_sl', 'pending_tp', 'pending_size',
                      'pending_dist', 'pending_mode', 'pending_placed_ts'):
                sc.pop(k, None)
        return

    # 2. Fill-time gap-through SL
    if slot['position'] is None or slot.get('entry_time') != ts:
        return

    gapped = (
        (slot['position'] == 'long'  and row.low  <= slot['stop_loss']) or
        (slot['position'] == 'short' and row.high >= slot['stop_loss'])
    )
    if gapped:
        if slot['position'] == 'long':
            ep = min(slot['stop_loss'], row.low) - hspd - slip
        else:
            ep = max(slot['stop_loss'], row.high) + hspd + slip
        _log_exit(bst, pair, slot, ts, ep, 'gap_stop')


# =============================================================================
# 11. RUN_BACKTEST
# =============================================================================

def run_backtest(subset_dfs: dict, spread_override=None, slippage_override=None,
                 registry=None, slot_managers=None, slot_entries=None,
                 cost_mult: float = 1.0,
                 session_hours: tuple[int, int] | None = None,
                 progress_callback=None):
    """Execute one backtest pass.

    Args:
        subset_dfs:       {pair: M1 DataFrame}
        spread_override:  {pair: spread} or None (use measured spread_mean per bar)
        slippage_override:{pair: slip} or None (derive as 20% of spread)
        registry:         list of strategy defs
        slot_managers:    {slot_class: manager_callable}
        slot_entries:     {slot_class: entry_callable}
        session_hours:    (lo, hi) inclusive UTC hour range — slice bars before
                          the iteration loop to cut per-bar work on
                          session-bounded strategies. Indicators with lookback
                          are precomputed columns in the prepared parquet, so
                          slicing post-prep is safe.

    Returns:
        (trades_df, final_balance, prop_summary)
    """
    if not registry:
        return pd.DataFrame(), float(INITIAL_BALANCE), {'account_blown': False, 'days_blocked': 0, 'profit_target_hit': False}

    if session_hours is not None:
        lo, hi = session_hours
        subset_dfs = {p: df[df['hour'].between(lo, hi)].reset_index(drop=True)
                      for p, df in subset_dfs.items()}

    # ── Vectorise per-bar gates (optimisation B, 2026-06-04) ────────────────
    # spread_gate() and _session_slip_mult() are called per-bar inside the
    # main loop. Both are pure functions of per-bar state; precompute as
    # columns once to eliminate 0.6+ s of per-call overhead per backtest.
    _base_slip_lookup = np.empty(24, dtype=np.float64)
    for _h in range(24):
        if 21 <= _h <= 23:
            _base_slip_lookup[_h] = SLIPPAGE_PROFILE['rollover']
        elif 0 <= _h < 7:
            _base_slip_lookup[_h] = SLIPPAGE_PROFILE['asian']
        elif 7 <= _h < 9:
            _base_slip_lookup[_h] = SLIPPAGE_PROFILE['london_open']
        elif 9 <= _h < 13:
            _base_slip_lookup[_h] = SLIPPAGE_PROFILE['london_ny']
        elif 13 <= _h < 15:
            _base_slip_lookup[_h] = SLIPPAGE_PROFILE['ny_open']
        elif 15 <= _h < 21:
            _base_slip_lookup[_h] = SLIPPAGE_PROFILE['ny_late']
        else:
            _base_slip_lookup[_h] = SLIPPAGE_PROFILE['default']

    for _pair, _df in subset_dfs.items():
        # spread_gate precompute
        _sn = _df['spread_mean'].fillna(0).values
        _sm_arr = _df['spread_median'].fillna(0).values
        _df['qp_spread_blocked'] = (_sm_arr > 0) & (_sn > SPREAD_GATE_MULT * _sm_arr)

        # slip_mult precompute (replaces per-bar _session_slip_mult call)
        if REALISTIC_FILLS:
            _hr = _df['hour'].astype(int).values
            _slip = _base_slip_lookup[_hr].copy()
            # Weekday overrides
            _wd = _df['timestamp'].dt.weekday.values
            _slip[(_wd == 4) & (_hr >= 21) & (_hr <= 22)] = 2.0  # Friday pre-weekend
            _slip[(_wd == 6) & (_hr >= 21) & (_hr <= 22)] = 1.2  # Sunday open
            # News slip overlay (bake the multiplier in)
            _nn = _df['near_news'].fillna(False).values.astype(bool) if 'near_news' in _df.columns else np.zeros(len(_df), dtype=bool)
            if _nn.any():
                _slip[_nn] = _slip[_nn] * NEWS_SLIPPAGE_MULT
            _df['qp_slip_mult'] = _slip
        else:
            _df['qp_slip_mult'] = 0.20

    _sm = slot_managers or {}
    _se = slot_entries  or {}
    use_dynamic_spread = (spread_override is None)
    pair_spread   = spread_override   or {}
    pair_slippage = slippage_override or {}

    bst = _BST()
    # Stash the progress callback on bst so _log_exit (called from many
    # code paths) can fire trade-closed events without a signature change.
    bst._progress_callback = progress_callback
    slot_states = {
        pair: [fresh_slot(sd) for sd in registry if pair in sd['pairs']]
        for pair in subset_dfs
    }
    day_sweep = {pair: {'low': False, 'high': False} for pair in subset_dfs}

    pair_lookups = {}
    for pair, df in subset_dfs.items():
        lk = {}
        for row in df.itertuples(index=False):
            lk[row.timestamp] = row
        pair_lookups[pair] = lk

    # Build sorted timestamp calendar. sorted(set().union(...)) is O(N log N)
    # but very Python-heavy (~0.33s per profile). pandas/numpy ops are 10-50×
    # faster on the same input. Single-pair path skips the union entirely
    # since DataFrame rows are already sorted by timestamp.
    if len(subset_dfs) == 1:
        df_single = next(iter(subset_dfs.values()))
        all_ts = df_single['timestamp'].tolist()
    else:
        _ts_series = pd.concat([df['timestamp'] for df in subset_dfs.values()])
        all_ts = _ts_series.drop_duplicates().sort_values().tolist()
    current_date = None
    total_bars = len(all_ts)

    for bar_counter, ts in enumerate(all_ts):
        if bst.account_blown:
            break
        # Progress tick every 1000 bars so the GUI's progress bar advances
        # on quiet stretches without trades. Cheap when callback is None.
        if progress_callback is not None and bar_counter % 1000 == 0:
            try:
                progress_callback({
                    "type": "tick",
                    "bar_idx": int(bar_counter),
                    "total_bars": int(total_bars),
                    "balance": float(bst.balance),
                })
            except Exception:
                pass

        bar_date = None
        for lk in pair_lookups.values():
            if ts in lk:
                bar_date = lk[ts].date; break
        if bar_date is None:
            continue

        if bar_date != current_date:
            # Weekend flatten (Phase 2): if the previous trading date was Friday
            # and any position survived into the next trading day, mark it closed
            # at the new bar's open — captures the weekend gap loss honestly.
            if (REALISTIC_FILLS and WEEKEND_FLATTEN
                    and current_date is not None
                    and hasattr(current_date, 'weekday')
                    and current_date.weekday() == 4):
                for fpair, fslots in slot_states.items():
                    if ts not in pair_lookups[fpair]:
                        continue
                    frow = pair_lookups[fpair][ts]
                    fhspd = (getattr(frow, 'spread_mean', 0) / 2 if use_dynamic_spread
                             else pair_spread.get(fpair, 0) / 2)
                    fslip = pair_slippage.get(
                        fpair,
                        fhspd * _session_slip_mult(getattr(frow, 'hour', None)),
                    )
                    for fslot in fslots:
                        if fslot['position']:
                            base = getattr(frow, 'open', frow.close)
                            fep = (base - fhspd - fslip if fslot['position'] == 'long'
                                   else base + fhspd + fslip)
                            _log_exit(bst, fpair, fslot, ts, fep, 'weekend_flatten')
            current_date = bar_date
            bst.day_start_bal = bst.balance
            day_sweep = {pair: {'low': False, 'high': False} for pair in subset_dfs}
            for fam in _ALL_FAMILIES:
                bst.family_day_pnl[fam]     = 0.0
                bst.family_day_blocked[fam] = False
            for pair_slots in slot_states.values():
                for slot in pair_slots:
                    if slot['position'] is None:
                        daily_reset_slot(slot)
                    else:
                        slot['opened_today'] = True; slot['session_exited'] = False

        for fam in _ALL_FAMILIES:
            if bst.family_blown[fam]:
                continue
            if -bst.family_total_pnl[fam] >= INITIAL_BALANCE * PROP_MAX_DRAWDOWN_LIMIT:
                bst.family_blown[fam] = True
                for pair, pair_slots in slot_states.items():
                    if ts not in pair_lookups[pair]:
                        continue
                    row  = pair_lookups[pair][ts]
                    hspd = (getattr(row, 'spread_mean', 0) / 2 if use_dynamic_spread
                            else pair_spread.get(pair, 0) / 2)
                    slip = pair_slippage.get(pair, hspd * 0.2)
                    for slot in pair_slots:
                        if slot['strategy_def']['family'] == fam and slot['position']:
                            ep = (row.close - hspd - slip if slot['position'] == 'long'
                                  else row.close + hspd + slip)
                            _log_exit(bst, pair, slot, ts, ep, 'family_blown')

        if all(bst.family_blown[f] for f in _ALL_FAMILIES):
            bst.account_blown = True; break

        _daily_limit = INITIAL_BALANCE * PROP_DAILY_LOSS_LIMIT
        for fam in _ALL_FAMILIES:
            if bst.family_day_blocked[fam] or bst.family_blown[fam]:
                continue
            if bst.family_day_pnl[fam] <= -_daily_limit:
                bst.family_day_blocked[fam] = True
                if bar_date not in bst.days_blocked:
                    bst.days_blocked.append(bar_date)
                for pair, pair_slots in slot_states.items():
                    if ts not in pair_lookups[pair]:
                        continue
                    row  = pair_lookups[pair][ts]
                    hspd = (getattr(row, 'spread_mean', 0) / 2 if use_dynamic_spread
                            else pair_spread.get(pair, 0) / 2)
                    slip = pair_slippage.get(pair, hspd * 0.2)
                    for slot in pair_slots:
                        if slot['strategy_def']['family'] == fam and slot['position']:
                            ep = (row.close - hspd - slip if slot['position'] == 'long'
                                  else row.close + hspd + slip)
                            _log_exit(bst, pair, slot, ts, ep, 'family_daily_limit')

        for pair in subset_dfs:
            lk = pair_lookups[pair]
            if ts not in lk:
                continue
            row = lk[ts]

            if use_dynamic_spread:
                # prefer spread_adj (includes news-time widening + commission)
                bar_spread = getattr(row, 'spread_adj',
                             getattr(row, 'spread_mean', float('nan')))
                if np.isnan(bar_spread) or bar_spread <= 0:
                    bar_spread = PAIR_PIP_SIZE.get(pair, 0.0001)
                # cost_mult now scales dynamic-mode spread too (review 2026-06-02).
                # Previously cost_mult was only applied to static spread_override.
                hspd = (bar_spread * cost_mult) / 2
                # 2026-06-04 optimisation B: slip_mult is precomputed per-bar
                # at the top of run_backtest (includes weekday overrides and
                # news multiplier). Fall back to the per-call function for
                # code paths that don't go through the precompute block.
                slip_mult = getattr(row, 'qp_slip_mult', None)
                if slip_mult is None:
                    _row_dt = getattr(row, 'date', None) or getattr(row, 'timestamp', None)
                    _wd = _row_dt.weekday() if hasattr(_row_dt, 'weekday') else None
                    slip_mult = (_session_slip_mult(getattr(row, 'hour', None), weekday=_wd)
                                 if REALISTIC_FILLS else 0.20)
                    if REALISTIC_FILLS and getattr(row, 'near_news', False):
                        slip_mult = slip_mult * NEWS_SLIPPAGE_MULT
                slip = hspd * slip_mult
            else:
                hspd = pair_spread.get(pair, 0) / 2
                slip = pair_slippage.get(pair, hspd * 0.2)

            sess_cfg = SESSION_CONFIG[PAIR_SESSION[pair]]
            if pd.isna(row.ma_trend):
                continue

            rng_h_val = getattr(row, 'range_high', float('nan'))
            if rng_h_val == rng_h_val:
                rng_l_val = row.range_low
                if row.low  < rng_l_val and row.close > rng_l_val:
                    day_sweep[pair]['low']  = True
                if row.high > rng_h_val and row.close < rng_h_val:
                    day_sweep[pair]['high'] = True

            regime = getattr(row, 'regime', 'UNDEFINED')
            if not isinstance(regime, str):
                regime = 'UNDEFINED'

            for slot in slot_states[pair]:
                sd          = slot['strategy_def']
                slot_class  = sd['slot_class']
                regime_mult = sd['regime_mult'].get(regime, 0.0)

                if slot['position'] is not None:
                    mgr = _sm.get(slot_class)
                    if mgr:
                        mgr(bst, slot, row, ts, pair, slip, hspd, sess_cfg, None, day_sweep)

                fam        = sd['family']
                fam_blocked = (bst.family_day_blocked.get(fam, False)
                               or bst.family_blown.get(fam, False))
                opens_today = slot['scratch'].get('opens_today', 0)
                if sd['allow_concurrent']:
                    can_enter = (opens_today < 2 and slot['position'] is None
                                 and regime_mult > 0 and not fam_blocked)
                else:
                    can_enter = (not slot['opened_today'] and slot['position'] is None
                                 and regime_mult > 0 and not fam_blocked)

                if can_enter:
                    entry_fn = _se.get(slot_class)
                    if entry_fn:
                        entry_fn(bst, slot, row, ts, pair, slip, hspd, sess_cfg,
                                 regime, regime_mult, None, day_sweep)
                        _apply_post_fill_realism(bst, pair, slot, row, ts, hspd, slip)

    prop_summary = {
        'account_blown':     bst.account_blown,
        'days_blocked':      len(bst.days_blocked),
        'profit_target_hit': bst.balance >= INITIAL_BALANCE * (1 + PROP_PROFIT_TARGET),
    }
    return pd.DataFrame(bst.trade_log), bst.balance, prop_summary


# =============================================================================
# 12. REPORTING UTILITIES
# =============================================================================

def _merge_partials(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or 'partial' not in trades_df.columns:
        return trades_df
    partials = (trades_df[trades_df['partial'] == True]
                .groupby(['instrument', 'strategy', 'entry_time'])['pnl']
                .sum().reset_index().rename(columns={'pnl': 'partial_pnl'}))
    full = trades_df[trades_df['partial'] == False].copy()
    if not partials.empty:
        full = full.merge(partials, on=['instrument', 'strategy', 'entry_time'], how='left')
        full['partial_pnl'] = full['partial_pnl'].fillna(0)
        full['pnl']         = full['pnl'] + full['partial_pnl']
    return full


def calc_stats(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty:
        return None
    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    n      = len(df)
    wr     = len(wins) / n * 100
    aw     = wins['pnl'].mean()   if len(wins)   > 0 else 0
    al     = losses['pnl'].mean() if len(losses) > 0 else 0
    rr     = abs(aw / al) if al != 0 else 0
    be     = (1 / (1 + rr)) * 100 if rr > 0 else 0
    margin = 1.96 * math.sqrt((wr / 100) * (1 - wr / 100) / n) * 100 if n > 1 else 0
    pnl_s  = df['pnl'].sum()
    avg_p  = df['pnl'].mean()

    df2 = df.copy()
    df2['equity']   = INITIAL_BALANCE + df2['pnl'].cumsum()
    df2['peak']     = df2['equity'].cummax()
    df2['drawdown'] = df2['equity'] - df2['peak']
    max_dd = df2['drawdown'].min()

    df2['exit_date'] = pd.to_datetime(df2['exit_time']).dt.tz_localize(None).dt.date
    daily_pnl  = df2.groupby('exit_date')['pnl'].sum()
    all_bdays  = pd.bdate_range(df2['exit_date'].min(), df2['exit_date'].max())
    daily_full = daily_pnl.reindex([d.date() for d in all_bdays], fill_value=0)
    sharpe     = (daily_full.mean() / daily_full.std() * math.sqrt(252)
                  if daily_full.std() != 0 else 0)

    return {'n': n, 'wr': wr, 'rr': rr, 'be': be, 'pnl': pnl_s, 'avg_pnl': avg_p,
            'max_dd': max_dd, 'sharpe': sharpe, 'ci_lo': wr - margin, 'ci_hi': wr + margin,
            'daily_pnl': daily_full}


def extended_risk_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {'sortino': 0, 'calmar': 0, 'kelly': 0, 'profit_factor': 0,
                'max_consec_loss': 0}
    df2 = df.copy()
    df2['exit_date'] = pd.to_datetime(df2['exit_time']).dt.tz_localize(None).dt.date
    daily = df2.groupby('exit_date')['pnl'].sum()
    all_b = pd.bdate_range(daily.index.min(), daily.index.max())
    daily = daily.reindex([d.date() for d in all_b], fill_value=0)

    neg_daily = daily[daily < 0]
    down_std  = neg_daily.std() if len(neg_daily) > 1 else 1e-8
    sortino   = daily.mean() / down_std * math.sqrt(252) if down_std > 0 else 0

    eq     = INITIAL_BALANCE + df2['pnl'].cumsum()
    max_dd = abs((eq - eq.cummax()).min())
    calmar = daily.mean() * 252 / max_dd if max_dd > 0 else 0

    wins   = df2[df2['pnl'] > 0]['pnl']
    losses = df2[df2['pnl'] <= 0]['pnl']
    pf     = wins.sum() / abs(losses.sum()) if not losses.empty and losses.sum() != 0 else 0
    wr     = len(wins) / len(df2)
    rr     = wins.mean() / abs(losses.mean()) if not losses.empty and losses.mean() != 0 else 0
    kelly  = wr - (1 - wr) / rr if rr > 0 else 0

    mcl = cur_l = 0
    for p in df2['pnl']:
        if p <= 0:
            cur_l += 1; mcl = max(mcl, cur_l)
        else:
            cur_l = 0

    return {'sortino': sortino, 'calmar': calmar, 'kelly': kelly,
            'profit_factor': pf, 'max_consec_loss': mcl}


# =============================================================================
# 12B. FAKE-EDGE FILTERS (Layers 3 & 4)
# =============================================================================

def deflated_sharpe_ratio(
    sharpe:   float,
    n_trials: int,
    n_obs:    int,
    skew:     float = 0.0,
    kurt:     float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Corrects the Sharpe Ratio for selection bias when the best of N strategies
    is chosen — the expected max Sharpe from N random trials grows with sqrt(log N).

    DSR > 0 means the strategy is better than what random selection bias alone
    predicts. DSR < 0 means it could easily be a lucky draw.

    Args:
        sharpe:   Observed (annualised) Sharpe of the best hypothesis.
        n_trials: Number of strategies tried (grid size or Optuna trials).
        n_obs:    Number of observations (daily returns or trades).
        skew:     Skewness of returns (negative = fat left tail).
        kurt:     Kurtosis of returns (3 = normal).

    Returns:
        DSR as a probability (0–1). Values near 1 are strong; near 0 are suspect.
    """
    try:
        from scipy.stats import norm as scipy_norm
    except ImportError:
        return float('nan')

    if n_trials <= 1 or n_obs <= 1 or sharpe == 0:
        return 0.5

    # Expected maximum Sharpe under H0 (no skill)
    gamma_euler = 0.5772156649
    e_max_sr = (
        (1.0 - gamma_euler) * scipy_norm.ppf(1.0 - 1.0 / n_trials)
        + gamma_euler * scipy_norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )

    # Non-normality adjustment
    sigma_sr = math.sqrt(
        (1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe ** 2)
        / (n_obs - 1.0)
    )

    dsr = scipy_norm.cdf((sharpe - e_max_sr) / sigma_sr)
    return round(float(dsr), 4)


def probabilistic_sharpe_ratio(
    sharpe:           float,
    n_obs:            int,
    benchmark_sharpe: float = 0.0,
    skew:             float = 0.0,
    kurt:             float = 3.0,
) -> float:
    """Phase 7 — Probabilistic Sharpe Ratio (Bailey & López de Prado).

    Probability that the *true* Sharpe exceeds `benchmark_sharpe` given the
    observed sample. Unlike DSR, this corrects only for finite-sample noise
    (not for selection bias across N trials). PSR > 0.95 is the standard
    'statistically reliable Sharpe' threshold.

    Returns a probability in [0, 1]. Returns 0.5 on degenerate inputs.
    """
    try:
        from scipy.stats import norm as scipy_norm
    except ImportError:
        return float('nan')

    if n_obs <= 1:
        return 0.5

    sigma_sr = math.sqrt(
        (1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe ** 2)
        / max(n_obs - 1.0, 1.0)
    )
    if sigma_sr <= 0:
        return 0.5
    return round(float(scipy_norm.cdf((sharpe - benchmark_sharpe) / sigma_sr)), 4)


def _stationary_bootstrap_indices(n: int, block_len: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap. Each step continues the current
    block with prob (1 - 1/block_len), else jumps to a new random start.
    Block length is therefore Geometric(1/block_len)-distributed; the
    resulting series stays stationary (unlike the moving-block bootstrap)."""
    p = 1.0 / max(float(block_len), 1.0)
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(0, n))
    for i in range(n):
        idx[i] = cur
        if rng.random() < p:
            cur = int(rng.integers(0, n))
        else:
            cur = (cur + 1) % n
    return idx


def bootstrap_sharpe_ci(
    trades_df: pd.DataFrame,
    n_boot:    int   = 500,
    ci_level:  float = 0.05,
    seed:      int   = 42,
    stationary: bool = True,
    block_len:  int  = 20,
) -> tuple:
    """Bootstrap confidence interval for the annualised Sharpe Ratio.

    Phase 7 — defaults to the Politis-Romano stationary bootstrap with
    block_len = 20. Daily PnL is autocorrelated (regime persistence,
    same-trade-overnight P&L, vol clustering); IID resampling collapses
    that structure and produces an over-tight CI that hides true tail risk.
    Pass stationary=False to revert to IID for unit tests / parity checks.

    An edge is robust if the lower CI bound stays above 0. A fake edge will
    have its CI straddling or below zero.

    Returns (ci_low, ci_high). Returns (nan, nan) if insufficient data.
    """
    if trades_df is None or trades_df.empty or len(trades_df) < 10:
        return float('nan'), float('nan')

    df = trades_df.copy()
    df['exit_date'] = pd.to_datetime(df['exit_time']).dt.tz_localize(None).dt.date
    daily = df.groupby('exit_date')['pnl'].sum()
    if len(daily) < 5:
        return float('nan'), float('nan')

    daily_vals = daily.values
    n          = len(daily_vals)
    rng        = np.random.default_rng(seed)
    sharpes    = []

    for _ in range(n_boot):
        if stationary:
            idx    = _stationary_bootstrap_indices(n, block_len, rng)
            sample = daily_vals[idx]
        else:
            sample = rng.choice(daily_vals, size=n, replace=True)
        std = float(np.std(sample))
        if std <= 0:
            continue
        sharpes.append(float(np.mean(sample) / std * math.sqrt(252)))

    if not sharpes:
        return float('nan'), float('nan')

    sharpes.sort()
    lo_idx = max(0, int(ci_level * len(sharpes)))
    hi_idx = min(len(sharpes) - 1, int((1 - ci_level) * len(sharpes)))
    return round(sharpes[lo_idx], 3), round(sharpes[hi_idx], 3)


def hac_sharpe(daily_pnl: np.ndarray, lag: int | None = None) -> float:
    """Phase 7 — Newey-West HAC-adjusted annualised Sharpe.

    Variance is inflated by the sum of weighted autocovariances out to lag
    q = floor(N^(1/3)) by default. Corrects Sharpe for serial correlation
    in daily PnL (overlap from same-trade overnight, vol clustering). Use
    this in place of the naive ratio when reporting Sharpe for promotion.
    Returns NaN on degenerate inputs.
    """
    x = np.asarray(daily_pnl, dtype=float)
    n = len(x)
    if n < 5:
        return float('nan')
    mean = float(np.mean(x))
    var  = float(np.var(x))
    if var <= 0:
        return float('nan')
    q = lag if lag is not None else max(1, int(np.floor(n ** (1.0 / 3.0))))
    s = var
    for k in range(1, min(q, n - 1) + 1):
        # Bartlett kernel weight
        w = 1.0 - k / (q + 1.0)
        cov = float(np.mean((x[k:] - mean) * (x[:-k] - mean)))
        s  += 2.0 * w * cov
    s = max(s, 1e-12)
    return float(mean / math.sqrt(s) * math.sqrt(252))


def winsorize_sharpe(daily_pnl: np.ndarray, p: float = 0.01) -> dict:
    """Phase 7 — winsorise daily PnL at p / (1-p) and report Sharpe both ways.
    A strategy whose un-winsorised Sharpe is more than 2× the winsorised one
    is leaning on a few outlier days — flag for manual review."""
    x = np.asarray(daily_pnl, dtype=float)
    if len(x) < 5:
        return {'sharpe_raw': float('nan'), 'sharpe_winsor': float('nan'),
                'outlier_ratio': float('nan')}
    lo, hi = np.quantile(x, [p, 1.0 - p])
    xw = np.clip(x, lo, hi)
    def _sh(v):
        s = float(np.std(v))
        return float(np.mean(v) / s * math.sqrt(252)) if s > 0 else 0.0
    raw, wn = _sh(x), _sh(xw)
    ratio = (raw / wn) if (wn != 0 and not math.isnan(wn)) else float('nan')
    return {'sharpe_raw': raw, 'sharpe_winsor': wn, 'outlier_ratio': ratio}


def pbo_score(trade_matrix: pd.DataFrame, n_blocks: int = 16) -> float:
    """Phase 7 — Probability of Backtest Overfitting (Bailey-LdP CSCV).

    trade_matrix: rows are time-aligned daily PnL observations, columns are
    candidate strategies. Splits the rows into n_blocks contiguous chunks,
    iterates over every C(n_blocks, n_blocks/2) train/test partition, picks
    the best strategy on train, and asks where it ranks on test. PBO is the
    fraction of partitions where the train-best falls in the bottom half on
    test. PBO ≈ 0.5 ⇒ the selection process is no better than chance —
    almost certainly overfit. Anything > 0.5 is a hard reject signal.

    Returns a value in [0, 1]; nan on insufficient data.
    """
    from itertools import combinations
    if trade_matrix is None or trade_matrix.empty:
        return float('nan')
    n_rows, n_strats = trade_matrix.shape
    if n_strats < 2 or n_rows < n_blocks * 2:
        return float('nan')

    blocks = np.array_split(np.arange(n_rows), n_blocks)
    half   = n_blocks // 2
    flags  = []
    for train_b in combinations(range(n_blocks), half):
        train_idx = np.concatenate([blocks[i] for i in train_b])
        test_idx  = np.concatenate([blocks[i] for i in range(n_blocks) if i not in train_b])

        train = trade_matrix.iloc[train_idx]
        test  = trade_matrix.iloc[test_idx]
        train_std = train.std()
        test_std  = test.std()
        train_sh  = (train.mean() / train_std.replace(0, np.nan))
        test_sh   = (test.mean()  / test_std.replace(0, np.nan))
        train_sh  = train_sh.dropna()
        test_sh   = test_sh.dropna()
        if train_sh.empty or test_sh.empty:
            continue

        best_strat = train_sh.idxmax()
        if best_strat not in test_sh.index:
            continue
        # rank in [0, 1] of best-train on test (1 = best). Below median ⇒ rank < 0.5.
        rank = float((test_sh < test_sh[best_strat]).sum()) / max(len(test_sh) - 1, 1)
        flags.append(1 if rank < 0.5 else 0)

    if not flags:
        return float('nan')
    return float(sum(flags) / len(flags))


def walk_forward_retrain(
    trades_df: pd.DataFrame,
    refit_fn,
    n_steps:   int = 4,
) -> pd.DataFrame:
    """Phase 7 — anchored walk-forward with retraining.

    Unlike walk_forward_test (which only splits the test set with frozen
    parameters), this calls refit_fn(train_slice) at the start of each fold
    to produce a fresh strategy callable, then evaluates it on the next OOS
    slice. Use for top-N candidates only — refit cost is non-trivial.

    refit_fn must accept a DataFrame and return a callable that produces
    the OOS PnL series for any test DataFrame. Caller owns the training
    procedure; this is a scaffold.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df['exit_date'] = pd.to_datetime(df['exit_time']).dt.tz_localize(None).dt.date
    chunk_size = max(1, len(df) // (n_steps + 1))

    rows = []
    for k in range(1, n_steps + 1):
        train = df.iloc[: k * chunk_size]
        test  = df.iloc[k * chunk_size : (k + 1) * chunk_size]
        if len(train) < 30 or len(test) < 5:
            continue
        try:
            fitted = refit_fn(train)
            oos    = fitted(test)
            sharpe = float(np.mean(oos) / np.std(oos) * math.sqrt(252)) if np.std(oos) > 0 else 0.0
        except Exception as e:
            print(f"walk_forward_retrain fold {k} failed: {e}")
            continue
        rows.append({'fold': k, 'train_n': len(train), 'test_n': len(test),
                     'oos_sharpe': sharpe})
    return pd.DataFrame(rows)


def is_regime_stable(
    trades_df:    pd.DataFrame,
    min_regimes:  int   = 2,
    min_frac:     float = 0.75,   # review#P2#2: was 0.6 — single-regime
                                  # strategies must dominate harder.
    min_trades:   int   = 10,     # review#P2#2: was 5 — need real evidence per regime.
) -> bool:
    """Check whether the edge is stable across market regimes (fake-edge Layer 4).

    review#P2#2 — tightened: a strategy with no regime evidence is no longer
    grandfathered. The previous version returned True (pass) when the regime
    column was missing, when no regimes had ≥5 trades, and when a single
    regime with as little as 60% of trades was profitable. This let
    fragile-by-construction strategies through. Now:
      * missing/empty regime data → False (must have regime tagging)
      * no regimes meeting min_trades → False
      * single profitable regime → must dominate ≥75% of trades to pass

    A strategy is regime_stable if it is profitable (positive mean PnL) in
    at least min_regimes distinct regimes, OR in exactly one regime that
    accounts for ≥ min_frac of all trades (declared single-regime specialist).
    """
    if trades_df is None or trades_df.empty or 'regime' not in trades_df.columns:
        return False   # review#P2#2: was True (grandfather)

    regime_stats = (trades_df.groupby('regime')['pnl']
                    .agg(['mean', 'count'])
                    .rename(columns={'mean': 'avg_pnl', 'count': 'n'}))
    regime_stats = regime_stats[regime_stats['n'] >= min_trades]

    if regime_stats.empty:
        return False   # review#P2#2: was True (grandfather)

    profitable  = regime_stats[regime_stats['avg_pnl'] > 0]
    n_total     = len(trades_df)
    n_profitable = len(profitable)

    if n_profitable >= min_regimes:
        return True

    if n_profitable == 1:
        dominant_frac = float(profitable['n'].iloc[0]) / n_total
        return dominant_frac >= min_frac

    return False


def portfolio_correlation(
    sweep_id:   str,
    min_sharpe: float = 0.0,
    split:      str   = 'test',
) -> dict:
    """Compute pairwise correlation of daily PnL for all survivors in a sweep.

    Useful for portfolio construction: highly correlated survivors add little
    diversification — you'd want to pick one, not both.

    Returns:
    {
        'corr_matrix':  pd.DataFrame,    (hypothesis_id × hypothesis_id)
        'avg_corr':     float,
        'n_survivors':  int,
        'high_corr_pairs': list[tuple],  (hid_a, hid_b, corr) where corr > 0.7
    }
    """
    _init_db()
    results = load_sweep_results(sweep_id)
    if results.empty:
        return {'corr_matrix': pd.DataFrame(), 'avg_corr': 0.0,
                'n_survivors': 0, 'high_corr_pairs': []}

    survivors = results[results.get('bh_sig', 0) == 1] if 'bh_sig' in results.columns else results
    survivors = survivors[survivors.get('test_sharpe', 0).fillna(0) >= min_sharpe] \
                if 'test_sharpe' in survivors.columns else survivors

    if survivors.empty:
        return {'corr_matrix': pd.DataFrame(), 'avg_corr': 0.0,
                'n_survivors': 0, 'high_corr_pairs': []}

    daily_series = {}
    for hid in survivors['hypothesis_id']:
        trades = load_hypothesis_trades(hid, split=split)
        if trades.empty:
            continue
        tm = _merge_partials(trades)
        tm['exit_date'] = pd.to_datetime(tm['exit_time']).dt.tz_localize(None).dt.date
        daily = tm.groupby('exit_date')['pnl'].sum()
        daily_series[hid] = daily

    if len(daily_series) < 2:
        return {'corr_matrix': pd.DataFrame(), 'avg_corr': 0.0,
                'n_survivors': len(daily_series), 'high_corr_pairs': []}

    panel     = pd.DataFrame(daily_series).fillna(0)
    corr_mat  = panel.corr()
    upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    flat      = upper_tri.stack()
    avg_corr  = float(flat.mean()) if not flat.empty else 0.0

    high_corr = []
    for (a, b), v in flat.items():
        if v > 0.70:
            high_corr.append((a, b, round(float(v), 3)))

    return {
        'corr_matrix':     corr_mat,
        'avg_corr':        round(avg_corr, 3),
        'n_survivors':     len(daily_series),
        'high_corr_pairs': sorted(high_corr, key=lambda x: -x[2]),
    }


# =============================================================================
# 13. MONTE CARLO (text summary — for the GUI use plot_mc_figure)
# =============================================================================

def run_monte_carlo(trades_df: pd.DataFrame, label: str = '',
                    n_sims: int = 10_000, challenge_days: int = 90) -> dict:
    if trades_df is None or trades_df.empty:
        return {}
    df = _merge_partials(trades_df.copy())
    df['exit_date'] = pd.to_datetime(df['exit_time']).dt.tz_localize(None).dt.date
    daily_pnl  = df.groupby('exit_date')['pnl'].sum()
    all_bdays  = pd.bdate_range(daily_pnl.index.min(), daily_pnl.index.max())
    daily_dist = daily_pnl.reindex([d.date() for d in all_bdays], fill_value=0).values

    pt    = INITIAL_BALANCE * PROP_PROFIT_TARGET
    mdd_l = INITIAL_BALANCE * PROP_MAX_DRAWDOWN_LIMIT
    dl    = INITIAL_BALANCE * PROP_DAILY_LOSS_LIMIT
    passed = blown = timeout = 0
    days_to_pass = []; peak_profits = []
    rng = np.random.default_rng(42)

    # review#9 — switch from IID bootstrap to stationary block bootstrap to
    # preserve daily-PnL autocorrelation and vol-clustering. IID resampling
    # underestimates tail risk by ~15-30% on FX. Block length ~20 days (a
    # standard rule-of-thumb for daily financial returns).
    _MC_BLOCK_LEN = 20
    n_dist = len(daily_dist)
    for _ in range(n_sims):
        bal  = float(INITIAL_BALANCE); peak = float(INITIAL_BALANCE); done = False
        if n_dist >= 5:
            idx = _stationary_bootstrap_indices(challenge_days, _MC_BLOCK_LEN, rng)
            # Wrap indices into the available daily_dist range.
            idx = idx % n_dist
            sampled = daily_dist[idx]
        else:
            sampled = rng.choice(daily_dist, size=challenge_days, replace=True)
        for day_idx, dpnl in enumerate(sampled):
            dpnl = max(dpnl, -dl); bal = bal + dpnl; peak = max(peak, bal)
            if peak - bal >= mdd_l:
                blown  += 1; done = True; break
            if bal - INITIAL_BALANCE >= pt:
                passed += 1; days_to_pass.append(day_idx + 1); done = True; break
        if not done:
            timeout += 1
        peak_profits.append(peak - INITIAL_BALANCE)

    pr = passed / n_sims * 100; br = blown / n_sims * 100
    # methodology back-port (crypto fork 2026-05-11): consult MC_MIN_PASS_PCT
    # instead of hardcoded 60/40 so the constant in agent/config.py actually
    # gates verdicts. Falls back to the previous 60/40 if import fails.
    try:
        from agent.config import MC_MIN_PASS_PCT as _MIN_PASS
    except ImportError:
        _MIN_PASS = 60.0
    if pr >= _MIN_PASS:
        verdict = 'VIABLE'
    elif pr >= _MIN_PASS * 0.5:
        verdict = 'MARGINAL'
    else:
        verdict = 'NOT VIABLE'
    return {
        'label': label, 'n_sims': n_sims,
        'pass_pct': pr, 'blown_pct': br, 'timeout_pct': 100 - pr - br,
        'median_days_to_pass': float(np.median(days_to_pass)) if days_to_pass else None,
        'median_peak_profit':  float(np.median(peak_profits)),
        'verdict': verdict,
    }


# =============================================================================
# 14. HYPOTHESIS TESTING
# =============================================================================

try:
    from scipy.stats import binomtest, ttest_1samp
    _scipy_ok = True
except ImportError:
    _scipy_ok = False

try:
    from statsmodels.stats.multitest import multipletests
    _sm_ok = True
except ImportError:
    _sm_ok = False


def _ht_sharpe(series):
    return series.mean() / series.std() * np.sqrt(252) if series.std() > 0 else 0


def ht_summary(trades_df: pd.DataFrame) -> dict:
    """Run all hypothesis tests on a trade log. Returns a summary dict."""
    if trades_df is None or trades_df.empty:
        return {}
    results = {}

    # Binomial
    if _scipy_ok and len(trades_df) >= 10:
        wins = (trades_df['pnl'] > 0).sum(); n = len(trades_df)
        al   = trades_df[trades_df['pnl'] <= 0]['pnl'].mean()
        aw   = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        rr   = abs(aw / al) if al != 0 else 0
        be   = 1 / (1 + rr) if rr > 0 else 0.5
        binom_p = float(binomtest(int(wins), int(n), be, alternative='greater').pvalue)
        results['binom_p']   = binom_p
        results['binom_sig'] = binom_p < 0.05
        results['wr']        = wins / n * 100
        results['be']        = be * 100

    # t-test
    if _scipy_ok and len(trades_df) >= 10:
        t_p = float(ttest_1samp(trades_df['pnl'], 0, alternative='greater').pvalue)
        results['ttest_p']   = t_p
        results['ttest_sig'] = t_p < 0.05

    # Permutation Sharpe
    if len(trades_df) >= 10:
        df2 = trades_df.copy()
        df2['exit_date'] = pd.to_datetime(df2['exit_time']).dt.tz_localize(None).dt.date
        daily    = df2.groupby('exit_date')['pnl'].sum()
        observed = _ht_sharpe(daily)
        rng      = np.random.default_rng(42)
        null     = [_ht_sharpe(pd.Series(rng.permutation(daily.values))) for _ in range(5000)]
        perm_p   = float((np.array(null) >= observed).mean())
        results['perm_p']    = perm_p
        results['perm_sig']  = perm_p < 0.05
        results['sharpe']    = observed

    return results


# =============================================================================
# ENTRY POINT — quick sanity check when run directly
# =============================================================================

if __name__ == '__main__':
    print("Edge Discovery Engine — infrastructure module")
    print(f"  Pairs:         {', '.join(ALL_PAIRS)}")
    print(f"  Train/Test:    {TRAIN_DAYS} / {TEST_DAYS} days")
    print(f"  Cache dirs:    {CACHE_DIR} | {PREPARED_CACHE_DIR}")
    print(f"  DB:            {DB_PATH}")
    print()
    print("  To load data:   train_dfs, test_dfs, spreads = load_all_data()")
    print("  To run the GUI: streamlit run edge_gui.py")
