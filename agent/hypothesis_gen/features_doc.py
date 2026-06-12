"""Feature dictionary — introspects the prepared parquet cache, groups columns
by category with a curated 1-line description per column."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import paths


# Curated descriptions. Keys are exact column names.
_DESCRIPTIONS: dict[str, str] = {
    # OHLC + time
    'open':  'M5 open price',
    'high':  'M5 high price',
    'low':   'M5 low price',
    'close': 'M5 close price',
    'volume': 'M5 volume (base asset)',
    'turnover': 'M5 turnover (USDT)',
    'timestamp': 'UTC timestamp (datetime64)',
    'mid':   'mid price (open+close)/2',
    'date':  'date (UTC)',
    'hour':  'UTC hour [0,23]',
    'minute': 'UTC minute [0,59]',
    'day_of_week': '0=Mon, 6=Sun',
    # Volatility
    'atr':           '14-period ATR (price units)',
    'atr_ratio':     'atr / atr.rolling(100).mean()',
    'atr_rank':      'percentile rank of atr vs trailing 30d',
    'realized_vol':  'rolling realized vol (close-to-close)',
    'rv_median':     'rolling median realized vol',
    'rv_delta':      'realized_vol - rv_median',
    'yz_vol':        'Yang-Zhang volatility estimator',
    'yz_vol_ratio':  'yz_vol / yz_vol_median',
    'spread_mean':   'mean bid-ask spread in bar',
    'spread_max':    'max spread in bar',
    'spread_median': 'rolling median spread',
    'spread_adj':    'spread-adjusted for cost modeling',
    # Trend
    'ma_trend':           'long-term MA (≈200-period equivalent)',
    'ma_dist':            'close - ma_trend',
    'h1_trend':           'H1 timeframe trend signal (positive = up)',
    'h1_trend_strength':  'magnitude of H1 trend',
    'adx':                'Average Directional Index',
    'hawkes_intensity':   'Hawkes self-excitation intensity (event clustering)',
    # Momentum
    'momentum_3':   '3-bar momentum (return)',
    'momentum_10':  '10-bar momentum',
    'rsi_14':       '14-period RSI',
    'bb_pct':       'Bollinger %B (where price sits within bands)',
    'bar_momentum': 'within-bar momentum (close-open)/(high-low)',
    # Regime
    'regime':           'string label: TRENDING/RANGING/TRANSITIONING/VOLATILE/UNDEFINED',
    'hurst':            'Hurst exponent (0.5 = random walk)',
    'hmm_state':        'HMM regime state (integer)',
    'hmm_prob_0':       'HMM probability of being in state 0',
    'hmm_prob_1':       'HMM probability of state 1',
    'hmm_prob_2':       'HMM probability of state 2',
    'hmm_prob_3':       'HMM probability of state 3',
    'hmm_transition':   'binary: did HMM state just change?',
    'perm_entropy_100': 'permutation entropy of last 100 bars',
    # Range / pivot
    'asian_high':       'Asian session high (used for FX; crypto: previous-day Asian-hours high)',
    'asian_low':        'Asian session low',
    'asian_valid':      'is Asian range tradeable today?',
    'swing_high5':      'rolling 5-bar swing high',
    'swing_low5':       '5-bar swing low',
    'swing_high_dist':  'close distance to swing_high5',
    'swing_low_dist':   'close distance to swing_low5',
    'range_high':       'session range high',
    'range_low':        'session range low',
    'range_size':       'range_high - range_low',
    'daily_pivot':      'classical daily pivot',
    'daily_pivot_r1':   'pivot R1 resistance',
    'daily_pivot_s1':   'pivot S1 support',
    'daily_pivot_position': 'where current price sits relative to pivot zones',
    'dist_pivot_r1':    'distance to R1',
    'dist_pivot_s1':    'distance to S1',
    'prev_day_high':    'previous calendar-day high',
    'prev_day_low':     'previous calendar-day low',
    'dist_prev_high':   'close - prev_day_high',
    'dist_prev_low':    'close - prev_day_low',
    'daily_gap':        "today's open vs yesterday's close",
    # Calendar
    'active_session':   'is current bar in any active session window?',
    'near_news':        'within news-window (any impact)',
    'near_news_impact': 'news-impact tier (1-3)',
    'bars_since_open':  'bars elapsed since current session opened',
    'bar_phase':        'phase index within current session',
    # Crypto-specific
    'funding_rate':           'current funding rate per 8h cycle (signed)',
    'funding_rate_mean_30d':  '30-day mean funding rate',
    'funding_rate_std_30d':   '30-day std of funding rate',
    'funding_z':              'z-score of current funding vs 30d distribution',
    'open_interest':          'open interest (USDT)',
    'oi_pct_change_1h':       '% change in OI over last 1h',
    'price_pct_change_1h':    '% change in price over last 1h',
    'spot_close':             'corresponding spot exchange close',
    'basis_pct':              '(perp - spot) / spot as %',
    'basis_pct_lag1':         'basis_pct lagged 1 bar',
    'basis_pct_lag2':         'basis_pct lagged 2 bars',
    'funding_cycle_phase':    'phase 0..1 within 8h funding cycle',
    'hours_to_funding':       'hours until next funding settlement',
    'bar_range_pct':          '(high-low) / close',
    # Microstructure (mostly FX but some carry to crypto)
    'tick_imbalance':       'buy-vol minus sell-vol normalised',
    'vol_imbalance':        'volume imbalance metric',
    'delta':                'cumulative buy minus sell volume',
    'aggressive_buy_ratio': 'fraction of aggressive buy prints',
    'cumulative_delta':     'cumulative delta over session',
    'delta_ma5':            'delta smoothed over 5 bars',
    'delta_momentum':       'rate of change of delta',
    'delta_divergence':     'price vs delta divergence flag',
    'persistent_imbalance': 'multi-bar imbalance persistence score',
    'close_location':       'where close sits within bar [0,1]',
    'stop_run_score':       'composite score for stop-run hunting setup',
    'tick_count':           'number of ticks in bar',
    'tick_imb_lag1':        'tick_imbalance lagged 1 bar',
    'tick_imb_roll5':       'tick_imbalance rolling mean (5)',
    'tick_imb_delta':       'change in tick_imbalance',
    'vol_imb_lag1':         'vol_imbalance lagged 1 bar',
    # Positioning (FX-only proxies, mostly N/A on crypto)
    'cot_net_position_pct':  '(FX) Commitments of Traders net position %',
    'cot_extreme_flag':      '(FX) flag for extreme positioning',
    'retail_long_pct':       '(FX) retail long %, contrarian signal',
}


CATEGORIES: dict[str, list[str]] = {
    'OHLC + time': [
        'timestamp', 'date', 'hour', 'minute', 'day_of_week',
        'open', 'high', 'low', 'close', 'volume', 'turnover', 'mid',
        'bar_range_pct',
    ],
    'Volatility': [
        'atr', 'atr_ratio', 'atr_rank',
        'realized_vol', 'rv_median', 'rv_delta',
        'yz_vol', 'yz_vol_ratio',
        'spread_mean', 'spread_max', 'spread_median', 'spread_adj',
    ],
    'Trend': [
        'ma_trend', 'ma_dist', 'h1_trend', 'h1_trend_strength',
        'adx', 'hawkes_intensity',
    ],
    'Momentum': [
        'momentum_3', 'momentum_10', 'rsi_14', 'bb_pct', 'bar_momentum',
    ],
    'Regime': [
        'regime', 'hurst', 'perm_entropy_100',
        'hmm_state', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'hmm_prob_3',
        'hmm_transition',
    ],
    'Range / pivot': [
        'range_high', 'range_low', 'range_size',
        'asian_high', 'asian_low', 'asian_valid',
        'swing_high5', 'swing_low5', 'swing_high_dist', 'swing_low_dist',
        'daily_pivot', 'daily_pivot_r1', 'daily_pivot_s1',
        'daily_pivot_position', 'dist_pivot_r1', 'dist_pivot_s1',
        'prev_day_high', 'prev_day_low', 'dist_prev_high', 'dist_prev_low',
        'daily_gap',
    ],
    'Calendar': [
        'active_session', 'near_news', 'near_news_impact',
        'bars_since_open', 'bar_phase',
    ],
    'Funding / OI / basis (crypto-only)': [
        'funding_rate', 'funding_rate_mean_30d', 'funding_rate_std_30d',
        'funding_z', 'funding_cycle_phase', 'hours_to_funding',
        'open_interest', 'oi_pct_change_1h', 'price_pct_change_1h',
        'spot_close', 'basis_pct', 'basis_pct_lag1', 'basis_pct_lag2',
    ],
    'Microstructure': [
        'tick_imbalance', 'vol_imbalance', 'delta', 'aggressive_buy_ratio',
        'cumulative_delta', 'delta_ma5', 'delta_momentum', 'delta_divergence',
        'persistent_imbalance', 'close_location', 'stop_run_score',
        'tick_count', 'tick_imb_lag1', 'tick_imb_roll5', 'tick_imb_delta',
        'vol_imb_lag1',
    ],
    'Positioning (FX-only)': [
        'cot_net_position_pct', 'cot_extreme_flag', 'retail_long_pct',
    ],
}


def _column_set_from_cache() -> tuple[set[str], dict[str, str]]:
    """Return (column names, dtype map) from the first parquet in PREPARED_CACHE.

    Falls back to the curated _DESCRIPTIONS keys if no parquet exists yet.
    """
    cache = paths.PREPARED_CACHE
    if not cache.exists():
        return set(_DESCRIPTIONS.keys()), {}
    parquets = sorted(cache.glob('*.parquet'))
    if not parquets:
        return set(_DESCRIPTIONS.keys()), {}
    try:
        import pandas as pd
        df = pd.read_parquet(parquets[0])
        return set(df.columns.tolist()), {c: str(df[c].dtype) for c in df.columns}
    except Exception:
        return set(_DESCRIPTIONS.keys()), {}


def generate(out_path: Optional[Path] = None) -> str:
    """Build features.md content."""
    cols, dtypes = _column_set_from_cache()

    body = ['# Features available in the prepared cache',
            '',
            f'{len(cols)} columns across the M5 prepared parquet for each pair.',
            'Grouped by conceptual category. Use exact column names in `row.<col>`.',
            '',
            ('**Use at least one feature from a category OTHER than Momentum, '
             'Trend, or OHLC.** Generic mean-reversion / breakout / MA-cross on '
             'those three categories has been exhaustively tested with zero '
             'survivors. The crypto-specific (funding / OI / basis), microstructure, '
             'and regime categories are under-explored relative to their information '
             'content.'),
            '']

    seen: set[str] = set()
    for cat, cols_in_cat in CATEGORIES.items():
        present = [c for c in cols_in_cat if c in cols]
        if not present:
            continue
        body.append(f'## {cat}')
        body.append('')
        body.append('| column | dtype | description |')
        body.append('|---|---|---|')
        for c in present:
            dt = dtypes.get(c, '')
            desc = _DESCRIPTIONS.get(c, '')
            body.append(f'| `{c}` | {dt} | {desc} |')
            seen.add(c)
        body.append('')

    remainder = sorted(cols - seen)
    if remainder:
        body.append('## Other')
        body.append('')
        body.append('| column | dtype | description |')
        body.append('|---|---|---|')
        for c in remainder:
            dt = dtypes.get(c, '')
            body.append(f'| `{c}` | {dt} | (uncurated; check before use) |')
        body.append('')

    out = '\n'.join(body)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding='utf-8')
    return out
