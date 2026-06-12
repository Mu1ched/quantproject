# -*- coding: utf-8 -*-
"""
Edge Miner — Automated pattern discovery for edge_engine.

Layers:
  A. feature_sweep()          — systematic: every feature × percentile × direction
  C. mine_patterns()          — LightGBM + SHAP → symbolic rules
     _walk_forward_cv()       — temporal 4-fold validation (fake-edge Layer 1)
  Gen. generate_hypotheses()  — convert top patterns → Python entry function code

Fake-edge filters (applied inside the miner):
  Layer 1: Walk-forward CV           — pattern must survive ≥3 of 4 temporal folds
  Layer 2: Adversarial validation    — warn/abort if train/test distributions diverge
  (Layers 3 & 4 are in edge_engine.py: Deflated Sharpe, Bootstrap CI, Regime stability)

Next-level additions:
  engineer_features()      — lag, delta, rolling, distance-to-level, ratio features
  check_distribution_shift()— KS-test per feature (train vs test)
  adversarial_validation() — LightGBM train/test classifier (AUC-based drift gate)
  PatternMemory.decay_weights() — exponential decay on stale feature evidence
  fit_meta_learner()       — logistic regression: what predicts condition robustness?

Usage:
    from edge_miner import run_miner, PatternMemory
    mem     = PatternMemory()
    results = run_miner(train_dfs, test_dfs=test_dfs, memory=mem)
"""

import ast
import json
import math
import sqlite3
import textwrap
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONSTANTS
# =============================================================================

# Base features that must exist in prepared pair_dfs (from prepare_df())
MINER_FEATURES = [
    'tick_imbalance',
    'vol_imbalance',
    'realized_vol',
    'rv_median',
    'atr',
    'atr_ratio',
    'adx',
    'spread_mean',
    'spread_median',
    # Microstructure causality features
    'delta',
    # 'cumulative_delta',   # REMOVED 2026-06-03: daily cumsum, non-stationary by construction → causes adversarial AUC > 0.90
    'delta_momentum',
    'delta_divergence',
    'persistent_imbalance',
    'stop_run_score',
    'bar_momentum',
    'close_location',
    'bar_range_pct',
    'bars_since_open',
    'bar_phase',
    'daily_gap',
    'aggressive_buy_ratio',
    # HMM regime features — REMOVED 2026-06-03: train-fit; misclassify under regime
    # shift in test → cause adversarial AUC drift even after running-total fix.
    # 'hmm_state',
    # 'hmm_prob_0',
    # 'hmm_prob_1',
    # 'hmm_prob_2',
    # 'hmm_prob_3',
    # 'hmm_transition',
    # Tier-1 statistical filters
    'hurst',           # rolling 200-bar Hurst exponent — >0.55 trending, <0.45 fade
    'yz_vol',          # Yang-Zhang realized vol (30-bar) — captures gaps + range
    'yz_vol_ratio',    # yz_vol / 60-bar median — vol regime indicator
    'perm_entropy_100', # Bandt-Pompe permutation entropy (window=100, order=4)
    'hawkes_intensity', # Self-exciting Hawkes intensity ratio of strong-flow events
]

# Engineered features computed by engineer_features() from the above
ENGINEERED_FEATURES = [
    'tick_imb_lag1',
    'tick_imb_delta',
    'tick_imb_roll5',
    'vol_imb_lag1',
    'spread_ratio',       # spread / realized_vol — liquidity relative to movement
    'ma_dist',            # (close - ma200) / atr — trend distance in ATR units
    'swing_high_dist',    # (swing_high5 - close) / atr — distance to overhead
    'swing_low_dist',     # (close - swing_low5) / atr — distance to support
    'rv_delta',           # realized_vol / rv_median — vol spike ratio
    # Derived from microstructure features
    # 'cum_delta_lag1',   # REMOVED 2026-06-03: lag of running total — inherits non-stationarity
    'delta_accel',        # second derivative of delta (acceleration)
    'stop_run_lag1',      # stop run score one bar ago
    'phase_x_imbalance',  # interaction: session phase × persistent imbalance
]

ALL_FEATURES = MINER_FEATURES + ENGINEERED_FEATURES

# Time features auto-added during dataset build
TIME_FEATURES = ['hour', 'day_of_week']

THRESHOLD_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

SESSION_HOURS = {
    'asian':  (0,  8),
    'london': (8,  13),
    'ny':     (13, 21),
    None:     (0,  24),
}

MEMORY_DB_PATH = Path('edge_memory.db')


# =============================================================================
# SECTION 0 — FEATURE ENGINEERING (Next-level #2)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features from base columns in a prepared M1 DataFrame.

    All operations are NaN-safe — missing input columns are skipped.
    Call this before building the label dataset.

    Adds:
        tick_imb_lag1     — previous bar's tick imbalance
        tick_imb_roll5    — 5-bar rolling mean of tick imbalance
        tick_imb_delta    — current vs 5-bar mean (momentum of order flow)
        vol_imb_lag1      — previous bar's volume imbalance
        spread_ratio      — spread_mean / realized_vol (high = wide/quiet, skip-worthy)
        ma_dist           — (close - ma_trend) / atr (trend position in ATR units)
        swing_high_dist   — (swing_high5 - close) / atr (headroom to resistance)
        swing_low_dist    — (close - swing_low5) / atr (cushion above support)
        rv_delta          — realized_vol / rv_median (vol spike ratio)
    """
    d = df.copy()

    if 'tick_imbalance' in d.columns:
        d['tick_imb_lag1']  = d['tick_imbalance'].shift(1)
        d['tick_imb_roll5'] = d['tick_imbalance'].rolling(5, min_periods=2).mean()
        d['tick_imb_delta'] = d['tick_imbalance'] - d['tick_imb_roll5']

    if 'vol_imbalance' in d.columns:
        d['vol_imb_lag1'] = d['vol_imbalance'].shift(1)

    if 'spread_mean' in d.columns and 'realized_vol' in d.columns:
        rv_safe = d['realized_vol'].replace(0, np.nan)
        d['spread_ratio'] = (d['spread_mean'] / rv_safe).clip(0, 100)

    if 'close' in d.columns and 'ma_trend' in d.columns and 'atr' in d.columns:
        atr_safe = d['atr'].replace(0, np.nan)
        d['ma_dist'] = (d['close'] - d['ma_trend']) / atr_safe

    if 'close' in d.columns and 'atr' in d.columns:
        atr_safe = d['atr'].replace(0, np.nan)
        if 'swing_high5' in d.columns:
            d['swing_high_dist'] = (d['swing_high5'] - d['close']) / atr_safe
        if 'swing_low5' in d.columns:
            d['swing_low_dist'] = (d['close'] - d['swing_low5']) / atr_safe

    if 'realized_vol' in d.columns and 'rv_median' in d.columns:
        rv_med_safe = d['rv_median'].replace(0, np.nan)
        d['rv_delta'] = (d['realized_vol'] / rv_med_safe).clip(0, 10)

    # Derived from microstructure causality features
    if 'cumulative_delta' in d.columns:
        d['cum_delta_lag1'] = d['cumulative_delta'].shift(1)

    if 'delta_momentum' in d.columns:
        d['delta_accel'] = d['delta_momentum'] - d['delta_momentum'].shift(1)

    if 'stop_run_score' in d.columns:
        d['stop_run_lag1'] = d['stop_run_score'].shift(1)

    if 'bar_phase' in d.columns and 'persistent_imbalance' in d.columns:
        d['phase_x_imbalance'] = d['bar_phase'] * d['persistent_imbalance']

    return d


# =============================================================================
# PATTERN MEMORY — persistent store for the self-improving loop
# =============================================================================

class PatternMemory:
    """SQLite-backed memory that persists across discovery rounds.

    Tables:
        feature_weights   — how often each feature appeared in surviving patterns
        proven_conditions — (feature, direction, threshold) tuples that survived
                            BH correction, with hit counts and average WR
        param_regions     — param dicts from significant results (Optuna warm-start)
        loop_rounds       — round-level summary log
    """

    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path) if db_path else MEMORY_DB_PATH
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS feature_weights (
                    feature     TEXT PRIMARY KEY,
                    appearances INT  DEFAULT 0,
                    total_wr    REAL DEFAULT 0.0,
                    last_seen   TEXT
                );
                CREATE TABLE IF NOT EXISTS proven_conditions (
                    feature     TEXT,
                    direction   TEXT,
                    threshold   REAL,
                    session     TEXT,
                    target      TEXT,
                    hit_count   INT  DEFAULT 1,
                    avg_wr      REAL DEFAULT 0.5,
                    last_seen   TEXT,
                    PRIMARY KEY (feature, direction, threshold, session, target)
                );
                CREATE TABLE IF NOT EXISTS param_regions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    sweep_name  TEXT,
                    params_json TEXT,
                    sharpe      REAL,
                    created_at  TEXT
                );
                CREATE TABLE IF NOT EXISTS loop_rounds (
                    round_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_n     INT,
                    sweep_id    TEXT,
                    n_tested    INT,
                    n_survivors INT,
                    session     TEXT,
                    created_at  TEXT
                );
            ''')

    # ── Write ─────────────────────────────────────────────────────────────────

    def update_from_patterns(self, patterns: list, session: str = 'ny'):
        """Record surviving patterns into memory (call after BH correction)."""
        if not patterns:
            return
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as con:
            for pat in patterns:
                feat = pat.get('feature')
                wr   = float(pat.get('win_rate', 0.5))
                if not feat:
                    continue

                con.execute('''
                    INSERT INTO feature_weights (feature, appearances, total_wr, last_seen)
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(feature) DO UPDATE SET
                        appearances = appearances + 1,
                        total_wr    = total_wr + excluded.total_wr,
                        last_seen   = excluded.last_seen
                ''', (feat, wr, now))

                all_conds = [{'feature': feat,
                               'direction': pat['direction'],
                               'threshold': pat['threshold']}] \
                           + pat.get('conditions', [])
                for cond in all_conds:
                    con.execute('''
                        INSERT INTO proven_conditions
                            (feature, direction, threshold, session, target,
                             hit_count, avg_wr, last_seen)
                        VALUES (?,?,?,?,?,1,?,?)
                        ON CONFLICT(feature, direction, threshold, session, target)
                        DO UPDATE SET
                            hit_count = hit_count + 1,
                            avg_wr    = (avg_wr * hit_count + excluded.avg_wr)
                                        / (hit_count + 1),
                            last_seen = excluded.last_seen
                    ''', (cond['feature'], cond['direction'],
                          round(float(cond['threshold']), 6),
                          session, pat.get('target', 'long_win'),
                          wr, now))

    def update_from_sweep_results(self, results_df: pd.DataFrame,
                                   sweep_name: str = ''):
        """Record param regions from BH-significant sweep results."""
        if results_df.empty:
            return
        survivors = results_df[results_df.get('bh_sig', pd.Series(dtype=int)) == 1] \
                    if 'bh_sig' in results_df.columns else results_df
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as con:
            for _, row in survivors.iterrows():
                params_json = row.get('params_json', '{}')
                sharpe      = float(row.get('test_sharpe', 0) or 0)
                con.execute(
                    'INSERT INTO param_regions (sweep_name, params_json, sharpe, created_at) '
                    'VALUES (?,?,?,?)',
                    (sweep_name, params_json, sharpe, now)
                )

    def record_round(self, round_n: int, sweep_id: str, n_tested: int,
                     n_survivors: int, session: str):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                'INSERT INTO loop_rounds '
                '(round_n, sweep_id, n_tested, n_survivors, session, created_at) '
                'VALUES (?,?,?,?,?,?)',
                (round_n, sweep_id, n_tested, n_survivors, session,
                 datetime.utcnow().isoformat())
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    @property
    def feature_weights(self) -> dict:
        """Dict of {feature: weight} where weight = sqrt(appearances)."""
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                'SELECT feature, appearances FROM feature_weights ORDER BY appearances DESC'
            ).fetchall()
        if not rows:
            return {}
        return {feat: math.sqrt(max(1, apps)) for feat, apps in rows}

    @property
    def proven_conditions(self) -> list:
        """Conditions sorted by hit_count descending — most-proven first."""
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute('''
                SELECT feature, direction, threshold, session, target, hit_count, avg_wr
                FROM proven_conditions
                ORDER BY hit_count DESC, avg_wr DESC
            ''').fetchall()
        return [
            {'feature': r[0], 'direction': r[1], 'threshold': r[2],
             'session': r[3], 'target': r[4], 'hit_count': r[5], 'avg_wr': r[6]}
            for r in rows
        ]

    def warm_start_params(self, sweep_name: str, top_n: int = 10) -> list:
        """Return list of param dicts (decoded JSON) for Optuna warm start."""
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute('''
                SELECT params_json FROM param_regions
                WHERE sweep_name = ? ORDER BY sharpe DESC LIMIT ?
            ''', (sweep_name, top_n)).fetchall()
        results = []
        for (pj,) in rows:
            try:
                results.append(json.loads(pj))
            except Exception:
                pass
        return results

    def rounds_df(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql('SELECT * FROM loop_rounds ORDER BY round_n', con)

    def feature_leaderboard_df(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql(
                'SELECT feature, appearances, ROUND(total_wr/MAX(appearances,1),4) as avg_wr '
                'FROM feature_weights ORDER BY appearances DESC',
                con
            )

    def proven_conditions_df(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql(
                'SELECT * FROM proven_conditions ORDER BY hit_count DESC',
                con
            )

    def decay_weights(self, decay_factor: float = 0.9, days_threshold: int = 7):
        """Exponentially decay feature_weights for features not seen recently.

        Features last updated more than `days_threshold` days ago are multiplied
        by `decay_factor`. This prevents old patterns from dominating the feature
        leaderboard — forces the miner to re-validate every feature every few rounds
        rather than trusting stale evidence indefinitely.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days_threshold)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute('''
                UPDATE feature_weights
                SET appearances = MAX(1, CAST(appearances * ? AS INTEGER))
                WHERE last_seen < ? OR last_seen IS NULL
            ''', (decay_factor, cutoff))

    def clear(self):
        """Wipe all memory (use with caution)."""
        with sqlite3.connect(self.db_path) as con:
            con.executescript('''
                DELETE FROM feature_weights;
                DELETE FROM proven_conditions;
                DELETE FROM param_regions;
                DELETE FROM loop_rounds;
            ''')


# =============================================================================
# SECTION 1 — Label Dataset Builder (Next-level #1: spread-adjusted labels)
# =============================================================================

def _compute_sweep_label(df: pd.DataFrame, horizon: int = 3) -> pd.Series:
    """Vectorised liquidity-sweep detection.

    A sweep bar is one where price extends beyond the recent swing high/low
    (triggering stops) then snaps back inside within `horizon` bars.
    Used as a third training target so the miner can learn WHERE stop runs
    occur — patterns that predict sweeps reveal liquidity trap zones.
    """
    close = df['close']
    sh5   = df['swing_high5'] if 'swing_high5' in df.columns else pd.Series(np.nan, index=df.index)
    sl5   = df['swing_low5']  if 'swing_low5'  in df.columns else pd.Series(np.nan, index=df.index)

    fwd_min = pd.concat([close.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1)
    fwd_max = pd.concat([close.shift(-k) for k in range(1, horizon + 1)], axis=1).max(axis=1)

    high_sweep = (close > sh5) & (fwd_min < sh5)
    low_sweep  = (close < sl5) & (fwd_max > sl5)
    return (high_sweep | low_sweep).fillna(False).astype(float)


def _simulate_forward(df: pd.DataFrame, tp_dist: pd.Series, sl_dist: pd.Series,
                       direction: str, horizon: int,
                       half_spread: float = 0.0) -> pd.Series:
    """Vectorised: for each row, scan forward 'horizon' bars to find which
    of TP or SL is hit first. Returns boolean Series (True = win).

    half_spread: applied to entry price, making wins harder to achieve
    (closer to what a real fill costs). This closes the gap between
    simulation-based labels and real backtest outcomes (Next-level #1).
    """
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n     = len(close)
    tp_d  = tp_dist.values
    sl_d  = sl_dist.values

    wins = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(tp_d[i]) or np.isnan(sl_d[i]) or tp_d[i] <= 0 or sl_d[i] <= 0:
            continue
        entry = close[i]
        if direction == 'long':
            entry_adj = entry + half_spread
            tp = entry_adj + tp_d[i]
            sl = entry_adj - sl_d[i]
            for j in range(i + 1, min(i + 1 + horizon, n)):
                if high[j] >= tp:
                    wins[i] = True
                    break
                if low[j] <= sl:
                    break
        else:
            entry_adj = entry - half_spread
            tp = entry_adj - tp_d[i]
            sl = entry_adj + sl_d[i]
            for j in range(i + 1, min(i + 1 + horizon, n)):
                if low[j] <= tp:
                    wins[i] = True
                    break
                if high[j] >= sl:
                    break
    return pd.Series(wins, index=df.index)


def build_label_dataset(
    pair_dfs:        dict,
    tp_r:            float = 2.0,
    sl_r:            float = 1.0,
    horizon_bars:    int   = 20,
    session_filter:  str   = None,
    use_engineered:  bool  = True,   # compute engineered features
) -> pd.DataFrame:
    """For every M1 bar across all pairs, simulate long + short trades and
    label whether each wins within horizon_bars.

    Labels include the median spread cost so they match real backtest
    costs more closely than pure mid-price simulation (Next-level #1).

    Returns DataFrame with columns:
      pair, ts, [ALL_FEATURES if use_engineered else MINER_FEATURES],
      hour, day_of_week, regime, long_win (0/1), short_win (0/1)
    """
    hour_lo, hour_hi = SESSION_HOURS.get(session_filter, (0, 24))
    chunks = []

    for pair, df in pair_dfs.items():
        if df is None or df.empty:
            continue
        df = df.copy()

        # Engineer features before session filter (needs full day context)
        if use_engineered:
            df = engineer_features(df)

        # Session filter
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour if hasattr(df.index, 'hour') else df['ts'].dt.hour
        mask = (df['hour'] >= hour_lo) & (df['hour'] < hour_hi)
        df = df[mask]
        if df.empty:
            continue

        if 'atr' not in df.columns:
            continue

        tp_dist = df['atr'] * tp_r
        sl_dist = df['atr'] * sl_r

        # Use median half-spread for label cost (realistic entry cost)
        med_half_spread = (float(df['spread_mean'].median()) / 2
                           if 'spread_mean' in df.columns else 0.0)

        long_win  = _simulate_forward(df, tp_dist, sl_dist, 'long',  horizon_bars,
                                       med_half_spread)
        short_win = _simulate_forward(df, tp_dist, sl_dist, 'short', horizon_bars,
                                       med_half_spread)

        feat_pool = ALL_FEATURES if use_engineered else MINER_FEATURES
        feature_cols = [c for c in feat_pool if c in df.columns]
        chunk = df[feature_cols].copy()
        chunk['pair']        = pair
        chunk['ts']          = df.index if hasattr(df.index, 'hour') else df['ts']
        chunk['hour']        = df['hour']
        chunk['day_of_week'] = (df.index.dayofweek
                                if hasattr(df.index, 'dayofweek')
                                else df['ts'].dt.dayofweek)
        chunk['regime']      = df['regime'] if 'regime' in df.columns else 'UNDEFINED'
        chunk['long_win']          = long_win.astype(int)
        chunk['short_win']         = short_win.astype(int)
        chunk['is_liquidity_sweep'] = _compute_sweep_label(df).values
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    dataset = pd.concat(chunks, ignore_index=True)
    dataset.dropna(subset=[c for c in MINER_FEATURES if c in dataset.columns],
                   inplace=True)
    return dataset


# =============================================================================
# SECTION 2 — Systematic Feature Sweeper (Layer A)
# =============================================================================

def feature_sweep(
    dataset:     pd.DataFrame,
    features:    list = None,
    min_samples: int  = 100,
    min_edge:    float = 0.03,
) -> pd.DataFrame:
    """For every (feature, percentile threshold, direction, long/short):
      - Filter rows where condition is met
      - Compute win rate, n, t-stat vs 50%

    Returns DataFrame sorted by |t_stat| descending.
    """
    if features is None:
        features = [c for c in ALL_FEATURES + TIME_FEATURES if c in dataset.columns]

    rows = []
    for feat in features:
        if feat not in dataset.columns:
            continue
        col = dataset[feat].dropna()
        if col.empty:
            continue
        percentile_vals = np.percentile(col, THRESHOLD_PERCENTILES)

        for pct, thresh in zip(THRESHOLD_PERCENTILES, percentile_vals):
            for direction, mask_fn in [('>', lambda v, f=feat: dataset[f] > v),
                                       ('<', lambda v, f=feat: dataset[f] < v)]:
                subset = dataset[mask_fn(thresh)]
                for target in ('long_win', 'short_win'):
                    if target not in subset.columns:
                        continue
                    y = subset[target].dropna()
                    n = len(y)
                    if n < min_samples:
                        continue
                    wr = y.mean()
                    if abs(wr - 0.5) < min_edge:
                        continue
                    t, p = scipy_stats.ttest_1samp(y, 0.5)
                    rows.append({
                        'feature':   feat,
                        'direction': direction,
                        'threshold': round(thresh, 6),
                        'pct':       pct,
                        'target':    target,
                        'n':         n,
                        'win_rate':  round(wr, 4),
                        't_stat':    round(t, 3),
                        'p_value':   round(p, 6),
                    })

    if not rows:
        return pd.DataFrame(columns=[
            'feature', 'direction', 'threshold', 'pct', 'target',
            'n', 'win_rate', 't_stat', 'p_value'
        ])

    result = pd.DataFrame(rows)
    result.sort_values('t_stat', key=abs, ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# =============================================================================
# SECTION 2B — Distribution Shift + Adversarial Validation (fake-edge Layer 2)
# =============================================================================

def check_distribution_shift(
    train_df:    pd.DataFrame,
    test_df:     pd.DataFrame,
    features:    list  = None,
    ks_p_thresh: float = 0.05,
) -> dict:
    """KS-test each feature comparing train vs test distributions.

    A significant KS-test means the two distributions differ, which can make
    patterns mined on train invalid on test — not because the edge doesn't
    exist, but because the market regime changed. This is different from, and
    complementary to, adversarial validation.

    Returns:
    {
        'shifted_features': list[str],  # features where train != test (p < thresh)
        'ks_results':       DataFrame,  # feature | ks_stat | p_value | shifted
        'max_ks_stat':      float,
        'any_shift':        bool,
    }
    """
    if features is None:
        features = [c for c in ALL_FEATURES + TIME_FEATURES
                    if c in train_df.columns and c in test_df.columns]

    rows = []
    for feat in features:
        tr = train_df[feat].dropna().values
        te = test_df[feat].dropna().values
        if len(tr) < 30 or len(te) < 30:
            continue
        ks_stat, p_val = scipy_stats.ks_2samp(tr, te)
        rows.append({
            'feature':  feat,
            'ks_stat':  round(float(ks_stat), 4),
            'p_value':  round(float(p_val), 6),
            'shifted':  bool(p_val < ks_p_thresh),
        })

    if not rows:
        return {'shifted_features': [], 'ks_results': pd.DataFrame(),
                'max_ks_stat': 0.0, 'any_shift': False}

    df_ks = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
    shifted = df_ks[df_ks['shifted']]['feature'].tolist()

    return {
        'shifted_features': shifted,
        'ks_results':       df_ks.reset_index(drop=True),
        'max_ks_stat':      float(df_ks['ks_stat'].max()),
        'any_shift':        bool(shifted),
    }


def adversarial_validation(
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    features:  list  = None,
    auc_warn:  float = 0.60,
    auc_abort: float = 0.75,
) -> dict:
    """Train a LightGBM classifier to distinguish train rows from test rows.

    If AUC > 0.5, the classifier has found systematic differences between train
    and test feature distributions. High AUC means any edge mined on train is
    suspect — the 'pattern' may just be the classifier learning which regime
    each sample came from.

    Soft by default: WARN down-weights drift features; ABORT skips mining entirely.

    Returns:
    {
        'auc':            float,
        'drift_features': list[str],  # top features the classifier exploits
        'shap_weights':   dict,        # {feature: mean_abs_shap}
        'verdict':        'OK' | 'WARN' | 'ABORT',
        'message':        str,
    }
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return {'auc': 0.5, 'drift_features': [], 'shap_weights': {},
                'verdict': 'OK', 'message': 'lightgbm not installed — skipping adversarial check'}

    if features is None:
        features = [c for c in ALL_FEATURES
                    if c in train_df.columns and c in test_df.columns]
    features = [f for f in features
                if f in train_df.columns and f in test_df.columns]
    if not features:
        return {'auc': 0.5, 'drift_features': [], 'shap_weights': {},
                'verdict': 'OK', 'message': 'No common features for adversarial check'}

    tr = train_df[features].dropna().copy()
    te = test_df[features].dropna().copy()
    tr['_label'] = 0
    te['_label'] = 1

    # Balance classes to avoid trivial AUC from class imbalance
    n = min(len(tr), len(te), 5000)
    tr_s = tr.sample(n=min(n, len(tr)), random_state=42)
    te_s = te.sample(n=min(n, len(te)), random_state=42)
    combined = pd.concat([tr_s, te_s], ignore_index=True).sample(frac=1, random_state=42)

    X = combined[features].values
    y = combined['_label'].values

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=15, random_state=42, verbose=-1,
        min_child_samples=20,
    )

    # 3-fold cross-validated AUC (prevents overfitting on the adversarial task itself)
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], preds))
        auc = float(np.mean(aucs))
    except ImportError:
        # Fall back to single-split if sklearn unavailable
        mid = len(X) // 2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X[:mid], y[:mid])
        try:
            from sklearn.metrics import roc_auc_score
            preds = model.predict_proba(X[mid:])[:, 1]
            auc = float(roc_auc_score(y[mid:], preds))
        except Exception:
            auc = 0.5

    # Re-fit on full dataset for SHAP importances
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, y)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals
        shap_imp = dict(zip(features, np.abs(sv).mean(axis=0)))
    except Exception:
        shap_imp = dict(zip(features,
                            model.feature_importances_ / max(model.feature_importances_.sum(), 1)))

    sorted_imp   = sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)
    drift_feats  = [f for f, _ in sorted_imp[:3]] if auc > auc_warn else []

    if auc >= auc_abort:
        verdict = 'ABORT'
        msg = (f"AUC={auc:.3f} — train/test distributions highly divergent. "
               f"Mined edges are likely regime artifacts, not structural. "
               f"Main drift axes: {', '.join(drift_feats)}")
    elif auc >= auc_warn:
        verdict = 'WARN'
        msg = (f"AUC={auc:.3f} — moderate train/test distribution shift. "
               f"Down-weighting drift features: {', '.join(drift_feats)}")
    else:
        verdict = 'OK'
        msg = f"AUC={auc:.3f} — distributions similar, mining is valid."

    return {
        'auc':            auc,
        'drift_features': drift_feats,
        'shap_weights':   shap_imp,
        'verdict':        verdict,
        'message':        msg,
    }


# =============================================================================
# SECTION 3 — ML Pattern Miner (Layer C)
# =============================================================================

def _apply_mask(X: pd.DataFrame, feat: str, direction: str, thresh: float) -> pd.Series:
    """Return boolean mask for (feat direction thresh)."""
    return X[feat] > thresh if direction == '>' else X[feat] < thresh


def _best_primary(X: pd.DataFrame, y: pd.Series, feat: str,
                  min_samples: int) -> tuple:
    """Scan percentile thresholds for the highest win rate on the primary feature.
    Returns (best_wr, best_thresh, best_dir) or (0.5, median, '>') if nothing found.
    """
    best_wr, best_thresh, best_dir = 0.5, float(X[feat].median()), '>'
    for pct in THRESHOLD_PERCENTILES:
        thresh = float(np.percentile(X[feat].dropna(), pct))
        for direction in ('>', '<'):
            mask  = _apply_mask(X, feat, direction, thresh)
            sub_y = y[mask]
            if len(sub_y) < min_samples:
                continue
            wr = float(sub_y.mean())
            if wr > best_wr:
                best_wr, best_thresh, best_dir = wr, thresh, direction
    return best_wr, best_thresh, best_dir


def _walk_forward_cv(
    X:                    pd.DataFrame,
    y:                    pd.Series,
    feat:                 str,
    direction:            str,
    threshold:            float,
    conditions:           list,
    n_folds:              int   = 4,
    min_folds:            int   = 3,
    min_wr:               float = 0.5,
    min_samples_per_fold: int   = 20,
) -> tuple:
    """Temporal walk-forward CV: split into n_folds and check whether the
    pattern holds (WR > min_wr) in at least min_folds of them.

    This is fake-edge filter Layer 1. A pattern that only holds in one
    time period but was lucky overall will fail this check.

    Returns (passes: bool, n_positive_folds: int).
    """
    n      = len(X)
    fold_n = max(1, n // n_folds)
    positive_folds = 0

    for i in range(n_folds):
        start = i * fold_n
        end   = (i + 1) * fold_n if i < n_folds - 1 else n
        X_f   = X.iloc[start:end]
        y_f   = y.iloc[start:end]

        mask = _apply_mask(X_f, feat, direction, threshold)
        for cond in conditions:
            if cond['feature'] not in X_f.columns:
                continue
            mask = mask & _apply_mask(X_f, cond['feature'],
                                       cond['direction'], cond['threshold'])

        sub_y = y_f[mask].dropna()
        if len(sub_y) < min_samples_per_fold:
            continue   # skip thin folds — don't count as failure
        if sub_y.mean() >= min_wr:
            positive_folds += 1

    return positive_folds >= min_folds, positive_folds


def _greedy_conjunction(
    X:               pd.DataFrame,
    y:               pd.Series,
    primary_mask:    pd.Series,
    primary_wr:      float,
    primary_feat:    str,
    feature_cols:    list,
    min_samples:     int,
    min_lift:        float,
    max_conds:       int,
    p_threshold:     float,
    proven_conds:    list = None,
) -> list:
    """Greedy search for secondary conditions that improve win rate.

    Overfitting guards at every level:
      min_lift    — WR must improve by at least this much per added condition
      min_samples — conjunction subset must still have ≥ min_samples rows
      p_threshold — t-test p-value (WR vs 50%) must be significant on conjunction
      max_conds   — hard cap on rule depth

    Phase 1 tries proven conditions from PatternMemory first (targeted search).
    Phase 2 scans all percentile thresholds for remaining features.
    """
    conditions   = []
    current_mask = primary_mask.copy()
    current_wr   = primary_wr

    for _ in range(max_conds):
        best_lift = min_lift
        best_cond = None
        best_mask = None

        # Phase 1: proven conditions first
        if proven_conds:
            for pc in proven_conds:
                sec_feat  = pc['feature']
                direction = pc['direction']
                thresh    = pc['threshold']
                if sec_feat == primary_feat:
                    continue
                if any(c['feature'] == sec_feat for c in conditions):
                    continue
                if sec_feat not in X.columns:
                    continue
                sec_mask = _apply_mask(X, sec_feat, direction, thresh)
                combo    = current_mask & sec_mask
                sub_y    = y[combo]
                if len(sub_y) < min_samples:
                    continue
                wr   = float(sub_y.mean())
                lift = wr - current_wr
                if lift <= best_lift:
                    continue
                _, p = scipy_stats.ttest_1samp(sub_y, 0.5)
                if p >= p_threshold:
                    continue
                best_lift = lift
                best_cond = {'feature': sec_feat, 'direction': direction,
                             'threshold': round(thresh, 6)}
                best_mask = combo

        # Phase 2: full percentile scan
        for sec_feat in feature_cols:
            if sec_feat == primary_feat:
                continue
            if any(c['feature'] == sec_feat for c in conditions):
                continue
            if best_cond and best_cond['feature'] == sec_feat:
                continue

            for pct in THRESHOLD_PERCENTILES:
                thresh = float(np.percentile(X[sec_feat].dropna(), pct))
                for direction in ('>', '<'):
                    sec_mask = _apply_mask(X, sec_feat, direction, thresh)
                    combo    = current_mask & sec_mask
                    sub_y    = y[combo]
                    if len(sub_y) < min_samples:
                        continue
                    wr   = float(sub_y.mean())
                    lift = wr - current_wr
                    if lift <= best_lift:
                        continue
                    _, p = scipy_stats.ttest_1samp(sub_y, 0.5)
                    if p >= p_threshold:
                        continue
                    best_lift = lift
                    best_cond = {'feature': sec_feat, 'direction': direction,
                                 'threshold': round(thresh, 6)}
                    best_mask = combo

        if best_cond is None:
            break

        conditions.append(best_cond)
        current_mask = best_mask
        current_wr   = current_wr + best_lift

    return conditions


def _cross_pair_validate(
    dataset:    pd.DataFrame,
    feat:       str,
    direction:  str,
    thresh:     float,
    conditions: list,
    target:     str,
    min_pairs:  int,
    min_wr:     float,
) -> bool:
    """Check that the pattern holds on at least min_pairs distinct pairs."""
    if 'pair' not in dataset.columns:
        return True

    pairs   = dataset['pair'].unique()
    passing = 0

    for pair in pairs:
        sub = dataset[dataset['pair'] == pair]
        if sub.empty:
            continue
        mask = _apply_mask(sub, feat, direction, thresh)
        for cond in conditions:
            if cond['feature'] not in sub.columns:
                continue
            mask = mask & _apply_mask(sub, cond['feature'],
                                       cond['direction'], cond['threshold'])
        sub_y = sub.loc[mask, target].dropna()
        if len(sub_y) < 20:
            continue
        if sub_y.mean() >= min_wr:
            passing += 1

    return passing >= min_pairs


def mine_patterns(
    dataset:         pd.DataFrame,
    target:          str             = 'long_win',
    top_n:           int             = 10,
    test_size:       float           = 0.3,
    purge_gap:       int             = 500,
    min_samples:     int             = 200,
    min_lift:        float           = 0.02,
    max_conditions:  int             = 2,
    p_threshold:     float           = 0.05,
    min_pairs:       int             = 2,
    drift_features:  list            = None,   # from adversarial_validation()
    memory:          'PatternMemory' = None,
) -> list:
    """Train LightGBM on dataset → target. SHAP ranks features. Greedy conjunction
    search builds multi-feature rules with overfitting guards at every step.

    Fake-edge filter Layer 1 (walk-forward CV) is applied here: each pattern
    must hold in ≥3 of 4 temporal folds of the OOS window.

    Fake-edge Layer 2 (adversarial) is applied upstream by passing drift_features
    which are down-weighted in the SHAP ranking so the miner avoids them.

    Each returned dict:
    {
        'feature':    str,
        'direction':  '>' or '<',
        'threshold':  float,
        'conditions': list[dict],
        'win_rate':   float,
        'n_samples':  int,
        'shap_value': float,
        'target':     str,
        'pairs_pass': int,
        'wf_folds':   int,   # how many of 4 temporal folds passed
    }
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("mine_patterns requires lightgbm: pip install lightgbm") from e

    try:
        import shap
    except ImportError as e:
        raise ImportError("mine_patterns requires shap: pip install shap") from e

    if target not in dataset.columns:
        return []

    feature_cols = [c for c in ALL_FEATURES + TIME_FEATURES if c in dataset.columns]
    if not feature_cols:
        return []

    X = dataset[feature_cols].copy()
    y = dataset[target].copy()

    # Temporal split with purge gap to prevent leakage
    split_idx = int(len(X) * (1 - test_size))
    purge_end = min(split_idx + purge_gap, len(X))
    X_train   = X.iloc[:split_idx]
    y_train   = y.iloc[:split_idx]
    X_oos     = X.iloc[purge_end:]
    y_oos     = y.iloc[purge_end:]

    if len(X_train) < min_samples:
        return []

    model = lgb.LGBMClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.03,
        num_leaves        = 15,
        subsample         = 0.7,
        colsample_bytree  = 0.7,
        min_child_samples = max(30, min_samples // 5),
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        class_weight      = 'balanced',
        random_state      = 42,
        verbose           = -1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_train, y_train)

    # SHAP on OOS window
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shap_vals = explainer.shap_values(X_oos if len(X_oos) > 0 else X)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    mean_shap = np.abs(sv).mean(axis=0)

    # Combine memory weights, SHAP, and adversarial down-weighting
    feat_weights  = memory.feature_weights if memory else {}
    drift_penalty = {f: 0.1 for f in (drift_features or [])}   # down-weight drift axes

    weighted_shap = []
    for feat, shap_s in zip(feature_cols, mean_shap):
        w = feat_weights.get(feat, 1.0) * drift_penalty.get(feat, 1.0)
        weighted_shap.append((feat, shap_s * w))

    feat_importance = sorted(weighted_shap, key=lambda t: t[1], reverse=True)
    proven_conds    = memory.proven_conditions if memory else []

    patterns  = []
    X_search  = X_oos if len(X_oos) >= min_samples else X
    y_search  = y_oos if len(X_oos) >= min_samples else y

    for feat, shap_val in feat_importance[:top_n]:
        if feat not in X_search.columns:
            continue

        best_wr, best_thresh, best_dir = _best_primary(
            X_search, y_search, feat, min_samples
        )
        if best_wr <= 0.5:
            continue

        primary_mask = _apply_mask(X_search, feat, best_dir, best_thresh)
        primary_n    = int(primary_mask.sum())
        if primary_n < min_samples:
            continue

        _, pval = scipy_stats.ttest_1samp(y_search[primary_mask].dropna(), 0.5)
        if pval >= p_threshold:
            continue

        conditions = _greedy_conjunction(
            X            = X_search,
            y            = y_search,
            primary_mask = primary_mask,
            primary_wr   = best_wr,
            primary_feat = feat,
            feature_cols = feature_cols,
            min_samples  = min_samples,
            min_lift     = min_lift,
            max_conds    = max_conditions,
            p_threshold  = p_threshold,
            proven_conds = proven_conds,
        )

        # Final conjunction mask + WR
        final_mask = primary_mask.copy()
        for cond in conditions:
            if cond['feature'] not in X_search.columns:
                continue
            final_mask = final_mask & _apply_mask(
                X_search, cond['feature'], cond['direction'], cond['threshold']
            )
        final_n  = int(final_mask.sum())
        final_wr = float(y_search[final_mask].mean()) if final_n > 0 else best_wr

        # ── Fake-edge Layer 1: Walk-forward CV ─────────────────────────────
        passes_wf, wf_pos_folds = _walk_forward_cv(
            X_search, y_search,
            feat, best_dir, best_thresh, conditions,
            n_folds=4, min_folds=3,
            min_wr=0.5 + min_lift / 2,
        )
        if not passes_wf:
            continue

        # Cross-pair generalisation check
        n_pairs          = dataset['pair'].nunique() if 'pair' in dataset.columns else 1
        effective_min_p  = min(min_pairs, max(1, n_pairs // 2))
        oos_slice        = dataset.iloc[purge_end:] if len(X_oos) > 0 else dataset

        passes_pairs = _cross_pair_validate(
            oos_slice, feat, best_dir, best_thresh, conditions,
            target, effective_min_p, 0.5,
        )
        if not passes_pairs:
            continue

        pairs_pass = sum(
            1 for p in (dataset['pair'].unique() if 'pair' in dataset.columns else [None])
            if p is None or _cross_pair_validate(
                dataset, feat, best_dir, best_thresh, conditions, target, 1, 0.5
            )
        )

        patterns.append({
            'feature':    feat,
            'direction':  best_dir,
            'threshold':  round(best_thresh, 6),
            'conditions': conditions,
            'win_rate':   round(final_wr, 4),
            'n_samples':  final_n,
            'shap_value': round(float(shap_val), 6),
            'target':     target,
            'pairs_pass': pairs_pass,
            'wf_folds':   wf_pos_folds,
        })

    return patterns


def mine_both_directions(dataset: pd.DataFrame, memory: 'PatternMemory' = None,
                         drift_features: list = None, **kwargs) -> list:
    """Run mine_patterns for long_win and short_win, merge and return all."""
    long_patterns  = mine_patterns(dataset, target='long_win',
                                   memory=memory, drift_features=drift_features, **kwargs)
    short_patterns = mine_patterns(dataset, target='short_win',
                                   memory=memory, drift_features=drift_features, **kwargs)
    return long_patterns + short_patterns


# =============================================================================
# SECTION 4 — Hypothesis Code Generator
# =============================================================================

_ENTRY_TEMPLATE = '''\
def entry_{name}(bst, slot, row, ts, pair, slip, hspd,
                 sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if sc.get('pending_dir') is not None:
        return check_and_fill(sc, row, slot, ts, regime)

    # --- auto-generated conditions ---
{conditions}
    # ---------------------------------

    atr = getattr(row, 'atr', float('nan'))
    if math.isnan(atr) or atr <= 0:
        return False

    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    entry = row.close {entry_offset}
    size  = rv_size(pair, bst.balance, risk, entry, sl_dist, row)
    sc['pending_dir']   = '{direction}'
    sc['pending_level'] = row.close
    sc['pending_entry'] = entry
    sc['pending_sl']    = entry {sl_offset} sl_dist
    sc['pending_tp']    = entry {tp_offset} tp_dist
    sc['pending_size']  = size
    sc['pending_dist']  = sl_dist
    return check_and_fill(sc, row, slot, ts, regime)
'''

_SWEEP_TEMPLATE = '''\
SWEEPS['{key}'] = {{
    'entry_fn':    entry_{name},
    'manager_fn':  make_manager(exit_hour={exit_hour}, use_breakeven=True),
    'pairs':       {pairs_repr},
    'session':     '{session}',
    'regime_mult': {{
        'TRENDING':      1.0,
        'RANGING':       0.5,
        'TRANSITIONING': 0.5,
        'VOLATILE':      0.0,
        'UNDEFINED':     0.3,
    }},
    'grid': OptunaGrid({{
        'tp_r': (1.0, 3.5),
        'sl_r': (0.5, 1.5),
    }}, n_trials=80),
}}
'''


def _build_conditions_code(pattern: dict, indent: str = '    ') -> str:
    lines = []
    primary = pattern
    feat    = primary['feature']
    thresh  = primary['threshold']
    direct  = primary['direction']
    lines.append(f"{indent}_{feat} = getattr(row, '{feat}', float('nan'))")
    lines.append(f"{indent}if math.isnan(_{feat}):")
    lines.append(f"{indent}    return False")
    lines.append(f"{indent}if not (_{feat} {direct} {thresh!r}):")
    lines.append(f"{indent}    return False")

    for cond in primary.get('conditions', []):
        cf, ct, cd = cond['feature'], cond['threshold'], cond['direction']
        lines.append(f"{indent}_{cf} = getattr(row, '{cf}', float('nan'))")
        lines.append(f"{indent}if math.isnan(_{cf}) or not (_{cf} {cd} {ct!r}):")
        lines.append(f"{indent}    return False")

    return '\n'.join(lines)


def generate_hypotheses(
    patterns:  list,
    top_n:     int  = 5,
    pairs:     list = None,
    session:   str  = 'ny',
    exit_hour: int  = 21,
) -> list:
    """Convert top_n patterns into runnable hypothesis dicts.

    Each returned dict:
    {
        'name':      str   — Python identifier for the entry function
        'key':       str   — SWEEPS registry key
        'code':      str   — complete Python source (function + SWEEPS entry)
        'sweep_def': dict  — (informational)
        'pattern':   dict  — source pattern
    }
    """
    from edge_engine import NY_PAIRS, LONDON_PAIRS

    if pairs is None:
        pairs = NY_PAIRS if session == 'ny' else LONDON_PAIRS

    results    = []
    seen_names = set()

    for pat in patterns[:top_n]:
        target    = pat['target']
        direction = 'long' if target == 'long_win' else 'short'
        feat_slug = pat['feature'].replace('_', '')[:12]
        base_name = f"mined_{feat_slug}_{direction}"

        name   = base_name
        suffix = 1
        while name in seen_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        seen_names.add(name)

        key             = name
        conditions_code = _build_conditions_code(pat, indent='    ')

        if direction == 'long':
            entry_offset = '+ hspd + slip'
            sl_offset    = '-'
            tp_offset    = '+'
        else:
            entry_offset = '- hspd - slip'
            sl_offset    = '+'
            tp_offset    = '-'

        fn_code = _ENTRY_TEMPLATE.format(
            name         = name,
            conditions   = conditions_code,
            entry_offset = entry_offset,
            direction    = direction,
            sl_offset    = sl_offset,
            tp_offset    = tp_offset,
        )

        pairs_repr = repr(pairs)
        sweep_code = _SWEEP_TEMPLATE.format(
            key        = key,
            name       = name,
            exit_hour  = exit_hour,
            pairs_repr = pairs_repr,
            session    = session,
        )

        full_code = fn_code + '\n\n' + sweep_code

        results.append({
            'name':      name,
            'key':       key,
            'code':      full_code,
            'sweep_def': {'session': session, 'pairs': pairs, 'pattern': pat},
            'pattern':   pat,
        })

    return results


def append_to_hypotheses_file(
    hypotheses: list,
    filepath:   str = None,
) -> None:
    """Append generated entry functions + SWEEPS entries to edge_hypotheses.py.
    Idempotent: skips any function name already present in the file.
    """
    if filepath is None:
        filepath = Path(__file__).parent / 'edge_hypotheses.py'
    filepath = Path(filepath)

    existing_text = filepath.read_text(encoding='utf-8')
    ts_str        = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    new_blocks = []
    for h in hypotheses:
        fn_name = f"def entry_{h['name']}"
        if fn_name in existing_text:
            continue
        new_blocks.append(h['code'])

    if not new_blocks:
        return

    header   = f"\n\n# === AUTO-GENERATED {ts_str} ===\n"
    addition = header + '\n\n'.join(new_blocks) + '\n'

    try:
        ast.parse(existing_text + addition)
    except SyntaxError as e:
        raise RuntimeError(
            f"Generated code has a syntax error, not writing to file: {e}"
        ) from e

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(addition)


# =============================================================================
# SECTION 5 — Meta-Learner: what makes a condition robust? (Next-level #6)
# =============================================================================

def fit_meta_learner(memory: 'PatternMemory') -> dict:
    """Train a logistic regression on proven conditions to predict robustness.

    Target: hit_count > 1 (condition has been re-confirmed at least once across
    rounds). Features: which base feature, direction, session, target.

    This answers: "ignoring what the conditions say about markets, which types
    of conditions tend to survive multiple rounds?" Useful for prioritising
    which mined patterns to test first.

    Returns:
    {
        'feature_ranking': list[(feature_name, score)],   sorted best → worst
        'n_trained':       int,
        'accuracy':        float,
        'note':            str,
    }
    """
    pc = memory.proven_conditions
    if len(pc) < 10:
        return {'feature_ranking': [], 'n_trained': 0, 'accuracy': 0.0,
                'note': 'Not enough proven conditions yet — run more rounds first.'}

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder
    except ImportError:
        return {'feature_ranking': [], 'n_trained': 0, 'accuracy': 0.0,
                'note': 'scikit-learn not installed (pip install scikit-learn)'}

    # Aggregate: for each unique feature, compute mean hit_count and mean avg_wr
    feat_stats: dict = {}
    for cond in pc:
        f = cond['feature']
        if f not in feat_stats:
            feat_stats[f] = {'hit_counts': [], 'avg_wrs': []}
        feat_stats[f]['hit_counts'].append(cond['hit_count'])
        feat_stats[f]['avg_wrs'].append(cond['avg_wr'])

    ranking = []
    for feat, stats in feat_stats.items():
        score = np.mean(stats['hit_counts']) * np.mean(stats['avg_wrs'])
        ranking.append((feat, round(float(score), 4)))
    ranking.sort(key=lambda x: x[1], reverse=True)

    # Simple per-condition logistic regression
    all_feats   = list(feat_stats.keys())
    feat_to_idx = {f: i for i, f in enumerate(all_feats)}
    rows, labels = [], []
    for cond in pc:
        fi = feat_to_idx.get(cond['feature'], 0) / max(len(all_feats) - 1, 1)
        di = 1 if cond['direction'] == '>' else 0
        ti = 1 if cond.get('target', '') == 'long_win' else 0
        rows.append([fi, di, ti, float(cond['avg_wr'])])
        labels.append(1 if cond['hit_count'] > 1 else 0)

    if len(rows) < 10 or len(set(labels)) < 2:
        return {'feature_ranking': ranking, 'n_trained': len(rows), 'accuracy': 0.0,
                'note': 'Insufficient label diversity for logistic regression.'}

    X_m = StandardScaler().fit_transform(np.array(rows))
    y_m = np.array(labels)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = LogisticRegression(random_state=42, max_iter=500)
        clf.fit(X_m, y_m)

    acc = float((clf.predict(X_m) == y_m).mean())

    return {
        'feature_ranking': ranking,
        'n_trained':       len(rows),
        'accuracy':        round(acc, 3),
        'note':            f"Logistic regression accuracy: {acc:.1%}",
    }


# =============================================================================
# SECTION 6 — Miner Orchestrator
# =============================================================================

def run_miner(
    pair_dfs:           dict,
    test_dfs:           dict           = None,   # for adversarial validation + KS-test
    session_filter:     str            = 'ny',
    tp_r:               float          = 2.0,
    sl_r:               float          = 1.0,
    horizon_bars:       int            = 20,
    top_n_patterns:     int            = 5,
    min_feature_edge:   float          = 0.03,
    use_engineered:     bool           = True,
    adversarial:        bool           = True,   # run adversarial validation
    adv_auc_warn:       float          = 0.60,
    adv_auc_abort:      float          = 0.75,
    # overfitting guards forwarded to mine_patterns
    purge_gap:          int            = 500,
    min_samples:        int            = 200,
    min_lift:           float          = 0.02,
    max_conditions:     int            = 2,
    p_threshold:        float          = 0.05,
    min_pairs:          int            = 2,
    memory:             'PatternMemory' = None,
    append_to_file:     bool           = True,
    hypotheses_path:    str            = None,
    progress_callback:  callable       = None,
) -> dict:
    """Full discovery pipeline: label → sweep → adversarial validation → mine → generate.

    Returns:
    {
        'feature_sweep':    pd.DataFrame,
        'patterns':         list[dict],
        'hypotheses':       list[dict],
        'adversarial':      dict,         result of adversarial_validation()
        'dist_shift':       dict,         result of check_distribution_shift()
        'meta_learner':     dict,         result of fit_meta_learner()
    }
    """
    def _progress(step, label):
        if progress_callback:
            progress_callback(step, 7, label)

    _progress(1, 'Building label dataset…')
    dataset = build_label_dataset(
        pair_dfs,
        tp_r            = tp_r,
        sl_r            = sl_r,
        horizon_bars    = horizon_bars,
        session_filter  = session_filter,
        use_engineered  = use_engineered,
    )

    if dataset.empty:
        return {'feature_sweep': pd.DataFrame(), 'patterns': [], 'hypotheses': [],
                'adversarial': {}, 'dist_shift': {}, 'meta_learner': {}}

    # ── Distribution shift check ──────────────────────────────────────────────
    dist_shift_result = {}
    if test_dfs is not None:
        _progress(2, 'Checking distribution shift (KS-test)…')
        test_dataset = build_label_dataset(
            test_dfs,
            tp_r           = tp_r,
            sl_r           = sl_r,
            horizon_bars   = horizon_bars,
            session_filter = session_filter,
            use_engineered = use_engineered,
        )
        if not test_dataset.empty:
            feat_cols = [c for c in ALL_FEATURES if c in dataset.columns and c in test_dataset.columns]
            dist_shift_result = check_distribution_shift(dataset, test_dataset, feat_cols)
            if dist_shift_result.get('any_shift'):
                print(f"[Miner] Distribution shift detected on: "
                      f"{', '.join(dist_shift_result['shifted_features'][:5])}")
    else:
        _progress(2, 'Skipping distribution shift (no test_dfs provided)…')

    # ── Adversarial validation ─────────────────────────────────────────────────
    adv_result   = {}
    drift_feats  = []
    if adversarial and test_dfs is not None and not dataset.empty:
        _progress(3, 'Running adversarial validation…')
        feat_cols = [c for c in ALL_FEATURES if c in dataset.columns]
        test_d    = build_label_dataset(test_dfs, tp_r=tp_r, sl_r=sl_r,
                                         session_filter=session_filter,
                                         use_engineered=use_engineered)
        if not test_d.empty:
            adv_result  = adversarial_validation(
                dataset, test_d, feat_cols,
                auc_warn=adv_auc_warn, auc_abort=adv_auc_abort,
            )
            drift_feats = adv_result.get('drift_features', [])
            print(f"[Miner] Adversarial validation: {adv_result.get('message', '')}")
            if adv_result.get('verdict') == 'ABORT':
                print("[Miner] WARNING: Aborting ML mining due to high distribution divergence.")
                return {
                    'feature_sweep': pd.DataFrame(), 'patterns': [], 'hypotheses': [],
                    'adversarial': adv_result, 'dist_shift': dist_shift_result,
                    'meta_learner': {},
                }
    else:
        _progress(3, 'Skipping adversarial validation…')

    _progress(4, 'Running feature sweep…')
    sweep_df = feature_sweep(dataset, min_edge=min_feature_edge)

    _progress(5, 'Mining patterns with LightGBM + SHAP + walk-forward CV…')
    try:
        patterns = mine_both_directions(
            dataset,
            memory         = memory,
            drift_features = drift_feats,
            top_n          = top_n_patterns * 2,
            min_samples    = max(min_samples, len(dataset) // 200),
            purge_gap      = purge_gap,
            min_lift       = min_lift,
            max_conditions = max_conditions,
            p_threshold    = p_threshold,
            min_pairs      = min_pairs,
        )
    except ImportError as e:
        patterns = []
        print(f"[Miner] Skipping ML mining: {e}")

    # Fill gaps with top feature_sweep results if ML gave nothing
    if not patterns and not sweep_df.empty:
        for _, row in sweep_df.head(top_n_patterns).iterrows():
            patterns.append({
                'feature':    row['feature'],
                'direction':  row['direction'],
                'threshold':  row['threshold'],
                'conditions': [],
                'win_rate':   row['win_rate'],
                'n_samples':  row['n'],
                'shap_value': abs(row['t_stat']),
                'target':     row['target'],
                'wf_folds':   0,
            })

    _progress(6, 'Generating hypothesis code…')
    session_exit = {'ny': 21, 'london': 13, 'asian': 8}.get(session_filter, 21)
    hypotheses   = generate_hypotheses(
        patterns,
        top_n     = top_n_patterns,
        session   = session_filter or 'ny',
        exit_hour = session_exit,
    )

    if append_to_file and hypotheses:
        try:
            append_to_hypotheses_file(hypotheses, filepath=hypotheses_path)
        except Exception as e:
            print(f"[Miner] Could not append to hypotheses file: {e}")

    # ── Meta-learner (requires memory) ────────────────────────────────────────
    meta_result = {}
    if memory is not None:
        meta_result = fit_meta_learner(memory)

    _progress(7, 'Done.')
    return {
        'feature_sweep': sweep_df,
        'patterns':      patterns,
        'hypotheses':    hypotheses,
        'adversarial':   adv_result,
        'dist_shift':    dist_shift_result,
        'meta_learner':  meta_result,
    }


# =============================================================================
# SECTION 7 — Pattern Mutation
# =============================================================================

def _mutate_patterns(patterns: list, mutation_rate: float = 0.15,
                     n_mutations: int = 2) -> list:
    """Perturb thresholds of surviving patterns ±mutation_rate (relative).

    Each pattern spawns n_mutations variants. Mutations are tested in the next
    round — if a shifted threshold also survives, the edge is robust to the
    exact threshold value (not overfit to it).
    """
    mutated = []
    for pat in patterns:
        for i in range(1, n_mutations + 1):
            sign  = 1 if i % 2 == 0 else -1
            delta = mutation_rate * sign

            new_thresh = round(pat['threshold'] * (1 + delta), 6)
            new_conds  = [
                {**c, 'threshold': round(c['threshold'] * (1 + delta), 6)}
                for c in pat.get('conditions', [])
            ]
            mutated.append({
                **pat,
                'threshold':  new_thresh,
                'conditions': new_conds,
                'shap_value': pat['shap_value'] * 0.9,
                '_mutated':   True,
            })
    return mutated


# =============================================================================
# SECTION 8 — Self-Improving Discovery Loop
# =============================================================================

def run_discovery_loop(
    pair_dfs:              dict,
    train_dfs:             dict,
    test_dfs:              dict,
    n_rounds:              int   = 5,
    session_filter:        str   = 'ny',
    hypotheses_per_round:  int   = 5,
    n_optuna_trials:       int   = 80,
    n_workers:             int   = 4,
    cost_mult:             float = 1.0,
    mutation_rate:         float = 0.15,
    # overfitting guards (forwarded to mine_patterns)
    purge_gap:             int   = 500,
    min_samples:           int   = 200,
    min_lift:              float = 0.02,
    max_conditions:        int   = 2,
    p_threshold:           float = 0.05,
    min_pairs:             int   = 2,
    # fake-edge filter thresholds applied to sweep survivors
    require_dsr_positive:  bool  = True,
    require_ci_positive:   bool  = True,
    require_regime_stable: bool  = False,  # strict — off by default
    # adversarial / concept drift
    run_adversarial:       bool  = True,
    decay_memory:          bool  = True,
    memory:                'PatternMemory' = None,
    append_to_file:        bool  = True,
    hypotheses_path:       str   = None,
    progress_callback:     callable = None,
) -> dict:
    """Run N rounds of: mine → generate → sweep → learn → mutate.

    Each round:
      1. (Optionally) decay stale feature weights in memory (Next-level #5)
      2. Mine patterns (walk-forward CV + optional adversarial gating)
      3. Generate entry functions for top patterns + mutations from last round
      4. Run Optuna sweep (warm-started from memory param regions)
      5. Collect survivors using combined fake-edge filter:
           BH-sig AND DSR>0 AND bootstrap CI_low>0 AND (optionally) regime_stable
      6. Update PatternMemory
      7. Mutate survivors for next round

    Returns:
    {
        'memory':     PatternMemory,
        'rounds':     list[dict],
        'all_sweeps': list[str],
        'survivors':  pd.DataFrame,
    }
    """
    from edge_engine import (
        run_sweep_optuna, OptunaGrid, load_sweep_results,
        NY_PAIRS, LONDON_PAIRS,
    )

    if memory is None:
        memory = PatternMemory()

    session_exit  = {'ny': 21, 'london': 13, 'asian': 8}.get(session_filter, 21)
    pairs         = NY_PAIRS if session_filter == 'ny' else LONDON_PAIRS
    all_rounds    = []
    all_sweep_ids = []
    all_survivors = []
    mutation_pool = []

    def _prog(msg: str, round_n: int = 0):
        if progress_callback:
            progress_callback(round_n, n_rounds, msg)

    for round_n in range(1, n_rounds + 1):
        _prog(f"Round {round_n}/{n_rounds} — decaying memory…", round_n)

        # ── Step 0: Concept drift — decay stale weights ──────────────────────
        if decay_memory and round_n > 1:
            memory.decay_weights(decay_factor=0.9, days_threshold=1)

        _prog(f"Round {round_n}/{n_rounds} — mining patterns…", round_n)

        # ── Step 1: Mine ─────────────────────────────────────────────────────
        mine_result = run_miner(
            pair_dfs          = pair_dfs,
            test_dfs          = test_dfs,
            session_filter    = session_filter,
            top_n_patterns    = hypotheses_per_round,
            purge_gap         = purge_gap,
            min_samples       = min_samples,
            min_lift          = min_lift,
            max_conditions    = max_conditions,
            p_threshold       = p_threshold,
            min_pairs         = min_pairs,
            adversarial       = run_adversarial,
            memory            = memory,
            append_to_file    = False,
            progress_callback = None,
        )

        fresh_patterns = mine_result.get('patterns', [])

        # ── Step 2: Mutations from last round ────────────────────────────────
        all_patterns = fresh_patterns + mutation_pool
        if not all_patterns:
            _prog(f"Round {round_n}: no patterns found, skipping.", round_n)
            all_rounds.append({'round': round_n, 'n_tested': 0, 'n_survivors': 0})
            continue

        hypotheses = generate_hypotheses(
            all_patterns,
            top_n     = hypotheses_per_round + len(mutation_pool),
            session   = session_filter or 'ny',
            exit_hour = session_exit,
            pairs     = pairs,
        )

        if append_to_file and hypotheses:
            try:
                append_to_hypotheses_file(hypotheses, filepath=hypotheses_path)
            except Exception as e:
                print(f"[Loop] Could not append round {round_n}: {e}")

        # ── Step 3: Sweep each hypothesis with Optuna ────────────────────────
        round_sweep_ids = []
        round_survivors = []

        for hyp_idx, hyp in enumerate(hypotheses):
            _prog(
                f"Round {round_n}/{n_rounds} — sweeping {hyp_idx+1}/{len(hypotheses)}: "
                f"{hyp['name']}",
                round_n,
            )

            sweep_name  = f"loop_r{round_n}_{hyp['name']}"
            warm_params = memory.warm_start_params(sweep_name)

            opt_grid = OptunaGrid(
                {'tp_r': (1.0, 3.5), 'sl_r': (0.5, 1.5)},
                n_trials=n_optuna_trials,
            )
            opt_grid.warm_params = warm_params

            try:
                import importlib, sys
                hyp_mod_name = 'edge_hypotheses'
                if hyp_mod_name in sys.modules:
                    hyp_mod = importlib.reload(sys.modules[hyp_mod_name])
                else:
                    import importlib.util
                    hyp_path = hypotheses_path or (Path(__file__).parent / 'edge_hypotheses.py')
                    spec = importlib.util.spec_from_file_location(hyp_mod_name, hyp_path)
                    hyp_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(hyp_mod)
                    sys.modules[hyp_mod_name] = hyp_mod

                entry_fn = getattr(hyp_mod, f"entry_{hyp['name']}", None)
                if entry_fn is None:
                    continue

                from edge_engine import make_manager
                manager_fn = make_manager(exit_hour=session_exit, use_breakeven=True)

                regime_mult = {
                    'TRENDING': 1.0, 'RANGING': 0.5, 'TRANSITIONING': 0.5,
                    'VOLATILE': 0.0, 'UNDEFINED': 0.3,
                }

                sweep_id = run_sweep_optuna(
                    sweep_name  = sweep_name,
                    entry_fn    = entry_fn,
                    manager_fn  = manager_fn,
                    grid        = opt_grid,
                    pairs       = pairs,
                    session     = session_filter or 'ny',
                    regime_mult = regime_mult,
                    train_dfs   = train_dfs,
                    test_dfs    = test_dfs,
                    n_workers   = n_workers,
                    cost_mult   = cost_mult,
                )
                round_sweep_ids.append(sweep_id)
                all_sweep_ids.append(sweep_id)

                results = load_sweep_results(sweep_id)
                if not results.empty and 'bh_sig' in results.columns:
                    # ── Combined fake-edge filter ─────────────────────────────
                    surv = results[results['bh_sig'] == 1].copy()

                    if require_dsr_positive and 'dsr' in surv.columns:
                        surv = surv[surv['dsr'].fillna(-1) > 0]

                    if require_ci_positive and 'sharpe_ci_low' in surv.columns:
                        surv = surv[surv['sharpe_ci_low'].fillna(-1) > 0]

                    if require_regime_stable and 'regime_stable' in surv.columns:
                        surv = surv[surv['regime_stable'].fillna(0) == 1]

                    if not surv.empty:
                        surv['round'] = round_n
                        surv['hyp']   = hyp['name']
                        round_survivors.append(surv)
                        all_survivors.append(surv)
                        memory.update_from_sweep_results(surv, sweep_name)

            except Exception as e:
                print(f"[Loop] Sweep failed for {hyp['name']}: {e}")

        # ── Step 4: Update memory from mined patterns ────────────────────────
        memory.update_from_patterns(fresh_patterns, session=session_filter or 'ny')

        # ── Step 5: Mutate top fresh patterns for next round ─────────────────
        n_survivors   = sum(len(s) for s in round_survivors)
        mutation_pool = _mutate_patterns(fresh_patterns[:3], mutation_rate=mutation_rate)

        memory.record_round(
            round_n     = round_n,
            sweep_id    = ','.join(round_sweep_ids),
            n_tested    = len(hypotheses),
            n_survivors = n_survivors,
            session     = session_filter or 'ny',
        )

        all_rounds.append({
            'round':       round_n,
            'n_tested':    len(hypotheses),
            'n_survivors': n_survivors,
            'sweep_ids':   round_sweep_ids,
        })

        _prog(
            f"Round {round_n} done — {n_survivors} real-edge survivors, "
            f"{len(mutation_pool)} mutations queued.",
            round_n,
        )

    survivors_df = pd.concat(all_survivors, ignore_index=True) \
                   if all_survivors else pd.DataFrame()

    return {
        'memory':     memory,
        'rounds':     all_rounds,
        'all_sweeps': all_sweep_ids,
        'survivors':  survivors_df,
    }