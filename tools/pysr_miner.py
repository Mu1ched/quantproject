"""
PySR symbolic-regression scaffold — mine human-readable predictive formulas.

PySR is Cranmer's evolutionary symbolic regression engine; on a feature
matrix X and target y it returns a Pareto front of (complexity, accuracy)
formulas like:
    y ≈ 0.34 · tanh(tick_imb_roll5) − 0.12 · log(yz_vol_ratio + 1) · sign(h1_trend)

These formulas can be:
  1. Treated as new features — append `f(row)` to MINER_FEATURES and let
     the rest of the pipeline (LLM, GP, miner) use them.
  2. Used as standalone entry conditions: trade when |f(row)| > θ.
  3. Acted on as sanity checks: if PySR finds nothing predictive, neither
     will Claude or the GP.

This module is a SCAFFOLD — it depends on PySR, which requires a working
Julia install. Import is lazy; the rest of the agent runs fine without it.

Usage:
    python -m agent.pysr_miner --pair EUR_USD --session london --out formulas.json
    # then feed formulas.json into your feature pipeline manually,
    # or call apply_pysr_formulas(df, formulas) inside prepare_df

Realistic runtime: 2-6 hours per (pair, session) on a 4-core CPU at the
default settings. Run it overnight, don't put it on the live loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Lazy import guard ─────────────────────────────────────────────────────────

def _import_pysr():
    """Import PySR lazily and emit a clear error if the env isn't ready."""
    try:
        from pysr import PySRRegressor  # noqa: F401
        return PySRRegressor
    except ImportError as e:
        raise ImportError(
            "PySR not installed. Install with `pip install pysr` and ensure "
            "Julia is installed (PySR will bootstrap its own Julia env on "
            "first use). Then run this module again."
        ) from e


# ── Feature subset for symbolic regression ────────────────────────────────────
# Kept smaller than MINER_FEATURES so PySR's combinatorial search stays
# tractable. Drop or extend per experiment.
PYSR_FEATURE_COLS: List[str] = [
    'tick_imbalance', 'tick_imb_roll5', 'persistent_imbalance',
    'vol_imbalance', 'aggressive_buy_ratio',
    'atr_ratio', 'rv_delta', 'yz_vol_ratio',
    'adx', 'ma_dist', 'momentum_3', 'bb_pct', 'rsi_14',
    'hurst', 'perm_entropy_100', 'hawkes_intensity',
    'h1_trend_strength',
]


# ── Target construction ───────────────────────────────────────────────────────

def _next_return_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Future log-return over `horizon` bars, with NaN at the tail."""
    if 'close' not in df.columns:
        raise ValueError("df missing 'close' column")
    ret = np.log(df['close'].clip(lower=1e-12)).diff(horizon).shift(-horizon)
    return ret


def _classify_target(df: pd.DataFrame, horizon: int = 5,
                     threshold: float = 0.0005) -> pd.Series:
    """Tri-state target: +1 if next-`horizon`-bar return > threshold,
    -1 if < -threshold, 0 otherwise. Useful for symbolic classification."""
    fut = _next_return_target(df, horizon)
    return np.sign(fut.where(fut.abs() > threshold, 0.0)).astype(float)


# ── Main run loop ─────────────────────────────────────────────────────────────

def run_pysr(
    df:                pd.DataFrame,
    feature_cols:      Optional[List[str]] = None,
    target:            str = 'next_return_5',
    n_iterations:      int = 200,
    population_size:   int = 33,
    populations:       int = 8,
    binary_operators:  Optional[List[str]] = None,
    unary_operators:   Optional[List[str]] = None,
    output_path:       Optional[str] = None,
):
    """
    Run PySR on a prepared df. Returns the fitted PySRRegressor.

    Conservative defaults — small grammar, modest iterations — to keep
    runtime reasonable. Increase n_iterations / populations for more
    thorough search at proportional time cost.
    """
    PySRRegressor = _import_pysr()

    feature_cols = feature_cols or [c for c in PYSR_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise ValueError("No PYSR_FEATURE_COLS present in df after intersection")

    if target == 'next_return_5':
        y = _next_return_target(df, horizon=5)
    elif target == 'next_return_1':
        y = _next_return_target(df, horizon=1)
    elif target == 'classify_5':
        y = _classify_target(df, horizon=5, threshold=0.0005)
    else:
        raise ValueError(f"Unknown target {target!r}")

    X = df[feature_cols].copy()
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[mask].astype(float)
    y = y.loc[mask].astype(float)
    log.info("PySR: training on %d rows × %d features → target=%s",
             len(X), X.shape[1], target)

    binary_operators = binary_operators or ['+', '-', '*', '/']
    unary_operators  = unary_operators  or ['sin', 'cos', 'exp', 'log', 'tanh', 'sqrt']

    model = PySRRegressor(
        niterations         = n_iterations,
        population_size     = population_size,
        populations         = populations,
        binary_operators    = binary_operators,
        unary_operators     = unary_operators,
        model_selection     = 'best',
        progress            = True,
        random_state        = 42,
        deterministic       = False,
    )
    model.fit(X.values, y.values, variable_names=feature_cols)

    if output_path:
        formulas = export_formulas(model, feature_cols)
        Path(output_path).write_text(json.dumps(formulas, indent=2))
        log.info("PySR: %d Pareto-front formulas written to %s",
                 len(formulas), output_path)

    return model


def export_formulas(model, feature_cols: List[str]) -> List[dict]:
    """
    Pull the Pareto front from a fitted PySRRegressor into a JSON-friendly
    list of {complexity, loss, equation, sympy} entries.
    """
    eqns = model.equations_
    if eqns is None or eqns.empty:
        return []
    out = []
    for _, row in eqns.iterrows():
        out.append({
            'complexity':  int(row.get('complexity', 0)),
            'loss':        float(row.get('loss', float('nan'))),
            'equation':    str(row.get('equation', '')),
            'features':    feature_cols,
        })
    return out


def apply_pysr_formulas(
    df: pd.DataFrame, formulas: List[dict], prefix: str = 'pysr_'
) -> pd.DataFrame:
    """
    Evaluate exported PySR formulas against `df` and append them as new columns.

    Each formula is a string like "0.34 * tanh(tick_imb_roll5) - 0.12 * log(...)"
    using only feature names that exist in `df`. We compile via numexpr-style
    eval inside a sandbox dict so no arbitrary code runs — only numpy ufuncs
    and the listed feature columns.

    Returns a copy of df with new columns prefix+0, prefix+1, ...
    """
    if not formulas:
        return df
    out = df.copy()
    safe_globals = {
        'np':    np,
        'sin':   np.sin,   'cos':   np.cos,    'exp':   np.exp,
        'log':   lambda x: np.log(np.clip(x, 1e-12, None)),
        'tanh':  np.tanh,  'sqrt':  lambda x: np.sqrt(np.abs(x)),
        'abs':   np.abs,   'sign':  np.sign,
        '__builtins__': {},
    }
    for i, f in enumerate(formulas):
        eq = f.get('equation', '')
        if not eq:
            continue
        try:
            local = {c: out[c].values for c in f.get('features', []) if c in out.columns}
            val = eval(eq, safe_globals, local)
            out[f'{prefix}{i}'] = val
        except Exception as e:
            log.warning("PySR formula %d evaluation failed: %s", i, e)
    return out


# ── CLI entry ─────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Mine PySR formulas from prepared data."
    )
    parser.add_argument('--pair',    required=True)
    parser.add_argument('--session', default='london')
    parser.add_argument('--target',  default='next_return_5',
                        choices=['next_return_1', 'next_return_5', 'classify_5'])
    parser.add_argument('--n-iters', type=int, default=200)
    parser.add_argument('--out',     default='pysr_formulas.json')
    args = parser.parse_args()

    # Late import to avoid pulling edge_engine into the agent loop.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import edge_engine as eng

    log.info("PySR: loading %s data...", args.pair)
    train_dfs, _, _ = eng.load_all_data()
    df = train_dfs.get(args.pair)
    if df is None or df.empty:
        log.error("No training data for pair %s", args.pair)
        return 1

    sess_lo, sess_hi = eng.SESSION_HOURS.get(args.session, (0, 24))
    df = df[(df.get('hour', 0) >= sess_lo) & (df.get('hour', 0) < sess_hi)]
    if df.empty:
        log.error("No rows in %s session for %s", args.session, args.pair)
        return 1

    run_pysr(
        df,
        target       = args.target,
        n_iterations = args.n_iters,
        output_path  = args.out,
    )
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_cli())
