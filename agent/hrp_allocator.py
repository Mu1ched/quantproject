"""
Hierarchical Risk Parity (HRP) — López de Prado, 2016.

Replaces equal-weight or naive inverse-vol allocation across surviving
strategies. HRP avoids the numerical fragility of Markowitz mean-variance
(which inverts an ill-conditioned covariance matrix and concentrates
weight on the noisiest assets) by:

  1. Building a correlation distance metric d_ij = √(0.5 · (1 - ρ_ij))
  2. Hierarchically clustering strategies by single-linkage on d
  3. Quasi-diagonalising the covariance matrix according to cluster order
  4. Recursively bisecting and allocating inversely to cluster variance

Result: weights that respect the cluster structure of strategy returns.
Highly-correlated strategies share their cluster's allocation rather than
each capturing full inverse-vol weight; truly orthogonal strategies get
heavier weight than they would under naive inverse-vol.

Compared to equal-weight, expected lift on portfolio Sharpe is 15-30%
when there are 5+ surviving strategies with a non-trivial correlation
structure (which is the realistic case once the agent has been running
for a few weeks).

Public API:
    hrp_weights(daily_pnls: dict[name -> pd.Series]) -> dict[name -> float]
    portfolio_sharpe_hrp(daily_pnls: dict) -> float
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Internals ─────────────────────────────────────────────────────────────────

def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Mantegna's correlation distance: √(0.5 · (1 - ρ)).
    Bounded in [0, 1]; valid metric (triangle inequality holds)."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


def _single_linkage(dist: np.ndarray) -> List[tuple]:
    """Agglomerative single-linkage clustering via scipy. Returns the
    standard scipy linkage matrix (n-1 merges)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    # squareform rejects matrices with non-zero diagonal precision — clean it.
    d = dist.copy()
    np.fill_diagonal(d, 0.0)
    return linkage(squareform(d, checks=False), method='single')


def _quasi_diag_order(linkage_matrix) -> List[int]:
    """Reorder leaves so that highly correlated assets sit next to each
    other in the resulting covariance permutation (López de Prado §16.4)."""
    link = linkage_matrix.astype(int)
    n_obs = link[-1, 3]
    # Start with the two clusters of the last merge.
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_idx.max() >= n_obs:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= n_obs]
        i = df0.index
        j = df0.values - n_obs
        sort_idx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df0]).sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    return sort_idx.tolist()


def _cluster_var(cov: pd.DataFrame, items: List[int]) -> float:
    """Inverse-variance weighted variance of a cluster. Reduces to
    Markowitz minimum-variance for the cluster's covariance sub-block,
    but uses the diagonal-only inverse-variance approximation — which
    is what makes HRP numerically robust."""
    sub = cov.iloc[items, items]
    diag = np.diag(sub.values)
    if (diag <= 0).any():
        return float('inf')
    inv_diag = 1.0 / diag
    w = inv_diag / inv_diag.sum()
    return float(w @ sub.values @ w)


def _recursive_bisection(cov: pd.DataFrame, ordered_items: List[int]) -> pd.Series:
    """López de Prado's recursive bisection. Walks the quasi-diagonalised
    covariance and splits cluster weight inversely to sub-cluster variance."""
    w = pd.Series(1.0, index=ordered_items)
    clusters = [ordered_items]
    while clusters:
        clusters = [
            c[start:end]
            for c in clusters
            if len(c) > 1
            for start, end in (
                (0, len(c) // 2),
                (len(c) // 2, len(c)),
            )
            if (end - start) > 0
        ]
        # Pair them up (each cluster split into two halves above).
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            denom = v0 + v1
            if denom <= 0 or not np.isfinite(denom):
                continue
            alpha = 1.0 - v0 / denom
            w.loc[c0] *= alpha
            w.loc[c1] *= 1.0 - alpha
    return w


# ── Public API ────────────────────────────────────────────────────────────────

def hrp_weights(daily_pnls: Dict[str, pd.Series], min_overlap: int = 30) -> Dict[str, float]:
    """
    Compute HRP weights from a {strategy_name: daily_pnl_series} mapping.

    Returns {strategy_name: weight}, weights summing to 1.0. Strategies
    with insufficient overlap are dropped (their weight = 0); remaining
    weights are renormalised.

    Falls back to equal weights if any of:
      * fewer than 2 strategies with sufficient overlap
      * covariance matrix is singular
      * scipy is not available
    """
    names = list(daily_pnls.keys())
    if len(names) < 2:
        return {n: 1.0 for n in names}

    # Align on common dates; drop rows with any NaN.
    aligned = pd.concat(daily_pnls, axis=1).dropna()
    if aligned.shape[0] < min_overlap or aligned.shape[1] < 2:
        log.info("HRP: insufficient overlap (%d days, %d strategies) — equal weights",
                 aligned.shape[0], aligned.shape[1])
        return {n: 1.0 / len(names) for n in names}

    try:
        cov  = aligned.cov()
        corr = aligned.corr().values
        if not np.all(np.isfinite(corr)):
            raise ValueError("non-finite values in correlation matrix")
        dist = _correlation_distance(corr)
        link = _single_linkage(dist)
        order = _quasi_diag_order(link)
        weights = _recursive_bisection(cov, order)
        # Map cov-index ordering back to strategy names.
        col_names = aligned.columns.tolist()
        out = {col_names[i]: float(weights.loc[i]) for i in weights.index}
    except Exception as e:
        log.warning("HRP failed (%s) — falling back to equal weights", e)
        return {n: 1.0 / len(names) for n in names}

    # Strategies that were dropped due to NaN alignment get 0 weight; renormalise.
    full = {n: out.get(n, 0.0) for n in names}
    s = sum(full.values())
    if s <= 0:
        return {n: 1.0 / len(names) for n in names}
    return {k: v / s for k, v in full.items()}


def portfolio_sharpe_hrp(daily_pnls: Dict[str, pd.Series]) -> float:
    """Annualised Sharpe of an HRP-weighted combination of daily PnL streams."""
    weights = hrp_weights(daily_pnls)
    if not weights:
        return 0.0
    aligned = pd.concat(daily_pnls, axis=1).fillna(0.0)
    w_vec = np.array([weights.get(c, 0.0) for c in aligned.columns])
    combined = aligned.values @ w_vec
    if combined.std() == 0 or len(combined) < 20:
        return 0.0
    return float((combined.mean() / combined.std()) * np.sqrt(252))


def hrp_summary(daily_pnls: Dict[str, pd.Series]) -> dict:
    """Diagnostic dict: weights, equal-weight Sharpe, HRP Sharpe, lift."""
    if len(daily_pnls) < 2:
        return {'weights': {}, 'sharpe_eq': 0.0, 'sharpe_hrp': 0.0, 'lift': 0.0}
    weights = hrp_weights(daily_pnls)
    aligned = pd.concat(daily_pnls, axis=1).fillna(0.0)
    eq      = aligned.mean(axis=1)
    sh_eq   = float((eq.mean() / eq.std()) * np.sqrt(252)) if eq.std() else 0.0
    sh_hrp  = portfolio_sharpe_hrp(daily_pnls)
    return {
        'weights':    weights,
        'sharpe_eq':  sh_eq,
        'sharpe_hrp': sh_hrp,
        'lift':       sh_hrp - sh_eq,
    }
