"""
Portfolio-level intelligence: redundancy filtering and portfolio Sharpe tracking.

When a new strategy passes all individual filters (static + robustness), this
module checks whether it adds genuine diversification to the existing survivor
portfolio, or merely duplicates an edge already captured.

Three checks (in order of cost):
  1. Pearson correlation pre-screen — kills strategies with linear redundancy.
     Cheap; fail-fast.
  2. Mutual-information gate (Kraskov k-NN estimator) — catches non-linear
     redundancy that Pearson misses. Two strategies with `corr ≈ 0` but
     mutual information ≈ 1 bit fire on the same underlying signal through
     a non-linear transform (think |x|>θ vs x²>θ²). Pearson admits both;
     MI rejects the redundant one.
  3. Portfolio Sharpe (equal-weight) is logged so the meta-learner can track it.

The MI gate fails open: if sklearn isn't available or the estimator errors,
the strategy is admitted with a warning rather than blocking the pipeline.
"""

import logging
import sqlite3

import numpy as np
import pandas as pd

from agent.config import (
    AGENT_DB_PATH, EDGE_DB_PATH,
    PORTFOLIO_MAX_CORR, PORTFOLIO_MIN_SURVIVORS,
)

# MI threshold (in nats). 0.35 nats ≈ 0.5 bits — at this level, knowing one
# series tells you ~50% of the entropy of the other. Higher → more redundant.
PORTFOLIO_MAX_MI = 0.35

log = logging.getLogger(__name__)


# ── Trade loading ──────────────────────────────────────────────────────────────

def load_daily_pnl(hypothesis_id: str) -> pd.Series:
    """Return a date-indexed daily PnL Series for a hypothesis's test trades."""
    if not hypothesis_id:
        return pd.Series(dtype=float)
    try:
        con = sqlite3.connect(EDGE_DB_PATH)
        df  = pd.read_sql(
            "SELECT exit_time, pnl FROM trades WHERE hypothesis_id = ? AND split = 'test'",
            con, params=(hypothesis_id,),
        )
        con.close()
        if df.empty:
            return pd.Series(dtype=float)
        df['date'] = pd.to_datetime(df['exit_time']).dt.normalize()
        return df.groupby('date')['pnl'].sum()
    except Exception as e:
        log.debug("load_daily_pnl(%s) failed: %s", hypothesis_id, e)
        return pd.Series(dtype=float)


def _load_all_survivor_hypothesis_ids() -> list:
    """Return (strategy_name, hypothesis_id) for all confirmed survivors."""
    try:
        con  = sqlite3.connect(AGENT_DB_PATH)
        rows = con.execute(
            "SELECT strategy_name, hypothesis_id FROM tested_strategies "
            "WHERE verdict = 'survivor' AND hypothesis_id IS NOT NULL AND hypothesis_id != ''"
        ).fetchall()
        con.close()
        return [(r[0], r[1]) for r in rows]
    except Exception:
        return []


# ── Statistics ─────────────────────────────────────────────────────────────────

def _pairwise_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    """Pearson correlation of two daily PnL series aligned by date."""
    if series_a.empty or series_b.empty:
        return 0.0
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 20:
        return 0.0
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    return corr if np.isfinite(corr) else 0.0


def _pairwise_mi(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Kraskov k-NN mutual-information estimator over two daily PnL series.

    Returns MI in nats (natural log). 0 = independent, ln(N) = identical
    after a measurable transform. Uses sklearn's mutual_info_regression
    which implements Kraskov-Stögbauer-Grassberger estimator I.

    Fails open (returns 0.0) on insufficient overlap or sklearn failure.
    """
    if series_a.empty or series_b.empty:
        return 0.0
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 30:
        return 0.0
    try:
        from sklearn.feature_selection import mutual_info_regression
        x = aligned.iloc[:, 0].values.astype(float).reshape(-1, 1)
        y = aligned.iloc[:, 1].values.astype(float)
        # n_neighbors=3 is the canonical choice for small samples (KSG paper);
        # discrete_features=False because daily PnL is continuous.
        mi = mutual_info_regression(
            x, y, n_neighbors=3, discrete_features=False, random_state=42,
        )
        return float(mi[0]) if np.isfinite(mi[0]) else 0.0
    except Exception as e:
        log.debug("MI estimation failed (%s) — failing open", e)
        return 0.0


def portfolio_sharpe(pnl_series_list: list) -> float:
    """Annualised Sharpe of equal-weighted combination of daily PnL streams."""
    if not pnl_series_list:
        return 0.0
    combined = pd.concat(pnl_series_list, axis=1).fillna(0.0).sum(axis=1)
    if combined.std() == 0 or len(combined) < 20:
        return 0.0
    return float((combined.mean() / combined.std()) * np.sqrt(252))


# ── Public API ─────────────────────────────────────────────────────────────────

def is_portfolio_additive(
    new_strategy_name: str,
    new_hypothesis_id: str,
) -> tuple:
    """
    Determine whether a new survivor adds genuine diversification.

    Returns (additive: bool, reason: str).

    Passes unconditionally if:
      - The new hypothesis's trade data can't be loaded (benefit of the doubt)
      - Fewer than PORTFOLIO_MIN_SURVIVORS exist (too early to enforce correlation)

    Fails if:
      - Daily PnL correlation with any existing survivor > PORTFOLIO_MAX_CORR
    """
    existing = _load_all_survivor_hypothesis_ids()

    if len(existing) < PORTFOLIO_MIN_SURVIVORS:
        return True, f"portfolio has {len(existing)} survivors — correlation filter not yet active"

    new_pnl = load_daily_pnl(new_hypothesis_id)
    if new_pnl.empty:
        return True, "could not load trade data for new strategy — accepting"

    existing_pnls = []
    for name, hyp_id in existing:
        if hyp_id == new_hypothesis_id:
            continue
        pnl = load_daily_pnl(hyp_id)
        if pnl.empty:
            continue

        # ── Layer 1: Pearson correlation (cheap, fails fast on linear redundancy) ──
        corr = _pairwise_corr(new_pnl, pnl)
        if corr > PORTFOLIO_MAX_CORR:
            return (
                False,
                f"daily PnL correlation {corr:.2f} with '{name}' exceeds limit {PORTFOLIO_MAX_CORR}"
            )

        # ── Layer 2: Mutual information (catches non-linear redundancy) ──
        # Only run if Pearson is borderline-but-passing — saves the k-NN cost on
        # obviously-orthogonal pairs (corr < 0.3 are unlikely to share information
        # through any monotone transform).
        if abs(corr) > 0.30:
            mi = _pairwise_mi(new_pnl, pnl)
            if mi > PORTFOLIO_MAX_MI:
                return (
                    False,
                    f"mutual information {mi:.2f} nats with '{name}' "
                    f"exceeds limit {PORTFOLIO_MAX_MI} (corr={corr:.2f})"
                )

        existing_pnls.append(pnl)

    # Log portfolio Sharpe improvement
    if existing_pnls:
        sharpe_before = portfolio_sharpe(existing_pnls)
        sharpe_after  = portfolio_sharpe(existing_pnls + [new_pnl])
        log.info(
            "Portfolio Sharpe: %.3f → %.3f (adding %s)",
            sharpe_before, sharpe_after, new_strategy_name,
        )
        return True, f"portfolio Sharpe {sharpe_before:.3f} → {sharpe_after:.3f}"

    return True, "all correlation checks passed"
