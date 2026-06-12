"""Metric correctness tests on canonical inputs (review#P2#5).

Covers:
  • deflated_sharpe_ratio — monotonicity in n_trials, sanity bounds
  • probabilistic_sharpe_ratio — same
  • _stationary_bootstrap_indices — return shape + value range
  • apply_bh_correction — family-wide behavior on a controlled DB
"""
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

import edge_engine as eng


# ── DSR ──────────────────────────────────────────────────────────────────────


def test_dsr_falls_with_more_trials():
    """More trials → more selection bias → lower DSR for the same Sharpe."""
    s = 1.5
    n_obs = 200
    dsr_few   = eng.deflated_sharpe_ratio(s, n_trials=5,   n_obs=n_obs)
    dsr_many  = eng.deflated_sharpe_ratio(s, n_trials=500, n_obs=n_obs)
    assert dsr_many < dsr_few, (dsr_few, dsr_many)


def test_dsr_in_zero_one_when_finite():
    s = 1.5
    val = eng.deflated_sharpe_ratio(s, n_trials=50, n_obs=200)
    assert 0.0 <= val <= 1.0


# ── PSR ──────────────────────────────────────────────────────────────────────


def test_psr_increases_with_n_obs():
    """Holding skew/kurt fixed, more observations → higher PSR confidence.
    Use a borderline Sharpe (0.3) so the cdf isn't saturated at 1.0."""
    s = 0.3
    psr_short = eng.probabilistic_sharpe_ratio(s, n_obs=30,  skew=0.0, kurt=3.0)
    psr_long  = eng.probabilistic_sharpe_ratio(s, n_obs=300, skew=0.0, kurt=3.0)
    assert psr_long > psr_short, (psr_short, psr_long)


# ── Stationary bootstrap indices ─────────────────────────────────────────────


def test_stationary_bootstrap_indices_shape():
    rng = np.random.default_rng(0)
    idx = eng._stationary_bootstrap_indices(n=100, block_len=10, rng=rng)
    assert idx.shape == (100,)
    assert idx.min() >= 0
    assert idx.max() < 100


def test_stationary_bootstrap_preserves_serial_correlation():
    """Block bootstrap preserves more autocorrelation than IID. With block_len=20
    and a strongly autocorrelated AR(1) series, the bootstrap variance of the
    sample mean should be markedly higher than under IID."""
    rng = np.random.default_rng(42)
    # AR(1) with phi=0.8: strong positive autocorrelation
    n = 500
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i-1] + eps[i]

    means_block = []
    means_iid   = []
    for _ in range(200):
        block_idx = eng._stationary_bootstrap_indices(n, 20, rng)
        means_block.append(np.mean(x[block_idx]))
        means_iid.append(np.mean(rng.choice(x, size=n, replace=True)))

    var_block = float(np.var(means_block))
    var_iid   = float(np.var(means_iid))
    # Block bootstrap variance should be at least 2× IID variance with phi=0.8.
    assert var_block > var_iid * 1.5, (var_block, var_iid)


# ── apply_bh_correction (family-wide, review#6) ──────────────────────────────


def test_apply_bh_correction_family_wide(monkeypatch, tmp_path):
    """Inserting one significant p-value alongside many uniformly-distributed
    null p-values should still produce a measurable BH gradient — i.e. the
    significant one's p_adj should be the smallest in the family."""
    # Redirect the engine's DB_PATH to a fresh temp file
    db_path = tmp_path / "test_edge.db"
    monkeypatch.setattr(eng, 'DB_PATH', str(db_path))

    eng._init_db()
    con = sqlite3.connect(str(db_path))
    rng = np.random.default_rng(0)
    rows = [(f"hyp_{i:04d}", 'sweep_test', '{}', float(rng.uniform(0.4, 1.0)))
            for i in range(99)]
    rows.append(('hyp_signal', 'sweep_test', '{}', 0.0001))
    con.executemany(
        "INSERT INTO hypotheses (hypothesis_id, sweep_id, params_json, p_raw) "
        "VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()

    eng.apply_bh_correction('sweep_test')

    con = sqlite3.connect(str(db_path))
    p_adj_rows = con.execute(
        "SELECT hypothesis_id, p_adj FROM hypotheses WHERE p_adj IS NOT NULL"
    ).fetchall()
    con.close()

    assert len(p_adj_rows) == 100  # family-wide: all populated
    by_id = dict(p_adj_rows)
    # The signal row's p_adj must be the smallest.
    assert by_id['hyp_signal'] == min(by_id.values())
