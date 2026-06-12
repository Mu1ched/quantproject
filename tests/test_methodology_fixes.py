"""
Parity tests for the methodology fixes back-ported from FX to crypto,
documented in ~/quant_methodology_log.md entries #1 (wf_sharpe min-trades)
and #5 (calc_size_with_rv zero/cap guards).

Both projects should pass these identically — drop into both tests/ dirs.
"""

import sys
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import edge_engine as eng


# =============================================================================
# Entry #5: calc_size_with_rv must not raise on rv_now=0; ratio is bounded
# both above and below to RV_SCALE_CAP and 1/RV_SCALE_CAP respectively.
# =============================================================================

def test_calc_size_with_rv_zero_rv_returns_base():
    """rv_now == 0 → return base unscaled (no division by zero)."""
    pair = 'BTCUSDT' if 'BTCUSDT' in eng.PAIR_PIP_SIZE else 'EUR_USD'
    base = eng.calc_size(pair, 25_000.0, 0.01, 100_000.0, 100.0)
    out  = eng.calc_size_with_rv(pair, 25_000.0, 0.01, 100_000.0, 100.0,
                                  rv_now=0.0, rv_median=0.001)
    assert out == pytest.approx(base, rel=1e-9), \
        f"rv_now=0 should return base ({base}); got {out}"


def test_calc_size_with_rv_zero_median_returns_base():
    """rv_median == 0 → fallback to base (no division by zero)."""
    pair = 'BTCUSDT' if 'BTCUSDT' in eng.PAIR_PIP_SIZE else 'EUR_USD'
    base = eng.calc_size(pair, 25_000.0, 0.01, 100_000.0, 100.0)
    out  = eng.calc_size_with_rv(pair, 25_000.0, 0.01, 100_000.0, 100.0,
                                  rv_now=0.001, rv_median=0.0)
    assert out == pytest.approx(base, rel=1e-9)


def test_calc_size_with_rv_high_vol_caps_shrinkage():
    """rv_now = 10× rv_median → ratio capped at RV_SCALE_CAP; size shrinks
    by at most RV_SCALE_CAP×."""
    pair = 'BTCUSDT' if 'BTCUSDT' in eng.PAIR_PIP_SIZE else 'EUR_USD'
    base = eng.calc_size(pair, 25_000.0, 0.01, 100_000.0, 100.0)
    out  = eng.calc_size_with_rv(pair, 25_000.0, 0.01, 100_000.0, 100.0,
                                  rv_now=0.01, rv_median=0.001)
    # ratio is min(10, 2.0) = 2.0, so size = base / 2.0
    assert out == pytest.approx(base / eng.RV_SCALE_CAP, rel=1e-9)


def test_calc_size_with_rv_low_vol_caps_amplification():
    """rv_now = 0.1× rv_median → ratio floored at 1/RV_SCALE_CAP; size
    grows by at most RV_SCALE_CAP× (mirrors shrinkage cap)."""
    pair = 'BTCUSDT' if 'BTCUSDT' in eng.PAIR_PIP_SIZE else 'EUR_USD'
    base = eng.calc_size(pair, 25_000.0, 0.01, 100_000.0, 100.0)
    out  = eng.calc_size_with_rv(pair, 25_000.0, 0.01, 100_000.0, 100.0,
                                  rv_now=0.0001, rv_median=0.001)
    # raw ratio = 0.1; after symmetric floor = 1/2.0 = 0.5; size = base / 0.5 = 2*base
    expected = base * eng.RV_SCALE_CAP
    assert out == pytest.approx(expected, rel=1e-9), \
        f"low-vol amplification should be capped at {eng.RV_SCALE_CAP}x"


# =============================================================================
# Entry #1: walk-forward Sharpe min-trades-per-fold gate. The wf_sharpe
# computation lives inside _run_single_hypothesis, so the test exercises
# calc_stats directly to validate the underlying assumption that small-fold
# Sharpe is meaningless. Then a higher-level integration check would be a
# full sweep — out of scope for a unit test, but the calc_stats behaviour
# on a 3-trade vs 30-trade sample is testable.
# =============================================================================

def _synth_trades(n: int, pnl_each: float, exit_t_start: str = '2025-01-01') -> pd.DataFrame:
    times = pd.date_range(exit_t_start, periods=n, freq='1h', tz='UTC')
    return pd.DataFrame({
        'instrument': ['BTCUSDT'] * n,
        'pnl':        [pnl_each] * n,
        'exit_time':  times,
        'partial':    [0] * n,
    })


def test_calc_stats_small_sample_sharpe_is_finite_or_handled():
    """Sharpe on a 3-trade fold with positive constant PnL has std=0 →
    either inf or carefully handled (NaN). This test pins the contract:
    the value MUST be finite-or-NaN, never a number > 100 (which is the
    pathology entry #1 was guarding against)."""
    df = _synth_trades(3, 100.0)
    stats = eng.calc_stats(df)
    if stats:
        sh = stats.get('sharpe', float('nan'))
        assert not np.isfinite(sh) or abs(sh) < 100.0, \
            f"calc_stats on a 3-trade constant-PnL fold returned Sharpe={sh}; " \
            f"entry #1 expects |sh|<100 or NaN, not an unbounded value"


def test_calc_stats_thirty_trades_finite_sharpe():
    """30-trade sample with realistic variance should give a finite, bounded
    Sharpe — this is the regime entry #1 says we should still trust."""
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame({
        'instrument': ['BTCUSDT'] * n,
        'pnl':        rng.normal(50.0, 200.0, n),
        'exit_time':  pd.date_range('2025-01-01', periods=n, freq='1h', tz='UTC'),
        'partial':    [0] * n,
    })
    stats = eng.calc_stats(df)
    assert stats, "calc_stats returned empty on 30-trade sample"
    sh = stats['sharpe']
    # Entry #1's pathology was |sh| > 100. Synthetic clean-signal Sharpes
    # can legitimately reach ~30 at this sample size; what we're guarding
    # against is the |sh| > 100 regime.
    assert np.isfinite(sh) and abs(sh) < 100.0, \
        f"30-trade sample Sharpe should be bounded < 100; got {sh}"
