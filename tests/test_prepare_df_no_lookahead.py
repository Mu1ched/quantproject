"""
Regression test for review#5 — train/test feature lookahead in prepare_df.

prepare_df() is called on the full series before the train/test split happens
in load_all_data. The audit comment at edge_engine.py:2381 claims every
rolling feature uses closed='right' past-only windows and every .shift() is
non-negative (i.e., past-only).

This test enforces that claim directly. We build a synthetic 1-minute series,
inject a sharp discontinuity at a chosen boundary B (price jumps, spread
spikes, volatility surges from bar B onward), and run prepare_df twice:

  (a) on the full series [0, N)
  (b) on the truncated series [0, B)

If any feature at row i < B in (a) depends on data at row j >= B (lookahead),
its value will differ from the same row in (b). The test asserts every shared
numeric column at every row < B is bit-equal (allowing NaN-NaN match and
float tolerance) between the two runs.

Note: the synthetic timestamps are placed entirely inside the live test
window (last TEST_DAYS days), which makes _fit_and_apply_hmm skip on
"insufficient training data" in both runs — so HMM is excluded from the
comparison. That's deliberate: HMM has its own explicit train/test split
guard at edge_engine.py:2316-2380 (review#5 first pass) and isn't part of
the rolling-feature audit this test enforces.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import edge_engine as eng


N_BARS   = 2000
BOUNDARY = 1500
PAIR     = 'EUR_USD'


def _make_synthetic_m1(n_bars: int, boundary: int) -> pd.DataFrame:
    """Build a deterministic synthetic 1-minute frame with a known
    discontinuity at `boundary`. From bar `boundary` onward, close prices
    jump +5% and spreads/realized vol spike 10x — large enough that any
    feature reaching across the boundary will leave a numeric fingerprint
    on rows < boundary."""
    rng = np.random.default_rng(42)

    end_ts   = datetime.now(timezone.utc) - timedelta(hours=2)
    start_ts = end_ts - timedelta(minutes=n_bars - 1)
    timestamps = pd.date_range(start_ts, end_ts, periods=n_bars, tz='UTC')

    close = 1.1000 + np.cumsum(rng.normal(0, 0.00005, n_bars))
    close[boundary:] += 0.055
    high = close + rng.uniform(0.00005, 0.0002, n_bars)
    low  = close - rng.uniform(0.00005, 0.0002, n_bars)
    open_ = close - rng.normal(0, 0.00003, n_bars)

    spread_mean = np.full(n_bars, 0.00012)
    spread_mean[boundary:] = 0.0012

    realized_vol = np.full(n_bars, 0.0001)
    realized_vol[boundary:] = 0.001

    df = pd.DataFrame({
        'timestamp':    timestamps,
        'open':         open_,
        'high':         high,
        'low':          low,
        'close':        close,
        'spread_mean':  spread_mean,
        'spread_max':   spread_mean * 1.5,
        'realized_vol': realized_vol,
        'volume':       np.full(n_bars, 100.0),
        'tick_count':   np.full(n_bars, 50, dtype=int),
    })
    return df


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


@pytest.fixture(scope='module')
def prepared_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run prepare_df on the full and truncated series. Suppress its
    informational print so test output stays clean."""
    m1 = _make_synthetic_m1(N_BARS, BOUNDARY)

    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    try:
        sys.stdout = devnull
        full  = eng.prepare_df(m1.copy(),                PAIR)
        trunc = eng.prepare_df(m1.iloc[:BOUNDARY].copy(), PAIR)
    finally:
        sys.stdout = old_stdout
        devnull.close()
    return full, trunc


def test_prepare_df_no_cross_boundary_leak(prepared_pair):
    """For every numeric column present in both runs, the values at rows
    [0, BOUNDARY) must be bit-equal (within float tolerance) regardless of
    whether the post-boundary rows existed when prepare_df was called.

    A leak in any rolling/window/groupby/shift operation that reaches
    forward will cause some row < BOUNDARY in the full-series run to be
    numerically different from the truncated-series run."""
    full, trunc = prepared_pair

    assert len(full)  >= BOUNDARY, "full run produced fewer rows than expected"
    assert len(trunc) == BOUNDARY, "truncated run row count must match boundary"

    full_pre = full.iloc[:BOUNDARY]

    full_cols  = set(_numeric_columns(full_pre))
    trunc_cols = set(_numeric_columns(trunc))
    shared     = sorted(full_cols & trunc_cols)
    assert shared, "no shared numeric columns between runs — fixture broken"

    leaks: list[tuple[str, int, float, float]] = []
    for col in shared:
        a = full_pre[col].to_numpy()
        b = trunc[col].to_numpy()
        if a.shape != b.shape:
            leaks.append((col, -1, float('nan'), float('nan')))
            continue
        finite = np.isfinite(a) | np.isfinite(b)
        diff   = np.where(
            finite,
            ~np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True),
            False,
        )
        if diff.any():
            first = int(np.argmax(diff))
            leaks.append((col, first, float(a[first]), float(b[first])))

    if leaks:
        lines = [f"  {col} at row {idx}: full={fv!r}  trunc={tv!r}"
                 for col, idx, fv, tv in leaks[:20]]
        pytest.fail(
            "review#5 violation — prepare_df features at rows < BOUNDARY "
            "depend on rows >= BOUNDARY (forward leakage detected). "
            f"{len(leaks)} affected column(s):\n" + "\n".join(lines)
        )
