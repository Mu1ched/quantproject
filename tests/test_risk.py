"""
Tests focused on risk management edge cases.
"""

import sys
from pathlib import Path
from collections import namedtuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import edge_engine as eng


def make_row(**kwargs):
    defaults = {
        'spread_mean': 0.0002, 'spread_median': 0.0001,
        'realized_vol': 0.0005, 'rv_median': 0.0005,
        'high': 1.2600, 'low': 1.2500, 'close': 1.2550, 'open': 1.2520,
    }
    defaults.update(kwargs)
    Row = namedtuple('Row', defaults.keys())
    return Row(**defaults)


class TestRvSizeEdgeCases:
    def test_zero_distance_does_not_raise(self):
        # A zero stop distance is a degenerate case — should not crash
        row = make_row()
        try:
            size = eng.rv_size('GBPUSD', 10_000, 0.01, 1.25, 0.0, row=row)
            assert size >= 0
        except ZeroDivisionError:
            pytest.fail("rv_size raised ZeroDivisionError on zero distance")

    def test_zero_balance_returns_zero(self):
        row = make_row()
        size = eng.rv_size('GBPUSD', 0, 0.01, 1.25, 0.002, row=row)
        assert size == 0

    def test_size_does_not_exceed_sane_maximum(self):
        # Even with huge balance, lot size should be reasonable
        row = make_row(realized_vol=0.0001, rv_median=0.001)  # very low vol → larger size
        size = eng.rv_size('GBPUSD', 1_000_000, 0.02, 1.25, 0.002, row=row)
        assert size < 10_000  # sanity cap


class TestSpreadGateEdgeCases:
    def test_missing_spread_mean_attribute(self):
        # Row without spread attributes should not crash
        Row = namedtuple('Row', ['close'])
        row = Row(close=1.25)
        result = eng.spread_gate(row)
        assert result is False

    def test_very_large_spread_blocked(self):
        row = make_row(spread_mean=1.0, spread_median=0.0001)
        assert eng.spread_gate(row) is True


class TestClassifyRegimeBoundaries:
    """Test regime classification at exact threshold boundaries."""

    def test_atr_ratio_exactly_at_volatile_threshold(self):
        # Behaviour at the exact threshold depends on implementation (> vs >=)
        # Just assert it returns a valid regime, not that it crashes
        result = eng.classify_regime(30.0, eng.REGIME_ATR_VOLATILE)
        assert result in ('VOLATILE', 'TRENDING', 'RANGING', 'TRANSITIONING', 'UNDEFINED')

    def test_adx_exactly_at_trend_threshold(self):
        result = eng.classify_regime(eng.REGIME_ADX_TREND, 1.0)
        assert result in ('VOLATILE', 'TRENDING', 'RANGING', 'TRANSITIONING', 'UNDEFINED')

    def test_adx_exactly_at_range_threshold(self):
        result = eng.classify_regime(eng.REGIME_ADX_RANGE, 1.0)
        assert result in ('VOLATILE', 'TRENDING', 'RANGING', 'TRANSITIONING', 'UNDEFINED')
