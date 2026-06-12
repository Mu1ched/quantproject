"""
Unit tests for edge_engine.py core functions.

Run with:
    pytest tests/ -v
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import edge_engine as eng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_row(**kwargs):
    """Return a simple namespace object that mimics a DataFrame row."""
    defaults = {
        'spread_mean':   0.0002,
        'spread_median': 0.0001,
        'realized_vol':  0.0005,
        'rv_median':     0.0005,
        'high':          1.2600,
        'low':           1.2500,
        'close':         1.2550,
        'open':          1.2520,
        'hour':          10,
        'adx':           30.0,
        'atr_ratio':     1.0,
    }
    defaults.update(kwargs)
    Row = namedtuple('Row', defaults.keys())
    return Row(**defaults)


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_volatile(self):
        # atr_ratio above threshold → VOLATILE regardless of adx
        result = eng.classify_regime(adx_val=50.0, atr_ratio_val=999.0)
        assert result == 'VOLATILE'

    def test_trending(self):
        # High ADX, normal vol → TRENDING
        result = eng.classify_regime(adx_val=35.0, atr_ratio_val=1.0)
        assert result == 'TRENDING'

    def test_ranging(self):
        # Low ADX, normal vol → RANGING
        result = eng.classify_regime(adx_val=10.0, atr_ratio_val=1.0)
        assert result == 'RANGING'

    def test_transitioning(self):
        # Mid ADX, normal vol → TRANSITIONING
        result = eng.classify_regime(adx_val=22.0, atr_ratio_val=1.0)
        assert result == 'TRANSITIONING'

    def test_nan_adx(self):
        result = eng.classify_regime(adx_val=float('nan'), atr_ratio_val=1.0)
        assert result == 'UNDEFINED'

    def test_nan_atr(self):
        result = eng.classify_regime(adx_val=30.0, atr_ratio_val=float('nan'))
        assert result == 'UNDEFINED'

    def test_both_nan(self):
        result = eng.classify_regime(adx_val=float('nan'), atr_ratio_val=float('nan'))
        assert result == 'UNDEFINED'


# ---------------------------------------------------------------------------
# spread_gate
# ---------------------------------------------------------------------------

class TestSpreadGate:
    def test_normal_spread_passes(self):
        # spread_mean <= mult * spread_median → gate open (False = do not skip)
        row = make_row(spread_mean=0.0001, spread_median=0.0001)
        assert eng.spread_gate(row) is False

    def test_elevated_spread_blocked(self):
        # spread_mean > 2× spread_median → gate closed (True = skip entry)
        row = make_row(spread_mean=0.0003, spread_median=0.0001)
        assert eng.spread_gate(row) is True

    def test_nan_spread_mean_passes(self):
        row = make_row(spread_mean=float('nan'), spread_median=0.0001)
        assert eng.spread_gate(row) is False

    def test_nan_spread_median_passes(self):
        row = make_row(spread_mean=0.0001, spread_median=float('nan'))
        assert eng.spread_gate(row) is False

    def test_zero_spread_median_passes(self):
        # Division by zero guard — should not raise
        row = make_row(spread_mean=0.0001, spread_median=0.0)
        assert eng.spread_gate(row) is False

    def test_exactly_at_threshold_passes(self):
        # Exactly at 2× median should still pass (not strictly greater)
        row = make_row(spread_mean=0.0002, spread_median=0.0001)
        assert eng.spread_gate(row) is False

    def test_custom_multiplier(self):
        # With mult=1.0 even a slightly elevated spread should be blocked
        row = make_row(spread_mean=0.00015, spread_median=0.0001)
        assert eng.spread_gate(row, mult=1.0) is True


# ---------------------------------------------------------------------------
# rv_size  (position sizing)
# ---------------------------------------------------------------------------

class TestRvSize:
    """rv_size scales down when realized vol is elevated vs. its rolling median."""

    def test_normal_vol_returns_positive_size(self):
        row = make_row(realized_vol=0.0005, rv_median=0.0005)
        size = eng.rv_size('GBPUSD', balance=10_000, risk=0.01,
                           entry=1.2500, dist=0.0020, row=row)
        assert size > 0

    def test_elevated_vol_smaller_than_normal(self):
        row_normal = make_row(realized_vol=0.0005, rv_median=0.0005)
        row_high   = make_row(realized_vol=0.0015, rv_median=0.0005)
        size_normal = eng.rv_size('GBPUSD', 10_000, 0.01, 1.25, 0.002, row=row_normal)
        size_high   = eng.rv_size('GBPUSD', 10_000, 0.01, 1.25, 0.002, row=row_high)
        assert size_high < size_normal

    def test_nan_vol_still_returns_size(self):
        # If vol data is missing, function should not crash and should return something
        row = make_row(realized_vol=float('nan'), rv_median=float('nan'))
        size = eng.rv_size('GBPUSD', 10_000, 0.01, 1.25, 0.002, row=row)
        assert size >= 0

    def test_higher_balance_scales_size(self):
        row = make_row(realized_vol=0.0005, rv_median=0.0005)
        size_small = eng.rv_size('GBPUSD', 5_000, 0.01, 1.25, 0.002, row=row)
        size_large = eng.rv_size('GBPUSD', 50_000, 0.01, 1.25, 0.002, row=row)
        assert size_large > size_small

    def test_wider_stop_reduces_size(self):
        row = make_row(realized_vol=0.0005, rv_median=0.0005)
        size_tight = eng.rv_size('GBPUSD', 10_000, 0.01, 1.25, 0.001, row=row)
        size_wide  = eng.rv_size('GBPUSD', 10_000, 0.01, 1.25, 0.010, row=row)
        assert size_wide < size_tight


# ---------------------------------------------------------------------------
# check_and_fill
# ---------------------------------------------------------------------------

class TestCheckAndFill:
    def _make_slot(self):
        return {
            'strategy_def': {'params': {}},
            'scratch': {},
            'pos': None,
            'position': None,
        }

    def _full_slot(self):
        slot = self._make_slot()
        slot.update({
            'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0,
            'pos_size': 0.0, 'partial_size': 0.0, 'remainder_size': 0.0,
            'sl_ref_dist': 0.0, 'entry_time': None, 'opened_today': False,
            'session_exited': False, 'breakeven_set': False,
            'profit_lock_set': False, 'partial_tp_done': False,
            'partial_pnl': 0.0, 'regime': 'UNDEFINED',
        })
        return slot

    def _ts(self, when='2024-01-02 10:00'):
        return pd.Timestamp(when, tz='UTC')

    def test_no_pending_returns_false(self):
        sc = {}
        row = make_row()
        result = eng.check_and_fill(sc, row, self._full_slot(),
                                    ts=self._ts(), regime='TRENDING')
        assert result is False

    def test_long_triggered_when_high_reaches_level(self):
        ts_placed = self._ts('2024-01-02 10:00')
        ts_fill   = self._ts('2024-01-02 10:01')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts_placed, 'long',
                          entry=1.2550, sl=1.2480, tp=1.2650,
                          size=0.01, dist=0.0070, level=1.2550)
        row = make_row(high=1.2560, low=1.2510, open=1.2540, close=1.2555)
        result = eng.check_and_fill(sc, row, slot, ts=ts_fill, regime='TRENDING')
        assert result is True
        assert slot['position'] == 'long'

    def test_long_not_triggered_when_high_below_level(self):
        ts_placed = self._ts('2024-01-02 10:00')
        ts_fill   = self._ts('2024-01-02 10:01')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts_placed, 'long',
                          entry=1.2600, sl=1.2480, tp=1.2700,
                          size=0.01, dist=0.0120, level=1.2600)
        row = make_row(high=1.2580, low=1.2510, open=1.2540, close=1.2560)
        result = eng.check_and_fill(sc, row, slot, ts=ts_fill, regime='TRENDING')
        assert result is False
        assert slot['position'] is None

    def test_short_triggered_when_low_reaches_level(self):
        ts_placed = self._ts('2024-01-02 10:00')
        ts_fill   = self._ts('2024-01-02 10:01')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts_placed, 'short',
                          entry=1.2500, sl=1.2570, tp=1.2400,
                          size=0.01, dist=0.0070, level=1.2500)
        row = make_row(high=1.2530, low=1.2490, open=1.2520, close=1.2505)
        result = eng.check_and_fill(sc, row, slot, ts=ts_fill, regime='TRENDING')
        assert result is True
        assert slot['position'] == 'short'


class TestNoSameBarFills:
    """Regression test for the look-ahead bias fix. A pending order placed on
    bar t MUST NOT fill on bar t — it can only fill on bar t+1 or later."""

    def _full_slot(self):
        return {
            'strategy_def': {'params': {}},
            'scratch': {},
            'position': None,
            'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0,
            'pos_size': 0.0, 'partial_size': 0.0, 'remainder_size': 0.0,
            'sl_ref_dist': 0.0, 'entry_time': None, 'opened_today': False,
            'session_exited': False, 'breakeven_set': False,
            'profit_lock_set': False, 'partial_tp_done': False,
            'partial_pnl': 0.0, 'regime': 'UNDEFINED',
        }

    def test_same_bar_long_never_fills(self):
        ts = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        # Stage a pending and immediately try to fill on the SAME bar.
        eng.place_pending(sc, ts, 'long', entry=1.2550, sl=1.2480,
                          tp=1.2650, size=0.01, dist=0.0070, level=1.2550)
        # Bar's high is well above the level — would have filled under old logic.
        row = make_row(high=1.2600, low=1.2510, open=1.2540, close=1.2580)
        result = eng.check_and_fill(sc, row, slot, ts=ts, regime='TRENDING')
        assert result is False
        assert slot['position'] is None
        assert sc.get('pending_placed_ts') == ts

    def test_same_bar_short_never_fills(self):
        ts = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts, 'short', entry=1.2500, sl=1.2570,
                          tp=1.2400, size=0.01, dist=0.0070, level=1.2500)
        row = make_row(high=1.2530, low=1.2470, open=1.2520, close=1.2490)
        result = eng.check_and_fill(sc, row, slot, ts=ts, regime='TRENDING')
        assert result is False

    def test_next_bar_long_fills_at_level(self):
        ts0 = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        ts1 = pd.Timestamp('2024-01-02 10:01', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts0, 'long', entry=1.2550, sl=1.2480,
                          tp=1.2650, size=0.01, dist=0.0070, level=1.2550)
        row = make_row(open=1.2540, high=1.2560, low=1.2530, close=1.2555)
        result = eng.check_and_fill(sc, row, slot, ts=ts1, regime='TRENDING',
                                    hspd=0.00005, slip=0.00002)
        assert result is True
        # Filled at level + hspd + slip = 1.2550 + 0.00005 + 0.00002
        assert abs(slot['entry_price'] - (1.2550 + 0.00005 + 0.00002)) < 1e-9

    def test_gap_through_long_fills_at_open(self):
        """If bar opens past the level, fill at adverse open price."""
        ts0 = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        ts1 = pd.Timestamp('2024-01-02 10:01', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts0, 'long', entry=1.2550, sl=1.2480,
                          tp=1.2650, size=0.01, dist=0.0070, level=1.2550)
        # Bar opens at 1.2570 — already past level. Fill at open + slip.
        row = make_row(open=1.2570, high=1.2580, low=1.2565, close=1.2575)
        result = eng.check_and_fill(sc, row, slot, ts=ts1, regime='TRENDING',
                                    hspd=0.00005, slip=0.00002)
        assert result is True
        assert abs(slot['entry_price'] - (1.2570 + 0.00005 + 0.00002)) < 1e-9

    def test_pending_persists_until_triggered(self):
        ts0 = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        ts_no_trigger = pd.Timestamp('2024-01-02 10:05', tz='UTC')
        ts_trigger    = pd.Timestamp('2024-01-02 10:10', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts0, 'long', entry=1.2550, sl=1.2480,
                          tp=1.2650, size=0.01, dist=0.0070, level=1.2550)
        # 5 minutes later, price hasn't reached the level.
        row1 = make_row(open=1.2530, high=1.2540, low=1.2525, close=1.2535)
        assert eng.check_and_fill(sc, row1, slot, ts=ts_no_trigger,
                                  regime='TRENDING') is False
        assert slot['position'] is None
        assert sc.get('pending_dir') == 'long'  # still staged
        # 10 minutes after placement, level is touched.
        row2 = make_row(open=1.2545, high=1.2560, low=1.2543, close=1.2555)
        assert eng.check_and_fill(sc, row2, slot, ts=ts_trigger,
                                  regime='TRENDING') is True
        assert slot['position'] == 'long'


class TestMarketNextOpen:
    """Confirmation-style entries fill at the next bar's open ± slip,
    independent of any level."""

    def _full_slot(self):
        return {
            'strategy_def': {'params': {}},
            'scratch': {},
            'position': None,
            'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0,
            'pos_size': 0.0, 'partial_size': 0.0, 'remainder_size': 0.0,
            'sl_ref_dist': 0.0, 'entry_time': None, 'opened_today': False,
            'session_exited': False, 'breakeven_set': False,
            'profit_lock_set': False, 'partial_tp_done': False,
            'partial_pnl': 0.0, 'regime': 'UNDEFINED',
        }

    def test_market_next_open_long_fills_at_open(self):
        ts0 = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        ts1 = pd.Timestamp('2024-01-02 10:01', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts0, 'long',
                          entry=1.2540, sl=1.2470, tp=1.2640,
                          size=0.01, dist=0.0070,
                          mode='market_next_open')
        row = make_row(open=1.2545, high=1.2560, low=1.2540, close=1.2555)
        ok = eng.check_and_fill(sc, row, slot, ts=ts1, regime='TRENDING',
                                hspd=0.00005, slip=0.00003)
        assert ok is True
        # Long market fill: bar.open + hspd + slip
        assert abs(slot['entry_price'] - (1.2545 + 0.00005 + 0.00003)) < 1e-9

    def test_market_next_open_short_fills_at_open(self):
        ts0 = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        ts1 = pd.Timestamp('2024-01-02 10:01', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts0, 'short',
                          entry=1.2540, sl=1.2610, tp=1.2440,
                          size=0.01, dist=0.0070,
                          mode='market_next_open')
        row = make_row(open=1.2535, high=1.2545, low=1.2520, close=1.2525)
        ok = eng.check_and_fill(sc, row, slot, ts=ts1, regime='TRENDING',
                                hspd=0.00005, slip=0.00003)
        assert ok is True
        # Short market fill: bar.open - hspd - slip
        assert abs(slot['entry_price'] - (1.2535 - 0.00005 - 0.00003)) < 1e-9

    def test_market_next_open_does_not_fill_same_bar(self):
        ts0 = pd.Timestamp('2024-01-02 10:00', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_pending(sc, ts0, 'long',
                          entry=1.2540, sl=1.2470, tp=1.2640,
                          size=0.01, dist=0.0070,
                          mode='market_next_open')
        row = make_row(open=1.2545, high=1.2560, low=1.2540, close=1.2555)
        ok = eng.check_and_fill(sc, row, slot, ts=ts0, regime='TRENDING')
        assert ok is False
        assert slot['position'] is None


class TestOcoBracket:
    """Pre-stage breakout brackets — both legs persist until one triggers."""

    def _full_slot(self):
        return {
            'strategy_def': {'params': {}},
            'scratch': {},
            'position': None,
            'entry_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0,
            'pos_size': 0.0, 'partial_size': 0.0, 'remainder_size': 0.0,
            'sl_ref_dist': 0.0, 'entry_time': None, 'opened_today': False,
            'session_exited': False, 'breakeven_set': False,
            'profit_lock_set': False, 'partial_tp_done': False,
            'partial_pnl': 0.0, 'regime': 'UNDEFINED',
        }

    def test_long_leg_triggers_first(self):
        ts0 = pd.Timestamp('2024-01-02 15:00', tz='UTC')
        ts1 = pd.Timestamp('2024-01-02 15:01', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_oco_pending(sc, ts0,
                              long_level=1.2600, long_sl=1.2580,
                              long_tp=1.2640, long_size=0.01, long_dist=0.0020,
                              short_level=1.2540, short_sl=1.2560,
                              short_tp=1.2500, short_size=0.01, short_dist=0.0020)
        # Bar pushes through long level.
        row = make_row(open=1.2580, high=1.2610, low=1.2575, close=1.2605)
        ok = eng.check_and_fill(sc, row, slot, ts=ts1, regime='TRENDING',
                                hspd=0.00005, slip=0.00002)
        assert ok is True
        assert slot['position'] == 'long'
        # Both pending legs were cleared.
        assert 'pending_long' not in sc
        assert 'pending_short' not in sc

    def test_short_leg_triggers_first(self):
        ts0 = pd.Timestamp('2024-01-02 15:00', tz='UTC')
        ts1 = pd.Timestamp('2024-01-02 15:01', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_oco_pending(sc, ts0,
                              long_level=1.2600, long_sl=1.2580,
                              long_tp=1.2640, long_size=0.01, long_dist=0.0020,
                              short_level=1.2540, short_sl=1.2560,
                              short_tp=1.2500, short_size=0.01, short_dist=0.0020)
        row = make_row(open=1.2570, high=1.2575, low=1.2535, close=1.2538)
        ok = eng.check_and_fill(sc, row, slot, ts=ts1, regime='TRENDING',
                                hspd=0.00005, slip=0.00002)
        assert ok is True
        assert slot['position'] == 'short'

    def test_oco_does_not_fill_same_bar(self):
        ts0 = pd.Timestamp('2024-01-02 15:00', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_oco_pending(sc, ts0,
                              long_level=1.2600, long_sl=1.2580,
                              long_tp=1.2640, long_size=0.01, long_dist=0.0020,
                              short_level=1.2540, short_sl=1.2560,
                              short_tp=1.2500, short_size=0.01, short_dist=0.0020)
        row = make_row(open=1.2570, high=1.2615, low=1.2535, close=1.2580)
        # Same bar as placement — must not fill even though both legs were touched.
        ok = eng.check_and_fill(sc, row, slot, ts=ts0, regime='TRENDING')
        assert ok is False
        assert slot['position'] is None

    def test_neither_leg_triggers_persists(self):
        ts0 = pd.Timestamp('2024-01-02 15:00', tz='UTC')
        ts1 = pd.Timestamp('2024-01-02 15:01', tz='UTC')
        slot = self._full_slot()
        sc = slot['scratch']
        eng.place_oco_pending(sc, ts0,
                              long_level=1.2600, long_sl=1.2580,
                              long_tp=1.2640, long_size=0.01, long_dist=0.0020,
                              short_level=1.2540, short_sl=1.2560,
                              short_tp=1.2500, short_size=0.01, short_dist=0.0020)
        # Bar contained — neither leg touched.
        row = make_row(open=1.2570, high=1.2585, low=1.2555, close=1.2575)
        ok = eng.check_and_fill(sc, row, slot, ts=ts1, regime='TRENDING')
        assert ok is False
        assert slot['position'] is None
        assert 'pending_long' in sc
        assert 'pending_short' in sc


# ---------------------------------------------------------------------------
# ticks_to_m1
# ---------------------------------------------------------------------------

class TestTicksToM1:
    def _make_tick_df(self, n=200):
        """Generate synthetic tick data."""
        times = pd.date_range('2024-01-02 09:00', periods=n, freq='3s', tz='UTC')
        price = 1.2500 + np.cumsum(np.random.randn(n) * 0.00005)
        return pd.DataFrame({
            'timestamp': times,
            'ask': price + 0.00010,
            'bid': price,
            'ask_vol': np.random.randint(1, 10, n).astype(float),
            'bid_vol': np.random.randint(1, 10, n).astype(float),
        })

    def test_returns_dataframe(self):
        ticks = self._make_tick_df()
        result = eng.ticks_to_m1(ticks)
        assert isinstance(result, pd.DataFrame)

    def test_has_ohlc_columns(self):
        ticks = self._make_tick_df()
        result = eng.ticks_to_m1(ticks)
        for col in ['open', 'high', 'low', 'close']:
            assert col in result.columns, f"Missing column: {col}"

    def test_high_gte_low(self):
        ticks = self._make_tick_df(500)
        result = eng.ticks_to_m1(ticks)
        assert (result['high'] >= result['low']).all()

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=['timestamp', 'ask', 'bid', 'ask_vol', 'bid_vol'])
        result = eng.ticks_to_m1(empty)
        assert result.empty

    def test_realized_vol_non_negative(self):
        ticks = self._make_tick_df(500)
        result = eng.ticks_to_m1(ticks)
        if 'realized_vol' in result.columns:
            assert (result['realized_vol'] >= 0).all()
