"""Index sizing test for MT5Live.compute_units' index branch (US500/NAS100/US30).

We can't import MT5Live.2.py in tests (MetaTrader5 is Windows-terminal only), so —
exactly like test_sizing_parity.py — we mirror the live index formula verbatim and
assert it risks the intended fraction of balance. If MT5Live.compute_units' index
branch diverges from this mirror, that's a bug to catch at review.

Mirror of MT5Live.2.py compute_units index branch:
    value_per_point = trade_tick_value / trade_tick_size
    lots = (balance * risk_pct) / (dist * value_per_point)
    units = int(lots * 100_000)            # units_to_lots(/100_000) recovers lots
"""
import pytest


def _live_index_units(balance, entry, sl, risk_pct, tick_value, tick_size):
    dist = abs(entry - sl)
    if dist == 0:
        return 0
    if tick_value <= 0 or tick_size <= 0:
        return 0                                  # refuse to size blind
    value_per_point = tick_value / tick_size
    lots = (balance * risk_pct) / (dist * value_per_point)
    return max(1, int(lots * 100_000))


def _units_to_lots(units):
    return max(0.01, round(units / 100_000, 2))


@pytest.mark.parametrize("sym,entry,sl,tick_value,tick_size", [
    ("US500",  7500.0, 7479.0, 1.0,  1.0),   # $1 / point / lot
    ("NAS100", 28800.0, 28684.0, 1.0, 1.0),  # 116 pt SL
    ("US30",   50700.0, 50509.0, 1.0, 1.0),  # 191 pt SL
    ("US500",  7500.0, 7479.0, 0.5,  0.1),   # $5 / point / lot (different contract)
])
def test_index_risk_is_correct(sym, entry, sl, tick_value, tick_size):
    balance, risk_pct = 10_000.0, 0.004        # 0.4% = $40
    units = _live_index_units(balance, entry, sl, risk_pct, tick_value, tick_size)
    lots  = _units_to_lots(units)
    dist  = abs(entry - sl)
    value_per_point = tick_value / tick_size
    dollar_risk = lots * dist * value_per_point
    target = balance * risk_pct
    # within one lot-rounding step of the target risk
    assert abs(dollar_risk - target) <= value_per_point * dist * 0.02 + 1.0, (
        f"{sym}: risked ${dollar_risk:.2f} vs target ${target:.2f} (lots={lots})")


def test_index_refuses_to_size_without_tick_data():
    # Broker not reporting tick economics -> size 0 (never size an index blind)
    assert _live_index_units(10_000, 7500, 7479, 0.004, 0.0, 0.0) == 0
    assert _live_index_units(10_000, 7500, 7479, 0.004, 1.0, 0.0) == 0


def test_zero_distance_returns_zero():
    assert _live_index_units(10_000, 7500, 7500, 0.004, 1.0, 1.0) == 0
