"""Sizing-parity tests (review#17).

Asserts that backtest position sizing (`edge_engine.calc_size`) returns the
same number of units the live trader (`MT5Live.compute_units`) would produce
for the same inputs. We can't import `MT5Live.2.py` directly in tests because
it depends on MetaTrader5 (Windows-only), so we reproduce its exact formula
here as `_live_compute_units` and treat any divergence between this test
helper and `MT5Live.compute_units` as a bug to be caught at code review.
"""
import math

import pytest

import edge_engine as eng


# ── Live formula mirror (verbatim from MT5Live.2.py:1605-1639) ────────────────


def _live_compute_units(balance, entry, sl, risk_pct, pair, usdjpy=150.0,
                        xau_contract_size=100.0):
    """Mirror of MT5Live.compute_units — returns integer units (min 1).

    `usdjpy` and `xau_contract_size` are injected so the test is deterministic.
    """
    dist = abs(entry - sl)
    if dist == 0:
        return 0
    if pair in ("GBPJPY", "EURJPY", "USDJPY"):
        return max(1, int((balance * risk_pct * usdjpy) / dist))
    # review#17 follow-up — mirror the USDCAD/USDCHF fix in MT5Live.
    if pair in ("USDCAD", "USDCHF"):
        return max(1, int((balance * risk_pct * entry) / dist))
    if pair == "XAUUSD":
        return max(1, int((balance * risk_pct * 100_000) / (dist * xau_contract_size)))
    return max(1, int((balance * risk_pct) / dist))


def _bt_pair(p: str) -> str:
    """Backtest uses underscore form; live uses compact form. Translate."""
    return p.replace('_', '')


# ── Parity scenarios ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("pair,entry,sl,balance,risk_pct", [
    # XXX/USD majors — both formulas should give units = balance*risk/dist
    ('EUR_USD', 1.10000, 1.09800, 10_000, 0.01),
    ('GBP_USD', 1.25000, 1.24700, 10_000, 0.01),
    ('AUD_USD', 0.65000, 0.64800, 10_000, 0.01),
    ('NZD_USD', 0.60000, 0.59800, 10_000, 0.01),
])
def test_sizing_parity_xxxusd(pair, entry, sl, balance, risk_pct):
    """XXX/USD pairs must match: both formulas reduce to balance*risk/dist."""
    bt_size   = eng.calc_size(pair, balance, risk_pct, entry, abs(entry - sl))
    live_size = _live_compute_units(balance, entry, sl, risk_pct, _bt_pair(pair))

    # Live floors at 1 and ints; backtest is float. Allow ±1 unit rounding.
    assert abs(bt_size - live_size) <= 1, (
        f"{pair}: backtest sized {bt_size:.1f} units, live would size "
        f"{live_size} units — divergence > 1 unit. Likely a missing entry "
        f"in edge_engine._USD_QUOTED."
    )


@pytest.mark.parametrize("pair,entry,sl,balance,risk_pct", [
    ('USD_CAD', 1.35000, 1.34800, 10_000, 0.01),
    ('USD_CHF', 0.90000, 0.89800, 10_000, 0.01),
])
def test_sizing_parity_usd_xxx(pair, entry, sl, balance, risk_pct):
    """USD/XXX pairs (USD_CAD, USD_CHF) — review#17 follow-up. Was previously
    xfailed because MT5Live.compute_units lacked the XXX→USD conversion;
    fixed so backtest entry-multiplied formula and live's pair-price
    formula now agree."""
    bt_size   = eng.calc_size(pair, balance, risk_pct, entry, abs(entry - sl))
    live_size = _live_compute_units(balance, entry, sl, risk_pct, _bt_pair(pair))
    assert abs(bt_size - live_size) <= 1, (
        f"{pair}: backtest sized {bt_size:.1f} units, live would size "
        f"{live_size} units — divergence > 1 unit."
    )


def test_sizing_parity_usd_jpy():
    # USD_JPY: backtest entry == live's usdjpy by construction.
    pair, entry, sl, balance, risk_pct = 'USD_JPY', 150.00, 149.85, 10_000, 0.01
    bt_size   = eng.calc_size(pair, balance, risk_pct, entry, abs(entry - sl))
    live_size = _live_compute_units(balance, entry, sl, risk_pct, 'USDJPY',
                                    usdjpy=entry)
    assert abs(bt_size - live_size) <= 1, (
        f"USD_JPY parity broken: bt={bt_size}, live={live_size}"
    )


def test_sizing_parity_eur_jpy_known_divergence():
    """Cross-JPY pairs (EUR_JPY, GBP_JPY) intentionally diverge — backtest uses
    entry (EUR/JPY ≈ 165), live uses USDJPY (≈ 150). The expected ratio is
    entry / USDJPY. Documenting the gap in a test rather than silently
    accepting it."""
    pair, entry, sl, balance, risk_pct = 'EUR_JPY', 165.00, 164.85, 10_000, 0.01
    usdjpy = 150.0
    bt_size   = eng.calc_size(pair, balance, risk_pct, entry, abs(entry - sl))
    live_size = _live_compute_units(balance, entry, sl, risk_pct, 'EURJPY',
                                    usdjpy=usdjpy)

    expected_ratio = entry / usdjpy
    actual_ratio   = bt_size / max(live_size, 1)
    # Expect ~1.10 ratio when EUR_JPY=165 and USD_JPY=150. Within 5% of that.
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.05, (
        f"EUR_JPY divergence ratio shifted: expected {expected_ratio:.3f}, "
        f"got {actual_ratio:.3f}. If this is intentional (sizing redesign), "
        f"update or delete this test."
    )


def test_sizing_parity_zero_distance():
    """Both implementations must return 0 (or 1 minimum, in live's case) on
    zero distance — never raise."""
    # Live returns 0 explicitly; backtest currently raises ZeroDivisionError
    # on zero distance (a known pre-existing test_risk failure). This test
    # documents the expectation that future fixes converge on a 0/1 return.
    live = _live_compute_units(10_000, 1.10, 1.10, 0.01, 'EURUSD')
    assert live == 0
    # NOT asserting on bt_size — test_risk.py already covers the
    # ZeroDivisionError gap and is left as a separate fix.
