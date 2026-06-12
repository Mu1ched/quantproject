"""Promotion-ramp transition tests (review#P2#5).

Covers `agent.promotion._ramp` — the pure transition function that decides
whether a strategy at a given mode advances, holds, or auto-kills based on
live trade count and decay metric.
"""
import pytest

from agent.promotion import (
    _ramp,
    KILL_DECAY_FLOOR,
)


# ── Auto-kill ───────────────────────────────────────────────────────────────


def test_decay_below_floor_auto_kills():
    new_mode, why = _ramp('LIVE_HALF', live_n=50, decay=0.2)
    assert new_mode == 'OFF'
    assert 'auto-KILL' in why


def test_decay_at_floor_does_not_kill():
    # Exactly at the floor — should not trigger auto-kill (strictly less than).
    new_mode, _ = _ramp('LIVE_QUARTER', live_n=20, decay=KILL_DECAY_FLOOR)
    assert new_mode != 'OFF'


# ── Hold conditions ─────────────────────────────────────────────────────────


def test_paused_and_off_never_progress():
    assert _ramp('PAUSED', live_n=999, decay=1.0) == ('PAUSED', '')
    assert _ramp('OFF',    live_n=999, decay=1.0) == ('OFF', '')


def test_shadow_holds_until_live_n_threshold():
    # SHADOW needs live_n >= 10 to advance; below that, hold.
    new_mode, _ = _ramp('SHADOW', live_n=9, decay=None)
    assert new_mode == 'SHADOW'


def test_quarter_holds_when_decay_too_low():
    # LIVE_QUARTER → LIVE_HALF requires live_n >= 20 AND decay >= 0.5.
    # decay=0.4 → hold, not advance.
    new_mode, _ = _ramp('LIVE_QUARTER', live_n=25, decay=0.4)
    assert new_mode == 'LIVE_QUARTER'


def test_quarter_holds_when_decay_missing():
    new_mode, _ = _ramp('LIVE_QUARTER', live_n=25, decay=None)
    assert new_mode == 'LIVE_QUARTER'


# ── Forward transitions ─────────────────────────────────────────────────────


def test_shadow_to_quarter_when_live_n_sufficient():
    new_mode, why = _ramp('SHADOW', live_n=12, decay=None)
    assert new_mode == 'LIVE_QUARTER'
    assert 'live_n=12' in why


def test_quarter_to_half_when_thresholds_met():
    new_mode, _ = _ramp('LIVE_QUARTER', live_n=25, decay=0.6)
    assert new_mode == 'LIVE_HALF'


def test_half_to_full_when_thresholds_met():
    new_mode, _ = _ramp('LIVE_HALF', live_n=35, decay=0.8)
    assert new_mode == 'LIVE_FULL'


def test_full_stays_full():
    # No further mode beyond LIVE_FULL.
    new_mode, _ = _ramp('LIVE_FULL', live_n=100, decay=1.2)
    assert new_mode == 'LIVE_FULL'
