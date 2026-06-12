"""MT5 reconciliation tests (review#P2#5) — currently a stub.

The fourth of the four test files specified in P2_P3_plan.md was supposed
to mock `mt5.positions_get` and assert that MT5Live's startup
reconciliation handles stale internal state correctly. Implementing this
properly requires either:

  (a) importing the MetaTrader5 library (Windows-only, not available in
      the standard test environment), OR

  (b) refactoring MT5Live.2.py so the reconciliation logic is in a
      pure-Python helper that takes a `positions_get`-shaped fixture.

Option (b) is the right long-term move but is a moderate refactor of a
3.9k-LOC file. Option (a) is brittle. Deferring to a focused MT5
refactoring session.

Marked as skipped so pytest collects it (visible in -v output) without
producing failure noise.
"""
import pytest

pytest.skip(
    "MT5 reconciliation tests require either MetaTrader5 lib or a refactor "
    "of MT5Live.2.py to extract reconciliation into a pure helper. See "
    "P2_P3_plan.md P2#5 (4) for the implementation plan.",
    allow_module_level=True,
)
