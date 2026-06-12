"""
Pre-sweep conditional check.

Before spending ~20 minutes on a full grid sweep (12 combos × 3 pairs +
robustness), run a minimal single-pair single-combo backtest. If the strategy
fires too few trades or produces a catastrophic Sharpe even at this minimal
scale, skip the full sweep entirely.

This is NOT a vectorized predicate check — the entry functions in
agent/generated/ are imperative (scratch state, place_pending side effects),
so we can't trivially call them as row→bool predicates. Instead we reuse
edge_engine.run_backtest on one pair with one params combo. Typical wall-time
on losers: 1–3 min vs ~20 min full sweep.
"""
from __future__ import annotations

import logging
import time

import edge_engine as eng

from agent.session_router import session_regime_mult

log = logging.getLogger(__name__)

# Reject gates — deliberately lenient. We're filtering catastrophes only,
# not selecting survivors (that's the scorer's job downstream).
# 2026-06-05: lowered MIN_TRADES from 20 → 10 to let near-miss strategies
# reach stage-1 for diagnostic measurement; the real survivor gate is still
# 20 (MIN_TEST_TRADES in agent/config.py).
PRESCREEN_MIN_TRADES = 10
PRESCREEN_MIN_SHARPE = -2.0


def pre_screen(
    name:          str,
    entry_fn,
    train_dfs:     dict,
    params:        dict,
    session:       str,
    stage1_pair:   str,
    exit_hour:     int,
    cost_mult:     float = 0.5,
    progress_callback=None,
) -> tuple[bool, str]:
    """Run a minimal one-pair-one-combo backtest. Return (passed, reason).

    `progress_callback`, when supplied, is forwarded to `eng.run_backtest` so
    the agent (or a GUI) can observe per-trade events as they happen. The
    callback receives `{"type": "trade_closed", "trade": {...}, "balance": ...}`
    on every closed trade, plus periodic `{"type": "tick", "bar_idx": N,
    "total_bars": M}` heartbeats.
    """
    if stage1_pair not in train_dfs:
        return True, f"skip pre-screen: {stage1_pair} not in train_dfs"

    slot_class = f"prescreen_{name[:14]}".replace("-", "_").lower()
    manager_fn = eng.make_manager(exit_hour=exit_hour, use_breakeven=True)
    registry = [{
        "id":               f"prescreen_{name}",
        "family":           "prescreen",
        "slot_class":       slot_class,
        "pairs":            [stage1_pair],
        "session":          session,
        "allow_concurrent": False,
        "regime_mult":      session_regime_mult(session),
        "params":           params,
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}

    train_subset = {stage1_pair: train_dfs[stage1_pair]}

    t0 = time.time()
    try:
        trades, _bal, _ = eng.run_backtest(
            train_subset, None, None,           # dynamic-spread mode
            registry, slot_managers, slot_entries,
            cost_mult=cost_mult,
            progress_callback=progress_callback,
        )
    except Exception as e:
        return False, f"backtest error: {type(e).__name__}: {e}"
    elapsed = time.time() - t0

    n_trades = 0 if trades is None or trades.empty else len(trades)
    if n_trades < PRESCREEN_MIN_TRADES:
        return False, (f"too few trades ({n_trades} < {PRESCREEN_MIN_TRADES}) "
                       f"on {stage1_pair} in {elapsed:.0f}s")

    stats = eng.calc_stats(trades) or {}
    sharpe = float(stats.get("sharpe", 0) or 0)
    if sharpe < PRESCREEN_MIN_SHARPE:
        return False, (f"catastrophic Sharpe ({sharpe:.2f} < "
                       f"{PRESCREEN_MIN_SHARPE}) on {stage1_pair} "
                       f"({n_trades} trades, {elapsed:.0f}s)")

    log.info("  pre_screen passed: %s on %s — %d trades, Sharpe=%.2f (%.0fs)",
             name, stage1_pair, n_trades, sharpe, elapsed)
    return True, f"passed: {n_trades} trades, Sharpe={sharpe:.2f}"
