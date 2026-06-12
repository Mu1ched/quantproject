"""Glue between MT5 live state and the entry-function contract used by the
backtester.

Entry functions (in edge_hypotheses.py and agent/generated/*.py) expect a
namedtuple-shaped `row` with attributes like .high, .low, .close, .hour,
.atr, .range_high, etc., and write a `pending_*` dict into slot['scratch'].

This adapter:
  * builds a row-equivalent object from live MT5 ticks + cached features
  * converts the slot's pending_* dict to the args of place_stop_order()

A tiny SimpleNamespace is sufficient — entry functions only read attributes,
never iterate.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def assemble_row(pair: str,
                 latest_bar: dict,
                 indicators: dict[str, Any] | None = None) -> SimpleNamespace:
    """Construct a row-equivalent object for entry_fn consumption.

    Args:
        pair:         broker pair symbol (e.g. 'GBPUSD' — caller normalises)
        latest_bar:   most recent M1 bar with keys: open, high, low, close, ts
        indicators:   precomputed feature dict from MT5Live's regime engine
                      (atr, adx, range_high, range_low, ma_trend, realized_vol,
                       rv_median, spread_mean, spread_median, tick_imbalance,
                       yz_vol_ratio, regime, hour, ...)
    """
    indicators = indicators or {}
    ts = latest_bar.get('ts')
    fields = {
        'open':         latest_bar['open'],
        'high':         latest_bar['high'],
        'low':          latest_bar['low'],
        'close':        latest_bar['close'],
        'timestamp':    ts,
        'date':         getattr(ts, 'date', lambda: None)() if ts is not None else None,
        'hour':         getattr(ts, 'hour', None),
        'minute':       getattr(ts, 'minute', None),
    }
    # Indicators take precedence over bar fields with the same name; live
    # feature engine is authoritative.
    fields.update(indicators)
    return SimpleNamespace(**fields)


def pending_to_mt5(pair: str, scratch: dict, expire_utc: int) -> dict | None:
    """Convert a slot scratch's pending_* fields to place_stop_order kwargs.

    Returns None if no pending order is set.
    """
    direction = scratch.get('pending_dir')
    if direction is None:
        return None
    return {
        'pair':       pair,
        'direction':  direction,
        'units':      int(scratch.get('pending_size') or 0),
        'trigger':    float(scratch['pending_level']),
        'sl':         float(scratch['pending_sl']),
        'tp':         float(scratch['pending_tp']),
        'expire_utc': int(expire_utc),
    }


def make_slot_for(strategy_def: dict) -> dict:
    """Construct a fresh slot the way edge_engine.fresh_slot() would.

    Live execution doesn't run the full bar loop, so we keep the schema minimal
    but matched: any field the entry function or check_and_fill() touches must
    exist.
    """
    return {
        'slot_id':         strategy_def['name'],
        'strategy_def': {
            'family':           'live',
            'params':           strategy_def.get('params', {}),
            'allow_concurrent': False,
        },
        'scratch':         {},
        'position':        None,
        'opened_today':    False,
        'session_exited':  False,
        'partial_tp_done': False,
        'partial_pnl':     0.0,
        'regime':          'UNDEFINED',
        'entry_price':     None,
        'stop_loss':       None,
        'take_profit':     None,
        'pos_size':        0,
        'sl_ref_dist':     0.0,
        'partial_size':    0,
        'remainder_size':  0,
        'entry_time':      None,
    }


class _LiveBst:
    """Minimal balance-state stand-in for live entry_fn calls."""
    def __init__(self, balance: float):
        self.balance             = float(balance)
        self.consecutive_wins    = 0
        self.consecutive_losses  = 0


def make_live_bst(balance: float, wins: int = 0, losses: int = 0) -> _LiveBst:
    bst = _LiveBst(balance)
    bst.consecutive_wins   = int(wins)
    bst.consecutive_losses = int(losses)
    return bst
