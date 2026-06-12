"""
Live bridge — runs auto-promoted survivor strategies inside MT5Live.2.py
alongside the existing hardcoded ORB, without touching the ORB code path.

Each promoted survivor:
  * Gets its own per-(pair, strategy) state object
  * Calls the same canonical entry_fn signature as backtest sweeps use
  * Sends orders through the same place_stop_order primitive
  * Logs trades to live_trades sqlite + live_trade_log.csv with strategy_name
  * Has its position size scaled by promotion mode (SHADOW=0, QUARTER=0.25, ...)

This module never raises out of public functions — survivor failures are
logged and swallowed so ORB continues trading regardless.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from agent import db

log = logging.getLogger(__name__)

# Ensure project root is on sys.path so generated entry modules can import
# from edge_engine etc.
_PROJ_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# review#2 — promotion-mode size multiplier. SHADOW=0 builds the trade and
# logs the decision but never calls place_stop_order; OFF/PAUSED also map to
# 0 so a stale spec can never trade at any size. .get(mode, 0.0) below ensures
# unknown modes default to paper-only (closed by construction).
MODE_SIZE_MULT = {
    'SHADOW':       0.0,
    'LIVE_QUARTER': 0.25,
    'LIVE_HALF':    0.50,
    'LIVE_FULL':    1.00,
    'OFF':          0.0,
    'PAUSED':       0.0,
}

# Cache of dynamically imported entry functions, keyed by strategy_name.
_ENTRY_FN_CACHE: dict[str, Callable] = {}


def _normalize_pair(pair: str) -> str:
    """MT5Live uses compact format ('GBPUSD'); edge_engine uses underscore
    format ('GBP_USD'). Translate compact → underscore for prepare_df and
    entry_fn calls. Idempotent: 'GBP_USD' passes through unchanged."""
    if '_' in pair:
        return pair
    if len(pair) == 6:
        return f"{pair[:3]}_{pair[3:]}"
    return pair  # XAUUSD etc — caller must have normalized already


# ---------------------------------------------------------------------------
# StrategyState — per (pair, strategy_name) lifecycle
# ---------------------------------------------------------------------------

@dataclass
class StrategyState:
    """Per-pair, per-strategy lifecycle. Mirrors PairState's phase machine but
    isolated from ORB so survivors can't corrupt ORB state."""
    pair:           str
    strategy_name:  str
    phase:          str = "IDLE"  # IDLE | ORDER_PLACED | IN_TRADE | DONE
    order_id:       str | None = None
    direction:      str | None = None
    entry_price:    float | None = None
    stop_loss:      float | None = None
    take_profit:    float | None = None
    units:          int | None = None
    opened_ts:      str | None = None
    pending_dist:   float | None = None
    # Scratch dict the entry_fn writes into across ticks (matches backtest convention)
    scratch:        dict = field(default_factory=dict)

    def reset(self) -> None:
        self.phase = "IDLE"
        self.order_id = None
        self.direction = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.units = None
        self.opened_ts = None
        self.pending_dist = None
        self.scratch = {}


# ---------------------------------------------------------------------------
# Loading promoted survivors
# ---------------------------------------------------------------------------

def _import_entry_fn(strategy_name: str) -> Callable | None:
    """Dynamically import agent.generated.entry_<name>:entry_<name>.

    Cached. Returns None on import failure (logged, never raised) so a single
    broken generated module doesn't kill the whole survivor surface.
    """
    if strategy_name in _ENTRY_FN_CACHE:
        return _ENTRY_FN_CACHE[strategy_name]
    mod_name  = f"agent.generated.entry_{strategy_name}"
    fn_name   = f"entry_{strategy_name}"
    try:
        mod = importlib.import_module(mod_name)
        fn  = getattr(mod, fn_name, None)
        if fn is None:
            log.error(f"[live_bridge] {mod_name} has no attribute {fn_name}")
            return None
        _ENTRY_FN_CACHE[strategy_name] = fn
        return fn
    except Exception as e:
        log.error(f"[live_bridge] failed to import {mod_name}: {e}")
        return None


def load_active_survivors() -> dict[str, dict]:
    """Return {strategy_name: {fn, mode, params, session, pairs, live_n}} for
    every promoted strategy in an active mode. Failures load to {} silently —
    ORB must keep trading even if the bridge breaks.
    """
    out: dict[str, dict] = {}
    try:
        promoted = db.list_promoted_survivors(active_only=True)
    except Exception as e:
        log.error(f"[live_bridge] list_promoted_survivors failed: {e}")
        return out
    for p in promoted:
        name = p['strategy_name']
        fn = _import_entry_fn(name)
        if fn is None:
            continue
        out[name] = {
            'fn':      fn,
            'mode':    p['mode'],
            'params':  p.get('params') or {},
            'session': p.get('session'),
            'pairs':   p.get('pairs') or [],
            'live_n':  p.get('live_n', 0),
        }
    return out


# ---------------------------------------------------------------------------
# Bookkeeping shim — entry fns call resolve_risk(bst, ...) which reads
# bst.balance, bst.consecutive_losses, etc. We give them a thin live shim so
# they don't need any code changes vs backtest.
# ---------------------------------------------------------------------------

class LiveBookkeeping:
    """Subset of edge_engine._BST sufficient for entry_fn / resolve_risk
    / rv_size to run unmodified against live state."""
    __slots__ = ['balance', 'consecutive_wins', 'consecutive_losses',
                 'day_start_bal', 'family_day_pnl', 'family_total_pnl',
                 'family_blown', 'family_day_blocked', 'account_blown',
                 'trade_log', 'days_blocked']

    def __init__(self, balance: float, day_start_bal: float | None = None,
                 wins: int = 0, losses: int = 0):
        self.balance            = float(balance)
        self.consecutive_wins   = int(wins)
        self.consecutive_losses = int(losses)
        self.day_start_bal      = float(day_start_bal if day_start_bal is not None else balance)
        self.family_day_pnl     = {}
        self.family_total_pnl   = {}
        self.family_blown       = {}
        self.family_day_blocked = {}
        self.account_blown      = False
        self.trade_log          = []
        self.days_blocked       = []


# ---------------------------------------------------------------------------
# assemble_row — build the row object generated entry fns expect
# ---------------------------------------------------------------------------

def _inject_live_defaults(df):
    """prepare_df assumes tick-aggregated input (spread_mean, tick_imbalance,
    vol_imbalance, realized_vol, delta come from groupby over ticks) AND a
    'timestamp' column. MT5 only gives us M1 OHLC indexed by time, so we
    inject defaults + materialise the timestamp column. Where a sensible live
    proxy exists (spread from bid/ask), use it. Where it doesn't (tick
    imbalance), NaN — and any strategy that gates on that feature will
    correctly skip via its NaN guard rather than firing on phantom data.
    """
    import pandas as pd
    df = df.copy()
    # Materialise 'timestamp' column from the DatetimeIndex if needed
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # Whatever the index was named (time / date / etc), rename to timestamp
            first_col = df.columns[0]
            if first_col != 'timestamp':
                df = df.rename(columns={first_col: 'timestamp'})
        else:
            # No datetime info to recover — let prepare_df fail loudly later
            df['timestamp'] = pd.NaT
    # Spread proxy: live bid-ask difference if both columns present.
    if 'spread_mean' not in df.columns:
        if 'ask' in df.columns and 'bid' in df.columns:
            df['spread_mean'] = (df['ask'] - df['bid']).abs()
        else:
            df['spread_mean'] = float('nan')
    if 'spread_max' not in df.columns:
        df['spread_max'] = df['spread_mean']
    for c in ('tick_imbalance', 'vol_imbalance', 'realized_vol', 'delta',
              'ask_vol_sum', 'bid_vol_sum', 'volume', 'tick_count'):
        if c not in df.columns:
            df[c] = float('nan')
    if 'mid' not in df.columns and 'close' in df.columns:
        df['mid'] = df['close']
    return df


def assemble_row(pair: str, bars_df, pair_state) -> SimpleNamespace | None:
    """Run prepare_df on recent live bars and return the latest row.

    Falls back to None if prepare_df fails or if there aren't enough bars
    to compute the rolling features (≥250 M1 bars recommended).
    """
    try:
        from edge_engine import prepare_df
    except Exception as e:
        log.error(f"[live_bridge] prepare_df import failed: {e}")
        return None
    if bars_df is None or getattr(bars_df, 'empty', True):
        return None
    edge_pair = _normalize_pair(pair)
    try:
        df_in = _inject_live_defaults(bars_df)
        prepped = prepare_df(df_in, edge_pair)
    except Exception as e:
        log.error(f"[live_bridge] prepare_df({edge_pair}) failed: {e}")
        return None
    if prepped.empty:
        return None
    last = prepped.iloc[-1].copy()
    # Override with live PairState range — its values are computed at the
    # actual session-open tick, while prepare_df's range_high comes from a
    # batch session-window aggregate that may be NaN on a partial buffer.
    rh = getattr(pair_state, 'cached_range_high', None)
    rl = getattr(pair_state, 'cached_range_low', None)
    if rh is not None:
        last['range_high'] = float(rh)
    if rl is not None:
        last['range_low']  = float(rl)
    return last


# ---------------------------------------------------------------------------
# run_survivor — analog of MT5Live.run_pair but per strategy
# ---------------------------------------------------------------------------

def run_survivor(
    name:      str,
    spec:      dict,
    pair:      str,
    st_strat:  StrategyState,
    *,
    bars_df,
    pair_state,
    balance:   float,
    regime:    str = 'UNDEFINED',
    regime_mult_map: dict | None = None,
    slip:      float = 0.0,
    hspd:      float = 0.0,
    sess_cfg:  dict | None = None,
    place_order_fn: Callable | None = None,
    expire_utc: int | None = None,
    consecutive_wins: int = 0,
    consecutive_losses: int = 0,
) -> dict | None:
    """Run one survivor's entry decision for one pair on one tick.

    Returns:
      - None if no decision was made (e.g. spread gate, no setup)
      - dict {'action': 'placed' | 'shadow', 'order_id', 'direction', ...}
        if a trade decision was reached this tick

    NEVER raises — failures are logged and swallowed so ORB keeps trading.
    """
    slog = logging.getLogger(f"strat.{name}")
    try:
        # Kill-switch gate (TCA)
        kill = db.get_live_kill(name)
        if kill and kill.get('verdict') == 'KILL':
            slog.debug(f"[{pair}] skip — TCA KILL verdict (decay={kill.get('decay')})")
            return None
        kill_size_mult = 0.5 if kill and kill.get('verdict') == 'REDUCE' else 1.0

        # Pair filter — strategies validated only on a subset of pairs.
        # Compare in normalized (underscore) form so 'GBPUSD' matches 'GBP_USD'.
        edge_pair = _normalize_pair(pair)
        if spec.get('pairs'):
            allowed = {_normalize_pair(p) for p in spec['pairs']}
            if edge_pair not in allowed:
                return None

        # Build the slot the entry_fn expects
        slot = {
            'strategy_def': {
                'family': name,
                'params': spec.get('params') or {},
            },
            'scratch':  st_strat.scratch,
            'position': None,
        }
        bst = LiveBookkeeping(balance=balance,
                              wins=consecutive_wins,
                              losses=consecutive_losses)

        # Build the row
        row = assemble_row(pair, bars_df, pair_state)
        if row is None:
            return None

        # Resolve the regime multiplier the entry fn will receive
        if regime_mult_map is None:
            regime_mult_map = {'TRENDING': 1.0, 'RANGING': 0.5, 'TRANSITIONING': 0.5,
                               'VOLATILE': 0.0, 'UNDEFINED': 0.3}
        regime_mult = float(regime_mult_map.get(regime, 0.3))

        # Call the strategy. Pass `edge_pair` (underscore form) so PAIR_PIP_SIZE
        # / SESSION_CONFIG lookups inside the entry fn succeed.
        try:
            spec['fn'](
                bst, slot, row, row.name if hasattr(row, 'name') else None,
                edge_pair, slip, hspd,
                sess_cfg or {}, regime, regime_mult,
                None, {},  # fvg_buf=None; day_sweep={} so strategies can do .get(pair, {})
            )
        except Exception as e:
            slog.error(f"[{pair}] entry_fn raised: {e}")
            return None

        # Where did the entry_fn end up?
        #   * slot['position'] filled  → check_and_fill triggered THIS tick
        #     (the pending_* keys were popped into slot fields). Use slot.
        #   * sc['pending_dir'] still set → resting stop order awaiting
        #     a future trigger. Use scratch.
        sc = slot['scratch']
        if slot.get('position') is not None:
            pending_dir = slot['position']
            entry = slot.get('entry_price')
            sl    = slot.get('stop_loss')
            tp    = slot.get('take_profit')
            size  = slot.get('pos_size')
            sl_dist = slot.get('sl_ref_dist')
        elif sc.get('pending_dir') is not None:
            pending_dir = sc.get('pending_dir')
            entry  = sc.get('pending_entry')
            sl     = sc.get('pending_sl')
            tp     = sc.get('pending_tp')
            size   = sc.get('pending_size')
            sl_dist = sc.get('pending_dist')
        else:
            return None
        if None in (entry, sl, tp, size):
            slog.warning(f"[{pair}] entry_fn set pending_dir={pending_dir} but missing fields")
            return None

        # Apply mode-based sizing (SHADOW=0 short-circuits to paper trade)
        mode_mult = MODE_SIZE_MULT.get(spec.get('mode', 'SHADOW'), 0.0)
        size_units = max(0, int(round(float(size) * mode_mult * kill_size_mult)))

        if spec.get('mode') == 'SHADOW' or size_units <= 0:
            slog.info(f"[{pair}] SHADOW {pending_dir.upper()} entry={entry:.5f} "
                      f"sl={sl:.5f} tp={tp:.5f} size_would_be={int(float(size))}")
            return {'action': 'shadow', 'direction': pending_dir,
                    'entry': entry, 'sl': sl, 'tp': tp,
                    'size_intended': int(float(size))}

        # Live mode — place the real order
        if place_order_fn is None:
            slog.error(f"[{pair}] mode={spec.get('mode')} but no place_order_fn provided")
            return None
        try:
            oid = place_order_fn(pair, pending_dir, size_units, entry, sl, tp, expire_utc)
        except Exception as e:
            slog.error(f"[{pair}] place_order_fn raised: {e}")
            return None
        if oid is None:
            slog.warning(f"[{pair}] place_order_fn returned None (broker rejected or stale tick)")
            return None

        st_strat.phase        = "ORDER_PLACED"
        st_strat.order_id     = oid
        st_strat.direction    = pending_dir
        st_strat.entry_price  = float(entry)
        st_strat.stop_loss    = float(sl)
        st_strat.take_profit  = float(tp)
        st_strat.units        = size_units
        st_strat.pending_dist = float(sl_dist or abs(entry - sl))
        slog.info(f"[{pair}] {spec.get('mode')} {pending_dir.upper()} order_id={oid} "
                  f"entry={entry:.5f} sl={sl:.5f} tp={tp:.5f} units={size_units}")
        return {'action': 'placed', 'order_id': oid, 'direction': pending_dir,
                'entry': entry, 'sl': sl, 'tp': tp, 'units': size_units,
                'mode': spec.get('mode')}

    except Exception as e:
        # Final safety net — never let a survivor strategy bug crash MT5Live
        slog.error(f"[{pair}] run_survivor unexpected error: {e}")
        return None


def record_exit(
    name:       str,
    pair:       str,
    direction:  str,
    ts_open:    str,
    ts_close:   str,
    entry:      float,
    exit_price: float,
    sl:         float,
    tp:        float | None,
    pnl_usd:    float,
    mode:       str,
    *,
    slip_open_pip:  float | None = None,
    slip_close_pip: float | None = None,
    regime:         str | None   = None,
    session:        str | None   = None,
) -> None:
    """Persist a closed live trade to the per-strategy ledger. Computes pnl_r
    from the entry-to-SL distance. Idempotent failure mode: logs error, returns."""
    try:
        sl_dist = abs(float(entry) - float(sl)) if sl is not None else 0.0
        per_unit = abs(float(exit_price) - float(entry))
        pnl_r = (per_unit / sl_dist) * (1 if pnl_usd >= 0 else -1) if sl_dist > 0 else None
        db.record_live_trade(
            strategy_name=name, pair=pair, side=direction,
            ts_open=ts_open, ts_close=ts_close,
            entry=float(entry), exit_price=float(exit_price),
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
            pnl_usd=float(pnl_usd), pnl_r=pnl_r,
            slip_open_pip=slip_open_pip, slip_close_pip=slip_close_pip,
            regime=regime, session=session, mode=mode,
        )
    except Exception as e:
        log.error(f"[live_bridge.record_exit] {name} {pair} failed: {e}")
