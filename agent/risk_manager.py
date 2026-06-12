"""
Phase 6 — portfolio-level risk integration.

Single entry point that MT5Live calls before placing each trade. Returns a
dict of risk multipliers and pass/fail flags so the caller can assemble the
final risk_pct without scattering policy across the live trader.

Components:
  1. HRP weight per active strategy        — capital allocation across survivors
  2. Currency exposure cap                  — limits aggregate single-CCY exposure
  3. Drawdown-adjusted sizing               — halves risk past 50% of daily loss limit
  4. Per-strategy consecutive-loss breaker  — pauses a strategy after K losses
  5. Heartbeat publish                      — emits a watchdog-readable file each call

This module owns NO MT5 calls; the live process is responsible for fetching
balance/equity and feeding them in. That keeps the policy testable in isolation
(tests/test_risk_manager.py) and reusable from the backtester.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable

from agent.config import AGENT_DB_PATH

log = logging.getLogger(__name__)


# ── Configurable policy thresholds ────────────────────────────────────────────

# Currency exposure cap — total notional risk on any one currency, across all
# open positions, expressed as a fraction of account equity. 6% covers a
# 3-position book at 2% each on the same CCY before any fourth gets blocked.
MAX_CCY_EXPOSURE_PCT  = 0.06

# Drawdown-adjusted sizing — once today's loss hits this fraction of the
# daily-loss limit, halve risk for remaining trades. De-risk before stop-out.
DD_HALVE_THRESHOLD    = 0.50

# Per-strategy consecutive-loss circuit breaker.
LOSS_HALT_K           = 4
LOSS_HALT_HOURS       = 4

# Heartbeat path — overwritten on every risk_check() call. A separate watchdog
# process tails this and flattens positions if the mtime is older than 60 s.
HEARTBEAT_FILE        = Path(AGENT_DB_PATH).parent / "live_heartbeat.json"


# ── Currency mapping ──────────────────────────────────────────────────────────

# Each pair contributes both legs to the exposure tally. XAUUSD treated as
# {XAU, USD}; metals roll into a single 'XAU' bucket.
def _currencies_of(pair: str) -> tuple:
    if pair.startswith("XAU") or pair.startswith("XAG"):
        return (pair[:3], pair[3:])
    if len(pair) >= 6:
        return (pair[:3], pair[3:6])
    return (pair, "")


# ── Currency exposure cap ─────────────────────────────────────────────────────

def currency_exposure_ok(
    new_pair:    str,
    new_risk:    float,
    open_book:   Iterable[dict],
    equity:      float,
) -> tuple:
    """
    open_book items: {'pair': str, 'risk_usd': float}.
    new_risk: USD-equivalent risk for the proposed trade.
    Returns (ok: bool, reason: str, exposure_after: dict[ccy, pct]).
    """
    if equity <= 0:
        return True, "no equity reference", {}

    exposure: Dict[str, float] = {}
    for pos in open_book:
        ccy_a, ccy_b = _currencies_of(pos.get('pair', ''))
        r = float(pos.get('risk_usd', 0.0))
        if ccy_a:
            exposure[ccy_a] = exposure.get(ccy_a, 0.0) + r
        if ccy_b:
            exposure[ccy_b] = exposure.get(ccy_b, 0.0) + r

    new_a, new_b = _currencies_of(new_pair)
    after = dict(exposure)
    if new_a:
        after[new_a] = after.get(new_a, 0.0) + new_risk
    if new_b:
        after[new_b] = after.get(new_b, 0.0) + new_risk

    cap = MAX_CCY_EXPOSURE_PCT * equity
    breached = {c: r for c, r in after.items() if r > cap}
    if breached:
        worst = max(breached.items(), key=lambda kv: kv[1])
        return (
            False,
            f"{worst[0]} exposure ${worst[1]:.0f} would exceed cap ${cap:.0f}",
            {c: r / equity for c, r in after.items()},
        )
    return True, "ok", {c: r / equity for c, r in after.items()}


# ── Drawdown-adjusted sizing ──────────────────────────────────────────────────

def dd_size_multiplier(realized_dd_pct: float, daily_loss_limit_pct: float) -> float:
    """
    Returns 1.0 normally, 0.5 once realized DD has chewed through
    DD_HALVE_THRESHOLD of the daily loss limit. Both inputs as positive
    decimals (e.g. 0.02 == 2%).
    """
    if daily_loss_limit_pct <= 0:
        return 1.0
    ratio = realized_dd_pct / daily_loss_limit_pct
    return 0.5 if ratio >= DD_HALVE_THRESHOLD else 1.0


# ── Per-strategy circuit breaker ──────────────────────────────────────────────

def _init_breaker_table():
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS strategy_breaker (
            strategy_name      TEXT PRIMARY KEY,
            consecutive_losses INTEGER NOT NULL DEFAULT 0,
            halted_until       TEXT,
            last_updated       TEXT
        )
    """)
    con.commit()
    con.close()


def record_trade_outcome(strategy_name: str, pnl: float):
    """Update the strategy's consecutive-loss counter and trip the breaker
    once K losses in a row land. Called by the live trader after each fill."""
    _init_breaker_table()
    now_iso = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute(
        "SELECT consecutive_losses FROM strategy_breaker WHERE strategy_name = ?",
        (strategy_name,),
    ).fetchone()
    streak = int(row[0]) if row else 0

    if pnl < 0:
        streak += 1
    else:
        streak = 0

    halted_until = None
    if streak >= LOSS_HALT_K:
        halted_until = (datetime.now(timezone.utc)
                        + timedelta(hours=LOSS_HALT_HOURS)).isoformat()
        log.warning("Circuit breaker tripped for '%s': %d consecutive losses, "
                    "halting until %s", strategy_name, streak, halted_until)
        streak = 0  # reset so we don't keep re-tripping after the halt expires

    con.execute("""
        INSERT INTO strategy_breaker (strategy_name, consecutive_losses, halted_until, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(strategy_name) DO UPDATE SET
            consecutive_losses=excluded.consecutive_losses,
            halted_until=excluded.halted_until,
            last_updated=excluded.last_updated
    """, (strategy_name, streak, halted_until, now_iso))
    con.commit()
    con.close()


def is_strategy_halted(strategy_name: str) -> tuple:
    """Returns (halted: bool, reason: str). Consults strategy_breaker AND the
    Phase-1 live_strategy_kills TCA verdict; either side can halt."""
    _init_breaker_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute(
        "SELECT halted_until FROM strategy_breaker WHERE strategy_name = ?",
        (strategy_name,),
    ).fetchone()
    con.close()
    if row and row[0]:
        try:
            halted_until = datetime.fromisoformat(row[0])
            if datetime.now(timezone.utc) < halted_until:
                return True, f"circuit breaker active until {halted_until.isoformat()}"
        except Exception:
            pass

    try:
        from agent import db
        kill = db.get_live_kill(strategy_name)
        if kill and (kill.get('verdict') or '').upper() == 'KILL':
            return True, f"TCA KILL verdict at {kill.get('kill_ts', '?')}"
    except Exception:
        pass

    return False, ""


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def emit_heartbeat(extra: dict | None = None):
    """Write a heartbeat file the watchdog can monitor. Atomic via tmp+rename."""
    payload = {
        'ts_utc': datetime.now(timezone.utc).isoformat(),
        'epoch':  time.time(),
    }
    if extra:
        payload.update(extra)
    tmp = HEARTBEAT_FILE.with_suffix('.tmp')
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload))
        tmp.replace(HEARTBEAT_FILE)
    except Exception as e:
        log.debug("heartbeat write failed: %s", e)


def heartbeat_age_seconds() -> float:
    """Watchdog helper: seconds since last heartbeat write. inf if missing."""
    try:
        return time.time() - HEARTBEAT_FILE.stat().st_mtime
    except FileNotFoundError:
        return float('inf')


# ── HRP weight helper ─────────────────────────────────────────────────────────

# Each call hits the agent DB; cache the weights and refresh at most every
# REFRESH_S so multiple intra-session lookups don't round-trip.
_HRP_CACHE: dict = {'weights': {}, 'ts': 0.0}
_HRP_REFRESH_S   = 600.0


def hrp_weight_for(strategy_key: str, lookback_days: int = 60) -> float:
    """
    Compute HRP weight for the given strategy_key, treating each row in
    live_trades.{pair} as one strategy stream (until Phase 1.6 fully unifies
    multi-strategy live, pair-keyed PnL is the best per-strategy proxy we
    have).

    Returns 1.0 fallback when:
      * fewer than 2 distinct streams exist
      * insufficient overlap for HRP
      * the requested key isn't in the live history yet
    """
    now = time.time()
    if now - _HRP_CACHE['ts'] < _HRP_REFRESH_S and _HRP_CACHE['weights']:
        return float(_HRP_CACHE['weights'].get(strategy_key, 1.0))

    try:
        import pandas as pd
        from agent.hrp_allocator import hrp_weights
    except Exception:
        return 1.0

    try:
        con = sqlite3.connect(AGENT_DB_PATH)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
        rows = con.execute("""
            SELECT date, pair, pnl FROM live_trades
            WHERE date >= ? AND pnl IS NOT NULL
        """, (cutoff,)).fetchall()
        con.close()
    except Exception as e:
        log.debug("hrp_weight_for live_trades read failed: %s", e)
        return 1.0

    if not rows:
        return 1.0

    df = pd.DataFrame(rows, columns=['date', 'pair', 'pnl'])
    daily_by_pair = (df.groupby(['pair', 'date'])['pnl'].sum()
                       .unstack('date').fillna(0.0))
    streams = {p: row.dropna() for p, row in daily_by_pair.iterrows()}
    if len(streams) < 2:
        return 1.0

    weights = hrp_weights(streams)
    # Cache the full weight map for cheap subsequent lookups.
    _HRP_CACHE['weights'] = weights
    _HRP_CACHE['ts']      = now
    return float(weights.get(strategy_key, 1.0))


# ── Combined risk gate ────────────────────────────────────────────────────────

def risk_check(
    strategy_name: str,
    pair:           str,
    proposed_risk:  float,
    open_book:      Iterable[dict],
    equity:         float,
    realized_dd_pct: float,
    daily_loss_limit_pct: float,
    hrp_weight:     float = 1.0,
) -> dict:
    """
    Single-call risk gate the live trader hits before placing an order.

    Returns:
        {
          'allowed':       bool,
          'risk_pct':      adjusted risk fraction to use,
          'reason':        explanation if not allowed,
          'multipliers':   {'hrp': .., 'dd': ..},
          'exposure_pct':  {ccy: fraction_of_equity},
        }
    """
    halted, why = is_strategy_halted(strategy_name)
    if halted:
        return {'allowed': False, 'risk_pct': 0.0, 'reason': why,
                'multipliers': {}, 'exposure_pct': {}}

    dd_mult  = dd_size_multiplier(realized_dd_pct, daily_loss_limit_pct)
    hrp_mult = float(max(0.0, hrp_weight))
    risk_pct = proposed_risk * hrp_mult * dd_mult

    if risk_pct <= 0:
        return {'allowed': False, 'risk_pct': 0.0,
                'reason': f"risk multipliers collapsed (hrp={hrp_mult}, dd={dd_mult})",
                'multipliers': {'hrp': hrp_mult, 'dd': dd_mult},
                'exposure_pct': {}}

    risk_usd = risk_pct * equity
    ok, reason, exposure_after = currency_exposure_ok(
        pair, risk_usd, open_book, equity,
    )
    if not ok:
        return {'allowed': False, 'risk_pct': 0.0, 'reason': reason,
                'multipliers': {'hrp': hrp_mult, 'dd': dd_mult},
                'exposure_pct': exposure_after}

    emit_heartbeat({'last_strategy': strategy_name, 'last_pair': pair,
                    'risk_pct': risk_pct})
    return {
        'allowed':      True,
        'risk_pct':     risk_pct,
        'reason':       'ok',
        'multipliers':  {'hrp': hrp_mult, 'dd': dd_mult},
        'exposure_pct': exposure_after,
    }
