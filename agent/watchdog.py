"""
Phase 6.5 — heartbeat / dead-man watchdog.

Run this as a separate, supervised process alongside MT5Live. Each iteration
of the live trader writes a heartbeat file via risk_manager.emit_heartbeat();
this watchdog tails the file's mtime. If the trader is unresponsive (mtime
older than HEARTBEAT_STALE_S) the watchdog:
  1. Logs the event
  2. Flattens all open ORB-magic positions on MT5
  3. Sends a Telegram alert if available

Usage:
    python -m agent.watchdog                     # runs forever, polls every 10 s
    python -m agent.watchdog --once              # one-shot probe (for cron)
    python -m agent.watchdog --stale-seconds 90  # override threshold
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Resolve project root so this module can be invoked directly via -m.
_PROJ = str(Path(__file__).parent.parent)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from agent import risk_manager

log = logging.getLogger("watchdog")


HEARTBEAT_STALE_S = 60.0
POLL_S            = 10.0


def _flatten_all_positions(reason: str) -> int:
    """Best-effort flatten of every ORB-magic position. Returns count closed.
    Imports MT5 lazily so the watchdog doesn't hard-require it for unit tests.
    """
    try:
        import MetaTrader5 as mt5  # noqa
        from MT5Live_2 import (    # type: ignore
            close_trade_market, ORB_MAGIC, _base,
        )
    except Exception:
        # Fallback: try the dotted module name (the live file is MT5Live.2.py).
        try:
            import importlib
            import MetaTrader5 as mt5  # noqa
            mod = importlib.import_module('MT5Live.2')
            close_trade_market = mod.close_trade_market
            ORB_MAGIC          = mod.ORB_MAGIC
            _base              = mod._base
        except Exception as e:
            log.error("Cannot import MT5 helpers — flatten skipped: %s", e)
            return 0

    if not mt5.initialize():
        log.error("mt5.initialize() failed — flatten skipped")
        return 0

    closed = 0
    try:
        positions = mt5.positions_get() or []
        for pos in positions:
            if getattr(pos, 'magic', 0) != ORB_MAGIC:
                continue
            try:
                close_trade_market(str(pos.ticket), _base(pos.symbol),
                                   f"watchdog:{reason}")
                closed += 1
            except Exception as e:
                log.error("Flatten failed for ticket %s: %s", pos.ticket, e)
    finally:
        mt5.shutdown()
    return closed


def _alert(message: str):
    try:
        from MT5Live_2 import send_telegram  # type: ignore
        send_telegram(message)
        return
    except Exception:
        pass
    try:
        from agent import telegram as _tg
        _tg.send_message(message)
    except Exception as e:
        log.debug("alert dispatch failed: %s", e)


def probe(stale_seconds: float = HEARTBEAT_STALE_S) -> bool:
    """One probe. Returns True if everything healthy, False if it tripped."""
    age = risk_manager.heartbeat_age_seconds()
    if age <= stale_seconds:
        log.debug("heartbeat fresh (%.1fs)", age)
        return True
    msg = (f"DEAD-MAN: heartbeat stale ({age:.0f}s > {stale_seconds:.0f}s) — "
           f"flattening all positions")
    log.warning(msg)
    _alert(msg)
    closed = _flatten_all_positions(reason=f"stale_{int(age)}s")
    log.warning("Watchdog flattened %d positions", closed)
    _alert(f"Watchdog flattened {closed} positions")
    return False


def run(stale_seconds: float = HEARTBEAT_STALE_S, poll: float = POLL_S):
    log.info("watchdog up — stale_threshold=%.0fs poll=%.0fs", stale_seconds, poll)
    last_tripped = False
    while True:
        try:
            healthy = probe(stale_seconds)
            if healthy and last_tripped:
                _alert("Watchdog: heartbeat recovered.")
                last_tripped = False
            elif not healthy:
                last_tripped = True
        except Exception as e:
            log.error("watchdog iteration failed: %s", e)
        time.sleep(poll)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    p = argparse.ArgumentParser()
    p.add_argument('--stale-seconds', type=float, default=HEARTBEAT_STALE_S)
    p.add_argument('--poll',          type=float, default=POLL_S)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    if args.once:
        probe(args.stale_seconds)
    else:
        run(args.stale_seconds, args.poll)


if __name__ == '__main__':
    main()
