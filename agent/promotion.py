"""
Auto-promotion daemon — bridges discovered survivors → live trader.

Runs once per agent-loop sweep (hooked from agent.loop.run_loop). Idempotent.

Flow:
    survivor (verdict='survivor' in tested_strategies)
        ─── _gauntlet_ok ───────► live_promoted(mode='SHADOW')
                                     │
                                  live_n ≥ 10  + decay never < 0.3
                                     ▼
                                  LIVE_QUARTER  (0.25× size)
                                     │
                                  live_n ≥ 20 + decay ≥ 0.5
                                     ▼
                                  LIVE_HALF     (0.50× size)
                                     │
                                  live_n ≥ 30 + decay ≥ 0.7
                                     ▼
                                  LIVE_FULL     (1.00× size)
                                     │
                                  decay < 0.3 at any point
                                     ▼
                                  OFF (auto-KILL, one-way)

CLI is inspection + emergency override only — promotion itself is automated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

_proj = str(Path(__file__).resolve().parent.parent)
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from agent import db
from agent.config import (
    MAX_PBO,
    MIN_PSR,
    MIN_TEST_SHARPE,
    MIN_DSR,
    MAX_TEST_DRAWDOWN,
)

log = logging.getLogger(__name__)


# ── Ramp gates ───────────────────────────────────────────────────────────────
# (current_mode, min_live_n, min_decay)  → next_mode
RAMP_RULES = [
    ('SHADOW',       10, None, 'LIVE_QUARTER'),
    ('LIVE_QUARTER', 20, 0.5,  'LIVE_HALF'),
    ('LIVE_HALF',    30, 0.7,  'LIVE_FULL'),
]
# Any mode where decay drops below this gets auto-KILLed.
KILL_DECAY_FLOOR = 0.3
# Backtest gauntlet — drop survivors with this many consecutive losses, since
# a single bad week can blow the prop daily-loss limit.
MAX_CONSECUTIVE_LOSSES = 6


# ── Pair lookup ──────────────────────────────────────────────────────────────

def _pairs_for_session(session: str | None) -> list[str]:
    """Default pair universe for a session — used when a survivor row doesn't
    carry an explicit pair list."""
    try:
        from edge_engine import NY_PAIRS, LONDON_PAIRS, ASIAN_PAIRS
    except Exception:
        return []
    s = (session or '').lower()
    if s == 'ny':
        return list(NY_PAIRS)
    if s == 'london':
        return list(LONDON_PAIRS)
    if s in ('asian', 'asia'):
        return list(ASIAN_PAIRS)
    return list(NY_PAIRS)  # safe default


# ── Hypothesis-id → winning params ───────────────────────────────────────────

def _fetch_winning_params(hypothesis_id: str) -> dict:
    """Return the params dict for one hypothesis_id, read from edge_results.db.
    Empty dict on miss — callers fall back to grid defaults."""
    if not hypothesis_id:
        return {}
    try:
        con = sqlite3.connect(db.EDGE_DB_PATH)
        row = con.execute(
            "SELECT params_json FROM hypotheses WHERE hypothesis_id = ? LIMIT 1",
            (hypothesis_id,),
        ).fetchone()
        con.close()
    except Exception as e:
        log.warning("[promotion] params fetch failed for %s: %s", hypothesis_id, e)
        return {}
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


def _candidate_hypothesis_id(strategy_name: str) -> str:
    """Most recent hypothesis_id for a survivor strategy_name."""
    try:
        con = sqlite3.connect(db.AGENT_DB_PATH)
        row = con.execute(
            """SELECT hypothesis_id FROM tested_strategies
               WHERE strategy_name = ? AND verdict = 'survivor'
               ORDER BY created_at DESC LIMIT 1""",
            (strategy_name,),
        ).fetchone()
        con.close()
        return row[0] if row else ''
    except Exception:
        return ''


# ── Gauntlet ─────────────────────────────────────────────────────────────────

def _gauntlet_ok(row: dict) -> tuple[bool, str]:
    """Statistical safety net before *any* live exposure.

    Returns (ok, reason). If False, reason explains the rejection (one short
    string suitable for logging).
    """
    name = row.get('strategy_name', '?')

    # 1. Re-check the gates we have (PBO/PSR are checked via hypothesis lookup
    #    further down — we don't recompute them here, we just demand they were
    #    populated when the verdict='survivor' row was written.)
    test_sharpe = float(row.get('test_sharpe') or 0)
    if test_sharpe < MIN_TEST_SHARPE:
        return False, f"test_sharpe {test_sharpe:.2f} < {MIN_TEST_SHARPE}"
    dsr = float(row.get('dsr') or 0)
    if dsr < MIN_DSR:
        return False, f"DSR {dsr:.2f} < {MIN_DSR}"
    max_dd = abs(float(row.get('max_dd') or 0))
    if max_dd > MAX_TEST_DRAWDOWN:
        return False, f"max_dd {max_dd:.1%} > {MAX_TEST_DRAWDOWN:.1%}"

    # 2. Pull PBO/PSR from edge_results.db if available — re-check in case
    #    scorer thresholds tightened since the row was written.
    hyp_id = row.get('hypothesis_id') or _candidate_hypothesis_id(name)
    if hyp_id:
        try:
            con = sqlite3.connect(db.EDGE_DB_PATH)
            r = con.execute(
                "SELECT pbo_score, psr FROM hypotheses WHERE hypothesis_id = ? LIMIT 1",
                (hyp_id,),
            ).fetchone()
            con.close()
        except Exception:
            r = None
        if r:
            pbo, psr = r
            if pbo is not None:
                try:
                    if float(pbo) > MAX_PBO:
                        return False, f"PBO {float(pbo):.2f} > {MAX_PBO}"
                except (TypeError, ValueError):
                    pass
            if psr is not None:
                try:
                    if float(psr) < MIN_PSR:
                        return False, f"PSR {float(psr):.2f} < {MIN_PSR}"
                except (TypeError, ValueError):
                    pass

    # 3. Smoke test — module must import. If the generated entry_<name>.py
    #    raises on import we'd silently fail in production; catch it now.
    try:
        from agent import live_bridge
        fn = live_bridge._import_entry_fn(name)
        if fn is None:
            return False, "module import failed"
    except Exception as e:
        return False, f"smoke test failed: {e}"

    # 4. Backtest equity-curve resilience — reject anything with a punishing
    #    losing streak that would breach prop daily-loss limits.
    if hyp_id:
        try:
            trades = db.load_test_trades(hyp_id)
            if trades is not None and not trades.empty:
                from edge_engine import extended_risk_metrics
                m = extended_risk_metrics(trades)
                mcl = int(m.get('max_consec_loss') or 0)
                if mcl > MAX_CONSECUTIVE_LOSSES:
                    return False, f"max_consec_loss {mcl} > {MAX_CONSECUTIVE_LOSSES}"
        except Exception as e:
            log.debug("[promotion] consec_loss check soft-fail for %s: %s", name, e)

    return True, ""


# ── Ramp ─────────────────────────────────────────────────────────────────────

def _decay_for(strategy_name: str, decay_df) -> float | None:
    """Pull this strategy's row out of per_strategy_decay()'s DataFrame."""
    if decay_df is None or getattr(decay_df, 'empty', True):
        return None
    sub = decay_df[decay_df['strategy_name'] == strategy_name]
    if sub.empty:
        return None
    val = sub.iloc[0].get('decay')
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _ramp(current_mode: str, live_n: int, decay: float | None) -> tuple[str, str]:
    """Return (next_mode, reason). next_mode == current_mode when no change."""
    # Dead-stop check first — never ramp a strategy whose live evidence is
    # actively bad. Decay below the floor → auto-KILL (one-way).
    if decay is not None and decay < KILL_DECAY_FLOOR:
        return 'OFF', f"decay {decay:.2f} < {KILL_DECAY_FLOOR} (auto-KILL)"

    # PAUSED/OFF — never auto-progressed. PAUSED must be manually --resumed;
    # OFF is permanent without manual --resume override.
    if current_mode in ('PAUSED', 'OFF'):
        return current_mode, ''

    for mode_from, min_n, min_decay, mode_to in RAMP_RULES:
        if current_mode != mode_from:
            continue
        if live_n < min_n:
            return current_mode, ''
        if min_decay is not None and (decay is None or decay < min_decay):
            return current_mode, ''
        return mode_to, f"live_n={live_n} decay={decay if decay is not None else 'n/a'}"
    return current_mode, ''


def _alert_auto_kill(name: str, prev_mode: str, decay: float | None, why: str) -> None:
    """Best-effort Telegram alert when a strategy is auto-killed by TCA.
    Silently no-op if telegram isn't importable or isn't configured.
    """
    try:
        from agent.telegram import send_message
    except Exception:
        return
    decay_str = f"{decay:.2f}" if decay is not None else "n/a"
    msg = (
        f"<b>⚠ AUTO-KILL</b>\n"
        f"Strategy: <code>{name}</code>\n"
        f"Was: <b>{prev_mode}</b> → <b>OFF</b>\n"
        f"Decay: {decay_str}\n"
        f"Reason: {why}\n"
        f"\n"
        f"To bring back online after review:\n"
        f"<code>python -m agent.promotion --resume {name}</code>"
    )
    try:
        send_message(msg)
    except Exception as e:
        log.debug("[promotion] auto-kill alert send failed: %s", e)


# ── Main entry point ─────────────────────────────────────────────────────────

def run_auto_promotion(*, dry_run: bool = False) -> dict:
    """Run one auto-promotion sweep. Idempotent — safe to call every loop.

    Returns a summary dict: counts of new promotions, ramp transitions, and
    auto-kills. Never raises; logs and continues on any failure.
    """
    out = {'promoted': [], 'ramped': [], 'killed': [], 'rejected': []}

    # ── Step 1: New survivors → SHADOW ────────────────────────────────────
    try:
        candidates = db.get_survivors_since(days_back=30)
    except Exception as e:
        log.error("[promotion] get_survivors_since failed: %s", e)
        candidates = []

    for cand in candidates:
        name = cand.get('strategy_name')
        if not name:
            continue
        ok, reason = _gauntlet_ok(cand)
        if not ok:
            out['rejected'].append((name, reason))
            log.info("[promotion] REJECT %s — %s", name, reason)
            continue

        hyp_id = cand.get('hypothesis_id') or _candidate_hypothesis_id(name)
        params = _fetch_winning_params(hyp_id)
        session = cand.get('session')
        pairs = _pairs_for_session(session)

        if dry_run:
            out['promoted'].append((name, 'SHADOW (dry-run)'))
            log.info("[promotion] DRY-RUN would promote %s → SHADOW", name)
            continue
        try:
            inserted = db.insert_promotion(
                name, mode='SHADOW',
                params=params, session=session, pairs=pairs,
            )
            if inserted:
                out['promoted'].append((name, 'SHADOW'))
                log.info("[promotion] AUTO-PROMOTE %s → SHADOW", name)
        except Exception as e:
            log.error("[promotion] insert_promotion(%s) failed: %s", name, e)

    # ── Step 2: Existing promotions → ramp/kill on live evidence ──────────
    try:
        from agent.tca import per_strategy_decay
        decay_df = per_strategy_decay()
    except Exception as e:
        log.warning("[promotion] per_strategy_decay failed: %s", e)
        decay_df = None

    try:
        active = db.list_promoted_survivors(active_only=True)
    except Exception as e:
        log.error("[promotion] list_promoted_survivors failed: %s", e)
        active = []

    for promo in active:
        name = promo['strategy_name']
        cur_mode = promo['mode']
        live_n = int(promo.get('live_n') or 0)
        decay = _decay_for(name, decay_df)
        new_mode, why = _ramp(cur_mode, live_n, decay)
        if new_mode == cur_mode:
            continue
        if dry_run:
            log.info("[promotion] DRY-RUN would ramp %s: %s → %s (%s)",
                     name, cur_mode, new_mode, why)
            out['ramped' if new_mode != 'OFF' else 'killed'].append((name, cur_mode, new_mode))
            continue
        try:
            db.set_promotion_mode(name, new_mode)
            log.info("[promotion] AUTO-%s %s: %s → %s (%s)",
                     'KILL' if new_mode == 'OFF' else 'RAMP',
                     name, cur_mode, new_mode, why)
            out['killed' if new_mode == 'OFF' else 'ramped'].append(
                (name, cur_mode, new_mode))
            if new_mode == 'OFF':
                _alert_auto_kill(name, cur_mode, decay, why)
        except Exception as e:
            log.error("[promotion] set_promotion_mode(%s, %s) failed: %s",
                      name, new_mode, e)

    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cmd_list() -> int:
    rows = db.list_promoted_survivors(active_only=False)
    if not rows:
        print("(no promotions yet)")
        return 0
    print(f"{'STRATEGY':40s} {'MODE':14s} {'LIVE_N':>7s}  PROMOTED_AT")
    for r in rows:
        print(f"{r['strategy_name']:40s} {r['mode']:14s} "
              f"{int(r.get('live_n') or 0):>7d}  {r.get('promoted_at', '')}")
    return 0


def _cmd_pause(name: str) -> int:
    promo = db.get_promotion(name)
    if not promo:
        print(f"unknown strategy: {name}")
        return 1
    db.set_promotion_mode(name, 'PAUSED', snapshot_prior=True)
    print(f"PAUSED {name} (was {promo['mode']})")
    return 0


def _cmd_resume(name: str) -> int:
    promo = db.get_promotion(name)
    if not promo:
        print(f"unknown strategy: {name}")
        return 1
    prior = promo.get('prior_mode') or 'SHADOW'
    db.set_promotion_mode(name, prior)
    # Lift any prior TCA kill — otherwise run_survivor would still skip
    # this strategy on the very next tick.
    db.clear_live_kill(name)
    print(f"RESUMED {name} -> {prior} (kill cleared)")
    return 0


def _cmd_kill(name: str) -> int:
    promo = db.get_promotion(name)
    if not promo:
        print(f"unknown strategy: {name}")
        return 1
    db.set_promotion_mode(name, 'OFF')
    print(f"KILLED {name}")
    return 0


def _cmd_dry_run() -> int:
    res = run_auto_promotion(dry_run=True)
    print(json.dumps({k: [list(t) if isinstance(t, tuple) else t for t in v]
                      for k, v in res.items()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Inspect/override the auto-promotion daemon. "
                    "Promotion itself is automatic — this CLI is for "
                    "monitoring and emergency intervention only.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--list',    action='store_true', help='show all promotions')
    g.add_argument('--pause',   metavar='NAME',
                   help='take a strategy off the wire without losing history')
    g.add_argument('--resume',  metavar='NAME',
                   help='restore a paused strategy to its prior mode')
    g.add_argument('--kill',    metavar='NAME', help='emergency manual KILL')
    g.add_argument('--dry-run', action='store_true',
                   help='show what auto-promotion would do this sweep')
    args = p.parse_args(argv)

    db.init_db()
    if args.list:    return _cmd_list()
    if args.pause:   return _cmd_pause(args.pause)
    if args.resume:  return _cmd_resume(args.resume)
    if args.kill:    return _cmd_kill(args.kill)
    if args.dry_run: return _cmd_dry_run()
    return 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    sys.exit(main())
