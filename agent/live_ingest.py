"""
Live trade ingest — closes the loop between MT5 live execution and backtest research.

Reads two CSVs maintained by MT5live.2.py:
  • live_trade_log.csv      — per-trade PnL on the live broker
  • execution_quality.csv   — fill slippage and observed spread on every fill

Writes both into agent_results.db (live_trades, live_executions tables) and
computes a per-pair execution-quality summary that the meta-learner can surface
to Claude as guidance ("EUR_USD live decay 0.78 vs backtest — model overstates
edge by 22%, prefer pairs with decay >= 0.9").

Designed to be cheap and idempotent: re-ingests the full CSV each run, replacing
the existing rows. CSV is append-only so we just rebuild from scratch.
"""

import csv
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agent.config import (
    AGENT_DB_PATH,
    LIVE_LOG_DIR,
    LIVE_TRADE_CSV,
    LIVE_EXEC_CSV,
    LIVE_MIN_TRADES_PER_PAIR,
    LIVE_DECAY_DEFAULT,
)

log = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

def _init_tables():
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT,
            pair         TEXT,
            direction    TEXT,
            entry        REAL,
            exit_price   REAL,
            pnl          REAL,
            sl           REAL,
            tp           REAL,
            exit_reason  TEXT,
            range_pips   REAL,
            be_set       INTEGER,
            pl_set       INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_executions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT,
            time          TEXT,
            pair          TEXT,
            direction     TEXT,
            expected_fill REAL,
            actual_fill   REAL,
            slippage_pips REAL,
            spread_pips   REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_decay (
            pair             TEXT PRIMARY KEY,
            n_trades         INTEGER,
            win_rate         REAL,
            avg_slippage     REAL,
            avg_spread       REAL,
            decay_factor     REAL,
            updated_at       TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_live_trades_pair ON live_trades(pair)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_live_exec_pair   ON live_executions(pair)")
    con.commit()
    con.close()


# ── CSV readers ───────────────────────────────────────────────────────────────

def _f(v, default=None):
    try:
        return float(v) if v not in ('', None) else default
    except (ValueError, TypeError):
        return default


def _i(v, default=0):
    s = str(v).strip().lower()
    if s in ('true', '1', 'yes'):
        return 1
    if s in ('false', '0', 'no', ''):
        return 0
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _ingest_trades(csv_path: Path) -> int:
    if not csv_path.exists():
        log.info("Live ingest: no %s found — skipping trades", csv_path.name)
        return 0

    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append((
                r.get('date', ''),
                r.get('pair', ''),
                r.get('direction', ''),
                _f(r.get('entry')),
                _f(r.get('exit_price')),
                _f(r.get('pnl')),
                _f(r.get('sl')),
                _f(r.get('tp')),
                r.get('exit_reason', ''),
                _f(r.get('range_size_pips')),
                _i(r.get('breakeven_set')),
                _i(r.get('profit_lock_set')),
            ))

    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("DELETE FROM live_trades")  # rebuild from CSV (source of truth)
    con.executemany("""
        INSERT INTO live_trades
            (date, pair, direction, entry, exit_price, pnl, sl, tp,
             exit_reason, range_pips, be_set, pl_set)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()
    con.close()
    return len(rows)


def _ingest_executions(csv_path: Path) -> int:
    if not csv_path.exists():
        log.info("Live ingest: no %s found — skipping executions", csv_path.name)
        return 0

    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append((
                r.get('date', ''),
                r.get('time', ''),
                r.get('pair', ''),
                r.get('direction', ''),
                _f(r.get('expected_fill')),
                _f(r.get('actual_fill')),
                _f(r.get('slippage_pips')),
                _f(r.get('spread_pips')),
            ))

    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("DELETE FROM live_executions")
    con.executemany("""
        INSERT INTO live_executions
            (date, time, pair, direction, expected_fill, actual_fill,
             slippage_pips, spread_pips)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()
    con.close()
    return len(rows)


# ── Decay computation ─────────────────────────────────────────────────────────

def _compute_decay():
    """
    For each pair, compute an empirical 'live decay factor' to compare against
    backtested expectations. Stored in live_decay table for use by meta_learner.

    Decay factor heuristic (range 0.0–1.5, neutral=1.0):
      • Start at 1.0
      • Subtract excess slippage cost (avg_slip pips × ~0.0001 per pip of edge)
      • Subtract penalty if win rate dropped vs typical 50% mean reversion
      • Floor at 0.3

    A pair with decay 0.7 means live execution is removing roughly 30% of
    backtested edge — Claude should be told to favour pairs with decay >= 0.9.
    """
    con = sqlite3.connect(AGENT_DB_PATH)

    pair_rows = con.execute("""
        SELECT pair, COUNT(*) as n, AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as wr
        FROM live_trades
        WHERE pair != '' AND pnl IS NOT NULL
        GROUP BY pair
    """).fetchall()

    exec_rows = con.execute("""
        SELECT pair, AVG(slippage_pips), AVG(spread_pips)
        FROM live_executions
        WHERE pair != ''
        GROUP BY pair
    """).fetchall()
    exec_map = {p: (s or 0.0, sp or 0.0) for p, s, sp in exec_rows}

    now = datetime.now(timezone.utc).isoformat()
    written = 0
    for pair, n, wr in pair_rows:
        if n < LIVE_MIN_TRADES_PER_PAIR:
            continue
        avg_slip, avg_spread = exec_map.get(pair, (0.0, 0.0))

        # Empirical decay model
        decay = 1.0
        decay -= min(avg_slip * 0.05, 0.30)        # heavy slip → up to -30%
        if wr is not None and wr < 0.40:
            decay -= (0.40 - wr) * 0.5             # win rate cliff penalty
        decay = max(0.30, min(1.30, decay))

        con.execute("""
            INSERT INTO live_decay (pair, n_trades, win_rate, avg_slippage,
                                    avg_spread, decay_factor, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pair) DO UPDATE SET
                n_trades     = excluded.n_trades,
                win_rate     = excluded.win_rate,
                avg_slippage = excluded.avg_slippage,
                avg_spread   = excluded.avg_spread,
                decay_factor = excluded.decay_factor,
                updated_at   = excluded.updated_at
        """, (pair, int(n), float(wr or 0.0), float(avg_slip),
              float(avg_spread), float(decay), now))
        written += 1

    con.commit()
    con.close()
    return written


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_all() -> dict:
    """Run a full ingest cycle. Safe to call repeatedly."""
    _init_tables()

    base = Path(LIVE_LOG_DIR)
    n_trades = _ingest_trades(base / LIVE_TRADE_CSV)
    n_exec   = _ingest_executions(base / LIVE_EXEC_CSV)
    n_decay  = _compute_decay()

    log.info("Live ingest: %d trades, %d executions, %d pairs with decay computed",
             n_trades, n_exec, n_decay)
    return {'trades': n_trades, 'executions': n_exec, 'pairs_with_decay': n_decay}


def get_live_decay_summary() -> dict:
    """
    Return per-pair live decay info for meta-learner / generation guidance.
    Returns empty dict if no live data ingested yet.
    """
    _init_tables()
    con = sqlite3.connect(AGENT_DB_PATH)
    rows = con.execute("""
        SELECT pair, n_trades, win_rate, avg_slippage, avg_spread, decay_factor
        FROM live_decay ORDER BY decay_factor ASC
    """).fetchall()
    con.close()
    return {
        r[0]: {
            'n_trades':     int(r[1]),
            'win_rate':     round(float(r[2]), 3),
            'avg_slippage': round(float(r[3]), 2),
            'avg_spread':   round(float(r[4]), 2),
            'decay_factor': round(float(r[5]), 3),
        }
        for r in rows
    }


def format_for_guidance() -> str:
    """Compact text block suitable for injection into Claude generation prompt."""
    summary = get_live_decay_summary()
    if not summary:
        return ""
    lines = ["LIVE EXECUTION QUALITY (from real broker fills — adjust conviction accordingly):"]
    for pair, d in sorted(summary.items(), key=lambda kv: kv[1]['decay_factor']):
        verdict = (
            "BACKTEST OVERSTATES" if d['decay_factor'] < 0.85
            else "WITHIN TOLERANCE"  if d['decay_factor'] < 1.05
            else "LIVE BETTER THAN MODEL"
        )
        lines.append(
            f"  {pair}: decay={d['decay_factor']:.2f} | wr={d['win_rate']:.1%} | "
            f"slip={d['avg_slippage']:.1f}pip | spread={d['avg_spread']:.1f}pip — {verdict}"
        )
    return "\n".join(lines)
