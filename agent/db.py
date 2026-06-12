"""
Agent-specific database (agent_results.db).

Tracks: generated strategy hashes for deduplication, survivor results with
composite scores, daily round telemetry, and sent reports.

Completely separate from edge_results.db — never modifies the main backtesting DB.
"""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone

from agent.config import AGENT_DB_PATH, EDGE_DB_PATH


def _normalise_code(code: str) -> str:
    """Strip comments and collapse whitespace before hashing (catches rephrased duplicates)."""
    code = re.sub(r'#[^\n]*', '', code)
    code = re.sub(r'[ \t]+', ' ', code)
    code = re.sub(r'\n{2,}', '\n', code)
    return code.strip()


def _code_hash(code: str) -> str:
    return hashlib.sha256(_normalise_code(code).encode()).hexdigest()


def init_db():
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        CREATE TABLE IF NOT EXISTS tested_strategies (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_hash     TEXT UNIQUE,
            strategy_name     TEXT,
            session           TEXT,
            sweep_id          TEXT,
            composite_score   REAL,
            test_sharpe       REAL,
            dsr               REAL,
            test_wr           REAL,
            regime_stable     INTEGER,
            n_trades          INTEGER,
            max_dd            REAL,
            rationale         TEXT,
            behaviour_type    TEXT,
            hypothesis_id     TEXT,
            code              TEXT,
            verdict           TEXT DEFAULT 'pending',
            best_params_json  TEXT,
            created_at        TEXT
        )
    """)
    # Idempotent migration: older DBs predate best_params_json.
    try:
        con.execute("ALTER TABLE tested_strategies ADD COLUMN best_params_json TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS agent_rounds (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            round_n        INTEGER,
            session        TEXT,
            n_generated    INTEGER,
            n_skipped_dup  INTEGER,
            n_swept        INTEGER,
            n_survivors    INTEGER,
            sweep_ids      TEXT,
            created_at     TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT UNIQUE,
            content     TEXT,
            sent_at     TEXT
        )
    """)
    # Phase 1 — kill switch consulted by MT5Live before placing any strategy's
    # order. Written by tca.per_strategy_decay() when it produces a verdict.
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_strategy_kills (
            strategy_name TEXT PRIMARY KEY,
            decay         REAL,
            verdict       TEXT,        -- KILL | REDUCE | HOLD | PROMOTE
            kill_ts       TEXT
        )
    """)
    # Phase 4 — global FDR ledger. Every distinct hypothesis (by code hash)
    # gets one row with its raw p-value plus the BY-adjusted p-value computed
    # over the entire historical population. Lets the agent enforce a cumulative
    # false-discovery budget across all sweeps and rounds.
    con.execute("""
        CREATE TABLE IF NOT EXISTS hypothesis_ledger (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_hash    TEXT UNIQUE,
            strategy_name      TEXT,
            sweep_id           TEXT,
            raw_pval           REAL,
            by_adjusted_pval   REAL,
            cumulative_tested  INTEGER,
            global_sig         INTEGER DEFAULT 0,
            created_at         TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ledger_hash ON hypothesis_ledger(hypothesis_hash)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ledger_sig  ON hypothesis_ledger(global_sig)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_hash   ON tested_strategies(strategy_hash)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_date   ON tested_strategies(created_at)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_verdict ON tested_strategies(verdict)")
    # Auto-promotion bridge — survivors discovered by the agent loop graduate
    # through SHADOW → LIVE_QUARTER → LIVE_HALF → LIVE_FULL based on accumulated
    # live evidence. PAUSED is a manual safety hatch; OFF is auto-KILL (one-way).
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_promoted (
            strategy_name TEXT PRIMARY KEY,
            mode          TEXT NOT NULL,    -- SHADOW|LIVE_QUARTER|LIVE_HALF|LIVE_FULL|PAUSED|OFF
            promoted_at   TEXT,
            last_ramp_at  TEXT,
            live_n        INTEGER DEFAULT 0,
            prior_mode    TEXT,             -- snapshot before PAUSED so --resume restores
            params_json   TEXT,             -- winning params at promotion (slot['strategy_def']['params'])
            session       TEXT,             -- ny|london|asian — which session the strategy targets
            pairs_json    TEXT              -- JSON list of pairs the strategy was validated on
        )
    """)
    # Migration guard for existing live_promoted rows
    for col_sql in [
        "ALTER TABLE live_promoted ADD COLUMN params_json TEXT",
        "ALTER TABLE live_promoted ADD COLUMN session TEXT",
        "ALTER TABLE live_promoted ADD COLUMN pairs_json TEXT",
    ]:
        try:
            con.execute(col_sql)
        except Exception:
            pass
    # Per-strategy live trade ledger. Phase 1 wired tca.per_strategy_decay() to
    # read this table; this is the table that finally populates it.
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name   TEXT NOT NULL,
            pair            TEXT,
            side            TEXT,           -- 'long'|'short'
            ts_open         TEXT,
            ts_close        TEXT,
            entry           REAL,
            exit            REAL,
            sl              REAL,
            tp              REAL,
            pnl_usd         REAL,
            pnl_r           REAL,           -- PnL in R-multiples (entry-to-SL distance)
            slip_open_pip   REAL,
            slip_close_pip  REAL,
            regime          TEXT,
            session         TEXT,
            mode            TEXT,           -- promotion mode at trade time (SHADOW means paper)
            created_at      TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_live_trades_name ON live_trades(strategy_name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_live_trades_open ON live_trades(ts_open)")
    # Migration guards for existing databases
    for col_sql in [
        "ALTER TABLE tested_strategies ADD COLUMN behaviour_type TEXT",
        "ALTER TABLE tested_strategies ADD COLUMN hypothesis_id TEXT",
        "ALTER TABLE tested_strategies ADD COLUMN code TEXT",
    ]:
        try:
            con.execute(col_sql)
        except Exception:
            pass
    con.commit()
    con.close()


def is_duplicate(code: str) -> bool:
    h   = _code_hash(code)
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute(
        "SELECT 1 FROM tested_strategies WHERE strategy_hash = ?", (h,)
    ).fetchone()
    con.close()
    return row is not None


def record_pending(strategy_name: str, code: str, session: str, rationale: str,
                   behaviour_type: str = ""):
    """Reserve the strategy slot before backtesting (also dedup gate)."""
    h   = _code_hash(code)
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        INSERT OR IGNORE INTO tested_strategies
            (strategy_hash, strategy_name, session, rationale, behaviour_type, verdict, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?)
    """, (h, strategy_name, session, rationale, behaviour_type,
          datetime.now(timezone.utc).isoformat()))
    con.commit()
    con.close()


def record_result(
    strategy_name:   str,
    code:            str,
    session:         str,
    sweep_id:        str,
    composite_score: float,
    metrics:         dict,
    rationale:       str = "",
    verdict:         str = "survivor",
    behaviour_type:  str = "",
    hypothesis_id:   str = "",
    best_params:     dict | None = None,
):
    h   = _code_hash(code)
    bp_json = json.dumps(best_params) if best_params else None
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        UPDATE tested_strategies
        SET sweep_id         = ?,
            composite_score  = ?,
            test_sharpe      = ?,
            dsr              = ?,
            test_wr          = ?,
            regime_stable    = ?,
            n_trades         = ?,
            max_dd           = ?,
            rationale        = ?,
            behaviour_type   = ?,
            hypothesis_id    = ?,
            code             = ?,
            verdict          = ?,
            best_params_json = ?
        WHERE strategy_hash = ?
    """, (
        sweep_id, composite_score,
        float(metrics.get('test_sharpe', 0) or 0),
        float(metrics.get('dsr', 0) or 0),
        float(metrics.get('test_wr', 0) or 0),
        int(metrics.get('regime_stable', 0) or 0),
        int(metrics.get('test_n', 0) or 0),
        float(metrics.get('test_max_dd', 0) or 0),
        rationale, behaviour_type, hypothesis_id, code, verdict, bp_json, h,
    ))
    con.commit()
    con.close()


def record_round(
    round_n:      int,
    session:      str,
    n_generated:  int,
    n_skipped:    int,
    n_swept:      int,
    n_survivors:  int,
    sweep_ids:    list,
):
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        INSERT INTO agent_rounds
            (round_n, session, n_generated, n_skipped_dup, n_swept, n_survivors, sweep_ids, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        round_n, session, n_generated, n_skipped, n_swept, n_survivors,
        json.dumps(sweep_ids), datetime.now(timezone.utc).isoformat(),
    ))
    con.commit()
    con.close()


def get_top_results(n: int = 5, days_back: int = 7) -> list:
    con  = sqlite3.connect(AGENT_DB_PATH)
    rows = con.execute(f"""
        SELECT strategy_name, session, test_sharpe, dsr, test_wr,
               regime_stable, n_trades, max_dd, composite_score, rationale, created_at,
               behaviour_type, hypothesis_id, code
        FROM tested_strategies
        WHERE verdict      = 'survivor'
          AND created_at  >= datetime('now', '-{days_back} days')
          AND composite_score IS NOT NULL
        ORDER BY composite_score DESC
        LIMIT ?
    """, (n,)).fetchall()
    con.close()
    cols = [
        'strategy_name', 'session', 'test_sharpe', 'dsr', 'test_wr',
        'regime_stable', 'n_trades', 'max_dd', 'composite_score', 'rationale', 'created_at',
        'behaviour_type', 'hypothesis_id', 'code',
    ]
    return [dict(zip(cols, r)) for r in rows]


def load_sweep_results(sweep_id: str) -> list:
    """Read hypothesis rows for sweep_id from the main edge_results.db."""
    try:
        con  = sqlite3.connect(EDGE_DB_PATH)
        rows = con.execute("""
            SELECT hypothesis_id,
                   test_sharpe, test_wr, test_n, test_max_dd,
                   dsr, regime_stable, bh_sig, p_adj, params_json,
                   train_sharpe, train_n,
                   sharpe_ci_low, sharpe_ci_high,
                   p_raw,
                   COALESCE(by_sig, 0)   AS by_sig,
                   p_adj_by,
                   psr, outlier_ratio, pbo_score
            FROM hypotheses
            WHERE sweep_id = ?
        """, (sweep_id,)).fetchall()
        con.close()
    except Exception:
        return []
    cols = [
        'hypothesis_id',
        'test_sharpe', 'test_wr', 'test_n', 'test_max_dd',
        'dsr', 'regime_stable', 'bh_sig', 'p_adj', 'params_json',
        'train_sharpe', 'train_n',
        'sharpe_ci_low', 'sharpe_ci_high',
        'p_raw', 'by_sig', 'p_adj_by',
        'psr', 'outlier_ratio', 'pbo_score',
    ]
    return [dict(zip(cols, r)) for r in rows]


def load_test_trades(hypothesis_id: str):
    """Load test-split trades for a hypothesis from edge_results.db. Returns DataFrame."""
    try:
        import pandas as pd
        con = sqlite3.connect(EDGE_DB_PATH)
        df  = pd.read_sql(
            "SELECT * FROM trades WHERE hypothesis_id = ? AND split = 'test'",
            con, params=(hypothesis_id,)
        )
        con.close()
        return df
    except Exception:
        import pandas as pd
        return pd.DataFrame()


def get_tested_count_since(since_iso: str) -> int:
    """Count strategies tested (any verdict) since a given ISO timestamp."""
    con = sqlite3.connect(AGENT_DB_PATH)
    n   = con.execute(
        "SELECT COUNT(*) FROM tested_strategies WHERE created_at >= ?", (since_iso,)
    ).fetchone()[0]
    con.close()
    return int(n)


def get_top_results_since(since_iso: str, n: int = 2) -> list:
    """Return the top-n survivors recorded since a given ISO timestamp."""
    con  = sqlite3.connect(AGENT_DB_PATH)
    rows = con.execute("""
        SELECT strategy_name, session, test_sharpe, dsr, test_wr,
               regime_stable, n_trades, max_dd, composite_score, rationale, created_at,
               behaviour_type, hypothesis_id, code
        FROM tested_strategies
        WHERE verdict     = 'survivor'
          AND created_at >= ?
          AND composite_score IS NOT NULL
        ORDER BY composite_score DESC
        LIMIT ?
    """, (since_iso, n)).fetchall()
    con.close()
    cols = ['strategy_name', 'session', 'test_sharpe', 'dsr', 'test_wr',
            'regime_stable', 'n_trades', 'max_dd', 'composite_score', 'rationale', 'created_at',
            'behaviour_type', 'hypothesis_id', 'code']
    return [dict(zip(cols, r)) for r in rows]


def report_sent_today() -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    con   = sqlite3.connect(AGENT_DB_PATH)
    row   = con.execute(
        "SELECT 1 FROM daily_reports WHERE report_date = ?", (today,)
    ).fetchone()
    con.close()
    return row is not None


def save_report(content: str):
    today = datetime.now(timezone.utc).date().isoformat()
    con   = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        INSERT OR REPLACE INTO daily_reports (report_date, content, sent_at)
        VALUES (?, ?, ?)
    """, (today, content, datetime.now(timezone.utc).isoformat()))
    con.commit()
    con.close()


def record_hypothesis_pval(hypothesis_hash: str, strategy_name: str,
                           sweep_id: str, raw_pval: float):
    """Insert a hypothesis into the global FDR ledger with its raw p-value.

    No-op if the same hash is already present (same hypothesis tested twice
    counts once toward the cumulative population).
    """
    if raw_pval is None:
        return
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        INSERT OR IGNORE INTO hypothesis_ledger
            (hypothesis_hash, strategy_name, sweep_id, raw_pval, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (hypothesis_hash, strategy_name, sweep_id, float(raw_pval),
          datetime.now(timezone.utc).isoformat()))
    con.commit()
    con.close()


def recompute_global_by_correction(alpha: float = 0.05) -> dict:
    """Run Benjamini-Yekutieli FDR over every recorded hypothesis.

    BY (vs BH) handles dependent hypotheses — which is the actual reality of
    correlated strategy variants generated by the agent. Returns a summary
    dict so callers can log/inject it.

    Bonferroni was rejected explicitly (at cumulative=10000, alpha drops to
    5e-6 and no hypothesis ever passes again).
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return {'error': 'statsmodels not installed'}

    con = sqlite3.connect(AGENT_DB_PATH)
    rows = con.execute(
        "SELECT id, raw_pval FROM hypothesis_ledger WHERE raw_pval IS NOT NULL"
    ).fetchall()
    if not rows:
        con.close()
        return {'cumulative': 0, 'n_significant': 0}

    ids   = [r[0] for r in rows]
    pvals = [r[1] for r in rows]
    _, p_adj, _, _ = multipletests(pvals, method='fdr_by', alpha=alpha)

    n_sig = 0
    for hid, padj in zip(ids, p_adj):
        sig = 1 if padj < alpha else 0
        n_sig += sig
        con.execute("""
            UPDATE hypothesis_ledger
            SET by_adjusted_pval  = ?,
                global_sig        = ?,
                cumulative_tested = ?
            WHERE id = ?
        """, (float(padj), sig, len(ids), hid))
    con.commit()
    con.close()
    return {'cumulative': len(ids), 'n_significant': n_sig}


def fdr_budget_remaining(alpha: float = 0.05) -> dict:
    """Return cumulative population, count of globally-significant hypotheses,
    and the fraction of FDR budget still available.

    budget_remaining = 1 - (n_significant * alpha) / cumulative
    Drops toward 0 as the agent burns through false-discovery budget by
    accumulating positives. Injected into Claude prompts so the model
    knows how parsimonious it needs to be.
    """
    con = sqlite3.connect(AGENT_DB_PATH)
    n_total = con.execute("SELECT COUNT(*) FROM hypothesis_ledger").fetchone()[0] or 0
    n_sig   = con.execute(
        "SELECT COUNT(*) FROM hypothesis_ledger WHERE global_sig = 1"
    ).fetchone()[0] or 0
    con.close()
    if n_total == 0:
        return {'cumulative': 0, 'n_significant': 0, 'budget_remaining': 1.0}
    burned   = (n_sig * alpha) / max(1, n_total)
    return {
        'cumulative':       int(n_total),
        'n_significant':    int(n_sig),
        'budget_remaining': max(0.0, 1.0 - burned),
    }


def upsert_live_strategy_kill(strategy_name: str, decay: float, verdict: str):
    """Write or update the kill-switch row for a strategy.

    MT5Live should consult get_live_kill(name) before placing any order;
    KILL = skip, REDUCE = halve sizing, HOLD/PROMOTE = full sizing.
    """
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        INSERT INTO live_strategy_kills (strategy_name, decay, verdict, kill_ts)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(strategy_name) DO UPDATE SET
            decay   = excluded.decay,
            verdict = excluded.verdict,
            kill_ts = excluded.kill_ts
    """, (strategy_name, float(decay) if decay is not None else None,
          verdict, datetime.now(timezone.utc).isoformat()))
    con.commit()
    con.close()


def get_live_kill(strategy_name: str) -> dict | None:
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute(
        "SELECT decay, verdict, kill_ts FROM live_strategy_kills WHERE strategy_name = ?",
        (strategy_name,),
    ).fetchone()
    con.close()
    if not row:
        return None
    return {'decay': row[0], 'verdict': row[1], 'kill_ts': row[2]}


def clear_live_kill(strategy_name: str) -> None:
    """Remove the kill-switch row so MT5Live no longer skips this strategy.
    Called by promotion --resume to lift a TCA-imposed KILL after manual review.
    """
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("DELETE FROM live_strategy_kills WHERE strategy_name = ?",
                (strategy_name,))
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Auto-promotion bridge
# ---------------------------------------------------------------------------

_VALID_MODES = {'SHADOW', 'LIVE_QUARTER', 'LIVE_HALF', 'LIVE_FULL', 'PAUSED', 'OFF'}


def insert_promotion(
    strategy_name: str,
    mode: str = 'SHADOW',
    *,
    params: dict | None = None,
    session: str | None = None,
    pairs: list[str] | None = None,
) -> bool:
    """Insert a fresh promotion row. Returns False if name already promoted (any mode).

    `params` is the winning hyperparameter dict from the sweep (becomes
    slot['strategy_def']['params'] when the entry fn runs live). `session` and
    `pairs` constrain when/where the strategy is allowed to fire.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode: {mode}")
    now = datetime.now(timezone.utc).isoformat()
    params_json = json.dumps(params or {})
    pairs_json  = json.dumps(pairs or [])
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    cur = con.execute("""
        INSERT OR IGNORE INTO live_promoted
            (strategy_name, mode, promoted_at, last_ramp_at, live_n,
             params_json, session, pairs_json)
        VALUES (?, ?, ?, ?, 0, ?, ?, ?)
    """, (strategy_name, mode, now, now, params_json, session, pairs_json))
    inserted = cur.rowcount > 0
    con.commit()
    con.close()
    return inserted


def set_promotion_mode(strategy_name: str, mode: str, *, snapshot_prior: bool = False) -> None:
    """Update promotion mode. If snapshot_prior=True (used by PAUSED), the current
    mode is saved into prior_mode so --resume can restore it."""
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode: {mode}")
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    if snapshot_prior:
        con.execute("""
            UPDATE live_promoted
            SET prior_mode   = mode,
                mode         = ?,
                last_ramp_at = ?
            WHERE strategy_name = ?
        """, (mode, now, strategy_name))
    else:
        con.execute("""
            UPDATE live_promoted
            SET mode         = ?,
                last_ramp_at = ?
            WHERE strategy_name = ?
        """, (mode, now, strategy_name))
    con.commit()
    con.close()


_PROMO_COLS = ['strategy_name', 'mode', 'promoted_at', 'last_ramp_at', 'live_n',
               'prior_mode', 'params_json', 'session', 'pairs_json']
_PROMO_SELECT = ', '.join(_PROMO_COLS)


def _promo_row_to_dict(row) -> dict:
    d = dict(zip(_PROMO_COLS, row))
    # Decode JSON columns to native types for callers
    try:
        d['params'] = json.loads(d.get('params_json') or '{}')
    except Exception:
        d['params'] = {}
    try:
        d['pairs'] = json.loads(d.get('pairs_json') or '[]')
    except Exception:
        d['pairs'] = []
    return d


def get_promotion(strategy_name: str) -> dict | None:
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute(
        f"SELECT {_PROMO_SELECT} FROM live_promoted WHERE strategy_name = ?",
        (strategy_name,),
    ).fetchone()
    con.close()
    return _promo_row_to_dict(row) if row else None


def list_promoted_survivors(active_only: bool = True) -> list[dict]:
    """Return all promoted strategies. active_only=True excludes OFF and PAUSED."""
    con = sqlite3.connect(AGENT_DB_PATH)
    if active_only:
        rows = con.execute(f"""
            SELECT {_PROMO_SELECT} FROM live_promoted
            WHERE mode NOT IN ('OFF', 'PAUSED')
            ORDER BY promoted_at
        """).fetchall()
    else:
        rows = con.execute(f"""
            SELECT {_PROMO_SELECT} FROM live_promoted
            ORDER BY promoted_at
        """).fetchall()
    con.close()
    return [_promo_row_to_dict(r) for r in rows]


def record_live_trade(
    strategy_name: str,
    pair:          str,
    side:          str,
    ts_open:       str,
    ts_close:      str | None,
    entry:         float,
    exit_price:    float | None,
    sl:            float,
    tp:            float | None,
    pnl_usd:       float,
    pnl_r:         float | None = None,
    slip_open_pip: float | None = None,
    slip_close_pip: float | None = None,
    regime:        str | None = None,
    session:       str | None = None,
    mode:          str | None = None,
) -> int:
    """Insert a live trade row and bump live_promoted.live_n. Returns row id.

    A SHADOW-mode trade is still recorded so `per_strategy_decay()` has data to
    chew on — but `mode='SHADOW'` lets analytics filter to real-money trades.
    """
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    cur = con.execute("""
        INSERT INTO live_trades
            (strategy_name, pair, side, ts_open, ts_close, entry, exit, sl, tp,
             pnl_usd, pnl_r, slip_open_pip, slip_close_pip, regime, session, mode, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (strategy_name, pair, side, ts_open, ts_close,
          entry, exit_price, sl, tp,
          pnl_usd, pnl_r, slip_open_pip, slip_close_pip,
          regime, session, mode, now))
    row_id = cur.lastrowid
    # Only count completed trades (with a close) toward live_n — open positions
    # don't yet count as "live evidence" for ramp gating.
    if ts_close is not None:
        con.execute("""
            UPDATE live_promoted
            SET live_n = live_n + 1
            WHERE strategy_name = ?
        """, (strategy_name,))
    con.commit()
    con.close()
    return row_id


def get_survivors_since(days_back: int = 30) -> list[dict]:
    """Survivor candidates the auto-promoter should consider — joins on
    live_promoted to surface only those not yet promoted."""
    con = sqlite3.connect(AGENT_DB_PATH)
    rows = con.execute(f"""
        SELECT t.strategy_name, t.session, t.test_sharpe, t.dsr, t.test_wr,
               t.regime_stable, t.n_trades, t.max_dd, t.composite_score,
               t.code, t.created_at
        FROM tested_strategies t
        LEFT JOIN live_promoted p ON p.strategy_name = t.strategy_name
        WHERE t.verdict = 'survivor'
          AND t.created_at >= datetime('now', '-{int(days_back)} days')
          AND p.strategy_name IS NULL
        ORDER BY t.composite_score DESC
    """).fetchall()
    con.close()
    cols = ['strategy_name', 'session', 'test_sharpe', 'dsr', 'test_wr',
            'regime_stable', 'n_trades', 'max_dd', 'composite_score', 'code', 'created_at']
    return [dict(zip(cols, r)) for r in rows]


def get_today_stats() -> dict:
    """Return counts of rounds, sweeps, and survivors created today."""
    today = datetime.now(timezone.utc).date().isoformat()
    con   = sqlite3.connect(AGENT_DB_PATH)

    rounds = con.execute(
        "SELECT COUNT(*) FROM agent_rounds WHERE created_at >= ?", (today,)
    ).fetchone()[0]
    swept = con.execute(
        "SELECT COALESCE(SUM(n_swept), 0) FROM agent_rounds WHERE created_at >= ?", (today,)
    ).fetchone()[0]
    survivors = con.execute(
        "SELECT COUNT(*) FROM tested_strategies WHERE verdict = 'survivor' AND created_at >= ?",
        (today,)
    ).fetchone()[0]
    tested = con.execute(
        "SELECT COUNT(*) FROM tested_strategies WHERE created_at >= ?", (today,)
    ).fetchone()[0]

    con.close()
    return {
        'rounds_today':             int(rounds),
        'sweeps_today':             int(swept),
        'survivors_today':          int(survivors),
        'strategies_tested_today':  int(tested),
    }
