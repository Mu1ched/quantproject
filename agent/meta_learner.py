"""
Meta-learner: closes the feedback loop between backtest results and generation.

Every META_UPDATE_EVERY strategies tested, it:
  1. Extracts which features appear in surviving strategy code
  2. Computes which gates are killing the most strategies
  3. Identifies best-performing sessions, tp_r/sl_r ranges, feature combinations
  4. Asks Claude to synthesise all of this into a compact guidance block
  5. Stores the guidance in agent_results.db
  6. That guidance is injected into every subsequent generation prompt

Effect: the generator is continuously taught by its own backtest results.
Without this, Claude generates in the dark. With it, each generation round
is steered toward the productive region of hypothesis space.
"""

import json
import logging
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_proj = str(Path(__file__).parent.parent)
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import anthropic

from agent.config import (
    AGENT_DB_PATH,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    META_UPDATE_EVERY,
    META_MAX_GUIDANCE_LEN,
)

log = logging.getLogger(__name__)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_used(code: str) -> list:
    """
    Parse generated code to find which row features it accesses.
    Matches: getattr(row, 'feature_name', ...) and row.feature_name patterns.
    """
    getattr_matches = re.findall(r"getattr\(row,\s*['\"](\w+)['\"]", code)
    dot_matches     = re.findall(r"\brow\.([a-z_][a-z0-9_]+)\b", code)
    # Filter out non-feature attributes (hour, minute, high, low, etc. are real features)
    all_features = list(set(getattr_matches + dot_matches))
    return all_features


# ── DB helpers ────────────────────────────────────────────────────────────────

def _init_meta_table():
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta_knowledge (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT,
            n_tested        INTEGER,
            n_survivors     INTEGER,
            stats_json      TEXT,
            guidance_text   TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS rejection_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name   TEXT,
            session         TEXT,
            gate            TEXT,
            detail          TEXT,
            created_at      TEXT
        )
    """)
    # Phase 5 — junction table mapping each rejection to the features it used.
    # Aggregating COUNT(*) by feature gives a "saturation" signal: features that
    # appear far more often in failures than in survivors are being over-fitted.
    con.execute("""
        CREATE TABLE IF NOT EXISTS feature_rejections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            rejection_id  INTEGER NOT NULL,
            feature       TEXT    NOT NULL,
            FOREIGN KEY (rejection_id) REFERENCES rejection_log(id)
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_rejections_feature "
        "ON feature_rejections(feature)"
    )
    # Migrate existing tested_strategies table
    for col_sql in [
        "ALTER TABLE tested_strategies ADD COLUMN features_json TEXT",
        "ALTER TABLE tested_strategies ADD COLUMN best_params_json TEXT",
        "ALTER TABLE tested_strategies ADD COLUMN rejection_gate TEXT",
    ]:
        try:
            con.execute(col_sql)
        except Exception:
            pass
    con.commit()
    con.close()


def record_rejection(strategy_name: str, session: str, gate: str, detail: str):
    """Record which gate killed a strategy (called from loop.py).

    Phase 5 — also fan out the strategy's feature list into the
    `feature_rejections` junction table so the meta-learner can detect
    features that consistently appear in failures (saturation signal).
    """
    _init_meta_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    cur = con.execute("""
        INSERT INTO rejection_log (strategy_name, session, gate, detail, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (strategy_name, session, gate, detail, datetime.now(timezone.utc).isoformat()))
    rejection_id = cur.lastrowid
    con.execute("""
        UPDATE tested_strategies SET rejection_gate = ? WHERE strategy_name = ?
    """, (gate, strategy_name))

    # Pull the strategy's stored code (recorded at record_pending) and fan out
    # the features it used. Skip silently if we can't find code — the rejection
    # row still records the gate, which is the primary signal.
    try:
        row = con.execute(
            "SELECT code FROM tested_strategies WHERE strategy_name = ? LIMIT 1",
            (strategy_name,),
        ).fetchone()
        code = row[0] if row else None
        if code:
            features = extract_features_used(code)
            if features:
                con.executemany(
                    "INSERT INTO feature_rejections (rejection_id, feature) VALUES (?, ?)",
                    [(rejection_id, f) for f in features],
                )
    except Exception as e:
        log.debug("feature_rejections fan-out failed for '%s': %s", strategy_name, e)

    con.commit()
    con.close()


def get_saturated_features(min_rejections: int = 5, top_n: int = 10) -> list:
    """
    Return the top-N features that appear most often in rejected strategies,
    expressed as (feature, percent_of_rejections_involving_feature).

    Used by the generator (claude_client) to steer away from features that
    are being over-used without producing edge. We compute the percent over
    total rejection rows so the number is comparable across runs as the DB
    grows. Features appearing in fewer than min_rejections rejections are
    excluded as noise.
    """
    _init_meta_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    try:
        total_rejections = con.execute(
            "SELECT COUNT(*) FROM rejection_log"
        ).fetchone()[0] or 0
        if total_rejections == 0:
            return []
        rows = con.execute("""
            SELECT feature, COUNT(*) AS n
            FROM feature_rejections
            GROUP BY feature
            HAVING n >= ?
            ORDER BY n DESC
            LIMIT ?
        """, (int(min_rejections), int(top_n))).fetchall()
    finally:
        con.close()

    return [(f, 100.0 * int(n) / total_rejections) for f, n in rows]


def update_survivor_metadata(strategy_name: str, features: list, best_params: dict):
    """Store feature list and best params for a confirmed survivor."""
    _init_meta_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        UPDATE tested_strategies
        SET features_json = ?, best_params_json = ?
        WHERE strategy_name = ?
    """, (json.dumps(features), json.dumps(best_params), strategy_name))
    con.commit()
    con.close()


def get_tested_count() -> int:
    con = sqlite3.connect(AGENT_DB_PATH)
    n   = con.execute("SELECT COUNT(*) FROM tested_strategies").fetchone()[0]
    con.close()
    return int(n)


def get_current_guidance() -> str:
    """Return the most recent synthesised guidance text, or empty string."""
    _init_meta_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute("""
        SELECT guidance_text FROM meta_knowledge
        ORDER BY id DESC LIMIT 1
    """).fetchone()
    con.close()
    return row[0] if row else ""


def get_last_update_n() -> int:
    """Return n_tested at the time of the last meta-update."""
    _init_meta_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute("""
        SELECT n_tested FROM meta_knowledge ORDER BY id DESC LIMIT 1
    """).fetchone()
    con.close()
    return int(row[0]) if row else 0


def should_update() -> bool:
    """Return True if enough new strategies have been tested since last update."""
    current = get_tested_count()
    last    = get_last_update_n()
    return (current - last) >= META_UPDATE_EVERY


# ── Statistics computation ────────────────────────────────────────────────────

def _compute_stats() -> dict:
    """Aggregate statistics from the DB for the meta-learner."""
    _init_meta_table()
    con = sqlite3.connect(AGENT_DB_PATH)

    # All tested strategies
    tested = con.execute("""
        SELECT strategy_name, session, verdict, features_json,
               best_params_json, rejection_gate, test_sharpe, composite_score,
               behaviour_type
        FROM tested_strategies
    """).fetchall()

    # Rejection log
    rejections = con.execute("""
        SELECT gate, COUNT(*) as n FROM rejection_log GROUP BY gate ORDER BY n DESC
    """).fetchall()

    # Survivors
    survivors = con.execute("""
        SELECT strategy_name, session, features_json, best_params_json,
               test_sharpe, composite_score, rationale, created_at
        FROM tested_strategies WHERE verdict = 'survivor'
        ORDER BY composite_score DESC
    """).fetchall()

    con.close()

    n_tested    = len(tested)
    n_survivors = len(survivors)

    # Gate failure distribution
    gate_counts = {g: int(n) for g, n in rejections}

    # Session survival rates
    session_total    = Counter()
    session_survived = Counter()
    for row in tested:
        session = row[1] or 'unknown'
        session_total[session] += 1
        if row[2] == 'survivor':
            session_survived[session] += 1

    session_rates = {
        s: round(session_survived[s] / max(session_total[s], 1) * 100, 1)
        for s in session_total
    }

    # Behaviour type survival rates
    bt_total    = Counter()
    bt_survived = Counter()
    for row in tested:
        bt = row[8] or 'unknown'
        bt_total[bt] += 1
        if row[2] == 'survivor':
            bt_survived[bt] += 1

    behaviour_rates = {
        bt: round(bt_survived[bt] / max(bt_total[bt], 1) * 100, 1)
        for bt in bt_total
    }

    # Feature frequency in survivors
    feature_counter = Counter()
    for row in survivors:
        try:
            feats = json.loads(row[2] or '[]')
            feature_counter.update(feats)
        except Exception:
            pass

    top_features = feature_counter.most_common(12)

    # Best param ranges from survivors
    tp_r_vals, sl_r_vals = [], []
    for row in survivors:
        try:
            params = json.loads(row[3] or '{}')
            if 'tp_r' in params:
                tp_r_vals.append(float(params['tp_r']))
            if 'sl_r' in params:
                sl_r_vals.append(float(params['sl_r']))
        except Exception:
            pass

    def _summarise_vals(vals):
        if not vals:
            return None
        vals.sort()
        n = len(vals)
        return {
            'median': round(vals[n // 2], 2),
            'p25':    round(vals[n // 4], 2),
            'p75':    round(vals[min(3 * n // 4, n - 1)], 2),
        }

    # Top survivor rationales
    top_rationales = [
        {'name': r[0], 'session': r[1], 'sharpe': round(float(r[4] or 0), 2),
         'rationale': r[6] or ''}
        for r in survivors[:8]
    ]

    return {
        'n_tested':        n_tested,
        'n_survivors':     n_survivors,
        'survival_rate':   round(n_survivors / max(n_tested, 1) * 100, 1),
        'gate_failures':   gate_counts,
        'session_rates':   session_rates,
        'behaviour_rates': behaviour_rates,
        'behaviour_counts': dict(bt_total),
        'top_features':    top_features,
        'tp_r_summary':    _summarise_vals(tp_r_vals),
        'sl_r_summary':    _summarise_vals(sl_r_vals),
        'top_survivors':   top_rationales,
    }


# ── Guidance synthesis ────────────────────────────────────────────────────────

def _format_stats_for_claude(stats: dict) -> str:
    lines = [
        f"SYSTEM STATS: {stats['n_tested']} tested, "
        f"{stats['n_survivors']} survivors "
        f"({stats['survival_rate']}% survival rate)",
        "",
        "GATE FAILURE BREAKDOWN (what's killing strategies):",
    ]
    for gate, n in sorted(stats['gate_failures'].items(), key=lambda x: -x[1])[:6]:
        lines.append(f"  {gate}: {n} rejections")

    lines.append("")
    lines.append("SESSION SURVIVAL RATES:")
    for session, rate in sorted(stats['session_rates'].items(), key=lambda x: -x[1]):
        lines.append(f"  {session}: {rate}%")

    brates  = stats.get('behaviour_rates', {})
    bcounts = stats.get('behaviour_counts', {})
    if brates:
        lines.append("")
        lines.append("BEHAVIOUR TYPE SURVIVAL RATES:")
        for bt, rate in sorted(brates.items(), key=lambda x: -x[1]):
            n = bcounts.get(bt, 0)
            lines.append(f"  {bt}: {rate}% ({n} tested)")

    lines.append("")
    lines.append("FEATURES MOST COMMON IN SURVIVORS:")
    for feat, count in stats['top_features'][:8]:
        lines.append(f"  {feat}: {count} survivors")

    tp = stats.get('tp_r_summary')
    sl = stats.get('sl_r_summary')
    if tp:
        lines.append(f"\nBEST tp_r RANGE: median={tp['median']}, IQR [{tp['p25']}–{tp['p75']}]")
    if sl:
        lines.append(f"BEST sl_r RANGE: median={sl['median']}, IQR [{sl['p25']}–{sl['p75']}]")

    if stats['top_survivors']:
        lines.append("\nTOP SURVIVOR RATIONALES:")
        for r in stats['top_survivors'][:5]:
            lines.append(
                f"  [{r['session'].upper()}] Sharpe={r['sharpe']:.2f} — {r['rationale'][:100]}"
            )

    return "\n".join(lines)


def _synthesise_guidance(stats: dict) -> str:
    """Ask Claude to turn the stats into actionable generation guidance."""
    stats_text = _format_stats_for_claude(stats)

    # Append live execution feedback if any has been ingested
    try:
        from agent import live_ingest
        live_block = live_ingest.format_for_guidance()
    except Exception:
        live_block = ""
    if live_block:
        stats_text = f"{stats_text}\n\n{live_block}"

    prompt = (
        f"You are analysing results from an automated forex strategy backtesting system "
        f"that applies 11 rigorous filters. Based on the statistics below, write concise "
        f"ACTIONABLE GUIDANCE for the next round of strategy generation. "
        f"Be specific — name features, ranges, conditions. Under {META_MAX_GUIDANCE_LEN} chars total.\n\n"
        f"Format as:\n"
        f"PURSUE: [2-3 specific feature/condition combos worth exploring]\n"
        f"AVOID: [2-3 patterns that consistently fail specific gates]\n"
        f"PARAMS: [tp_r and sl_r guidance based on survivor distribution]\n"
        f"UNEXPLORED: [1-2 directions not yet tried]\n"
        f"UNDEREXPLORED_BEHAVIOURS: [behaviour types with 0 survivors or not yet tested — "
        f"suggest a fresh angle for 1-2 of them]\n"
        f"LIVE_DECAY: [if live data is present, name pairs the model overstates and tell "
        f"the generator to favour pairs with decay >= 0.9; if no live data, say 'no live data yet']\n\n"
        f"{stats_text}"
    )

    # Route through claude_client so the call respects the budget cap and
    # records spend in the same api_spend table as hypothesis generation.
    from agent import claude_client
    if not claude_client._budget_allows():
        return ""
    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 500,
            messages   = [{"role": "user", "content": prompt}],
        )
        try:
            claude_client._record_spend(getattr(response, "model", "") or CLAUDE_MODEL,
                                        getattr(response, "usage", None))
        except Exception as e:
            log.debug("spend recording failed: %s", e)
        return response.content[0].text.strip()
    except Exception as e:
        log.warning("Meta-learner Claude call failed: %s", e)
        return ""


def _store_guidance(stats: dict, guidance: str):
    _init_meta_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        INSERT INTO meta_knowledge (created_at, n_tested, n_survivors, stats_json, guidance_text)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        stats['n_tested'],
        stats['n_survivors'],
        json.dumps(stats),
        guidance,
    ))
    con.commit()
    con.close()


# ── Public entry point ─────────────────────────────────────────────────────────

def update() -> str:
    """
    Run a full meta-learning update cycle.
    Computes stats, asks Claude to synthesise guidance, stores it.
    Returns the new guidance text.
    """
    log.info("Meta-learner: computing stats...")
    stats    = _compute_stats()
    log.info(
        "Meta-learner: %d tested, %d survivors (%.1f%%), gate kills: %s",
        stats['n_tested'], stats['n_survivors'], stats['survival_rate'],
        dict(list(stats['gate_failures'].items())[:3]),
    )
    guidance = _synthesise_guidance(stats)
    if guidance:
        _store_guidance(stats, guidance)
        log.info("Meta-learner: guidance updated (%d chars)", len(guidance))
    return guidance
