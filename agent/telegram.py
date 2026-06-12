"""
Thin Telegram HTTP client. No SDK — just requests.
"""

import logging
import sqlite3
from datetime import datetime, timezone

import requests

from agent.config import AGENT_DB_PATH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)

_MAX_LEN = 4096


def _record_status(ok: bool, error: str = '') -> None:
    """review#P3#2 — persist last send outcome so the GUI can surface it.

    Schema: telegram_status(ts_utc TEXT, ok INTEGER, last_error TEXT).
    Single-row table — UPSERT on a fixed sentinel id.
    """
    try:
        con = sqlite3.connect(AGENT_DB_PATH)
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("""
            CREATE TABLE IF NOT EXISTS telegram_status (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                ts_utc      TEXT,
                ok          INTEGER,
                last_error  TEXT
            )
        """)
        con.execute("""
            INSERT OR REPLACE INTO telegram_status (id, ts_utc, ok, last_error)
            VALUES (1, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(),
              1 if ok else 0,
              (error or '')[:500]))
        con.commit()
        con.close()
    except Exception as e:
        log.debug("telegram_status persist failed: %s", e)


def get_status() -> dict | None:
    """Return last-send status from DB, or None if never recorded."""
    try:
        con = sqlite3.connect(AGENT_DB_PATH)
        row = con.execute(
            "SELECT ts_utc, ok, last_error FROM telegram_status WHERE id=1"
        ).fetchone()
        con.close()
    except Exception:
        return None
    if not row:
        return None
    return {'ts_utc': row[0], 'ok': bool(row[1]), 'last_error': row[2] or ''}


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the configured Telegram chat. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured — add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
        _record_status(False, 'not configured')
        return False

    text = text[:_MAX_LEN]
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=15,
        )
        if not resp.ok:
            log.warning("Telegram send failed (%s): %s", resp.status_code, resp.text[:200])
            _record_status(False, f"{resp.status_code}: {resp.text[:200]}")
            return False
        _record_status(True)
        return True
    except requests.RequestException as e:
        log.warning("Telegram error: %s", e)
        _record_status(False, str(e)[:500])
        return False


def format_batch_report(top_results: list, batch_n: int, total_tested: int, n_survivors: int) -> str:
    """
    Short alert sent every BATCH_REPORT_EVERY strategies.
    Shows top TOP_PER_BATCH survivors from this batch window.
    """
    lines = [
        f"<b>Batch #{batch_n} complete — {total_tested} strategies tested</b>",
        f"Survivors this batch: {n_survivors}",
        "",
    ]

    if top_results:
        lines.append(f"<b>Top {len(top_results)} this batch:</b>")
        for i, r in enumerate(top_results, 1):
            wr_pct   = f"{float(r.get('test_wr', 0)) * 100:.1f}%"
            session  = r.get('session', '?').upper()
            score    = float(r.get('composite_score', 0))
            sharpe   = float(r.get('test_sharpe', 0))
            dsr      = float(r.get('dsr', 0))
            max_dd   = abs(float(r.get('max_dd', 0)))
            n        = int(r.get('n_trades', 0))
            lines.append(
                f"{i}. <b>{r['strategy_name']}</b> [{session}]\n"
                f"   Score: {score:.3f} | Sharpe: {sharpe:.2f} | DSR: {dsr:.2f} | "
                f"WR: {wr_pct} | DD: {max_dd:.1%} | N: {n}"
            )
            if r.get('rationale'):
                lines.append(f"   <i>{r['rationale'][:150]}</i>")
            lines.append("")
    else:
        lines += ["No survivors passed all filters in this batch.", ""]

    lines.append("<i>⚠ Review required before live deployment.</i>")
    return "\n".join(lines)


def format_daily_report(top_results: list, narrative: str, stats: dict) -> str:
    """Build the HTML-formatted daily report message."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [f"<b>Edge Agent Daily Report — {date_str} 18:00 UTC</b>", ""]

    if narrative:
        lines += [narrative, ""]

    if top_results:
        lines.append("<b>Top Strategies (last 7 days):</b>")
        for i, r in enumerate(top_results, 1):
            wr_pct = f"{float(r.get('test_wr', 0)) * 100:.1f}%"
            session = r.get('session', '?').upper()
            score   = float(r.get('composite_score', 0))
            sharpe  = float(r.get('test_sharpe', 0))
            dsr     = float(r.get('dsr', 0))
            n       = int(r.get('n_trades', 0))
            lines.append(
                f"{i}. <b>{r['strategy_name']}</b> [{session}]"
            )
            lines.append(
                f"   Score: {score:.3f} | Sharpe: {sharpe:.2f} | "
                f"DSR: {dsr:.2f} | WR: {wr_pct} | N: {n}"
            )
            rationale = r.get('rationale', '')
            if rationale:
                lines.append(f"   <i>{rationale[:150]}</i>")
            lines.append("")
    else:
        lines += ["No surviving strategies found yet.", ""]

    rounds    = stats.get('rounds_today', 0)
    sweeps    = stats.get('sweeps_today', 0)
    tested    = stats.get('strategies_tested_today', 0)
    survivors = stats.get('survivors_today', 0)

    lines += [
        "<b>Today's activity:</b>",
        f"Rounds: {rounds} | Sweeps: {sweeps} | Tested: {tested} | Survivors: {survivors}",
        "",
        "<i>⚠ Human review required before live deployment.</i>",
    ]
    return "\n".join(lines)
