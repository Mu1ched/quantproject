"""
Orchestrates the 18:00 UTC daily top-5 report.
Reads from agent_results.db, generates narrative via Claude, sends via Telegram.
"""

import logging
from datetime import datetime, timezone

import agent.claude_client as claude_client
import agent.db as db
import agent.telegram as telegram
from agent.config import DAILY_REPORT_UTC_HOUR, TOP_PER_BATCH

log = logging.getLogger(__name__)


def send_batch_report(batch_n: int, since_iso: str, total_tested: int):
    """
    Send a short alert with the top TOP_PER_BATCH survivors from this batch window.
    Called every BATCH_REPORT_EVERY strategies tested.
    """
    top_results = db.get_top_results_since(since_iso, n=TOP_PER_BATCH)
    n_survivors = len(top_results)

    log.info("Batch #%d: %d survivors from window since %s", batch_n, n_survivors, since_iso)

    message = telegram.format_batch_report(top_results, batch_n, total_tested, n_survivors)
    success = telegram.send_message(message)

    if not success:
        log.warning("Batch #%d report failed to send", batch_n)


def should_send_report(last_report_time) -> bool:
    """Return True if it is past 18:00 UTC and we have not yet reported today."""
    now = datetime.now(timezone.utc)
    if now.hour < DAILY_REPORT_UTC_HOUR:
        return False
    if last_report_time is None:
        return True
    return last_report_time.date() < now.date()


def send_daily_report(stats: dict) -> datetime:
    """
    Build and send the daily report.
    Returns the datetime of the send (used to prevent double-sends).
    """
    log.info("Building daily report...")
    top_results = db.get_top_results(n=5, days_back=7)
    narrative   = claude_client.generate_daily_report(top_results, stats)

    current_stats = db.get_today_stats()
    current_stats.update(stats)

    message = telegram.format_daily_report(top_results, narrative, current_stats)
    success = telegram.send_message(message)

    if success:
        db.save_report(message)
        log.info("Daily report sent (%d top results)", len(top_results))
    else:
        log.warning("Daily report failed to send via Telegram")

    return datetime.now(timezone.utc)
