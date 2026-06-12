"""
Entry point for the autonomous edge-discovery agent.

Usage:
    python -m agent.main                  # Run the continuous loop
    python -m agent.main --dry-run        # Generate one hypothesis and print it (no backtest)
    python -m agent.main --report-now     # Send today's report immediately and exit
    python -m agent.main --round-once     # Run exactly one round then exit
    python -m agent.main --analyse        # Deep-dive analysis on recent survivors
    python -m agent.main --analyse --analyse-days 30 --analyse-top 10 --analyse-html

Run from the quantproject directory:
    cd C:/Users/malac/Downloads/quantproject
    python -m agent.main
"""

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows defaults to cp1252 which can't encode
# the Unicode arrows/dashes that edge_engine.py uses in its progress prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

_proj = str(Path(__file__).parent.parent)
if _proj not in sys.path:
    sys.path.insert(0, _proj)


def _configure_logging():
    from agent.config import LOG_PATH
    fmt      = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(str(LOG_PATH), maxBytes=10_000_000, backupCount=3,
                                 encoding='utf-8')
        )
    except Exception as e:
        print(f"Warning: could not open log file: {e}")
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    # Quieten noisy libraries
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Phase 8 — scrub secrets (API keys, bot tokens, .env dumps) from every record
    try:
        from agent.log_filter import install_global_redaction
        install_global_redaction()
    except Exception:
        pass


def _check_env():
    from agent.config import ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Add them to the .env file in the project root:\n"
            + "\n".join(f"  {k}=your_value" for k in missing)
        )


def _dry_run():
    import agent.session_router as sr
    import agent.claude_client as cc

    session, pairs, exit_hour = sr.get_current_session()
    print(f"\nDRY RUN — generating 1 hypothesis for {session.upper()} session")
    print(f"Pairs: {', '.join(pairs)}\n")

    hypotheses = cc.generate_hypotheses(
        session=session, pairs=pairs,
        proven_conditions=[], top_results=[], n=1,
    )
    if hypotheses:
        h = hypotheses[0]
        print(f"Function name: entry_{h['function_name']}")
        print(f"Rationale: {h['rationale']}")
        extra = h.get('additional_params') or {}
        if extra:
            print(f"Extra params: {extra}")
        print(f"\n--- Generated code ---\n{h['code']}\n---")
    else:
        print("No hypothesis returned from Claude.")


def _report_now():
    import agent.db as db
    import agent.reporter as rpt
    db.init_db()
    rpt.send_daily_report(stats={})
    print("Report sent.")


def _round_once():
    import agent.db as db
    import agent.loop as loop
    db.init_db()
    loop._load_data()
    summary = loop.run_one_round(1)
    print(f"\nRound complete: {summary}")


def _analyse(days: int, top, strategy, out, html: bool):
    import agent.db as db
    import agent.analyse as analyse
    db.init_db()
    n = analyse.run(days=days, top=top, strategy_name=strategy,
                    out_root=out, emit_html=html)
    print(f"\nAnalysed {n} survivor(s).")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Edge Discovery Agent")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Generate one hypothesis and print it (no backtest)")
    parser.add_argument("--report-now", action="store_true",
                        help="Send daily report immediately and exit")
    parser.add_argument("--round-once", action="store_true",
                        help="Run one full round then exit")
    parser.add_argument("--analyse",    action="store_true",
                        help="Deep-dive analyse recent survivors and exit")
    parser.add_argument("--analyse-days", type=int, default=7,
                        help="Look-back window for --analyse (default 7)")
    parser.add_argument("--analyse-top", type=int, default=None,
                        help="Only analyse top-N by composite_score")
    parser.add_argument("--analyse-strategy", default=None,
                        help="Analyse one specific strategy by name")
    parser.add_argument("--analyse-out", default=None,
                        help="Output root for reports (default agent/reports/)")
    parser.add_argument("--analyse-html", action="store_true",
                        help="Also render HTML alongside markdown")
    args = parser.parse_args()

    _configure_logging()

    try:
        _check_env()
    except EnvironmentError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    import agent.db as db
    db.init_db()

    if args.dry_run:
        _dry_run()
        return

    if args.report_now:
        _report_now()
        return

    if args.round_once:
        _round_once()
        return

    if args.analyse:
        from pathlib import Path as _P
        _analyse(
            days     = args.analyse_days,
            top      = args.analyse_top,
            strategy = args.analyse_strategy,
            out      = _P(args.analyse_out) if args.analyse_out else None,
            html     = args.analyse_html,
        )
        return

    import agent.loop as loop
    loop.run_loop()


if __name__ == "__main__":
    main()
