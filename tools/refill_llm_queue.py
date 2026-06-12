"""
Bulk LLM hypothesis refill — amortises web_search cost across many strategies.

Calls Claude once per session with web_search enabled, asks for N diverse
fully-coded strategies grounded in retail-FX literature, and writes the
combined result to tools/llm_hypotheses_queue.json. The agent loop drains
that queue during LLM-path rounds instead of calling Claude per round.

Cost economics:
  - Per-round LLM call (current): ~$0.025 for 2 strategies = $0.0125/strategy
  - This script (3 sessions × N=10): ~$0.15-0.25 for 30 strategies = ~$0.007/strategy
  - Roughly 2-4x cheaper per strategy, depending on actual token usage.

Run manually whenever the queue runs low, or on a cron (e.g. once a day).

Usage:
    python tools/refill_llm_queue.py [--sessions asian,london,ny] \\
                                      [--pairs EUR_USD,GBP_USD] \\
                                      [--n-per-session 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import claude_client

OUT_JSON = PROJECT_ROOT / "tools" / "llm_hypotheses_queue.json"

ALL_SESSIONS = ["asian", "london", "ny"]


def refill(
    sessions:        list[str],
    pairs:           list[str],
    n_per_session:   int,
    max_tokens:      int,
    append:          bool,
) -> dict:
    """Run the bulk generation per session and combine the results."""
    print(f"[refill] sessions={sessions} pairs={pairs} n_per_session={n_per_session}")

    all_hyps: list = []
    per_session_counts: dict[str, int] = {}

    for session in sessions:
        print(f"[refill] generating for {session.upper()}...")
        out = claude_client.bulk_generate_hypotheses_for_session(
            session    = session,
            pairs      = pairs,
            n          = n_per_session,
            max_tokens = max_tokens,
        )
        per_session_counts[session] = len(out)
        for h in out:
            h["target_session"] = session
            h["target_pairs"]   = list(pairs)
            all_hyps.append(h)
        print(f"[refill] {session}: got {len(out)} non-empty hypothesis(es)")

    # Optional append mode — merge with existing queue, deduplicating by function_name
    if append and OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            existing_hyps = existing.get("hypotheses") or []
            existing_names = {h.get("function_name") for h in existing_hyps}
            new_count_before_dedup = len(all_hyps)
            all_hyps = existing_hyps + [
                h for h in all_hyps if h.get("function_name") not in existing_names
            ]
            print(f"[refill] appended {len(all_hyps) - len(existing_hyps)} new "
                  f"(of {new_count_before_dedup}) to existing queue of "
                  f"{len(existing_hyps)}")
        except Exception as e:
            print(f"[refill] WARN: could not merge with existing queue: {e}")

    payload = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "pairs":               list(pairs),
        "n_per_session":       n_per_session,
        "per_session_counts":  per_session_counts,
        "hypotheses":          all_hyps,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[refill] DONE — {len(all_hyps)} total hypothesis(es) written to {OUT_JSON}")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--sessions", default="asian,london,ny",
                    help="Comma-separated sessions to refill (default: asian,london,ny)")
    ap.add_argument("--pairs", default="EUR_USD,GBP_USD",
                    help="Comma-separated pairs (default: EUR_USD,GBP_USD)")
    ap.add_argument("--n-per-session", type=int, default=10,
                    help="How many strategies per session (default: 10 → 30 total for 3 sessions)")
    ap.add_argument("--max-tokens", type=int, default=32000,
                    help="Per-call max_tokens; raise for higher --n-per-session (default: 32000)")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing queue (dedup by function_name) instead of replacing")
    args = ap.parse_args(argv)

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    pairs    = [p.strip() for p in args.pairs.split(",") if p.strip()]

    for s in sessions:
        if s not in ALL_SESSIONS:
            raise SystemExit(f"unknown session '{s}'; expected one of {ALL_SESSIONS}")

    refill(
        sessions      = sessions,
        pairs         = pairs,
        n_per_session = args.n_per_session,
        max_tokens    = args.max_tokens,
        append        = args.append,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
