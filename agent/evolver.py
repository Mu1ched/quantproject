"""
Genetic operators on surviving strategy code.

Every EVOLVER_EVERY_N_ROUNDS, generates hypotheses by mutating and crossing
over the top survivors rather than generating from scratch. This exploits
known-good regions of hypothesis space rather than randomly exploring them.

Two operators:
  mutate_survivors()  — makes ONE targeted change to a proven strategy
  crossover_survivors() — combines entry logic from two survivors
"""

import logging
import time

import anthropic

from agent.config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS,
    EVOLVER_N_MUTATIONS, EVOLVER_N_CROSSOVERS,
)

log = logging.getLogger(__name__)

# ── Tool schema (shared with claude_client submit_hypothesis format) ───────────

_EVOLVER_TOOL = {
    "name": "submit_hypothesis",
    "description": (
        "Submit one evolved strategy hypothesis. "
        "Call this tool once per hypothesis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "function_name": {
                "type": "string",
                "description": (
                    "Python identifier (without 'entry_' prefix). "
                    "Must differ from the parent strategy name."
                ),
            },
            "code": {
                "type": "string",
                "description": "Complete entry function code following the mandatory pattern.",
            },
            "rationale": {
                "type": "string",
                "description": (
                    "What specifically was changed from the parent, and why this change "
                    "should improve or diversify the edge."
                ),
            },
            "behaviour_type": {
                "type": "string",
                "enum": [
                    "breakout_continuation",
                    "false_breakout_liquidity_grab",
                    "mean_reversion_low_volatility",
                    "momentum_ignition_after_compression",
                    "trend_pullback_continuation",
                    "stop_run_reversal",
                ],
            },
            "additional_params": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
        },
        "required": ["function_name", "code", "rationale", "behaviour_type"],
    },
}

_MUTATION_INSTRUCTIONS = """\
You are a quantitative researcher performing genetic mutation on a proven forex strategy.

MUTATION RULES — make EXACTLY ONE of the following changes per mutation:
  (a) Replace one feature with a stronger correlated signal from the same category
  (b) Add ONE new orthogonal condition from a different feature category
  (c) Tighten the regime filter (e.g. require TRENDING instead of any regime)
  (d) Narrow the time window to the highest-signal portion of the session
  (e) Adjust parameter grid to explore a flatter, more robust region

FORBIDDEN:
  - Reproducing the parent code unchanged
  - Making 2+ changes in one mutation (defeats the purpose)
  - Changing the function to a completely different thesis

RULES:
  - Wide, round parameter grids: [0.1, 0.2, 0.3] not [0.137, 0.152]
  - All hard rules from the system framework still apply
  - New function name must be different from parent
"""

_CROSSOVER_INSTRUCTIONS = """\
You are a quantitative researcher performing genetic crossover on two proven forex strategies.

CROSSOVER RULES:
  - Take the ENTRY SIGNAL logic from Strategy A (the core condition that triggers an entry)
  - Take the FILTER/GATE logic from Strategy B (regime check, time filter, flow confirmation)
  - Combine them into ONE coherent new function
  - The result must make microstructural sense — not just concatenation

FORBIDDEN:
  - Simply appending all conditions from both strategies
  - Producing a function that is a copy of either parent
  - Producing an over-complex function with 5+ conditions

RULES:
  - Wide, round parameter grids
  - All mandatory function pattern rules apply
  - Name must differ from both parents
"""


# ── Client helpers ─────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _retry(fn, retries: int = 3):
    for attempt in range(retries):
        try:
            return fn()
        except anthropic.RateLimitError:
            wait = 60 * (2 ** attempt)
            log.warning("Evolver rate limit — waiting %ds", wait)
            time.sleep(wait)
        except anthropic.APIConnectionError:
            log.warning("Evolver connection error (attempt %d/%d)", attempt + 1, retries)
            time.sleep(30)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(30)
            else:
                raise
    return None


def _extract(response) -> list:
    """Pull hypothesis dicts from a tool_use response."""
    if response is None:
        return []
    results = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_hypothesis":
            inp = block.input
            results.append({
                "function_name":     inp.get("function_name", ""),
                "code":              inp.get("code", ""),
                "rationale":         inp.get("rationale", ""),
                "behaviour_type":    inp.get("behaviour_type", ""),
                "additional_params": inp.get("additional_params") or {},
            })
    return results


# ── Public operators ───────────────────────────────────────────────────────────

def mutate_survivors(survivors: list, session: str = "") -> list:
    """
    Produce EVOLVER_N_MUTATIONS mutations from the top survivors.

    survivors: list of dicts with keys strategy_name, code, rationale,
               test_sharpe, behaviour_type (from db.get_top_results).
    Returns list of hypothesis dicts (same format as claude_client output).
    """
    if not survivors:
        return []

    candidates = [s for s in survivors if s.get('code')][:3]
    if not candidates:
        log.warning("Evolver: no survivors with code available for mutation")
        return []

    survivor_blocks = "\n\n".join(
        f"=== PARENT: {s['strategy_name']} | "
        f"Sharpe={s.get('test_sharpe', 0):.2f} | "
        f"behaviour={s.get('behaviour_type', '?')} ===\n"
        f"Rationale: {s.get('rationale', '')}\n\n"
        f"{s['code']}"
        for s in candidates
    )

    n = EVOLVER_N_MUTATIONS
    user_msg = (
        f"{_MUTATION_INSTRUCTIONS}\n\n"
        f"Session: {session.upper()}\n"
        f"Produce {n} mutations by calling submit_hypothesis {n} times. "
        f"Each mutation must modify a DIFFERENT parent strategy or a DIFFERENT aspect.\n\n"
        f"{survivor_blocks}"
    )

    def _call():
        return _client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            tools=[_EVOLVER_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )

    response = _retry(_call)
    results  = _extract(response)
    log.info("Evolver: %d mutations generated", len(results))
    return results


def crossover_survivors(survivors: list, session: str = "") -> list:
    """
    Produce EVOLVER_N_CROSSOVERS crossovers from the top 2 survivors.

    Returns list of hypothesis dicts.
    """
    candidates = [s for s in survivors if s.get('code')]
    if len(candidates) < 2:
        log.info("Evolver: need ≥2 survivors with code for crossover (have %d)", len(candidates))
        return []

    a, b = candidates[0], candidates[1]
    n    = EVOLVER_N_CROSSOVERS

    user_msg = (
        f"{_CROSSOVER_INSTRUCTIONS}\n\n"
        f"Session: {session.upper()}\n"
        f"Produce {n} crossover(s) by calling submit_hypothesis {n} time(s).\n\n"
        f"=== STRATEGY A: {a['strategy_name']} | "
        f"Sharpe={a.get('test_sharpe', 0):.2f} | "
        f"behaviour={a.get('behaviour_type', '?')} ===\n"
        f"Rationale: {a.get('rationale', '')}\n\n"
        f"{a['code']}\n\n"
        f"=== STRATEGY B: {b['strategy_name']} | "
        f"Sharpe={b.get('test_sharpe', 0):.2f} | "
        f"behaviour={b.get('behaviour_type', '?')} ===\n"
        f"Rationale: {b.get('rationale', '')}\n\n"
        f"{b['code']}"
    )

    def _call():
        return _client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            tools=[_EVOLVER_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )

    response = _retry(_call)
    results  = _extract(response)
    log.info("Evolver: %d crossover(s) generated", len(results))
    return results
