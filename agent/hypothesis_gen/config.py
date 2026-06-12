"""Hypothesis-gen-specific configuration.

Kept here, not in `agent/config.py`, so this submodule stays independent
of `agent.main`'s import surface.
"""

# ── Model + API ──────────────────────────────────────────────────────
MODEL                  = 'claude-sonnet-4-6'   # default; Opus 4.7 selectable
MODEL_OPUS             = 'claude-opus-4-7'
MAX_TOKENS_OUTPUT      = 16_000

# Cost per 1M tokens (Anthropic pricing, USD)
COST_TABLE = {
    'claude-sonnet-4-6': {'input': 3.0,  'output': 15.0,
                          'cache_read': 0.3, 'cache_write': 3.75},
    'claude-opus-4-7':   {'input': 15.0, 'output': 75.0,
                          'cache_read': 1.5, 'cache_write': 18.75},
}

# ── Loop control ─────────────────────────────────────────────────────
MAX_ROUNDS             = 20
BUDGET_USD             = 25.00
ROUND_PAUSE_SEC        = 30
AUTO_SWEEP             = False

# ── Retry policy ─────────────────────────────────────────────────────
RETRY_MAX_ATTEMPTS     = 3
RETRY_BASE_DELAY_S     = 60

# ── Cache + truncation ───────────────────────────────────────────────
# When tried_list / rejection_digest grow large, truncate to most recent
# families so the prompt stays under the cache-friendly threshold.
TRIED_LIST_MAX_FAMILIES     = 200
REJECTION_DIGEST_MAX_ROWS   = 500
