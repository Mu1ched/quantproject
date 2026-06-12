"""Configuration for the autonomous edge-discovery agent system."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).parent.parent
AGENT_DIR     = Path(__file__).parent
GENERATED_DIR = AGENT_DIR / "generated"
AGENT_DB_PATH = AGENT_DIR / "agent_results.db"
EDGE_DB_PATH  = PROJECT_DIR / "edge_results.db"
LOG_PATH      = AGENT_DIR / "agent.log"

# ── Environment ───────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Schedule ──────────────────────────────────────────────────────────────────
DAILY_REPORT_UTC_HOUR = 18
# 120s sleep keeps spend predictable on a small API balance — at Haiku 4.5
# pricing, ~30 rounds/hr × 5 hyps × ~$0.005 ≈ $0.75/hr. Tighten only if budget
# headers say you're nowhere near the cap.
LOOP_SLEEP_SECONDS    = 120

# ── Generation ────────────────────────────────────────────────────────────────
HYPOTHESES_PER_BATCH = 2
MAX_CODE_RETRIES     = 3

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_WORKERS     = 2     # 8 GB Windows VPS — borderline: 2 workers ≈ 5–6 GB RSS. Drop back to 1 if MemoryError returns.
DATA_CACHE_TTL_HOURS = 6

# ── Execution realism ─────────────────────────────────────────────────────────
SWEEP_USE_DYNAMIC_SPREAD = True    # use per-bar spread_adj instead of fixed median
NEWS_SPREAD_MULT         = 2.0     # spread widens 2.0× during high-impact news bars (was 2.5 pre 2026-06-02 recalibration)
COMMISSION_PIPS          = 0.3     # round-trip commission equivalent in pips per side

# ── Portfolio intelligence ────────────────────────────────────────────────────
PORTFOLIO_MAX_CORR       = 0.65    # reject survivor if daily PnL corr with any existing > this
PORTFOLIO_MIN_SURVIVORS  = 3       # don't apply correlation filter until we have this many

# ── Genetic evolution ─────────────────────────────────────────────────────────
# Phase 5b throughput lift — evolver paths (mutate / crossover / GP) are zero-
# API-cost, so cranking their cadence is free throughput. Was 4 (1:4 evolver
# rounds : Claude rounds), now 2 (1:2). The GP path also covers an orthogonal
# slice of hypothesis space that Claude doesn't reach.
EVOLVER_EVERY_N_ROUNDS   = 2       # run genetic batch every N standard rounds
EVOLVER_N_MUTATIONS      = 3       # mutations to generate per evolver run
EVOLVER_N_CROSSOVERS     = 2       # crossovers to generate per evolver run

# ── Parameter grid ────────────────────────────────────────────────────────────
GRID_TP_VALUES = [1.5, 2.5, 4.0]  # 4.0 added 2026-06-07 so the sweep can find survivors for 30-60 bar drift / mean-reversion plays
GRID_SL_VALUES = [0.75, 1.25, 2.0]  # 2.0 added 2026-06-07 to absorb noise on wide-TP variants

# ── Composite score weights ───────────────────────────────────────────────────
COMPOSITE_WEIGHTS = {
    # §1 Tier 1 back-port (TIER1_3_PLAN.md) — same rebalance as crypto so
    # cross-project comparisons score on identical criteria. mc_eval_pass_pct
    # backfills to NULL on the existing 1,423 FX rows; _safe(None)=0 means
    # they score 0 on that dimension. USE_MC_GATE=False (below) prevents
    # those rows from being retro-rejected; only ranking shifts.
    'test_sharpe':       0.25,   # was 0.35
    'dsr':               0.20,   # was 0.25
    'mc_eval_pass_pct':  0.20,   # NEW
    'test_wr':           0.10,   # was 0.15
    'regime_stable':     0.10,   # unchanged
    'n_trades':          0.10,   # unchanged
    'max_dd':            0.05,   # unchanged
}

# ── Batch reporting ───────────────────────────────────────────────────────────
BATCH_REPORT_EVERY = 100   # send top-N alert after every N strategies tested
TOP_PER_BATCH      = 2     # how many to surface per batch

# ── Survivor filters — ALL must pass to appear in any report ─────────────────
# Target: roughly top 2% of tested strategies survive
MIN_TEST_TRADES      = 20       # pooled across pairs; lower-n strategies still penalised by Bayesian shrinkage (BAYES_PRIOR_N=50) and the full PSR/DSR/BH gauntlet
MIN_DSR              = 0.10     # meaningfully positive deflated Sharpe (not just > 0)
MIN_TEST_SHARPE      = 0.50     # minimum acceptable out-of-sample Sharpe
REQUIRE_BH_SIG       = True     # must pass Benjamini-Hochberg FDR correction
REQUIRE_REGIME_STABLE = True    # edge must hold across multiple market regimes
MAX_TRAIN_TEST_DECAY = 0.50     # test Sharpe can't be less than 50% of train Sharpe
MAX_TEST_DRAWDOWN    = 0.20     # max acceptable drawdown on test set

# ── Meta-learning ────────────────────────────────────────────────────────────
META_UPDATE_EVERY    = 20    # re-synthesise guidance every N strategies tested
META_MAX_GUIDANCE_LEN = 600  # max chars injected into each generation prompt

# ── Robustness checks (run on trades after sweep completes) ───────────────────
# Monte Carlo — simulates prop challenge 10k times on resampled daily PnL
MC_N_SIMS              = 10_000
MC_CHALLENGE_DAYS      = 90
MC_MIN_PASS_PCT        = 60.0   # ≥ 60% of simulations must pass the challenge
MC_MAX_BLOWN_PCT       = 20.0   # ≤ 20% of simulations may blow the account

# Bootstrap Sharpe CI — lower bound must be positive (95th percentile is still an edge)
MIN_SHARPE_CI_LOW      = 0.0

# Walk-forward — split test trades into N time folds, check consistency
WF_N_FOLDS             = 4
WF_MIN_PROFITABLE_FOLDS = 3    # ≥ 3 of 4 folds must show positive PnL

# review#18 — feature flag for walk-forward in the gauntlet. When True, the
# scorer additionally rejects strategies whose worst WF fold Sharpe < this
# threshold. Default False so existing behaviour is preserved while the
# walked metrics are populated for new hypotheses; flip to True after a
# round of post-restart hypotheses have wf_sharpe_min populated and you've
# eyeballed the distribution against a defensible cutoff.
USE_WF_GATE            = False
WF_GATE_MIN_SHARPE     = 0.0   # min wf_sharpe_min to pass when USE_WF_GATE is on

# §1 Tier 1 back-port (TIER1_3_PLAN.md) — MC pass-rate gate in is_survivor.
# **Default False on FX** so the 1,423 existing rows aren't retro-rejected
# when MC numbers backfill on the next sweep. The numbers populate regardless
# of the flag; the flag only controls whether they gate survivor status.
# Reuses the existing MC_MIN_PASS_PCT (60.0) and MC_MAX_BLOWN_PCT (20.0)
# defined above at lines 98-99.
USE_MC_GATE            = False

# Parameter sensitivity — best combo's parameter neighbours must also be viable
PARAM_SENS_MIN_VIABLE_FRAC = 0.50  # ≥ 50% of adjacent parameter combos must survive basic filter

# Phase 7 — Probability of Backtest Overfitting (Bailey-LdP CSCV) and PSR gates.
# PBO > 0.5 = backtest is more likely overfit than not. PSR is the probability
# the true Sharpe exceeds zero given observed skew/kurtosis; 0.95 = 95% confident.
MAX_PBO = 0.5
MIN_PSR = 0.95

# ── Regime multipliers ────────────────────────────────────────────────────────
DEFAULT_REGIME_MULT = {
    'TRENDING':      1.0,
    'RANGING':       0.5,
    'TRANSITIONING': 0.5,
    'VOLATILE':      0.0,
    'UNDEFINED':     0.3,
}

# ── Claude API ────────────────────────────────────────────────────────────────
# Two-tier model strategy: Haiku does the wide first-pass generation (cheap +
# fast), Sonnet runs evolver, syntax fixes, and meta-learner synthesis where
# reasoning quality matters more than raw throughput.
CLAUDE_MODEL_FAST = "claude-haiku-4-5-20251001"  # generation tier
CLAUDE_MODEL_DEEP = "claude-sonnet-4-6"          # evolver / meta / fix tier
CLAUDE_MODEL      = CLAUDE_MODEL_DEEP             # legacy alias (default = deep)
# 8192 was a holdover from earlier multi-strategy responses. Five tool calls
# averaging ~400 tokens each plus a small preface fits comfortably in 3000.
# Smaller cap = lower spend on truncated responses and tighter latency.
MAX_TOKENS        = 3000

# ── Spend budget (per Anthropic API key, NOT linked to Claude Max) ───────────
# When a call's accumulated daily/total spend exceeds these caps, the client
# refuses to call and returns None. Budget is persisted in agent_results.db.
# Approximate USD using current public pricing (see _PRICE_PER_MTOKEN below).
BUDGET_DAILY_USD = 1.50      # ~£1.20/day ceiling
BUDGET_TOTAL_USD = 9.00      # ~£7.20 total — matches the £8 starting balance
BUDGET_WARN_PCT  = 0.80      # log a loud warning at 80% of either cap

# ── Live → backtest feedback ─────────────────────────────────────────────────
LIVE_LOG_DIR             = PROJECT_DIR        # MT5live writes CSVs here
LIVE_TRADE_CSV           = "live_trade_log.csv"
LIVE_EXEC_CSV            = "execution_quality.csv"
LIVE_INGEST_EVERY_ROUNDS = 10        # re-scan CSVs every N agent rounds
LIVE_MIN_TRADES_PER_PAIR = 20        # need this many before computing a per-pair decay
LIVE_DECAY_DEFAULT       = 1.0       # neutral when no live data yet

# ── Bayesian Sharpe shrinkage ────────────────────────────────────────────────
# Shrinks raw test Sharpe toward a prior of 0 weighted by trade count.
# At BAYES_PRIOR_N trades, posterior is 50% raw / 50% prior. Higher trade
# counts get progressively closer to the raw value.
BAYES_PRIOR_SHARPE = 0.0
BAYES_PRIOR_N      = 50


# ── GUI overrides ────────────────────────────────────────────────────────────
# At module-load, apply any user-set overrides from gui_config.json so a
# single source of truth (the GUI) controls the live agent. Knob changes take
# effect on the next agent start.
def _apply_gui_overrides() -> None:
    try:
        from agent import gui_config as _gc
        cfg = _gc.load() or {}
    except Exception:
        return
    g = globals()
    _map = {
        "cost_mult":            ("COST_MULT", float),
        "hypotheses_per_batch": ("HYPOTHESES_PER_BATCH", int),
        "backtest_workers":     ("BACKTEST_WORKERS", int),
        "budget_daily_usd":     ("BUDGET_DAILY_USD", float),
        "budget_total_usd":     ("BUDGET_TOTAL_USD", float),
        "loop_sleep_seconds":   ("LOOP_SLEEP_SECONDS", int),
        "min_test_sharpe":      ("MIN_TEST_SHARPE", float),
        "min_dsr":              ("MIN_DSR", float),
        "max_test_drawdown":    ("MAX_TEST_DRAWDOWN", float),
        "min_test_trades":      ("MIN_TEST_TRADES", int),
    }
    for src, (target, caster) in _map.items():
        v = cfg.get(src)
        if v is None:
            continue
        try:
            g[target] = caster(v)
        except Exception:
            pass


# Default cost multiplier (overridable via GUI). 1.0 = realistic, 0.5 =
# optimistic (tighter spreads), 1.5 = pessimistic (worst-case).
COST_MULT = 1.0

_apply_gui_overrides()
