"""
Main orchestration loop for the autonomous edge-discovery agent.

Each round:
  1. Determine session from UTC time (Asian / London / NY)
  2. Ask Claude to generate HYPOTHESES_PER_BATCH novel entry functions
     (every EVOLVER_EVERY_N_ROUNDS, also run genetic mutation/crossover on survivors)
  3. Write each to disk, validate, load as callable
  4. Run run_sweep() for each — results written to edge_results.db
  5. Score survivors → portfolio correlation filter → write to agent_results.db
  6. At 18:00 UTC: generate and send daily top-5 Telegram report
"""

import json as _json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_proj = str(Path(__file__).parent.parent)
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import edge_engine as eng

import agent.claude_client as claude_client
import agent.code_writer as code_writer
import agent.db as db
import agent.evolver as evolver
import agent.gp_evolver as gp_evolver
import agent.live_ingest as live_ingest
import agent.portfolio as portfolio
import agent.promotion as promotion
import agent.reporter as reporter
import agent.robustness as robustness
import agent.scorer as scorer
import agent.meta_learner as meta_learner
import agent.pre_screen as pre_screen_mod
from agent.scorer import rejection_reason
import agent.session_router as session_router
import agent.telegram as telegram
import agent.thompson_bandit as thompson_bandit
import agent.runtime_state as runtime_state

from agent.config import (
    BACKTEST_WORKERS,
    BATCH_REPORT_EVERY,
    COST_MULT,
    DATA_CACHE_TTL_HOURS,
    EVOLVER_EVERY_N_ROUNDS,
    GRID_SL_VALUES,
    GRID_TP_VALUES,
    HYPOTHESES_PER_BATCH,
    LIVE_INGEST_EVERY_ROUNDS,
    LOOP_SLEEP_SECONDS,
    MAX_CODE_RETRIES,
    SWEEP_USE_DYNAMIC_SPREAD,
    TOP_PER_BATCH,
)

log = logging.getLogger(__name__)

# ── Data cache ────────────────────────────────────────────────────────────────

_cache: dict     = {}
_cache_loaded_at = None

# ── Persistent worker pool (optimisation #4) ──────────────────────────────────
# A single ProcessPoolExecutor is created the first time _load_data() succeeds
# and reused across every run_sweep call. Avoids the ~10–30 sec per-sweep cost
# of spawning fresh workers + re-writing parquets on Windows.
#
# Lifecycle:
#   - Created in _init_persistent_pool() after _cache is populated.
#   - Torn down in _teardown_persistent_pool() on cache refresh and atexit.
#   - Passed to eng.run_sweep via the executor= and data_dir= kwargs.
import concurrent.futures as _futures
import multiprocessing as _mp
import shutil as _shutil

_persistent_executor: "_futures.ProcessPoolExecutor | None" = None
_persistent_data_dir = None  # type: Path | None


def _init_persistent_pool() -> None:
    """Create the persistent data_dir + executor for the current _cache."""
    global _persistent_executor, _persistent_data_dir
    if _persistent_executor is not None or not _cache:
        return
    _persistent_data_dir = eng._prepare_sweep_data_dir(
        _cache['train_dfs'], _cache['test_dfs'],
    )
    ctx = _mp.get_context('spawn')
    _persistent_executor = _futures.ProcessPoolExecutor(
        max_workers=BACKTEST_WORKERS,
        mp_context=ctx,
        initializer=eng._worker_init,
        initargs=(str(_persistent_data_dir),),
    )
    log.info("Persistent worker pool initialised: %d workers, data_dir=%s",
             BACKTEST_WORKERS, _persistent_data_dir)


def _teardown_persistent_pool() -> None:
    """Shut down the executor + delete the data_dir. Safe to call multiple times."""
    global _persistent_executor, _persistent_data_dir
    if _persistent_executor is not None:
        try:
            _persistent_executor.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            log.warning("Persistent pool shutdown error: %s", e)
        _persistent_executor = None
    if _persistent_data_dir is not None:
        _shutil.rmtree(_persistent_data_dir, ignore_errors=True)
        _persistent_data_dir = None


import atexit as _atexit
_atexit.register(_teardown_persistent_pool)


def _load_data() -> bool:
    global _cache, _cache_loaded_at
    log.info("Loading market data from Dukascopy cache...")
    runtime_state.update_status(phase="downloading")
    # Tear down stale pool before refreshing data — workers are pinned to the
    # previous data_dir and will read stale parquets if we don't recycle them.
    _teardown_persistent_pool()
    try:
        from agent.data_inventory import _update_multi_status as _dl_progress

        def _cb(p):
            try:
                _dl_progress(p.get("pair", ""), p)
            except Exception:
                pass
        train_dfs, test_dfs, spreads = eng.load_all_data(progress_callback=_cb)
        _cache = {'train_dfs': train_dfs, 'test_dfs': test_dfs, 'spreads': spreads}
        _cache_loaded_at = datetime.now(timezone.utc)
        log.info("Data ready: %d pairs loaded", len(train_dfs))
        _init_persistent_pool()
        runtime_state.update_status(phase="idle", last_error=None)
        return True
    except Exception as e:
        log.error("Data load failed: %s", e)
        runtime_state.update_status(phase="idle", last_error=f"data load: {e}")
        return False


def _cache_stale() -> bool:
    if _cache_loaded_at is None or not _cache:
        return True
    return (datetime.now(timezone.utc) - _cache_loaded_at) > timedelta(hours=DATA_CACHE_TTL_HOURS)


# ── Grid builder ──────────────────────────────────────────────────────────────

_GRID_MAX_COMBOS = 12


def _build_grid(additional_params: dict) -> eng.ParameterGrid:
    grid = {'tp_r': GRID_TP_VALUES, 'sl_r': GRID_SL_VALUES}
    for k, v in (additional_params or {}).items():
        if (
            isinstance(v, list)
            and len(v) <= 8
            and all(isinstance(x, (int, float)) for x in v)
        ):
            grid[k] = [float(x) for x in v]

    def _total(g):
        t = 1
        for vals in g.values():
            t *= max(len(vals), 1)
        return t

    while _total(grid) > _GRID_MAX_COMBOS:
        widest = max(
            (k for k in grid if k not in ('tp_r', 'sl_r')),
            key=lambda k: len(grid[k]),
            default=None,
        )
        if widest is None or len(grid[widest]) <= 2:
            break
        grid[widest] = grid[widest][::2]
        log.warning("Grid too large — thinned '%s' to %d values (total combos now %d)",
                    widest, len(grid[widest]), _total(grid))

    return eng.ParameterGrid(grid)


# ── 2026-06-05: Two-stage thesis → validate → implement ─────────────────────

# Pre-validated mechanic queue, written offline by tools/discover_mechanics.py.
# When non-empty for the current session, drained ahead of the LLM proposal so
# Stage 1 becomes deterministic and zero-cost. consumed_mechanics.json tracks
# which mechanic_ids have already been fed to the loop so they don't fire twice.
_DISCOVERY_JSON = Path(__file__).resolve().parent.parent / "tools" / "validated_mechanics.json"
_CONSUMED_JSON  = Path(__file__).resolve().parent.parent / "tools" / "consumed_mechanics.json"

# Live-trades JSONL: written by pre_screen's progress_callback, read by the
# GUI to render an equity curve as trades close. Truncated at the start of
# every backtest (single-file convention — the file always reflects the
# CURRENT in-flight strategy, never accumulates across runs).
_LIVE_TRADES_JSONL = Path(__file__).resolve().parent.parent / "runtime" / "live_trades.jsonl"


def _live_trades_start(strategy: str, pair: str, session: str,
                        stage: str, params: dict) -> None:
    """Truncate and write the header line for a new live backtest stream."""
    try:
        _LIVE_TRADES_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with _LIVE_TRADES_JSONL.open("w", encoding="utf-8") as f:
            f.write(_json.dumps({
                "type": "start",
                "strategy": strategy,
                "pair":     pair,
                "session":  session,
                "stage":    stage,
                "params":   params,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
    except Exception:
        pass  # observability is best-effort


def _live_trades_callback(event: dict) -> None:
    """Forwarded to eng.run_backtest. Appends per-trade / tick events to the
    live JSONL so the GUI can tail it. Best-effort — never raises."""
    try:
        t = event.get("type")
        if t == "trade_closed":
            trade = event.get("trade") or {}
            with _LIVE_TRADES_JSONL.open("a", encoding="utf-8") as f:
                f.write(_json.dumps({
                    "type":     "trade",
                    "pnl":      float(trade.get("pnl", 0.0) or 0.0),
                    "balance":  float(event.get("balance", 0.0) or 0.0),
                    "exit_ts":  str(trade.get("exit_ts", "")),
                    "side":     str(trade.get("side", "")),
                }) + "\n")
        # Skip tick events — they fire per-bar and would swamp the file.
        # The GUI's elapsed counter already gives bar-level progress feel.
    except Exception:
        pass


def _live_trades_done(passed: bool, reason: str) -> None:
    """Append a terminal `done` event so the GUI knows the run finished."""
    try:
        with _LIVE_TRADES_JSONL.open("a", encoding="utf-8") as f:
            f.write(_json.dumps({
                "type":        "done",
                "passed":      bool(passed),
                "reason":      str(reason),
                "ended_at":    datetime.now(timezone.utc).isoformat(),
            }) + "\n")
    except Exception:
        pass


# LLM hypothesis queue, written offline by tools/refill_llm_queue.py.
# Hypotheses are already coded — drain them straight into _process_hypotheses
# (no Stage 2, no Stage 3 — Claude wrote the code in the bulk refill call).
# Amortises the web_search surcharge across many strategies (~3-5x cheaper
# per strategy than calling generate_hypotheses every round).
_LLM_QUEUE_JSON     = Path(__file__).resolve().parent.parent / "tools" / "llm_hypotheses_queue.json"
_LLM_CONSUMED_JSON  = Path(__file__).resolve().parent.parent / "tools" / "consumed_llm_hypotheses.json"


def _drain_discovered_mechanics(needed: int, session: str, pairs: list) -> list:
    """Pop up to `needed` unconsumed mechanics matching session/pairs from
    validated_mechanics.json. Returns a list of {mechanic, validation_result}
    dicts. Appends consumed mechanic_ids to consumed_mechanics.json atomically
    so a crash mid-round doesn't double-consume.
    """
    if not _DISCOVERY_JSON.exists() or needed <= 0:
        return []
    try:
        payload = _json.loads(_DISCOVERY_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read %s: %s", _DISCOVERY_JSON, e)
        return []
    mechanics = payload.get("mechanics") or []
    if not mechanics:
        return []

    # Load already-consumed names
    consumed: set = set()
    if _CONSUMED_JSON.exists():
        try:
            consumed = set(_json.loads(_CONSUMED_JSON.read_text(encoding="utf-8")) or [])
        except Exception:
            consumed = set()

    pairs_set = set(pairs)
    picked: list = []
    for entry in mechanics:
        if len(picked) >= needed:
            break
        mech = entry.get("mechanic") or {}
        mech_id = mech.get("mechanic_id")
        if not mech_id or mech_id in consumed:
            continue
        # Session filter: discovered_session must match the loop's session.
        if entry.get("discovered_session") != session:
            continue
        # Pair compatibility: all of this round's pairs must be in the
        # mechanic's passing pair list (so validation evidence applies).
        passing = set(entry.get("passing_pairs") or [])
        if not pairs_set.issubset(passing):
            continue
        picked.append(entry)

    if not picked:
        return []

    # Atomically append the picked names to consumed_mechanics.json BEFORE we
    # return them, so a mid-round crash still leaves the queue advanced.
    new_consumed = list(consumed) + [e["mechanic"]["mechanic_id"] for e in picked]
    tmp = _CONSUMED_JSON.with_suffix(".tmp")
    tmp.write_text(_json.dumps(sorted(set(new_consumed)), indent=2), encoding="utf-8")
    tmp.replace(_CONSUMED_JSON)

    return picked


def _drain_llm_hypothesis_queue(needed: int, session: str, pairs: list) -> list:
    """Pop up to `needed` unconsumed LLM-sourced hypotheses from
    llm_hypotheses_queue.json. Returns hypothesis dicts ready to feed
    _process_hypotheses (function_name, code, rationale, behaviour_type,
    additional_params).

    Filters by `target_session == session` AND `set(pairs).issubset(target_pairs)`
    so the queue can be refilled for all sessions but each round only consumes
    appropriate entries.

    Atomically appends consumed function_names to consumed_llm_hypotheses.json
    before returning so a mid-round crash doesn't double-consume.
    """
    if not _LLM_QUEUE_JSON.exists() or needed <= 0:
        return []
    try:
        payload = _json.loads(_LLM_QUEUE_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read %s: %s", _LLM_QUEUE_JSON, e)
        return []
    queue = payload.get("hypotheses") or []
    if not queue:
        return []

    consumed: set = set()
    if _LLM_CONSUMED_JSON.exists():
        try:
            consumed = set(_json.loads(_LLM_CONSUMED_JSON.read_text(encoding="utf-8")) or [])
        except Exception:
            consumed = set()

    pairs_set = set(pairs)
    picked: list = []
    for entry in queue:
        if len(picked) >= needed:
            break
        fn = entry.get("function_name")
        if not fn or fn in consumed:
            continue
        if entry.get("target_session") != session:
            continue
        target_pairs = set(entry.get("target_pairs") or [])
        # If the entry doesn't specify pairs, accept it; otherwise require overlap.
        if target_pairs and not pairs_set.issubset(target_pairs):
            continue
        picked.append(entry)

    if not picked:
        return []

    new_consumed = list(consumed) + [e["function_name"] for e in picked]
    tmp = _LLM_CONSUMED_JSON.with_suffix(".tmp")
    tmp.write_text(_json.dumps(sorted(set(new_consumed)), indent=2), encoding="utf-8")
    tmp.replace(_LLM_CONSUMED_JSON)

    return picked


def _generate_validated_hypotheses(
    session:           str,
    pairs:             list,
    proven_conditions: list,
    top_results:       list,
    n:                 int,
    meta_guidance:     str = "",
    bandit_weights:    dict | None = None,
    saturated_features: list | None = None,
) -> list:
    """Hypothesis generation with two paths.

    Discovery-queue path (when tools/validated_mechanics.json has unconsumed
    mechanics matching this session+pairs):
      Stage 1a: drain the queue (zero LLM cost).
      Stage 2:  reuse pre-validated stats — no re-validation.
      Stage 3:  ask Claude to implement each mechanic.

    LLM path (queue empty for this session):
      Single-shot call to `generate_hypotheses` with the server-side
      web_search tool enabled. Claude reads retail-FX strategy literature
      online and returns full entry-function code in one round-trip. No
      Stage 2 empirical gate — the strategy flows straight into
      _process_hypotheses → pre-screen → sweep → robustness.

    The 2026-06-05 two-stage propose_mechanic → validate → implement flow
    is retired on the LLM path (Stage 2 was rejecting ~100% of LLM
    proposals). `propose_mechanic` remains in claude_client.py as dead
    code in case we want to revisit it.
    """
    from agent.mechanic_validator import validate_mechanic

    runtime_state.update_status(activity="Checking discovery queue")
    discovered = _drain_discovered_mechanics(needed=n, session=session, pairs=pairs)

    if not discovered:
        # 1st choice: drain the offline LLM queue (tools/refill_llm_queue.py).
        # These are already coded; skip Stage 3 too — straight into the sweep.
        runtime_state.update_status(activity="Checking LLM hypothesis queue")
        queued = _drain_llm_hypothesis_queue(needed=n, session=session, pairs=pairs)
        if queued:
            log.info("[llm-path] drained %d hypothesis(es) from llm_hypotheses_queue.json "
                     "(no API call this round)", len(queued))
            runtime_state.update_status(
                activity=f"Drained {len(queued)} pre-coded hypothesis(es) from queue"
            )
            # Strip queue-only metadata fields; _process_hypotheses ignores
            # unknown keys but keeping the dicts clean avoids surprises.
            for h in queued:
                h.pop("target_session", None)
                h.pop("target_pairs", None)
            return queued

        # 2nd choice: per-round web-search call (slower + more expensive).
        log.info("[llm-path] queue empty for %s — falling back to per-round "
                 "web-search hypothesis generation", session)
        runtime_state.update_status(
            activity=f"LLM web-search generation ({session}, n={n})"
        )
        return claude_client.generate_hypotheses(
            session            = session,
            pairs              = pairs,
            proven_conditions  = proven_conditions,
            top_results        = top_results,
            n                  = n,
            meta_guidance      = meta_guidance,
            bandit_weights     = bandit_weights,
            saturated_features = saturated_features,
        )

    # Discovery-queue path
    log.info("[stage1] using %d pre-discovered mechanic(s), skipping LLM proposal",
             len(discovered))
    runtime_state.update_status(
        activity=f"Drained {len(discovered)} discovered mechanic(s) for Stage 2/3"
    )
    proposed = [d["mechanic"] for d in discovered]
    pre_validated = {d["mechanic"]["mechanic_id"]: d["validation_result"]
                     for d in discovered}

    # Stage 2 — reuse discovery's validation results
    log.info("[stage2] validating %d mechanic(s) against train data...", len(proposed))
    runtime_state.update_status(
        activity=f"Stage 2: validating {len(proposed)} mechanic(s)"
    )
    validated: list[tuple[dict, dict]] = []
    for mech in proposed:
        mech_id = mech.get('mechanic_id', '?')
        if mech_id in pre_validated:
            val = pre_validated[mech_id]
            log.info("[stage2] ✓ '%s' PRE-VALIDATED (discovery) — n=%d, t=%+.2f, mean_atr=%+.3f",
                     mech_id, val['n_events'], val['t_stat'], val['mean_fwd_return'])
            validated.append((mech, val))
            continue
        # Defence-in-depth: re-validate anything that slipped through without
        # a cached result. Should not happen with the current drain code.
        try:
            result = validate_mechanic(mech, _cache['train_dfs'])
        except Exception as e:
            log.warning("[stage2] '%s' validation crashed: %s", mech_id, e)
            meta_learner.record_rejection(
                mech_id, session, 'mechanic_validate_error', str(e)[:200],
            )
            continue
        if result.passed:
            validated.append((mech, {
                'n_events':         result.n_events,
                't_stat':           result.t_stat,
                'mean_fwd_return':  result.mean_fwd_return,
                'per_pair':         result.per_pair,
            }))
        else:
            log.info("[stage2] ✗ '%s' rejected: %s", mech_id, result.reason[:200])
            meta_learner.record_rejection(
                mech_id, session, 'mechanic_no_edge', result.reason[:200],
            )

    if not validated:
        log.warning("[stage2] 0 of %d discovered mechanics validated", len(proposed))
        return []

    # Stage 3 — implement each validated mechanic
    log.info("[stage3] implementing %d validated mechanic(s)...", len(validated))
    hypotheses = []
    for i, (mech, val) in enumerate(validated, 1):
        runtime_state.update_status(
            activity=f"Stage 3: Claude implementing mechanic {i}/{len(validated)} "
                     f"({mech.get('mechanic_id', '?')[:40]})"
        )
        impl = claude_client.implement_validated_mechanic(
            mechanic          = mech,
            validation_result = val,
            session           = session,
            pairs             = pairs,
        )
        hypotheses.extend(impl)

    log.info("[stage3] generated %d implementation(s) from %d validated mechanic(s)",
             len(hypotheses), len(validated))
    return hypotheses


# ── Single strategy sweep ─────────────────────────────────────────────────────

def _sweep_one(
    name:              str,
    entry_fn,
    additional_params: dict,
    session:           str,
    pairs:             list,
    exit_hour:         int,
) -> tuple:
    """
    Run a full parameter grid sweep for one entry function.

    When more than one pair is in scope, we do a two-stage sweep: stage 1 runs
    only on the cheapest live-calibrated pair (EUR_USD if available) to weed
    out strategies that produce 0 trades or catastrophic Sharpe even on the
    easy pair. Stage 2 runs the full-pair sweep — its sweep_id and metrics are
    the canonical result used downstream.

    Returns (sweep_id, best_metrics, best_score, n_survivors, robust_passed, survivor_metrics).
    Returns (None, {}, 0.0, 0, False, {}) on failure.
    """
    grid        = _build_grid(additional_params)
    manager_fn  = eng.make_manager(exit_hour=exit_hour, use_breakeven=True)
    regime_mult = session_router.session_regime_mult(session)

    def _run(pairs_subset: list, label: str) -> str | None:
        log.info("  Sweeping '%s' %s: %d combos × %d pairs",
                 name, label, len(grid), len(pairs_subset))
        try:
            return eng.run_sweep(
                sweep_name          = f"{name}{'_s1' if label == 'stage1' else ''}",
                entry_fn            = entry_fn,
                manager_fn          = manager_fn,
                grid                = grid,
                pairs               = pairs_subset,
                session             = session,
                regime_mult         = regime_mult,
                train_dfs           = _cache['train_dfs'],
                test_dfs            = _cache['test_dfs'],
                measured_spreads    = _cache['spreads'],
                n_workers           = BACKTEST_WORKERS,
                cost_mult           = COST_MULT,
                use_dynamic_spread  = SWEEP_USE_DYNAMIC_SPREAD,
                executor            = _persistent_executor,
                data_dir            = _persistent_data_dir,
            )
        except Exception as e:
            log.error("  run_sweep %s failed for '%s': %s", label, name, e)
            return None

    # ── Stage 1: single-pair cheap qualifier (only when multi-pair scope) ────
    if len(pairs) > 1:
        stage1_pair = "EUR_USD" if "EUR_USD" in pairs else pairs[0]
        s1_id = _run([stage1_pair], "stage1")
        if s1_id is None:
            return None, {}, 0.0, 0, False, {}
        s1_rows = db.load_sweep_results(s1_id)
        if not s1_rows:
            log.info("  '%s': stage1 produced 0 rows → skip", name)
            meta_learner.record_rejection(name, session, 'stage1_empty',
                                           'no hypothesis results on stage1 pair')
            return s1_id, {}, 0.0, 0, False, {}
        # 2026-06-06 bugfix — db.load_sweep_results doesn't ORDER BY, so
        # s1_rows[0] used to be the first-inserted hypothesis (often the
        # worst), not the best. Sort here so the rejected-strategy row in
        # tested_strategies reflects the BEST hypothesis from the sweep
        # rather than an arbitrary one.
        s1_rows = sorted(s1_rows,
                          key=lambda r: float(r.get('test_sharpe') or 0),
                          reverse=True)
        s1_max_sharpe = float(s1_rows[0].get('test_sharpe') or 0)
        s1_max_trades = max(int(r.get('test_n') or 0) for r in s1_rows)
        if s1_max_sharpe < -2.0 or s1_max_trades < 10:
            reason = (f"stage1 fail: max_sharpe={s1_max_sharpe:.2f}, "
                      f"max_trades={s1_max_trades} (on {stage1_pair})")
            log.info("  '%s': STAGE1 REJECT — %s", name, reason)
            meta_learner.record_rejection(name, session, 'stage1_reject', reason)
            return s1_id, s1_rows[0], 0.0, 0, False, {}
        log.info("  '%s': stage1 passed (max_sharpe=%.2f, max_trades=%d) → stage2",
                 name, s1_max_sharpe, s1_max_trades)

    # ── Stage 2: full pair sweep (canonical result) ──────────────────────────
    sweep_id = _run(pairs, "stage2" if len(pairs) > 1 else "")
    if sweep_id is None:
        return None, {}, 0.0, 0, False, {}

    rows = db.load_sweep_results(sweep_id)
    best_metrics, best_score, n_static = scorer.pick_best_from_sweep(rows)

    if n_static == 0:
        if rows:
            gate = rejection_reason(rows[0])
            log.info("  '%s': %d rows, 0 passed static filters. Sample: %s",
                     name, len(rows), gate)
            meta_learner.record_rejection(name, session, gate, f"static filter: {gate}")
            # Surface the top hypothesis's metrics even though it didn't survive,
            # so the Rejected panel shows real numbers next to the gate string
            # instead of None. Caller uses this to call db.record_result.
            return sweep_id, rows[0], 0.0, 0, False, {}
        return sweep_id, {}, 0.0, 0, False, {}

    hyp_id    = best_metrics.get('hypothesis_id', '')
    trades_df = db.load_test_trades(hyp_id) if hyp_id else None

    robust_passed, robust_report = robustness.run_all_checks(best_metrics, rows, trades_df)

    if not robust_passed:
        failed_gates = [k for k, v in robust_report.items() if not v.get('passed', True)]
        gate_str = ', '.join(failed_gates)
        meta_learner.record_rejection(
            name, session, gate_str,
            '; '.join(robust_report[g].get('detail', '') for g in failed_gates),
        )

    n_survivors = 1 if robust_passed else 0
    log.info("  '%s': static OK | robustness %s | score=%.3f",
             name, "PASSED" if robust_passed else "FAILED", best_score)

    return sweep_id, best_metrics, best_score, n_survivors, robust_passed, \
           best_metrics if robust_passed else {}


# ── Hypothesis processing pipeline ────────────────────────────────────────────

def _process_hypotheses(
    hypotheses: list,
    session:    str,
    pairs:      list,
    exit_hour:  int,
    source:     str = 'llm',
) -> dict:
    """
    Run the full write → validate → sweep → score → portfolio pipeline
    for a list of hypothesis dicts.  Returns summary counters.
    """
    n_generated = len(hypotheses)
    n_skipped   = 0
    n_swept     = 0
    n_survivors = 0
    sweep_ids   = []

    for hyp in hypotheses:
        runtime_state.checkpoint()  # honour pause/stop between hypotheses
        fn_name        = hyp.get('function_name', '').strip()
        code           = hyp.get('code', '').strip()
        rationale      = hyp.get('rationale', '').strip()
        behaviour_type = hyp.get('behaviour_type', '').strip()
        extra          = hyp.get('additional_params') or {}
        runtime_state.update_status(phase="backtesting",
                                    current_strategy=fn_name or None,
                                    current_session=session,
                                    activity=f"Compiling entry function: {fn_name}")

        if not fn_name or not code:
            log.warning("  Empty hypothesis (%s) — skipping", source)
            n_skipped += 1
            continue

        if db.is_duplicate(code):
            log.info("  Duplicate '%s' (%s) — skipping", fn_name, source)
            runtime_state.update_status(activity=f"Skipped (duplicate): {fn_name}")
            n_skipped += 1
            continue

        db.record_pending(fn_name, code, session, rationale, behaviour_type)

        entry_fn     = None
        last_error   = None
        current_code = code
        current_name = fn_name

        for attempt in range(MAX_CODE_RETRIES):
            try:
                code_writer.write_entry_module(current_name, current_code, session)
                entry_fn = code_writer.load_entry_fn(current_name)
                break
            except SyntaxError as e:
                last_error = str(e)
                log.warning("  Syntax error in '%s' (attempt %d): %s",
                            current_name, attempt + 1, e)
                if attempt < MAX_CODE_RETRIES - 1:
                    fixes = claude_client.fix_syntax_error(current_code, str(e), current_name)
                    if fixes:
                        current_code = fixes[0].get('code', current_code)
                        current_name = fixes[0].get('function_name', current_name)
            except (ValueError, AttributeError) as e:
                last_error = str(e)
                log.warning("  Code validation failed for '%s': %s", current_name, e)
                break
            except Exception as e:
                # 2026-06-06 — catch-all for bulk-LLM output with
                # module-level NameError / TypeError / etc. so one bad
                # generation doesn't kill the round.
                last_error = f"{type(e).__name__}: {e}"
                log.warning("  Load error in '%s': %s", current_name, last_error)
                break

        if entry_fn is None:
            log.error("  Could not compile '%s' after %d attempts: %s",
                      fn_name, MAX_CODE_RETRIES, last_error)
            meta_learner.record_rejection(current_name, session, 'syntax_error', last_error or '')
            # 2026-06-07 — update verdict from 'pending' to 'rejected' so the
            # GUI Results tab reflects the true state instead of every
            # compile-fail looking like a still-running strategy.
            db.record_result(
                strategy_name   = current_name,
                code            = current_code,
                session         = session,
                sweep_id        = '',
                composite_score = 0.0,
                metrics         = {},
                rationale       = f"{rationale}\n\n[REJECTED gate=syntax_error] {last_error or ''}",
                verdict         = 'rejected',
                behaviour_type  = behaviour_type,
                hypothesis_id   = '',
                best_params     = None,
            )
            continue

        # Pre-sweep conditional check: one-pair, one-combo minimal backtest.
        # Skip the full ~20-min sweep on strategies that fire too few trades
        # or produce catastrophic Sharpe on the cheapest pair.
        # 2026-06-07: test the WIDEST grid combo (largest tp_r + largest sl_r)
        # instead of the first (= tightest). With the old "first combo" choice,
        # mean-reversion / drift theses were getting catastrophic-Sharpe-
        # rejected because TP=1.5ATR fires before a +0.45ATR average drift
        # plays out. The widest combo gives the strategy its best chance to
        # show edge — if it still loses, the strategy is truly broken.
        try:
            ps_combos = list(_build_grid(extra)) or [{}]
            ps_first_combo = ps_combos[-1]
        except Exception:
            ps_first_combo = {}
        ps_pair = "EUR_USD" if "EUR_USD" in pairs else (pairs[0] if pairs else None)
        if ps_pair and ps_first_combo:
            runtime_state.update_status(
                current_pair=ps_pair,
                activity=f"Pre-screening {current_name} on {ps_pair}",
            )
            # Per-trade live emission to runtime/live_trades.jsonl so the
            # GUI can render a live equity curve while the backtest runs.
            # File is truncated + a "start" header written here; the
            # callback below appends "trade_closed" / "tick" events.
            _live_trades_start(
                strategy=current_name, pair=ps_pair, session=session,
                stage="pre_screen", params=ps_first_combo,
            )
            ps_passed, ps_reason = pre_screen_mod.pre_screen(
                name        = current_name,
                entry_fn    = entry_fn,
                train_dfs   = _cache['train_dfs'],
                params      = ps_first_combo,
                session     = session,
                stage1_pair = ps_pair,
                exit_hour   = exit_hour,
                cost_mult   = COST_MULT,
                progress_callback = _live_trades_callback,
            )
            _live_trades_done(passed=ps_passed, reason=ps_reason)
            if not ps_passed:
                log.info("  '%s': PRE_SCREEN REJECT — %s", current_name, ps_reason)
                runtime_state.update_status(
                    activity=f"Pre-screen REJECT {current_name}: {ps_reason[:80]}"
                )
                meta_learner.record_rejection(
                    current_name, session, 'pre_screen', ps_reason,
                )
                # 2026-06-07 — flip verdict from 'pending' to 'rejected' so
                # pre_screen rejects don't show up as still-running in the GUI.
                db.record_result(
                    strategy_name   = current_name,
                    code            = current_code,
                    session         = session,
                    sweep_id        = '',
                    composite_score = 0.0,
                    metrics         = {},
                    rationale       = f"{rationale}\n\n[REJECTED gate=pre_screen] {ps_reason}",
                    verdict         = 'rejected',
                    behaviour_type  = behaviour_type,
                    hypothesis_id   = '',
                    best_params     = None,
                )
                n_skipped += 1
                continue

        runtime_state.update_status(
            activity=f"Full sweep: {current_name} across {len(pairs)} pair(s)"
        )
        result = _sweep_one(current_name, entry_fn, extra, session, pairs, exit_hour)
        if result[0] is None:
            continue

        sweep_id, best_metrics, best_score, n_s, robust_passed, survivor_metrics = result
        sweep_ids.append(sweep_id)
        n_swept += 1

        # Extract best-params JSON once so every record_result branch can
        # persist it (lets retests and "what won?" inspections work without
        # joining tested_strategies → edge_results.db via sweep_id).
        try:
            best_params = _json.loads(
                (survivor_metrics or best_metrics or {}).get('params_json') or '{}'
            )
        except Exception:
            best_params = {}

        # Bookkeeping: when a strategy was rejected (static-filter or robustness)
        # but the sweep did produce a top-hypothesis row, copy its metrics into
        # tested_strategies so the Rejected panel shows real numbers next to the
        # gate string instead of None. Survivors and correlated/adversarial
        # branches further down handle their own record_result calls.
        if not robust_passed and best_metrics:
            db.record_result(
                strategy_name   = current_name,
                code            = current_code,
                session         = session,
                sweep_id        = sweep_id,
                composite_score = float(best_score or 0.0),
                metrics         = best_metrics,
                rationale       = rationale,
                verdict         = 'rejected',
                behaviour_type  = behaviour_type,
                hypothesis_id   = best_metrics.get('hypothesis_id', ''),
                best_params     = best_params,
            )

        # Phase 4 — record this hypothesis's raw p-value into the global FDR
        # ledger. Uses the best parameter combo's p_raw (one row per generated
        # strategy, not per param combo — avoids inflating cumulative count
        # with correlated grid neighbours).
        try:
            p_raw = best_metrics.get('p_raw') if best_metrics else None
            if p_raw is not None:
                code_hash = db._code_hash(current_code)
                db.record_hypothesis_pval(
                    code_hash, current_name, sweep_id, float(p_raw),
                )
        except Exception as e:
            log.debug("FDR ledger write failed for '%s': %s", current_name, e)

        if robust_passed and survivor_metrics:
            hyp_id = best_metrics.get('hypothesis_id', '')

            # ── Portfolio correlation filter ──────────────────────────────────
            portfolio_ok, portfolio_reason = portfolio.is_portfolio_additive(
                current_name, hyp_id,
            )

            if not portfolio_ok:
                log.info("  '%s': CORRELATED — %s", current_name, portfolio_reason)
                meta_learner.record_rejection(current_name, session, 'correlated', portfolio_reason)
                db.record_result(
                    strategy_name   = current_name,
                    code            = current_code,
                    session         = session,
                    sweep_id        = sweep_id,
                    composite_score = best_score,
                    metrics         = survivor_metrics,
                    rationale       = rationale,
                    verdict         = 'correlated',
                    behaviour_type  = behaviour_type,
                    hypothesis_id   = hyp_id,
                    best_params     = best_params,
                )
            else:
                # Phase 5.3 — adversarial review immediately before promotion.
                # The adversarial reviewer is intentionally skeptical and looks
                # specifically for look-ahead, peeking, and overfitting. We only
                # block on REJECT; on any reviewer failure (budget out, API
                # exception) it returns PASS so primary statistical filters
                # remain authoritative.
                review = claude_client.review_strategy_for_bias(
                    current_name, current_code, rationale,
                )
                if (review.get('verdict') or 'PASS').upper() == 'REJECT':
                    cat = review.get('category', 'unknown')
                    reason = review.get('reason', '')
                    log.info("  '%s': ADVERSARIAL REJECT (%s) — %s",
                             current_name, cat, reason[:200])
                    meta_learner.record_rejection(
                        current_name, session,
                        f'adversarial_{cat}',
                        reason,
                    )
                    db.record_result(
                        strategy_name   = current_name,
                        code            = current_code,
                        session         = session,
                        sweep_id        = sweep_id,
                        composite_score = best_score,
                        metrics         = survivor_metrics,
                        rationale       = rationale,
                        verdict         = 'adversarial_reject',
                        behaviour_type  = behaviour_type,
                        hypothesis_id   = hyp_id,
                        best_params     = best_params,
                    )
                else:
                    log.info("  '%s': SURVIVOR (%s) — %s", current_name, source, portfolio_reason)
                    n_survivors += 1

                    features = meta_learner.extract_features_used(current_code)

                    db.record_result(
                        strategy_name   = current_name,
                        code            = current_code,
                        session         = session,
                        sweep_id        = sweep_id,
                        composite_score = best_score,
                        metrics         = survivor_metrics,
                        rationale       = rationale,
                        verdict         = 'survivor',
                        behaviour_type  = behaviour_type,
                        hypothesis_id   = hyp_id,
                        best_params     = best_params,
                    )
                    meta_learner.update_survivor_metadata(current_name, features, best_params)

                    # Auto deep-dive: write a full markdown report whenever a new survivor
                    # passes all gates so the user can review without running the CLI manually.
                    try:
                        import agent.analyse as _analyse
                        _analyse.run(days=1, strategy_name=current_name)
                    except Exception as _e:
                        log.warning("Auto-analyse failed for %s: %s", current_name, _e)

    return {
        'session':     session,
        'n_generated': n_generated,
        'n_skipped':   n_skipped,
        'n_swept':     n_swept,
        'n_survivors': n_survivors,
        'sweep_ids':   sweep_ids,
    }


# ── One full round ────────────────────────────────────────────────────────────

def run_one_round(round_n: int) -> dict:
    """
    Full generation → validation → backtest → scoring → meta-feedback cycle.
    Returns summary dict. Never raises.
    """
    session, pairs, exit_hour = session_router.get_current_session(round_n)

    top_results       = db.get_top_results(n=10, days_back=30)
    proven_conditions = [r.get('rationale', '') for r in top_results if r.get('rationale')]
    meta_guidance     = meta_learner.get_current_guidance()

    # Phase 5 — closed-loop generation signals.
    #   bandit_weights:     posterior capital weight per recent survivor; tells
    #                       Claude which families are paying off right now.
    #   saturated_features: features over-represented in failures; tells Claude
    #                       to demote them to secondary filters or skip them.
    bandit_weights: dict | None = None
    try:
        survivor_names = [r['strategy_name'] for r in top_results if r.get('strategy_name')]
        if survivor_names:
            bandit_weights = thompson_bandit.allocate(survivor_names)
    except Exception as e:
        log.debug("bandit allocate failed: %s", e)

    saturated_features: list | None = None
    try:
        sf = meta_learner.get_saturated_features()
        if sf:
            saturated_features = sf
    except Exception as e:
        log.debug("saturated_features query failed: %s", e)

    log.info("Round %d | %s | pairs: %s | meta_guidance: %s | bandit: %s | saturated: %s",
             round_n, session.upper(), ', '.join(pairs),
             f"{len(meta_guidance)} chars" if meta_guidance else "none yet",
             f"{len(bandit_weights)} weights" if bandit_weights else "none",
             f"{len(saturated_features)} features" if saturated_features else "none")

    # 2026-06-05 — Two-stage generation: propose mechanic → validate against
    # historical data → implement only if validated. Drops Claude calls for
    # ~60-80% of generated theses whose claimed bias doesn't exist in the data,
    # so the agent stops burning budget on hypotheses with no chance.
    try:
        hypotheses = _generate_validated_hypotheses(
            session            = session,
            pairs              = pairs,
            proven_conditions  = proven_conditions,
            top_results        = top_results,
            n                  = HYPOTHESES_PER_BATCH,
            meta_guidance      = meta_guidance,
            bandit_weights     = bandit_weights,
            saturated_features = saturated_features,
        )
    except Exception as e:
        log.error("Hypothesis generation failed: %s", e)
        return {'session': session, 'n_generated': 0, 'n_skipped': 0,
                'n_swept': 0, 'n_survivors': 0}

    result = _process_hypotheses(hypotheses, session, pairs, exit_hour, source='llm')

    db.record_round(
        round_n     = round_n,
        session     = session,
        n_generated = result['n_generated'],
        n_skipped   = result['n_skipped'],
        n_swept     = result['n_swept'],
        n_survivors = result['n_survivors'],
        sweep_ids   = result['sweep_ids'],
    )
    code_writer.cleanup_old_modules(keep_n=200)

    # Phase 4 — recompute global Benjamini-Yekutieli FDR every 10 rounds.
    # BY adjusts dependent hypotheses, so the population must be rescored
    # whenever new tests land. Frequency = 10 rounds is a balance: cheap
    # enough that each call covers <100 new rows, but rare enough that
    # we're not paying for it every round.
    if round_n % 10 == 0:
        try:
            summary = db.recompute_global_by_correction()
            log.info("Global FDR recomputed: %s", summary)
        except Exception as e:
            log.warning("Global FDR recompute failed: %s", e)

    log.info("Round %d complete: %s", round_n, result)
    return result


def run_evolver_round(round_n: int) -> dict:
    """
    Generate hypotheses via genetic operators (mutation + crossover) on top survivors.
    Runs through the same pipeline as standard rounds.
    """
    session, pairs, exit_hour = session_router.get_current_session(round_n)
    top_survivors = [r for r in db.get_top_results(n=10, days_back=30) if r.get('code')]

    if not top_survivors:
        log.info("Evolver round %d: no survivors with code yet — skipping", round_n)
        return {'session': session, 'n_generated': 0, 'n_swept': 0, 'n_survivors': 0}

    hypotheses = []
    try:
        hypotheses += evolver.mutate_survivors(top_survivors, session=session)
    except Exception as e:
        log.error("Evolver mutation failed: %s", e)
    try:
        hypotheses += evolver.crossover_survivors(top_survivors, session=session)
    except Exception as e:
        log.error("Evolver crossover failed: %s", e)
    # Typed-GP candidates — zero-API-cost; orthogonal hypothesis space coverage.
    try:
        hypotheses += gp_evolver.gp_generate(top_survivors, n=4)
    except Exception as e:
        log.error("GP evolver failed: %s", e)

    if not hypotheses:
        return {'session': session, 'n_generated': 0, 'n_swept': 0, 'n_survivors': 0}

    log.info("Evolver round %d: %d hypotheses generated", round_n, len(hypotheses))
    result = _process_hypotheses(hypotheses, session, pairs, exit_hour, source='evolver')

    db.record_round(
        round_n     = round_n,
        session     = session,
        n_generated = result['n_generated'],
        n_skipped   = result['n_skipped'],
        n_swept     = result['n_swept'],
        n_survivors = result['n_survivors'],
        sweep_ids   = result['sweep_ids'],
    )
    return result


# ── Main loop ─────────────────────────────────────────────────────────────────

_last_reset_date = None


def _reset_daily_stats_if_new_day(stats: dict):
    global _last_reset_date
    today = datetime.now(timezone.utc).date()
    if _last_reset_date != today:
        stats.clear()
        _last_reset_date = today


def run_loop():
    """Infinite loop — runs until killed. Safe to Ctrl-C."""
    last_report_time     = None
    round_n              = 0
    stats                = {}
    batch_n              = 0
    batch_window_start   = datetime.now(timezone.utc).isoformat()
    tested_at_last_batch = 0

    telegram.send_message(
        "🚀 <b>Edge Discovery Agent started.</b>\n"
        "Generating and backtesting novel entry strategies continuously.\n"
        f"Batch alert every {BATCH_REPORT_EVERY} strategies tested (top {TOP_PER_BATCH} survivors).\n"
        f"Genetic evolution every {EVOLVER_EVERY_N_ROUNDS} rounds.\n"
        f"Daily summary at 18:00 UTC."
    )

    runtime_state.reset_status()
    runtime_state.update_status(phase="downloading")
    if not _load_data():
        log.critical("Initial data load failed — exiting")
        runtime_state.update_status(phase="stopped",
                                    last_error="initial data load failed")
        return

    runtime_state.set_command("run")  # clear any stale stop from a previous run

    while True:
        try:
            runtime_state.checkpoint()
        except runtime_state.StopRequested:
            log.info("Stop command received — exiting agent loop cleanly.")
            runtime_state.update_status(phase="stopped")
            return

        round_n += 1
        try:
            tested_today = db.get_tested_count_since(
                datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
            )
        except Exception:
            tested_today = 0
        runtime_state.update_status(phase="generating", round=round_n,
                                    tested_today=tested_today)
        _reset_daily_stats_if_new_day(stats)

        if _cache_stale():
            _load_data()

        # Standard LLM generation round
        try:
            run_one_round(round_n)
        except runtime_state.StopRequested:
            log.info("Stop received mid-round — exiting.")
            runtime_state.update_status(phase="stopped")
            return
        except Exception as e:
            log.exception("Round %d crashed (continuing loop): %s", round_n, e)
            runtime_state.update_status(last_error=f"round {round_n}: {e}")

        # Genetic evolution round — every EVOLVER_EVERY_N_ROUNDS standard rounds
        if round_n % EVOLVER_EVERY_N_ROUNDS == 0:
            try:
                run_evolver_round(round_n)
            except Exception as e:
                log.error("Evolver round %d crashed: %s", round_n, e)

        # Live trade ingest — pulls fresh MT5 trade/exec CSVs into the agent DB
        # so the meta-learner can compare live execution to backtest expectation.
        if round_n % LIVE_INGEST_EVERY_ROUNDS == 0:
            try:
                live_ingest.ingest_all()
            except Exception as e:
                log.error("Live ingest round %d failed: %s", round_n, e)

        # Auto-promotion: discovered survivors → SHADOW → ramp on live evidence.
        # Idempotent — safe to call every round.
        try:
            promotion.run_auto_promotion()
        except Exception as e:
            log.error("Auto-promotion round %d failed: %s", round_n, e)

        # Batch report: fire every BATCH_REPORT_EVERY strategies tested
        try:
            total_tested = db.get_tested_count_since(
                datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
            )
            if total_tested - tested_at_last_batch >= BATCH_REPORT_EVERY:
                batch_n += 1
                reporter.send_batch_report(
                    batch_n      = batch_n,
                    since_iso    = batch_window_start,
                    total_tested = total_tested,
                )
                tested_at_last_batch = total_tested
                batch_window_start   = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            log.error("Batch report check failed: %s", e)

        # Meta-learner: synthesise new guidance every META_UPDATE_EVERY strategies
        if meta_learner.should_update():
            try:
                meta_learner.update()
            except Exception as e:
                log.error("Meta-learner update failed: %s", e)

        if reporter.should_send_report(last_report_time):
            try:
                last_report_time = reporter.send_daily_report(stats)
            except Exception as e:
                log.error("Daily report failed: %s", e)

        runtime_state.update_status(phase="idle")
        # Sleep in 1-second increments so a stop/pause command takes effect quickly.
        slept = 0.0
        while slept < LOOP_SLEEP_SECONDS:
            try:
                runtime_state.checkpoint(poll_secs=1.0)
            except runtime_state.StopRequested:
                runtime_state.update_status(phase="stopped")
                return
            time.sleep(1.0)
            slept += 1.0
