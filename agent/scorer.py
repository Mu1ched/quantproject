"""
Composite scoring and survivor filtering. Pure functions — no I/O.
"""

import math

from agent.config import (
    COMPOSITE_WEIGHTS,
    MIN_TEST_TRADES, MIN_DSR, MIN_TEST_SHARPE,
    REQUIRE_BH_SIG, REQUIRE_REGIME_STABLE,
    MAX_TRAIN_TEST_DECAY, MAX_TEST_DRAWDOWN,
    MIN_SHARPE_CI_LOW,
    BAYES_PRIOR_SHARPE, BAYES_PRIOR_N,
    MAX_PBO, MIN_PSR,
    USE_WF_GATE, WF_GATE_MIN_SHARPE,  # review#18
    USE_MC_GATE, MC_MIN_PASS_PCT, MC_MAX_BLOWN_PCT,  # §1 Tier 1
)


def _safe(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bayesian_shrunk_sharpe(raw_sharpe: float, n_trades: float,
                           prior: float = BAYES_PRIOR_SHARPE,
                           prior_n: float = BAYES_PRIOR_N) -> float:
    """
    Shrink a raw Sharpe estimate toward a prior using sample size.

        posterior = (n × raw + prior_n × prior) / (n + prior_n)

    With prior_n = 50, a strategy with 50 trades gets 50/50 raw/prior weighting
    and a strategy with 500 trades gets 91% raw weight. Penalises high Sharpes
    coming from thin trade samples — exactly the failure mode that survives
    backtest then dies live.
    """
    n = max(float(n_trades or 0), 0.0)
    if n <= 0:
        return float(prior)
    return (n * float(raw_sharpe) + prior_n * float(prior)) / (n + prior_n)


def composite_score(metrics: dict) -> float:
    """
    Score a hypothesis result on a [0, 1] scale.

    Weights (from config.COMPOSITE_WEIGHTS):
      test_sharpe  0.35 — primary quality signal
      dsr          0.25 — corrects for selection bias across the sweep
      test_wr      0.15 — excess win rate above 40% floor
      regime_stable 0.10 — edge works across multiple market regimes
      n_trades     0.10 — logarithmic confidence from trade count
      max_dd       0.05 — lower drawdown is marginally preferred
    """
    raw_sharpe  = _safe(metrics.get('test_sharpe'))
    dsr         = _safe(metrics.get('dsr'))
    mc_pass_raw = _safe(metrics.get('mc_eval_pass_pct'))   # §1 Tier 1, 0-100 scale
    wr          = _safe(metrics.get('test_wr'))
    regime      = _safe(metrics.get('regime_stable'))
    n           = _safe(metrics.get('test_n'))
    max_dd      = abs(_safe(metrics.get('test_max_dd')))

    # Bayesian shrinkage: collapses thin-sample Sharpe spikes that would
    # otherwise rank above structurally-sound strategies with more trades.
    sharpe = bayesian_shrunk_sharpe(raw_sharpe, n)

    sharpe_score  = _clip(sharpe / 3.0, 0.0, 1.0)
    dsr_score     = _clip(dsr, 0.0, 1.0)
    mc_pass_score = _clip(mc_pass_raw / 100.0, 0.0, 1.0)   # §1 Tier 1
    wr_score      = _clip((wr - 0.40) / 0.30, 0.0, 1.0)
    regime_score  = _clip(regime, 0.0, 1.0)
    n_score       = _clip(math.log(max(n, 1.0)) / math.log(500.0), 0.0, 1.0)
    dd_score      = _clip(1.0 - max_dd / 0.20, 0.0, 1.0)

    w = COMPOSITE_WEIGHTS
    return (
        sharpe_score   * w['test_sharpe']        +
        dsr_score      * w['dsr']                +
        mc_pass_score  * w.get('mc_eval_pass_pct', 0.0)  +   # §1 Tier 1
        wr_score       * w['test_wr']            +
        regime_score   * w['regime_stable']      +
        n_score        * w['n_trades']           +
        dd_score       * w['max_dd']
    )


def is_survivor(metrics: dict) -> bool:
    """
    Six-gate quality filter. All must pass. Targets roughly the top 2% of
    tested hypotheses — equivalent to what a rigorous human researcher would
    shortlist for further review.

    Gates (in order of cheapest to compute):
      1. Minimum trade count           — statistical floor
      2. BH-corrected significance     — controls false discovery rate
      3. Regime stability              — edge must hold across market conditions
      4. Minimum test Sharpe           — out-of-sample quality floor
      5. Minimum DSR                   — corrects for selection bias in the sweep
      6. Train→test decay              — rejects overfitted strategies
      7. Max drawdown                  — rejects strategies with unacceptable risk
    """
    # 1. Trade count
    if _safe(metrics.get('test_n')) < MIN_TEST_TRADES:
        return False

    # 2. Per-sweep FDR significance — prefer Benjamini-Yekutieli (handles
    # dependent hypotheses, which is what correlated grid neighbours actually
    # are) and fall back to Benjamini-Hochberg only if BY hasn't been
    # populated yet (older rows from before Phase 4).
    if REQUIRE_BH_SIG:
        by_sig = metrics.get('by_sig')
        if by_sig is not None:
            if int(by_sig or 0) != 1:
                return False
        elif int(metrics.get('bh_sig') or 0) != 1:
            return False

    # 3. Regime stability
    if REQUIRE_REGIME_STABLE and int(metrics.get('regime_stable') or 0) != 1:
        return False

    # 4. Test Sharpe floor
    # review#P2#3 — gate on Bayesian-shrunk Sharpe, not raw. Thin-sample
    # strategies with high raw Sharpe are exactly the failure mode the
    # shrinkage was designed for; comparing raw against MIN_TEST_SHARPE
    # admitted them. Composite ranking already used the shrunk value; the
    # gate now matches.
    test_sharpe = _safe(metrics.get('test_sharpe'), -99.0)
    n_trades    = _safe(metrics.get('test_n'))
    test_sharpe_shrunk = bayesian_shrunk_sharpe(test_sharpe, n_trades)
    if test_sharpe_shrunk < MIN_TEST_SHARPE:
        return False

    # 5. Deflated Sharpe floor
    if _safe(metrics.get('dsr'), -99.0) < MIN_DSR:
        return False

    # 6. Train→test decay: test Sharpe must be ≥ MAX_TRAIN_TEST_DECAY × train Sharpe
    #    Guards against in-sample overfitting masquerading as a good result
    train_sharpe = _safe(metrics.get('train_sharpe'), 0.0)
    if train_sharpe > 0 and test_sharpe < train_sharpe * MAX_TRAIN_TEST_DECAY:
        return False

    # 7. Drawdown ceiling
    if abs(_safe(metrics.get('test_max_dd'))) > MAX_TEST_DRAWDOWN:
        return False

    # 8. Bootstrap Sharpe CI lower bound — even in unlucky draws, edge must exist
    ci_low = metrics.get('sharpe_ci_low')
    if ci_low is not None and not math.isnan(float(ci_low)):
        if float(ci_low) <= MIN_SHARPE_CI_LOW:
            return False

    # 9. Probability of Backtest Overfitting (Phase 7). Only applied when the
    # field has been populated — older rows pre-Phase-7 stay grandfathered.
    pbo = metrics.get('pbo_score')
    if pbo is not None and not math.isnan(float(pbo)):
        if float(pbo) > MAX_PBO:
            return False

    # 10. Probabilistic Sharpe Ratio (Phase 7). Same grandfathering rule.
    psr = metrics.get('psr')
    if psr is not None and not math.isnan(float(psr)):
        if float(psr) < MIN_PSR:
            return False

    # 11. Walk-forward minimum Sharpe (review#18). Gated behind USE_WF_GATE
    # so existing population isn't retro-rejected; flip the flag once a
    # round of post-restart hypotheses has wf_sharpe_min populated and the
    # cutoff has been calibrated against the observed distribution. NULL is
    # grandfathered (older rows or strategies with too few trades to fold).
    if USE_WF_GATE:
        wf_min = metrics.get('wf_sharpe_min')
        if wf_min is not None and not math.isnan(float(wf_min)):
            if float(wf_min) < WF_GATE_MIN_SHARPE:
                return False

    # §1 Tier 1 back-port (TIER1_3_PLAN.md) — MC pass-rate gate. On FX,
    # USE_MC_GATE defaults to False so the existing 1,423 rows aren't
    # retro-rejected. Numbers backfill on next sweep regardless of flag.
    if USE_MC_GATE:
        mc_pass = metrics.get('mc_eval_pass_pct')
        if mc_pass is not None and not math.isnan(float(mc_pass)):
            if float(mc_pass) < MC_MIN_PASS_PCT:
                return False
        mc_blown = metrics.get('mc_blown_pct')
        if mc_blown is not None and not math.isnan(float(mc_blown)):
            if float(mc_blown) > MC_MAX_BLOWN_PCT:
                return False

    return True


def rejection_reason(metrics: dict) -> str:
    """Return a short string explaining why a hypothesis was rejected (for logging)."""
    if _safe(metrics.get('test_n')) < MIN_TEST_TRADES:
        return f"too few trades ({int(_safe(metrics.get('test_n')))} < {MIN_TEST_TRADES})"
    if REQUIRE_BH_SIG and int(metrics.get('bh_sig') or 0) != 1:
        p_adj = metrics.get('p_adj')
        p_str = f"{p_adj:.3f}" if isinstance(p_adj, (int, float)) else str(p_adj)
        return f"failed BH correction (p_adj={p_str})"
    if REQUIRE_REGIME_STABLE and int(metrics.get('regime_stable') or 0) != 1:
        return "not regime-stable"
    test_sharpe = _safe(metrics.get('test_sharpe'), -99.0)
    n_trades_for_shrink = _safe(metrics.get('test_n'))
    test_sharpe_shrunk  = bayesian_shrunk_sharpe(test_sharpe, n_trades_for_shrink)
    if test_sharpe_shrunk < MIN_TEST_SHARPE:
        return (f"test Sharpe (shrunk) too low ({test_sharpe_shrunk:.2f} < "
                f"{MIN_TEST_SHARPE}; raw={test_sharpe:.2f}, n={int(n_trades_for_shrink)})")
    if _safe(metrics.get('dsr'), -99.0) < MIN_DSR:
        return f"DSR too low ({_safe(metrics.get('dsr')):.2f} < {MIN_DSR})"
    train_sharpe = _safe(metrics.get('train_sharpe'), 0.0)
    test_sharpe  = _safe(metrics.get('test_sharpe'), -99.0)
    if train_sharpe > 0 and test_sharpe < train_sharpe * MAX_TRAIN_TEST_DECAY:
        return f"train→test decay ({test_sharpe:.2f} vs train {train_sharpe:.2f})"
    if abs(_safe(metrics.get('test_max_dd'))) > MAX_TEST_DRAWDOWN:
        return f"drawdown too large ({abs(_safe(metrics.get('test_max_dd'))):.1%})"
    ci_low = metrics.get('sharpe_ci_low')
    if ci_low is not None and not math.isnan(float(ci_low)) and float(ci_low) <= MIN_SHARPE_CI_LOW:
        return f"bootstrap CI lower bound {float(ci_low):.3f} ≤ {MIN_SHARPE_CI_LOW}"
    pbo = metrics.get('pbo_score')
    if pbo is not None and not math.isnan(float(pbo)) and float(pbo) > MAX_PBO:
        return f"PBO too high ({float(pbo):.2f} > {MAX_PBO})"
    psr = metrics.get('psr')
    if psr is not None and not math.isnan(float(psr)) and float(psr) < MIN_PSR:
        return f"PSR too low ({float(psr):.2f} < {MIN_PSR})"
    if USE_WF_GATE:  # review#18
        wf_min = metrics.get('wf_sharpe_min')
        if (wf_min is not None and not math.isnan(float(wf_min))
                and float(wf_min) < WF_GATE_MIN_SHARPE):
            return f"WF min Sharpe too low ({float(wf_min):.2f} < {WF_GATE_MIN_SHARPE})"
    if USE_MC_GATE:  # §1 Tier 1
        mc_pass = metrics.get('mc_eval_pass_pct')
        if (mc_pass is not None and not math.isnan(float(mc_pass))
                and float(mc_pass) < MC_MIN_PASS_PCT):
            return f"MC eval pass rate too low ({float(mc_pass):.1f}% < {MC_MIN_PASS_PCT}%)"
        mc_blown = metrics.get('mc_blown_pct')
        if (mc_blown is not None and not math.isnan(float(mc_blown))
                and float(mc_blown) > MC_MAX_BLOWN_PCT):
            return f"MC blown pct too high ({float(mc_blown):.1f}% > {MC_MAX_BLOWN_PCT}%)"
    return "unknown"


def pick_best_from_sweep(rows: list) -> tuple:
    """
    From a list of hypothesis result dicts for one sweep, return
    (best_metrics, best_score, n_survivors).
    """
    n_survivors  = 0
    best_score   = -1.0
    best_metrics = {}
    for row in rows:
        if is_survivor(row):
            n_survivors += 1
            score = composite_score(row)
            if score > best_score:
                best_score   = score
                best_metrics = row
    return best_metrics, best_score, n_survivors
