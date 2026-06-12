"""
Post-backtest robustness checks for agent-discovered strategies.

These run after is_survivor() passes the static filter gates, and require
loading the actual trade-level data. All four checks must pass for a strategy
to be recorded as a survivor.

Checks:
  1. Monte Carlo simulation  — resamples daily PnL 10k times against prop rules
  2. Walk-forward consistency — edge must be profitable in ≥ 3 of 4 time folds
  3. Bootstrap Sharpe CI     — lower 95% CI bound must be positive
  4. Parameter sensitivity   — best combo's neighbours must also be viable
"""

import json
import logging
import sys
from pathlib import Path

_proj = str(Path(__file__).parent.parent)
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import edge_engine as eng

from agent.config import (
    MC_N_SIMS, MC_CHALLENGE_DAYS, MC_MIN_PASS_PCT, MC_MAX_BLOWN_PCT,
    MIN_SHARPE_CI_LOW,
    WF_N_FOLDS, WF_MIN_PROFITABLE_FOLDS,
    PARAM_SENS_MIN_VIABLE_FRAC,
    MIN_TEST_TRADES,
)

log = logging.getLogger(__name__)


# ── 1. Monte Carlo ─────────────────────────────────────────────────────────────

def check_monte_carlo(trades_df) -> dict:
    """
    Run 10,000 prop-challenge simulations on resampled daily PnL.

    Resamples trade-level PnL into daily buckets then bootstraps sequences of
    MC_CHALLENGE_DAYS trading days. A simulation "passes" when cumulative profit
    hits the prop target before hitting the max drawdown limit.

    Returns:
        {passed, pass_pct, blown_pct, verdict, detail}
    """
    if trades_df is None or trades_df.empty:
        return {'passed': False, 'detail': 'no trades data'}

    try:
        result = eng.run_monte_carlo(
            trades_df,
            label          = 'agent_check',
            n_sims         = MC_N_SIMS,
            challenge_days = MC_CHALLENGE_DAYS,
        )
    except Exception as e:
        return {'passed': False, 'detail': f'MC error: {e}'}

    if not result:
        return {'passed': False, 'detail': 'MC returned empty'}

    pass_pct  = result.get('pass_pct', 0.0)
    blown_pct = result.get('blown_pct', 100.0)
    verdict   = result.get('verdict', 'NOT VIABLE')

    passed = pass_pct >= MC_MIN_PASS_PCT and blown_pct <= MC_MAX_BLOWN_PCT

    return {
        'passed':    passed,
        'pass_pct':  round(pass_pct, 1),
        'blown_pct': round(blown_pct, 1),
        'verdict':   verdict,
        'detail':    (
            f"MC: {pass_pct:.1f}% pass / {blown_pct:.1f}% blown "
            f"({MC_N_SIMS:,} sims, {MC_CHALLENGE_DAYS}d)"
        ),
    }


# ── 2. Walk-forward consistency ────────────────────────────────────────────────

def check_walk_forward(trades_df) -> dict:
    """
    Split test trades into WF_N_FOLDS time-ordered folds and check that
    at least WF_MIN_PROFITABLE_FOLDS are profitable.

    Catches strategies that only worked in one market period — a real edge
    shows positive expectancy across the full out-of-sample window.

    Returns:
        {passed, profitable_folds, total_folds, detail}
    """
    if trades_df is None or trades_df.empty:
        return {'passed': False, 'detail': 'no trades data'}

    try:
        folds_df = eng.walk_forward_test(trades_df, n_folds=WF_N_FOLDS)
    except Exception as e:
        return {'passed': False, 'detail': f'walk-forward error: {e}'}

    if folds_df is None or folds_df.empty:
        return {'passed': False, 'detail': 'walk-forward returned empty'}

    profitable = int((folds_df['pnl'] > 0).sum())
    total      = len(folds_df)
    passed     = profitable >= WF_MIN_PROFITABLE_FOLDS

    fold_summary = ', '.join(
        f"F{int(r['fold'])}:{r['pnl']:+.0f}"
        for _, r in folds_df.iterrows()
    )

    return {
        'passed':           passed,
        'profitable_folds': profitable,
        'total_folds':      total,
        'detail':           f"WF: {profitable}/{total} folds profitable [{fold_summary}]",
    }


# ── 3. Bootstrap Sharpe CI lower bound ────────────────────────────────────────

def check_bootstrap_ci(sharpe_ci_low) -> dict:
    """
    The lower 95% CI bound of the bootstrap Sharpe distribution must be
    above MIN_SHARPE_CI_LOW (default 0).

    A strategy where the lower CI dips below zero means that in unlucky
    but plausible draws, the edge disappears entirely.

    Returns:
        {passed, ci_low, detail}
    """
    import math
    if sharpe_ci_low is None or (isinstance(sharpe_ci_low, float) and math.isnan(sharpe_ci_low)):
        return {'passed': False, 'ci_low': None, 'detail': 'bootstrap CI not available'}

    ci_low = float(sharpe_ci_low)
    passed = ci_low > MIN_SHARPE_CI_LOW

    return {
        'passed': passed,
        'ci_low': round(ci_low, 3),
        'detail': f"CI lower bound: {ci_low:.3f} (min {MIN_SHARPE_CI_LOW})",
    }


# ── 4. Parameter sensitivity ──────────────────────────────────────────────────

def check_parameter_sensitivity(sweep_rows: list, best_params: dict) -> dict:
    """
    Check that the best parameter combo is not an isolated lucky peak.

    Finds all sweep rows that differ from best_params by exactly one parameter
    step (adjacent neighbours in parameter space). Requires that at least
    PARAM_SENS_MIN_VIABLE_FRAC of those neighbours also have test_sharpe > 0
    and test_n >= MIN_TEST_TRADES.

    Real edges are broad plateaus. Curve-fits are sharp peaks.

    Returns:
        {passed, n_neighbours, n_viable, frac_viable, detail}
    """
    import math

    if not best_params or len(sweep_rows) < 3:
        return {
            'passed': True,
            'detail': 'insufficient sweep rows for sensitivity check (skipped)',
        }

    # Build lookup: params_json → metrics
    param_to_metrics = {}
    for row in sweep_rows:
        try:
            p = json.loads(row.get('params_json') or '{}')
            param_to_metrics[json.dumps(p, sort_keys=True)] = row
        except Exception:
            continue

    # Collect the value lists for each parameter
    param_values = {}
    for row in sweep_rows:
        try:
            p = json.loads(row.get('params_json') or '{}')
            for k, v in p.items():
                param_values.setdefault(k, set()).add(v)
        except Exception:
            continue
    param_values = {k: sorted(v) for k, v in param_values.items()}

    # Find neighbours: combos that differ by exactly one parameter step
    neighbours = []
    for param_name, values in param_values.items():
        current_val = best_params.get(param_name)
        if current_val is None:
            continue
        try:
            idx = values.index(current_val)
        except ValueError:
            continue
        for adj_idx in [idx - 1, idx + 1]:
            if 0 <= adj_idx < len(values):
                neighbour_params = dict(best_params)
                neighbour_params[param_name] = values[adj_idx]
                key = json.dumps(neighbour_params, sort_keys=True)
                if key in param_to_metrics:
                    neighbours.append(param_to_metrics[key])

    if not neighbours:
        return {
            'passed': True,
            'detail': 'no neighbours found in sweep (single-combo grid)',
        }

    def _viable(row):
        sharpe = row.get('test_sharpe')
        n      = row.get('test_n', 0)
        if sharpe is None or (isinstance(sharpe, float) and math.isnan(sharpe)):
            return False
        return float(sharpe) > 0 and float(n or 0) >= MIN_TEST_TRADES

    n_viable   = sum(1 for nb in neighbours if _viable(nb))
    n_total    = len(neighbours)
    frac       = n_viable / n_total if n_total > 0 else 0.0
    passed     = frac >= PARAM_SENS_MIN_VIABLE_FRAC

    return {
        'passed':      passed,
        'n_neighbours': n_total,
        'n_viable':    n_viable,
        'frac_viable': round(frac, 2),
        'detail':      f"Param sensitivity: {n_viable}/{n_total} neighbours viable ({frac:.0%})",
    }


# ── Combined entry point ───────────────────────────────────────────────────────

def run_all_checks(best_metrics: dict, sweep_rows: list, trades_df) -> tuple:
    """
    Run all four robustness checks against the best hypothesis in a sweep.

    Args:
        best_metrics:  the hypothesis row dict that passed is_survivor()
        sweep_rows:    all hypothesis rows from the sweep (for param sensitivity)
        trades_df:     test-split trade log for the best hypothesis

    Returns:
        (all_passed: bool, report: dict)
        report keys: mc, walk_forward, bootstrap_ci, param_sensitivity
    """
    import json as _json
    try:
        best_params = _json.loads(best_metrics.get('params_json') or '{}')
    except Exception:
        best_params = {}

    sharpe_ci_low = best_metrics.get('sharpe_ci_low')

    mc   = check_monte_carlo(trades_df)
    wf   = check_walk_forward(trades_df)
    ci   = check_bootstrap_ci(sharpe_ci_low)
    sens = check_parameter_sensitivity(sweep_rows, best_params)

    report = {
        'mc':                mc,
        'walk_forward':      wf,
        'bootstrap_ci':      ci,
        'param_sensitivity': sens,
    }

    all_passed = mc['passed'] and wf['passed'] and ci['passed'] and sens['passed']

    if not all_passed:
        failures = [
            name for name, r in report.items() if not r.get('passed', True)
        ]
        log.info("  Robustness FAILED: %s", ', '.join(failures))
        for name in failures:
            log.info("    %s: %s", name, report[name].get('detail', ''))
    else:
        log.info(
            "  Robustness PASSED: MC %.1f%% | WF %d/%d folds | CI low %.3f",
            mc.get('pass_pct', 0),
            wf.get('profitable_folds', 0), wf.get('total_folds', WF_N_FOLDS),
            ci.get('ci_low', 0) or 0,
        )

    return all_passed, report
