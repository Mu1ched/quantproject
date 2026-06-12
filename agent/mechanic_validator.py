"""
Stage 2 — deterministic empirical validator.

Takes a structured "mechanic" specification (from Claude in Stage 1), checks
whether the mechanic's claimed forward-return bias actually exists in the
training data. No LLM involved — pure pandas/NumPy. If validation fails,
Stage 3 (code generation) is skipped entirely.

Why this exists:
  Phase 2 of the agent loop kept generating strategies with hundreds of trades
  but catastrophic Sharpe (-6+). That means Claude was right that the entry
  condition fires often, but wrong about which direction has expectancy. By
  testing the directional bias BEFORE writing code, we filter out theses
  whose claimed mechanic doesn't actually exist.

  Cost: one API call (Stage 1) instead of two (Stage 1 + Stage 3) on rejected
  theses — typically 60-80% of generations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Whitelist of features Claude may reference in trigger/context filters.
# Mirrors columns we verified exist in edge_prepared_cache/*.parquet earlier.
# Any feature outside this list = validation fails immediately.
ALLOWED_FEATURES: set[str] = {
    # Time features
    "hour", "minute", "dow",
    # Price + range
    "open", "high", "low", "close",
    "range_high", "range_low",
    "asian_high", "asian_low",
    # ATR / vol
    "atr", "atr_ratio", "realized_vol", "rv_median", "yz_vol", "yz_vol_ratio",
    # Trend
    "ma_trend", "ma_dist", "adx", "hurst",
    # Order flow
    "tick_imbalance", "vol_imbalance", "persistent_imbalance",
    "delta", "delta_momentum",
    # Spread / liquidity
    "spread_mean", "spread_median", "spread_adj",
    # Regime
    "regime", "hmm_state",
    # News / calendar
    "near_news",
}

# Pre-existing JSON-schema for the structured mechanic spec we want Claude
# to emit. Used by claude_client.py to constrain Stage 1's tool output.
MECHANIC_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanic_id": {
            "type": "string",
            "description": "Short snake_case identifier, e.g. 'ny_atr_spike_revert'.",
        },
        "rationale": {
            "type": "string",
            "description": "Plain-English mechanical thesis: who is on the other "
                           "side, why they take it, why edge persists.",
        },
        "trigger": {
            "type": "object",
            "description": "Primary trigger — the condition whose firing defines "
                           "an event. Must reference an ALLOWED_FEATURES column.",
            "properties": {
                "feature":    {"type": "string"},
                "comparison": {"type": "string", "enum": [">", "<", ">=", "<="]},
                "threshold":  {"type": "number"},
            },
            "required": ["feature", "comparison", "threshold"],
        },
        "context_filters": {
            "type": "array",
            "description": "Additional context that must also be true (e.g. "
                           "hour-of-day, near_news=False). Empty array = no extra filtering.",
            "items": {
                "type": "object",
                "properties": {
                    "feature":    {"type": "string"},
                    "comparison": {"type": "string",
                                   "enum": [">", "<", ">=", "<=", "==", "!=", "in"]},
                    "threshold":  {"type": ["number", "string", "boolean"]},
                    "values":     {"type": "array",
                                   "description": "Used with comparison='in'."},
                },
                "required": ["feature", "comparison"],
            },
        },
        "direction_hypothesis": {
            "type": "string",
            "enum": ["long", "short"],
            "description": "Predicted direction of forward return. The validator "
                           "checks that mean forward return has this sign.",
        },
        "forward_horizon_bars": {
            "type": "integer",
            "enum": [5, 15, 30, 60],
            "description": "How many bars after the event we measure return over.",
        },
        "pair_universe": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Pairs the mechanic should hold on. Use ['EUR_USD'] for "
                           "single-pair, multiple for cross-pair confirmation.",
        },
    },
    "required": [
        "mechanic_id", "rationale", "trigger",
        "direction_hypothesis", "forward_horizon_bars",
    ],
}


@dataclass
class ValidationResult:
    passed:          bool
    reason:          str
    n_events:        int = 0
    mean_fwd_return: float = 0.0
    t_stat:          float = 0.0
    direction_match: bool = False
    per_pair:        dict = field(default_factory=dict)


# ── Internal helpers ────────────────────────────────────────────────────────

def _apply_filter(df: pd.DataFrame, feat: str, comp: str,
                  threshold=None, values=None) -> pd.Series:
    """Vectorized filter over a DataFrame. Returns boolean mask of len(df).

    Type-coercion robustness (2026-06-05): Claude sometimes proposes a string
    threshold for a numeric column (or vice-versa). For numeric columns we
    cast the threshold to float; if that fails we return an all-False mask
    so the mechanic is gracefully rejected as "no events" rather than
    crashing the round.
    """
    if feat not in df.columns:
        return pd.Series(False, index=df.index)
    col = df[feat]
    # Numeric coercion for numeric columns when threshold is the wrong type.
    if pd.api.types.is_numeric_dtype(col) and threshold is not None:
        if isinstance(threshold, str):
            try:
                threshold = float(threshold)
            except (ValueError, TypeError):
                return pd.Series(False, index=df.index)
    try:
        if comp == ">":   return col >  threshold
        if comp == "<":   return col <  threshold
        if comp == ">=":  return col >= threshold
        if comp == "<=":  return col <= threshold
        if comp == "==":  return col == threshold
        if comp == "!=":  return col != threshold
        if comp == "in":  return col.isin(values or [])
    except (TypeError, ValueError):
        # Mismatched dtype that we couldn't coerce — reject mechanic
        return pd.Series(False, index=df.index)
    return pd.Series(False, index=df.index)


def _build_event_mask(df: pd.DataFrame, mechanic: dict) -> pd.Series:
    """Apply trigger + all context filters, AND-combined."""
    trig = mechanic["trigger"]
    mask = _apply_filter(df, trig["feature"], trig["comparison"],
                         trig.get("threshold"))
    for ctx in mechanic.get("context_filters") or []:
        mask = mask & _apply_filter(df, ctx["feature"], ctx["comparison"],
                                     ctx.get("threshold"), ctx.get("values"))
    return mask.fillna(False).astype(bool)


def _stats_for_pair(df: pd.DataFrame, mechanic: dict) -> dict:
    """Per-pair statistics. df is one pair's full train DataFrame."""
    horizon = mechanic["forward_horizon_bars"]
    mask = _build_event_mask(df, mechanic)
    # Drop events too close to end (forward return NaN)
    mask.iloc[-horizon:] = False
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "mean_fwd": 0.0, "t_stat": 0.0,
                "mean_fwd_atr": 0.0, "reason": "0 events"}

    fwd_ret = df["close"].shift(-horizon) - df["close"]
    atr     = df["atr"]
    sub_fwd = fwd_ret[mask].dropna()
    sub_atr = atr[mask].reindex(sub_fwd.index).ffill().fillna(1e-4)

    # Normalised forward return = price-units / ATR (unitless, comparable)
    norm = (sub_fwd / sub_atr).dropna()
    n = int(len(norm))
    if n < 2:
        return {"n": n, "mean_fwd": 0.0, "t_stat": 0.0,
                "mean_fwd_atr": 0.0, "reason": f"only {n} valid samples"}

    mean = float(norm.mean())
    std  = float(norm.std())
    t    = mean / (std / np.sqrt(n)) if std > 0 else 0.0
    return {
        "n": n,
        "mean_fwd_atr": round(mean, 4),
        "mean_fwd": round(float(sub_fwd.mean()), 6),
        "t_stat": round(float(t), 3),
        "reason": "ok",
    }


# ── Public API ──────────────────────────────────────────────────────────────

def validate_mechanic(
    mechanic:        dict,
    train_dfs:       dict[str, pd.DataFrame],
    min_n_per_pair:  int = 50,
    min_abs_t_stat:  float = 1.5,
    min_abs_mean_atr: float = 0.20,
    require_cross_pair: bool = False,
) -> ValidationResult:
    """Empirically validate a mechanic against train data.

    Pass criteria (all must hold on at least one pair, or both pairs if
    require_cross_pair=True):
      - Trigger + context features must all be in ALLOWED_FEATURES.
      - On each pair in pair_universe (intersected with train_dfs.keys()):
        * n_events ≥ min_n_per_pair                   (sample size)
        * |t_stat| ≥ min_abs_t_stat                   (statistical significance)
        * |mean_fwd_atr| ≥ min_abs_mean_atr           (economic magnitude)
        * sign(mean_fwd_return) matches direction_hypothesis  (correct side)
      - If require_cross_pair: at least 2 pairs must pass with same direction.

    The magnitude threshold (min_abs_mean_atr) is the key economic gate —
    statistically-significant bias below the cost floor produces losing
    strategies in backtest. 0.20 ATR units ≈ 3× typical retail cost floor
    on majors, giving margin for slippage + TP/SL inefficiency.
    """
    # ── Schema validation: feature whitelist ────────────────────────────────
    referenced = {mechanic["trigger"]["feature"]}
    for ctx in mechanic.get("context_filters") or []:
        referenced.add(ctx["feature"])
    unknown = referenced - ALLOWED_FEATURES
    if unknown:
        return ValidationResult(
            passed=False,
            reason=f"unknown features in mechanic: {sorted(unknown)} "
                   f"(allowed: {sorted(ALLOWED_FEATURES)})",
        )

    # ── Direction hypothesis must have expected sign ────────────────────────
    expected_sign = 1 if mechanic["direction_hypothesis"] == "long" else -1

    # ── Pair universe ───────────────────────────────────────────────────────
    requested_pairs = mechanic.get("pair_universe") or list(train_dfs.keys())
    pairs = [p for p in requested_pairs if p in train_dfs]
    if not pairs:
        return ValidationResult(
            passed=False,
            reason=f"none of pair_universe {requested_pairs} in train data "
                   f"(have: {list(train_dfs.keys())})",
        )

    # ── Compute per-pair statistics ─────────────────────────────────────────
    per_pair = {}
    passing_pairs = []
    for p in pairs:
        s = _stats_for_pair(train_dfs[p], mechanic)
        per_pair[p] = s
        if s["n"] < min_n_per_pair:
            continue
        if abs(s["t_stat"]) < min_abs_t_stat:
            continue
        # Sign of normalised fwd return must match direction hypothesis
        if np.sign(s["mean_fwd_atr"]) != expected_sign:
            continue
        # Magnitude must clear the cost floor — statistically significant
        # but economically trivial mechanics produce losing backtests.
        if abs(s["mean_fwd_atr"]) < min_abs_mean_atr:
            continue
        passing_pairs.append(p)

    # ── Aggregate verdict ───────────────────────────────────────────────────
    if not passing_pairs:
        # Provide concise per-pair summary
        summary = "; ".join(
            f"{p}: n={per_pair[p]['n']}, t={per_pair[p]['t_stat']:+.2f}, "
            f"mean_atr={per_pair[p]['mean_fwd_atr']:+.3f}"
            for p in pairs
        )
        return ValidationResult(
            passed=False,
            reason=f"no pair passed (need n≥{min_n_per_pair}, |t|≥{min_abs_t_stat}, "
                   f"|mean_atr|≥{min_abs_mean_atr}, sign={mechanic['direction_hypothesis']}). "
                   f"{summary}",
            per_pair=per_pair,
        )

    if require_cross_pair and len(passing_pairs) < 2:
        return ValidationResult(
            passed=False,
            reason=f"only 1 pair passed ({passing_pairs[0]}); cross-pair "
                   f"required ≥2",
            per_pair=per_pair,
            n_events=per_pair[passing_pairs[0]]["n"],
        )

    # Aggregate stats across passing pairs
    total_n = sum(per_pair[p]["n"] for p in passing_pairs)
    weighted_t = sum(per_pair[p]["t_stat"] * per_pair[p]["n"] for p in passing_pairs) / total_n
    weighted_mean = sum(per_pair[p]["mean_fwd_atr"] * per_pair[p]["n"] for p in passing_pairs) / total_n

    return ValidationResult(
        passed=True,
        reason=f"validated on {len(passing_pairs)} pair(s): {passing_pairs}",
        n_events=total_n,
        mean_fwd_return=round(float(weighted_mean), 4),
        t_stat=round(float(weighted_t), 3),
        direction_match=True,
        per_pair=per_pair,
    )
