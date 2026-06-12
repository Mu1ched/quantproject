"""
Deterministic mechanic discovery — replaces LLM Stage-1 proposal.

Brute-forces the (feature × threshold × session × horizon) tuple space against
train data and emits the survivors as MECHANIC_SCHEMA-shaped JSON that the
agent's existing _generate_validated_hypotheses loop can consume verbatim.

Filters (pre-committed — do NOT loosen):
  - n_events ≥ 50 per pair
  - |t_stat| ≥ 2.0
  - |mean_atr| ≥ 0.20
  - 6-window stability (≥4 same-signed windows, t_window > 1.5)
  - ≥2 pairs same-signed direction

Every emitted mechanic round-trips through `agent.mechanic_validator.validate_mechanic`
as a self-check, so the discovery filters cannot drift from the live validator.

Usage:
    python tools/discover_mechanics.py --pairs EUR_USD,GBP_USD --max-out 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import edge_engine as eng
from agent.mechanic_validator import (
    ALLOWED_FEATURES,
    validate_mechanic,
)

OUT_JSON   = PROJECT_ROOT / "tools" / "validated_mechanics.json"
OUT_REPORT = PROJECT_ROOT / "tools" / "discovery_report.md"


# ── Pre-committed filter thresholds ─────────────────────────────────────────
MIN_N_PER_PAIR        = 50
MIN_ABS_T_STAT        = 2.0
MIN_ABS_MEAN_ATR      = 0.20
N_WINDOWS             = 6
MIN_PASSING_WINDOWS   = 4
MIN_EVENTS_PER_WINDOW = 10
MIN_WINDOW_T_STAT     = 1.5
MIN_PAIRS_CROSS       = 2

HORIZONS = [5, 15, 30, 60]

# Session hour ranges mirror agent/session_router._LIVE_SCHEDULE
SESSIONS = {
    "asian":  (0,  8),
    "london": (8,  13),
    "ny":     (13, 24),
}

# Numeric features eligible for percentile threshold sweep. Excludes raw
# price levels (open/high/low/close/asian_high/low/range_high/low) because
# their absolute scale isn't a meaningful threshold.
NUMERIC_FEATURES = [
    "atr", "atr_ratio",
    "realized_vol", "rv_median",
    "yz_vol", "yz_vol_ratio",
    "ma_dist", "ma_trend",
    "adx", "hurst",
    "tick_imbalance", "vol_imbalance", "persistent_imbalance",
    "delta", "delta_momentum",
    "spread_mean", "spread_median", "spread_adj",
]

# Categorical features eligible for equality sweep
CATEGORICAL_FEATURES = ["regime", "hmm_state", "near_news"]

# Percentile thresholds — emit (">=", p75) high, (">=", p90) very high,
# ("<=", p25) low, ("<=", p10) very low.
PERCENTILE_PAIRS = [
    (">", 75),
    (">", 90),
    ("<", 25),
    ("<", 10),
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _session_mask(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.Series:
    return (df["hour"] >= start_hour) & (df["hour"] < end_hour)


def _apply_trigger(df: pd.DataFrame, feature: str, op: str, threshold) -> pd.Series:
    col = df[feature]
    if op == ">":  return col >  threshold
    if op == "<":  return col <  threshold
    if op == ">=": return col >= threshold
    if op == "<=": return col <= threshold
    if op == "==": return col == threshold
    raise ValueError(f"bad op {op}")


def _score_tuple(df: pd.DataFrame, mask: pd.Series, horizon: int) -> dict | None:
    """Per-pair score: n, mean_atr, t_stat, per-window signs."""
    # Drop bars too close to end (forward return NaN)
    mask = mask.copy()
    mask.iloc[-horizon:] = False
    n_raw = int(mask.sum())
    if n_raw < MIN_N_PER_PAIR:
        return None

    fwd_ret = df["close"].shift(-horizon) - df["close"]
    atr     = df["atr"]
    sub_fwd = fwd_ret[mask].dropna()
    sub_atr = atr[mask].reindex(sub_fwd.index).ffill().fillna(1e-4)
    norm    = (sub_fwd / sub_atr).dropna()
    n       = int(len(norm))
    if n < MIN_N_PER_PAIR:
        return None

    mean = float(norm.mean())
    std  = float(norm.std())
    if std <= 0:
        return None
    t = mean / (std / np.sqrt(n))
    if abs(t) < MIN_ABS_T_STAT:
        return None
    if abs(mean) < MIN_ABS_MEAN_ATR:
        return None

    # 6-window stability — split norm by timestamp into equal-duration windows.
    ts = df["timestamp"].iloc[norm.index]
    if len(ts) == 0:
        return None
    t_start = ts.iloc[0]
    t_end   = ts.iloc[-1]
    total_sec = (t_end - t_start).total_seconds()
    if total_sec <= 0:
        return None
    win_sec = total_sec / N_WINDOWS

    win_means = []
    win_ns    = []
    for w in range(N_WINDOWS):
        lo = t_start + pd.Timedelta(seconds=w * win_sec)
        hi = t_start + pd.Timedelta(seconds=(w + 1) * win_sec)
        m = (ts >= lo) & (ts < hi)
        w_norm = norm[m.values]
        if len(w_norm) < MIN_EVENTS_PER_WINDOW:
            win_means.append(np.nan); win_ns.append(int(len(w_norm)))
            continue
        win_means.append(float(w_norm.mean()))
        win_ns.append(int(len(w_norm)))

    valid_means = [m for m in win_means if not np.isnan(m)]
    if len(valid_means) < MIN_PASSING_WINDOWS:
        return None
    dominant_sign = 1 if mean > 0 else -1
    same_sign_count = sum(1 for m in valid_means if (1 if m > 0 else -1) == dominant_sign)
    if same_sign_count < MIN_PASSING_WINDOWS:
        return None
    wm_arr = np.array(valid_means)
    if wm_arr.std() <= 0:
        return None
    t_window = float(wm_arr.mean() / (wm_arr.std() / np.sqrt(len(wm_arr))))
    if abs(t_window) < MIN_WINDOW_T_STAT:
        return None

    return {
        "n": n,
        "mean_atr": round(mean, 4),
        "t_stat": round(float(t), 3),
        "direction": "long" if mean > 0 else "short",
        "t_window": round(t_window, 3),
        "window_signs": [1 if (m or 0) > 0 else (-1 if (m or 0) < 0 else 0) for m in win_means],
        "window_ns": win_ns,
    }


def _build_mechanic_dict(
    feature: str, op: str, threshold, threshold_label: str,
    session: str, hour_lo: int, hour_hi: int,
    horizon: int, direction: str, passing_pairs: list,
    per_pair_stats: dict,
) -> dict:
    """Assemble a MECHANIC_SCHEMA dict (compatible with validate_mechanic)."""
    # Round numeric thresholds for prettier mechanic_id + rationale.
    thr_repr = (f"{threshold:.4g}" if isinstance(threshold, (int, float, np.floating))
                else str(threshold))
    n_total = sum(per_pair_stats[p]["n"] for p in passing_pairs)
    avg_t   = sum(per_pair_stats[p]["t_stat"] for p in passing_pairs) / len(passing_pairs)
    avg_m   = sum(per_pair_stats[p]["mean_atr"] for p in passing_pairs) / len(passing_pairs)

    mechanic_id = (
        f"disc_{session}_{feature}_{op_to_slug(op)}_{threshold_label}_h{horizon}_{direction}"
    )[:60]
    rationale = (
        f"Discovered mechanic for {session.upper()} session ({hour_lo:02d}-{hour_hi:02d} UTC). "
        f"Trigger: when {feature} {op} {thr_repr} (pooled train {threshold_label}), "
        f"price drifts {direction} over the next {horizon} M1 bars. "
        f"Statistical evidence on train data: n={n_total}, "
        f"t={avg_t:+.2f}, mean_atr={avg_m:+.3f}. "
        f"Cross-pair confirmed on pairs {passing_pairs}. "
        f"Source: deterministic single-feature search, not LLM-proposed."
    )

    return {
        "mechanic_id": mechanic_id,
        "rationale":   rationale,
        "trigger": {
            "feature":    feature,
            "comparison": op,
            "threshold":  float(threshold) if isinstance(threshold, (int, np.integer, float, np.floating)) else threshold,
        },
        "context_filters": [
            {"feature": "hour", "comparison": ">=", "threshold": int(hour_lo)},
            {"feature": "hour", "comparison": "<",  "threshold": int(hour_hi)},
        ],
        "direction_hypothesis": direction,
        "forward_horizon_bars": int(horizon),
        "pair_universe":        list(passing_pairs),
    }


def op_to_slug(op: str) -> str:
    return {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq"}.get(op, "x")


# ── Search ──────────────────────────────────────────────────────────────────

def _build_triggers_pooled(train_dfs: dict[str, pd.DataFrame]) -> list:
    """Compute pooled-distribution percentiles across all pairs' train data.

    Using POOLED percentiles (not per-pair) so that the emitted mechanic has a
    single shared threshold that fires consistently on every pair. Per-pair
    percentiles diverge and break the validate_mechanic round-trip when one
    pair's pX is far outside another pair's distribution.
    """
    triggers = []

    for feat in NUMERIC_FEATURES:
        cols = []
        for df in train_dfs.values():
            if feat not in df.columns:
                continue
            c = df[feat].dropna()
            if len(c) >= 1000:
                cols.append(c.values)
        if not cols:
            continue
        pooled = np.concatenate(cols)
        for op, pct in PERCENTILE_PAIRS:
            thr = float(np.percentile(pooled, pct))
            if not np.isfinite(thr):
                continue
            triggers.append((feat, op, thr, f"p{pct}"))

    # Categorical: union of distinct values across pairs, with global support filter
    for feat in CATEGORICAL_FEATURES:
        seen = {}
        total = 0
        for df in train_dfs.values():
            if feat not in df.columns:
                continue
            vc = df[feat].value_counts(dropna=True)
            for v, c in vc.items():
                seen[v] = seen.get(v, 0) + int(c)
                total += int(c)
        for v, c in seen.items():
            if total > 0 and c / total < 0.01:
                continue
            triggers.append((feat, "==", _coerce_cat_value(v), f"eq_{str(v)[:8]}"))

    return triggers


def _coerce_cat_value(v):
    """Return value in a form acceptable to MECHANIC_SCHEMA's threshold field."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return str(v)


def discover(
    pairs: list[str],
    sessions: list[str],
    horizons: list[int],
    max_out: int,
) -> dict:
    """Main entry — load data, search, filter, return result payload."""
    print(f"[discover] loading train data for {pairs}")
    t0 = time.time()
    train_dfs, test_dfs, _ = eng.load_all_data(pairs=pairs)
    print(f"[discover] data loaded in {time.time()-t0:.1f}s")

    # Pooled-distribution triggers (single shared threshold per shape)
    pooled_triggers = _build_triggers_pooled(train_dfs)
    print(f"[discover] {len(pooled_triggers)} unique trigger-shapes × "
          f"{len(sessions)} sessions × {len(horizons)} horizons "
          f"= {len(pooled_triggers) * len(sessions) * len(horizons)} tuples per pair")

    # ── Score every (trigger, session, horizon, pair) ────────────────────────
    scored: dict = {}   # {(feat, op, threshold, lbl, session, horizon): {pair: stats}}
    n_tested = 0
    n_per_pair_passed = 0
    for (feat, op, threshold, lbl) in sorted(pooled_triggers,
                                             key=lambda x: (x[0], x[1], x[3])):
        for session in sessions:
            s_lo, s_hi = SESSIONS[session]
            for horizon in horizons:
                per_pair_stats = {}
                for p in pairs:
                    df = train_dfs[p]
                    if feat not in df.columns:
                        continue
                    sess_mask = _session_mask(df, s_lo, s_hi)
                    try:
                        trig_mask = _apply_trigger(df, feat, op, threshold)
                    except (TypeError, ValueError):
                        continue
                    mask = (sess_mask & trig_mask).fillna(False)
                    n_tested += 1
                    stats = _score_tuple(df, mask, horizon)
                    if stats is None:
                        continue
                    n_per_pair_passed += 1
                    per_pair_stats[p] = stats
                if per_pair_stats:
                    scored[(feat, op, threshold, lbl, session, horizon)] = per_pair_stats

    print(f"[discover] tested {n_tested} (key, pair) combos, "
          f"{n_per_pair_passed} passed per-pair filter, "
          f"{len(scored)} keys had ≥1 passing pair")

    # ── Cross-pair confirmation ──────────────────────────────────────────────
    survivors = []
    for key, per_pair in scored.items():
        if len(per_pair) < MIN_PAIRS_CROSS:
            continue
        dirs = [s["direction"] for s in per_pair.values()]
        if len(set(dirs)) > 1:
            continue
        feat, op, threshold, lbl, session, horizon = key
        direction = dirs[0]
        s_lo, s_hi = SESSIONS[session]
        mechanic = _build_mechanic_dict(
            feature=feat, op=op, threshold=threshold, threshold_label=lbl,
            session=session, hour_lo=s_lo, hour_hi=s_hi,
            horizon=horizon, direction=direction,
            passing_pairs=list(per_pair.keys()),
            per_pair_stats=per_pair,
        )
        # ── Round-trip self-check ────────────────────────────────────────────
        vr = validate_mechanic(
            mechanic, train_dfs,
            min_n_per_pair=MIN_N_PER_PAIR,
            min_abs_t_stat=MIN_ABS_T_STAT,
            min_abs_mean_atr=MIN_ABS_MEAN_ATR,
            require_cross_pair=True,
        )
        if not vr.passed:
            print(f"[discover] WARN: discovered mechanic '{mechanic['mechanic_id']}' "
                  f"failed validator round-trip: {vr.reason}")
            continue
        survivors.append({
            "mechanic": mechanic,
            "validation_result": {
                "n_events":        vr.n_events,
                "t_stat":          vr.t_stat,
                "mean_fwd_return": vr.mean_fwd_return,
                "per_pair":        vr.per_pair,
            },
            "discovered_session": session,
            "passing_pairs":      list(per_pair.keys()),
            "min_abs_t_stat":     min(abs(s["t_stat"]) for s in per_pair.values()),
        })

    # Rank by min |t_stat| across passing pairs (most conservative first)
    survivors.sort(key=lambda d: d["min_abs_t_stat"], reverse=True)
    survivors = survivors[:max_out]

    print(f"[discover] {len(survivors)} mechanics passed all filters "
          f"(cross-pair, validator round-trip)")

    # Train window for the payload — read off the first pair
    sample_df = train_dfs[pairs[0]]
    train_window = [
        str(sample_df["timestamp"].iloc[0].date()),
        str(sample_df["timestamp"].iloc[-1].date()),
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_window": train_window,
        "pairs":        list(pairs),
        "config": {
            "min_n_per_pair":   MIN_N_PER_PAIR,
            "min_abs_t_stat":   MIN_ABS_T_STAT,
            "min_abs_mean_atr": MIN_ABS_MEAN_ATR,
            "n_windows":        N_WINDOWS,
            "min_passing_windows": MIN_PASSING_WINDOWS,
            "min_pairs_cross":  MIN_PAIRS_CROSS,
            "sessions":         sessions,
            "horizons":         horizons,
        },
        "n_tested":     n_tested,
        "n_pre_cross":  len(scored),
        "mechanics":    survivors,
    }


# ── Reporting ───────────────────────────────────────────────────────────────

def write_report(payload: dict) -> None:
    lines = []
    lines.append("# Deterministic mechanic discovery report\n")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Train window: {payload['train_window'][0]} → {payload['train_window'][1]}")
    lines.append(f"- Pairs: {payload['pairs']}")
    lines.append(f"- Tuples tested: {payload['n_tested']:,}")
    lines.append(f"- Keys with ≥1 passing pair (pre cross-pair): {payload['n_pre_cross']}")
    lines.append(f"- Surviving mechanics (post cross-pair + validator round-trip): "
                 f"**{len(payload['mechanics'])}**\n")
    if not payload["mechanics"]:
        lines.append("## NO SURVIVORS\n")
        lines.append("Single-feature single-threshold edges fail the pre-committed filters "
                     "(n≥50, |t|≥2.0, |mean_atr|≥0.20, stability, cross-pair).\n")
        lines.append("**Do not loosen the filters.** Pivot to two-feature combinations or "
                     "abandon this universe.\n")
    else:
        lines.append("## Surviving mechanics\n")
        lines.append("| rank | mechanic_id | session | feature | op | horizon | dir | "
                     "pairs | n_total | t_stat | mean_atr |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, m in enumerate(payload["mechanics"], 1):
            mech = m["mechanic"]
            vr = m["validation_result"]
            lines.append(
                f"| {i} | `{mech['mechanic_id']}` | {m['discovered_session']} | "
                f"`{mech['trigger']['feature']}` | {mech['trigger']['comparison']} | "
                f"{mech['forward_horizon_bars']} | {mech['direction_hypothesis']} | "
                f"{','.join(m['passing_pairs'])} | {vr['n_events']} | "
                f"{vr['t_stat']:+.2f} | {vr['mean_fwd_return']:+.3f} |"
            )
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[discover] report written: {OUT_REPORT}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pairs", default="EUR_USD,GBP_USD",
                    help="Comma-separated pairs (default: EUR_USD,GBP_USD)")
    ap.add_argument("--sessions", default="asian,london,ny",
                    help="Comma-separated sessions (default: asian,london,ny)")
    ap.add_argument("--horizons", default="5,15,30,60",
                    help="Comma-separated forward horizons in bars (default: 5,15,30,60)")
    ap.add_argument("--max-out", type=int, default=50,
                    help="Max survivors to keep in the output (default: 50)")
    args = ap.parse_args(argv)

    pairs    = [p.strip() for p in args.pairs.split(",") if p.strip()]
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",")]

    for s in sessions:
        if s not in SESSIONS:
            raise SystemExit(f"unknown session '{s}'; expected one of {list(SESSIONS)}")

    payload = discover(pairs=pairs, sessions=sessions, horizons=horizons,
                       max_out=args.max_out)

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[discover] JSON written: {OUT_JSON}")
    write_report(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
