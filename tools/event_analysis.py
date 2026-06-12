"""
Phase 2 — Stability + cross-pair analysis.

Reads tools/event_log.parquet from Phase 1 and tests every event type for
statistically reliable forward-return bias across multiple time windows and
pairs. Only events that survive both the stability filter (4 of 6 same-signed
windows + t-stat > 1.5) AND the cross-pair filter (≥2 pairs same-signed)
become "validated."

Outputs:
  - tools/validated_events.json — machine-readable list of survivors
  - tools/event_analysis_report.md — human-readable per-event summary

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/event_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats as scs

CACHE_DIR    = PROJECT_ROOT / "edge_prepared_cache"
EVENT_LOG    = PROJECT_ROOT / "tools" / "event_log.parquet"
OUT_JSON     = PROJECT_ROOT / "tools" / "validated_events.json"
OUT_REPORT   = PROJECT_ROOT / "tools" / "event_analysis_report.md"

HORIZONS    = [5, 15, 30, 60]
N_WINDOWS   = 6           # number of non-overlapping windows
WINDOW_MONTHS = 4         # length of each window
MIN_EVENTS_PER_WINDOW = 20
MIN_PASSING_WINDOWS   = 4  # of 6
MIN_WINDOW_T_STAT     = 1.5
MIN_EVENTS_TOTAL      = 100    # drop ultra-rare events from analysis
MIN_PAIRS_FOR_CROSS   = 2


def load_baselines() -> dict:
    """Compute unconditional baseline forward returns per pair, per horizon,
    normalised by ATR. Used as the comparison floor for event-conditional stats.
    Returns {(pair, N): (mean, std, n)}."""
    out = {}
    for fp in sorted(CACHE_DIR.glob("*_m1.parquet")):
        pair = fp.stem.replace("_m1", "")
        print(f"  baseline: {pair}", end=" ")
        df = pd.read_parquet(fp, columns=["close", "atr"])
        atr = df["atr"]
        for N in HORIZONS:
            fwd = df["close"].shift(-N) - df["close"]
            norm = fwd / atr
            norm = norm.dropna()
            out[(pair, N)] = (float(norm.mean()), float(norm.std()), int(len(norm)))
        print("done")
    return out


def analyse_event_pair(event_rows: pd.DataFrame, baselines: dict,
                       pair: str, event_type: str) -> dict:
    """Compute per-(event_type, pair, horizon) statistics + stability test.
    Returns dict with shape used by the JSON serializer."""
    rows = event_rows[event_rows["pair"] == pair]
    if rows.empty:
        return None
    timestamps = pd.to_datetime(rows["timestamp"])
    atrs = rows["atr_at_event"].values

    per_horizon = {}
    for N in HORIZONS:
        fwd = rows[f"forward_ret_{N}"].values
        norm = fwd / atrs
        norm = norm[~np.isnan(norm)]
        if len(norm) < MIN_EVENTS_TOTAL // 10:
            per_horizon[N] = None
            continue
        mean_norm = float(norm.mean())
        std_norm  = float(norm.std()) if len(norm) > 1 else 0.0
        n         = len(norm)
        # t-stat vs zero
        t_zero = mean_norm / (std_norm / np.sqrt(n)) if std_norm > 0 else 0.0
        # t-stat vs unconditional baseline
        base_mean, base_std, base_n = baselines.get((pair, N), (0.0, 1.0, 1))
        if base_std > 0 and std_norm > 0:
            # Welch's t-test approximation: (mean_event - mean_baseline) / sqrt(var_event/n_event + var_baseline/n_baseline)
            denom = np.sqrt(std_norm**2 / n + base_std**2 / base_n)
            t_base = (mean_norm - base_mean) / denom if denom > 0 else 0.0
        else:
            t_base = 0.0
        per_horizon[N] = {
            "n": n,
            "mean_norm": round(mean_norm, 4),
            "std_norm":  round(std_norm, 4),
            "t_zero":    round(float(t_zero), 3),
            "t_vs_baseline": round(float(t_base), 3),
            "baseline_mean": round(base_mean, 4),
        }

    # Stability filter: split rows into 6 non-overlapping 4-month windows by timestamp
    # Use the rows' actual time range
    if len(rows) < N_WINDOWS * MIN_EVENTS_PER_WINDOW:
        stability_by_h = {N: {"pass": False, "reason": f"too few events ({len(rows)}) for {N_WINDOWS} windows of {MIN_EVENTS_PER_WINDOW}"} for N in HORIZONS}
    else:
        ts_sorted = timestamps.sort_values()
        t_start = ts_sorted.iloc[0]
        t_end   = ts_sorted.iloc[-1]
        # Build 6 equal-duration windows over the actual range
        total_seconds = (t_end - t_start).total_seconds()
        win_seconds   = total_seconds / N_WINDOWS

        stability_by_h = {}
        for N in HORIZONS:
            if per_horizon.get(N) is None:
                stability_by_h[N] = {"pass": False, "reason": "no horizon data"}
                continue
            win_means = []
            win_signs = []
            win_ns    = []
            for w in range(N_WINDOWS):
                w_lo = t_start + pd.Timedelta(seconds=w * win_seconds)
                w_hi = t_start + pd.Timedelta(seconds=(w + 1) * win_seconds)
                mask = (timestamps >= w_lo) & (timestamps < w_hi)
                w_rows = rows[mask.values]
                if len(w_rows) < MIN_EVENTS_PER_WINDOW:
                    win_means.append(np.nan); win_signs.append(0); win_ns.append(len(w_rows))
                    continue
                w_norm = (w_rows[f"forward_ret_{N}"] / w_rows["atr_at_event"]).dropna()
                if len(w_norm) == 0:
                    win_means.append(np.nan); win_signs.append(0); win_ns.append(0)
                    continue
                wm = float(w_norm.mean())
                win_means.append(wm)
                win_signs.append(1 if wm > 0 else -1 if wm < 0 else 0)
                win_ns.append(int(len(w_norm)))
            wm_arr = np.array([m for m in win_means if not np.isnan(m)])
            valid_signs = [s for s in win_signs if s != 0]
            if len(valid_signs) < MIN_PASSING_WINDOWS:
                stability_by_h[N] = {"pass": False, "reason": "not enough windows with data",
                                     "window_signs": win_signs, "window_ns": win_ns}
                continue
            dominant_sign = 1 if sum(s == 1 for s in valid_signs) >= len(valid_signs) // 2 else -1
            same_sign_count = sum(s == dominant_sign for s in valid_signs)
            # One-sample t-test of per-window means vs 0
            if len(wm_arr) > 1 and wm_arr.std() > 0:
                t_window = float(wm_arr.mean() / (wm_arr.std() / np.sqrt(len(wm_arr))))
            else:
                t_window = 0.0
            passes = (same_sign_count >= MIN_PASSING_WINDOWS
                      and abs(t_window) > MIN_WINDOW_T_STAT)
            stability_by_h[N] = {
                "pass": bool(passes),
                "dominant_sign": dominant_sign,
                "same_sign_count": same_sign_count,
                "window_signs": win_signs,
                "window_means": [round(m, 4) if not np.isnan(m) else None for m in win_means],
                "window_ns": win_ns,
                "t_window": round(t_window, 3),
            }
    return {
        "pair": pair,
        "event_type": event_type,
        "n_total": int(len(rows)),
        "horizons": per_horizon,
        "stability": stability_by_h,
    }


def main() -> int:
    if not EVENT_LOG.exists():
        print(f"ERROR: {EVENT_LOG} not found. Run event_catalog.py first.")
        return 2

    print("[analysis] Loading event log...")
    events = pd.read_parquet(EVENT_LOG)
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    print(f"  Loaded {len(events):,} events across {events['event_type'].nunique()} types and {events['pair'].nunique()} pairs")

    # Drop ultra-rare events
    event_counts = events.groupby("event_type").size()
    rare_events = event_counts[event_counts < MIN_EVENTS_TOTAL].index.tolist()
    if rare_events:
        print(f"  Dropping {len(rare_events)} rare events (< {MIN_EVENTS_TOTAL} total): {rare_events}")
        events = events[~events["event_type"].isin(rare_events)]

    print("\n[analysis] Computing unconditional baselines...")
    baselines = load_baselines()

    print("\n[analysis] Analysing events...")
    per_event_pair = []   # list of dicts
    failed = []
    event_types = sorted(events["event_type"].unique())
    pairs = sorted(events["pair"].unique())
    for ev in event_types:
        ev_rows = events[events["event_type"] == ev]
        for pair in pairs:
            res = analyse_event_pair(ev_rows, baselines, pair, ev)
            if res is not None:
                per_event_pair.append(res)

    # Cross-pair aggregation: which (event_type, horizon) tuples passed stability
    # on ≥2 pairs with consistent sign?
    validated = []
    by_event_horizon = {}   # {(event_type, horizon): [(pair, sign, t_vs_baseline, n)]}
    for r in per_event_pair:
        ev = r["event_type"]
        for N in HORIZONS:
            stab = r["stability"].get(N, {})
            horiz = r["horizons"].get(N)
            if horiz is None:
                continue
            if stab.get("pass", False):
                key = (ev, N)
                by_event_horizon.setdefault(key, []).append({
                    "pair": r["pair"],
                    "sign": stab["dominant_sign"],
                    "t_vs_baseline": horiz["t_vs_baseline"],
                    "mean_norm": horiz["mean_norm"],
                    "n": horiz["n"],
                    "window_signs": stab["window_signs"],
                })

    for (ev, N), pair_results in by_event_horizon.items():
        if len(pair_results) < MIN_PAIRS_FOR_CROSS:
            continue
        signs = [pr["sign"] for pr in pair_results]
        if len(set(signs)) > 1:
            # Inconsistent direction across pairs
            continue
        dominant = signs[0]
        validated.append({
            "event_type": ev,
            "horizon_bars": N,
            "passing_pairs": [pr["pair"] for pr in pair_results],
            "direction": "long" if dominant > 0 else "short",
            "mean_norm_per_pair": {pr["pair"]: pr["mean_norm"] for pr in pair_results},
            "t_vs_baseline_per_pair": {pr["pair"]: pr["t_vs_baseline"] for pr in pair_results},
            "n_total_events": sum(pr["n"] for pr in pair_results),
            "window_signs_per_pair": {pr["pair"]: pr["window_signs"] for pr in pair_results},
        })

    # Sort validated by max |t_vs_baseline| across passing pairs
    validated.sort(
        key=lambda v: max(abs(t) for t in v["t_vs_baseline_per_pair"].values()),
        reverse=True,
    )

    # ── Failed-events summary ─────────────────────────────────────────────────
    seen_validated = {(v["event_type"], v["horizon_bars"]) for v in validated}
    for r in per_event_pair:
        ev = r["event_type"]
        for N in HORIZONS:
            key = (ev, N)
            stab = r["stability"].get(N, {})
            horiz = r["horizons"].get(N)
            if horiz is None:
                failed.append({"event_type": ev, "pair": r["pair"], "horizon": N,
                               "reason": "no horizon data"})
                continue
            if not stab.get("pass", False):
                failed.append({"event_type": ev, "pair": r["pair"], "horizon": N,
                               "reason": stab.get("reason", "stability_fail"),
                               "t_vs_baseline": horiz["t_vs_baseline"],
                               "window_signs": stab.get("window_signs")})

    # Cross-pair-only failures (passed stability but no other pair confirmed)
    for (ev, N), pair_results in by_event_horizon.items():
        if len(pair_results) >= MIN_PAIRS_FOR_CROSS:
            continue
        failed.append({"event_type": ev, "horizon": N,
                       "reason": "only 1 pair passed stability",
                       "passing_pair": pair_results[0]["pair"]})

    payload = {
        "validated": validated,
        "n_validated": len(validated),
        "n_event_types_tested": len(event_types),
        "n_pairs_tested": len(pairs),
        "n_dropped_rare": len(rare_events),
        "config": {
            "horizons": HORIZONS,
            "n_windows": N_WINDOWS,
            "min_events_per_window": MIN_EVENTS_PER_WINDOW,
            "min_passing_windows": MIN_PASSING_WINDOWS,
            "min_window_t_stat": MIN_WINDOW_T_STAT,
            "min_events_total": MIN_EVENTS_TOTAL,
            "min_pairs_for_cross": MIN_PAIRS_FOR_CROSS,
        },
        "failed_summary": failed[:50],   # cap to avoid blowing up the file
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ── Markdown report ──────────────────────────────────────────────────────
    lines = []
    lines.append("# Event-conditional return analysis\n")
    lines.append(f"- Event types tested: **{len(event_types)}**")
    lines.append(f"- Pairs tested: {pairs}")
    lines.append(f"- Total event occurrences in log: {len(events):,}")
    lines.append(f"- Rare events dropped (< {MIN_EVENTS_TOTAL}): {rare_events}\n")
    lines.append(f"## Validated events (passed stability + cross-pair)\n")
    if not validated:
        lines.append("_NONE PASSED._\n")
        lines.append("**Phase 2 stop condition triggered**: no event type produced statistically reliable forward-return bias across multiple windows and pairs. Document and pivot.\n")
    else:
        lines.append(f"**{len(validated)} validated** — ranked by max |t_vs_baseline| across passing pairs.\n")
        lines.append("| event_type | horizon | direction | pairs | mean_norm | max \\|t_vs_baseline\\| | n |")
        lines.append("|---|---|---|---|---|---|---|")
        for v in validated:
            max_t = max(abs(t) for t in v["t_vs_baseline_per_pair"].values())
            mean_norm_avg = np.mean(list(v["mean_norm_per_pair"].values()))
            pairs_str = ",".join(v["passing_pairs"])
            lines.append(
                f"| `{v['event_type']}` | {v['horizon_bars']} | {v['direction']} | "
                f"{pairs_str} | {mean_norm_avg:+.3f} | {max_t:.2f} | "
                f"{v['n_total_events']} |"
            )

    lines.append(f"\n## Per-(event,pair) details\n")
    lines.append("| event_type | pair | horizon | n | mean_norm | t_vs_baseline | stability |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in per_event_pair:
        for N in HORIZONS:
            horiz = r["horizons"].get(N)
            stab  = r["stability"].get(N, {})
            if horiz is None:
                continue
            pass_str = "✓" if stab.get("pass") else "✗"
            lines.append(
                f"| `{r['event_type']}` | {r['pair']} | {N} | {horiz['n']} | "
                f"{horiz['mean_norm']:+.3f} | {horiz['t_vs_baseline']:+.2f} | {pass_str} |"
            )
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[analysis] DONE")
    print(f"  Validated events: {len(validated)}")
    print(f"  Report: {OUT_REPORT}")
    print(f"  JSON:   {OUT_JSON}")
    if validated:
        print("\nTop validated:")
        for v in validated[:5]:
            max_t = max(abs(t) for t in v["t_vs_baseline_per_pair"].values())
            print(f"  {v['event_type']:35s} horizon={v['horizon_bars']:>3} "
                  f"dir={v['direction']:5s} max|t|={max_t:.2f} "
                  f"pairs={v['passing_pairs']}")
    else:
        print("\n  No events passed both stability + cross-pair filters.")
        print("  Phase 2 stop condition triggered. Don't proceed to Phase 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
