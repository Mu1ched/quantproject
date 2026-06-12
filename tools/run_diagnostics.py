"""
Run cost-stack diagnostics — null strategy + cost-sensitivity — in either
static or dynamic spread mode (or both).

The autonomous agent uses dynamic mode (per-bar spreads + session-hour
slippage + news multipliers); the GUI's `cost_mult` slider only affects
static mode. Run both to verify the 2026-06-02 recalibration is sensible
under each.

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/run_diagnostics.py --mode both --seeds 50
    python tools/run_diagnostics.py --mode static --seeds 200       # full first diagnostic
    python tools/run_diagnostics.py --mode dynamic --seeds 50       # agent's actual cost model

In static mode, the script also runs a cost-sensitivity sweep over
{0.0, 0.25, 0.5, 1.0, 1.5}. In dynamic mode, `cost_mult` is ignored by the
engine, so only a single null + ORB run is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_engine import (
    load_all_data, run_sweep, load_sweep_results, NY_PAIRS,
    SLIP_RATIO_STATIC, NEWS_SLIPPAGE_MULT,
)
from edge_hypotheses import (
    SWEEP_NULL_NY, SWEEP_ORB_NY,
    ParameterGrid,
)


def cached_pairs() -> list[str]:
    cache_dir = PROJECT_ROOT / "edge_prepared_cache"
    return sorted({p.stem.replace("_m1", "") for p in cache_dir.glob("*_m1.parquet")})


def describe_sharpe(series) -> dict:
    if len(series) == 0:
        return {k: float("nan") for k in
                ["n", "mean", "median", "std", "min", "p25", "p75", "max", "frac_pos"]}
    s = series.astype(float)
    return {
        "n":        int(len(s)),
        "mean":     round(float(s.mean()), 3),
        "median":   round(float(s.median()), 3),
        "std":      round(float(s.std()), 3),
        "min":      round(float(s.min()), 3),
        "p25":      round(float(s.quantile(0.25)), 3),
        "p75":      round(float(s.quantile(0.75)), 3),
        "max":      round(float(s.max()), 3),
        "frac_pos": round(float((s > 0).mean()), 3),
    }


def trimmed_null_sweep(n_seeds: int) -> dict:
    """Return a copy of SWEEP_NULL_NY whose grid uses only the first n_seeds."""
    base = dict(SWEEP_NULL_NY)
    base["grid"] = ParameterGrid({
        "seed": list(range(1, n_seeds + 1)),
        "tp_r": [2.0],
    })
    return base


def run_mode(mode: str, seeds: int, default_cm: float,
             train_dfs: dict, test_dfs: dict, measured_spreads: dict,
             ny_pairs: list[str], n_workers: int) -> dict:
    """Run null + (static-only) cost-sensitivity for one mode. Returns summary dict."""
    use_dynamic = (mode == "dynamic")
    tag = mode
    null = trimmed_null_sweep(seeds)

    print(f"\n[diag/{tag}] Null sweep ({seeds} seeds, cost_mult={default_cm}, "
          f"use_dynamic_spread={use_dynamic})...")
    null_sid = run_sweep(
        sweep_name=f"null_ny_after_{tag}",
        entry_fn=null["entry_fn"], manager_fn=null["manager_fn"],
        grid=null["grid"], pairs=ny_pairs,
        session=null["session"], regime_mult=null["regime_mult"],
        train_dfs=train_dfs, test_dfs=test_dfs,
        measured_spreads=measured_spreads,
        cost_mult=default_cm, n_workers=n_workers,
        use_dynamic_spread=use_dynamic,
    )
    null_df = load_sweep_results(null_sid)
    null_test = describe_sharpe(null_df["test_sharpe"])
    null_trades = (round(float(null_df["test_n"].mean()), 1)
                   if "test_n" in null_df.columns and len(null_df) else 0)
    print(f"[diag/{tag}] Null mean test Sharpe = {null_test['mean']} "
          f"frac_pos = {null_test['frac_pos']} mean_trades = {null_trades}")

    out = {
        "mode": mode,
        "use_dynamic_spread": use_dynamic,
        "default_cost_mult":  default_cm,
        "null_sweep_id":      null_sid,
        "null_test_sharpe":   null_test,
        "null_mean_trades":   null_trades,
    }

    # Cost-sensitivity only meaningful in static mode (dynamic ignores cost_mult).
    if not use_dynamic:
        cost_mults = [0.0, 0.25, 0.5, 1.0, 1.5]
        rows = []
        for cm in cost_mults:
            print(f"[diag/{tag}] Cost sweep cm={cm} ...")
            sid = run_sweep(
                sweep_name=f"cost_sens_orb_ny_after_{tag}_cm{cm:g}",
                entry_fn=SWEEP_ORB_NY["entry_fn"], manager_fn=SWEEP_ORB_NY["manager_fn"],
                grid=SWEEP_ORB_NY["grid"], pairs=ny_pairs,
                session=SWEEP_ORB_NY["session"], regime_mult=SWEEP_ORB_NY["regime_mult"],
                train_dfs=train_dfs, test_dfs=test_dfs,
                measured_spreads=measured_spreads,
                cost_mult=cm, n_workers=n_workers,
                use_dynamic_spread=False,
            )
            df = load_sweep_results(sid)
            summ = describe_sharpe(df["test_sharpe"])
            summ["cost_mult"] = cm
            summ["sweep_id"]  = sid
            rows.append(summ)
            print(f"[diag/{tag}]   cm={cm}: mean={summ['mean']} best={summ['max']} "
                  f"frac_pos={summ['frac_pos']}")
        out["cost_sensitivity"] = rows
    else:
        # In dynamic mode, just run ORB once at default and report
        print(f"[diag/{tag}] ORB sweep (default, dynamic spread)...")
        sid = run_sweep(
            sweep_name=f"orb_ny_after_dynamic",
            entry_fn=SWEEP_ORB_NY["entry_fn"], manager_fn=SWEEP_ORB_NY["manager_fn"],
            grid=SWEEP_ORB_NY["grid"], pairs=ny_pairs,
            session=SWEEP_ORB_NY["session"], regime_mult=SWEEP_ORB_NY["regime_mult"],
            train_dfs=train_dfs, test_dfs=test_dfs,
            measured_spreads=measured_spreads,
            cost_mult=default_cm, n_workers=n_workers,
            use_dynamic_spread=True,
        )
        df = load_sweep_results(sid)
        summ = describe_sharpe(df["test_sharpe"])
        summ["sweep_id"] = sid
        out["orb_dynamic"] = summ
        print(f"[diag/{tag}] ORB dynamic: mean={summ['mean']} best={summ['max']} "
              f"frac_pos={summ['frac_pos']}")

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--mode", choices=["static", "dynamic", "both"], default="both")
    p.add_argument("--seeds", type=int, default=50,
                   help="Null-strategy seeds per mode (default 50; 200 for full diagnostic)")
    p.add_argument("--cost-mult", type=float, default=0.5,
                   help="Default cost_mult for the null and dynamic-mode runs (default 0.5)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--report", default=str(PROJECT_ROOT / "tools" / "diagnostics_after_fix.md"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    avail = cached_pairs()
    ny_pairs = [p for p in NY_PAIRS if p in avail]
    if not ny_pairs:
        print("[diag] ERROR: no NY_PAIRS in cache.")
        return 2
    print(f"[diag] NY pairs in cache: {ny_pairs}")
    print(f"[diag] Engine cost knobs: SLIP_RATIO_STATIC={SLIP_RATIO_STATIC}, "
          f"NEWS_SLIPPAGE_MULT={NEWS_SLIPPAGE_MULT}")

    train_dfs, test_dfs, measured_spreads = load_all_data(pairs=ny_pairs)

    modes = ["static", "dynamic"] if args.mode == "both" else [args.mode]
    summaries = []
    for m in modes:
        summaries.append(run_mode(
            mode=m, seeds=args.seeds, default_cm=args.cost_mult,
            train_dfs=train_dfs, test_dfs=test_dfs,
            measured_spreads=measured_spreads,
            ny_pairs=ny_pairs, n_workers=args.workers,
        ))

    # ----- Build report (with before/after if static results exist) -----
    BEFORE_STATIC = {
        "null_mean":     -0.967,
        "null_frac_pos": 0.195,
        "null_trades":   109.5,
        "orb_at_cm05":   0.124,
        "orb_at_cm10": -0.260,
        "orb_at_cm00":   0.483,
    }

    lines = []
    lines.append("# Quantproject diagnostics — after 2026-06-02 cost recalibration\n")
    lines.append(f"Pairs: `{ny_pairs}`. Seeds per null sweep: `{args.seeds}`. "
                 f"Default cost_mult tested: `{args.cost_mult}`.\n")
    lines.append(f"Engine knobs at runtime: `SLIP_RATIO_STATIC={SLIP_RATIO_STATIC}`, "
                 f"`NEWS_SLIPPAGE_MULT={NEWS_SLIPPAGE_MULT}`.\n")

    for s in summaries:
        m = s["mode"]
        lines.append(f"## {m.capitalize()} mode\n")
        lines.append(f"`use_dynamic_spread = {s['use_dynamic_spread']}`, "
                     f"`cost_mult = {s['default_cost_mult']}`. "
                     f"Null sweep id: `{s['null_sweep_id']}`.\n")
        nt = s["null_test_sharpe"]
        lines.append("**Null test-Sharpe distribution:**\n")
        lines.append("| n | mean | median | std | p25 | p75 | max | frac_pos | mean_trades |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines.append(f"| {nt['n']} | {nt['mean']} | {nt['median']} | {nt['std']} | "
                     f"{nt['p25']} | {nt['p75']} | {nt['max']} | {nt['frac_pos']} | "
                     f"{s['null_mean_trades']} |\n")

        if m == "static":
            lines.append("**Cost-sensitivity (after recalibration):**\n")
            lines.append("| cost_mult | n | mean | median | max | frac_pos |")
            lines.append("|---|---|---|---|---|---|")
            for r in s["cost_sensitivity"]:
                lines.append(f"| {r['cost_mult']} | {r['n']} | {r['mean']} | "
                             f"{r['median']} | {r['max']} | {r['frac_pos']} |")
            lines.append("")

            # ---- Before/After comparison ----
            cm05 = next((r["mean"] for r in s["cost_sensitivity"] if r["cost_mult"] == 0.5), None)
            cm10 = next((r["mean"] for r in s["cost_sensitivity"] if r["cost_mult"] == 1.0), None)
            cm00 = next((r["mean"] for r in s["cost_sensitivity"] if r["cost_mult"] == 0.0), None)
            lines.append("### Before/After (static mode)\n")
            lines.append("| metric | before (cm=1.0, slip=0.20) | after (cm=0.5, slip=0.10) |")
            lines.append("|---|---|---|")
            lines.append(f"| Null mean test Sharpe | {BEFORE_STATIC['null_mean']} | {nt['mean']} |")
            lines.append(f"| Null frac positive | {BEFORE_STATIC['null_frac_pos']} | {nt['frac_pos']} |")
            lines.append(f"| ORB mean Sharpe @ cm=0.0 | {BEFORE_STATIC['orb_at_cm00']} | {cm00} |")
            lines.append(f"| ORB mean Sharpe @ cm=0.5 | {BEFORE_STATIC['orb_at_cm05']} | {cm05} |")
            lines.append(f"| ORB mean Sharpe @ cm=1.0 | {BEFORE_STATIC['orb_at_cm10']} | {cm10} |")
            lines.append("")
            delta = (nt["mean"] - BEFORE_STATIC["null_mean"]) if nt["mean"] is not None else None
            lines.append(f"Null mean test Sharpe shifted by **{delta:+.2f}** vs pre-recal baseline.\n")

        elif m == "dynamic":
            od = s.get("orb_dynamic", {})
            lines.append("**ORB performance under dynamic spreads:**\n")
            lines.append("| n | mean | median | best | frac_pos |")
            lines.append("|---|---|---|---|---|")
            lines.append(f"| {od.get('n')} | {od.get('mean')} | {od.get('median')} | "
                         f"{od.get('max')} | {od.get('frac_pos')} |\n")
            lines.append(f"ORB dynamic sweep id: `{od.get('sweep_id')}`.\n")

    # Verdict
    nm_static = next((s["null_test_sharpe"]["mean"] for s in summaries if s["mode"] == "static"), None)
    nm_dynamic = next((s["null_test_sharpe"]["mean"] for s in summaries if s["mode"] == "dynamic"), None)
    lines.append("## Verdict\n")
    if nm_static is not None:
        if -0.5 <= nm_static <= 0.0:
            lines.append(f"Static null mean = **{nm_static}** is within target range "
                         "[−0.5, 0.0]. Static cost stack recalibration looks healthy.\n")
        elif nm_static > 0:
            lines.append(f"Static null mean = **{nm_static}** is *positive* — costs may "
                         "now be too generous, or a structural quirk exists. Inspect.\n")
        else:
            lines.append(f"Static null mean = **{nm_static}** is still below target. "
                         "Consider further trimming SLIP_RATIO_STATIC or default cost_mult.\n")
    if nm_dynamic is not None:
        lines.append(f"Dynamic null mean = **{nm_dynamic}**. No pre-recal baseline (the "
                     "first diagnostic ran static-only), but a value in [−0.6, +0.2] is "
                     "expected; large negatives suggest SLIPPAGE_PROFILE or news multipliers "
                     "still bite. Compare against your live MT5 PnL once you have trades.\n")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[diag] Report written: {out}")

    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps({
        "ny_pairs":  ny_pairs,
        "engine":    {"SLIP_RATIO_STATIC": SLIP_RATIO_STATIC,
                      "NEWS_SLIPPAGE_MULT": NEWS_SLIPPAGE_MULT},
        "args":      vars(args),
        "summaries": summaries,
        "baseline_static": BEFORE_STATIC,
    }, indent=2), encoding="utf-8")
    print(f"[diag] JSON summary: {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
