"""
Re-test every Claude-generated strategy in agent/generated/ under the new
live-calibrated dynamic-mode cost model. Records test Sharpe per strategy
and counts survivors at various thresholds.

Each strategy is run with the agent's default 4-combo grid (tp_r × sl_r),
cost_mult=0.5, use_dynamic_spread=True. No additional_params (we don't have
the original best_params_json — only the strategy's internal defaults via
params.get(key, default) apply).

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/retest_generated.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_engine import (
    load_all_data, run_sweep, load_sweep_results,
    ParameterGrid, make_manager,
)
from agent.session_router import _resolve_session, session_regime_mult
from agent.code_writer import load_entry_fn  # picklable loader for workers


GEN_DIR = PROJECT_ROOT / "agent" / "generated"
CACHE_DIR = PROJECT_ROOT / "edge_prepared_cache"

# Default grid mirrors agent.loop._build_grid base values.
BASE_GRID = ParameterGrid({"tp_r": [1.5, 2.5], "sl_r": [0.75, 1.25]})

THRESHOLDS = [0.0, 0.25, 0.5]


def cached_pairs() -> list[str]:
    return sorted({p.stem.replace("_m1", "") for p in CACHE_DIR.glob("*_m1.parquet")})


def parse_session(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"#\s*Session:\s*(\w+)", text)
    return m.group(1).strip() if m else "ny"


def import_entry_fn(path: Path):
    """Load module via the agent's canonical loader so worker processes can
    re-import it by dotted path (avoids PicklingError under spawn workers)."""
    # path.stem is "entry_<name>"; load_entry_fn wants just "<name>"
    name = path.stem.removeprefix("entry_")
    return load_entry_fn(name)


def main() -> int:
    t0 = time.time()

    avail = set(cached_pairs())
    print(f"[retest] Cached pairs: {sorted(avail)}")

    # Load data for all cached pairs (we'll subset per-strategy by session pairs).
    print("[retest] Loading data...")
    train_dfs, test_dfs, measured_spreads = load_all_data(pairs=sorted(avail))

    files = sorted(GEN_DIR.glob("entry_*.py"))
    print(f"[retest] Found {len(files)} generated strategy files\n")

    results = []
    skipped = []
    failed = []

    for i, fp in enumerate(files, 1):
        name = fp.stem  # entry_<name>
        sess = parse_session(fp)
        try:
            default_pairs, exit_hour = _resolve_session(sess)
            avail_pairs = [p for p in default_pairs if p in avail]
            if not avail_pairs:
                skipped.append((name, sess, "no cached pairs for session"))
                continue
            entry_fn = import_entry_fn(fp)
        except Exception as e:
            failed.append((name, sess, f"import/setup error: {e!s}"))
            continue

        # Run a small grid sweep in dynamic mode at the new default cost_mult.
        sweep_name = f"retest_{name}"
        try:
            sweep_id = run_sweep(
                sweep_name=sweep_name,
                entry_fn=entry_fn,
                manager_fn=make_manager(exit_hour=exit_hour, use_breakeven=True),
                grid=BASE_GRID,
                pairs=avail_pairs,
                session=sess,
                regime_mult=session_regime_mult(sess),
                train_dfs=train_dfs, test_dfs=test_dfs,
                measured_spreads=measured_spreads,
                cost_mult=0.5, n_workers=2,
                use_dynamic_spread=True,
            )
            df = load_sweep_results(sweep_id)
            if df.empty:
                skipped.append((name, sess, "no hypothesis results"))
                continue
            best_sharpe = float(df["test_sharpe"].max())
            best_row = df.iloc[df["test_sharpe"].astype(float).idxmax()]
            n_trades = int(best_row.get("test_n", 0) or 0)
            results.append({
                "name": name, "session": sess,
                "pairs": avail_pairs,
                "best_test_sharpe": round(best_sharpe, 3),
                "mean_test_sharpe": round(float(df["test_sharpe"].mean()), 3),
                "n_hypotheses": int(len(df)),
                "best_n_trades": n_trades,
                "sweep_id": sweep_id,
            })
            elapsed = time.time() - t0
            avg_per = elapsed / i
            eta = avg_per * (len(files) - i)
            print(f"[{i:3d}/{len(files)}] {name[:50]:50s} best={best_sharpe:+.3f} "
                  f"(eta {eta/60:.0f}m)")
        except Exception as e:
            failed.append((name, sess, f"sweep error: {type(e).__name__}: {e!s}"))
            traceback.print_exc()
            continue

    # ─── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n[retest] Done in {elapsed/60:.1f} min")
    print(f"[retest] Results: {len(results)} ok, {len(skipped)} skipped, {len(failed)} failed")

    counts = {}
    for thr in THRESHOLDS:
        counts[thr] = sum(1 for r in results if r["best_test_sharpe"] > thr)

    lines = []
    lines.append("# Retest of generated strategies under live-calibrated costs\n")
    lines.append(f"Total strategies: {len(files)}  |  Ok: {len(results)}  |  "
                 f"Skipped: {len(skipped)}  |  Failed: {len(failed)}  |  "
                 f"Wall: {elapsed/60:.1f} min\n")
    lines.append("## Survivor counts by best test Sharpe threshold\n")
    lines.append("| threshold | survivors | survivor rate |")
    lines.append("|---|---|---|")
    for thr, c in counts.items():
        rate = c / len(results) if results else 0
        lines.append(f"| > {thr} | {c} | {rate:.1%} |")
    lines.append("")

    # Top 25 by best test Sharpe
    top = sorted(results, key=lambda r: r["best_test_sharpe"], reverse=True)[:25]
    lines.append("## Top 25 strategies (by best test Sharpe)\n")
    lines.append("| rank | strategy | session | best_sharpe | mean_sharpe | n_trades |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(top, 1):
        lines.append(f"| {i} | `{r['name']}` | {r['session']} | "
                     f"{r['best_test_sharpe']} | {r['mean_test_sharpe']} | "
                     f"{r['best_n_trades']} |")
    lines.append("")

    if skipped:
        lines.append(f"## Skipped ({len(skipped)})\n")
        for n, s, why in skipped[:20]:
            lines.append(f"- `{n}` (session={s}): {why}")
        if len(skipped) > 20:
            lines.append(f"- … +{len(skipped) - 20} more")
        lines.append("")

    if failed:
        lines.append(f"## Failed ({len(failed)})\n")
        for n, s, why in failed[:20]:
            lines.append(f"- `{n}` (session={s}): {why}")
        if len(failed) > 20:
            lines.append(f"- … +{len(failed) - 20} more")
        lines.append("")

    report_path = PROJECT_ROOT / "tools" / "retest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = PROJECT_ROOT / "tools" / "retest_results.json"
    json_path.write_text(json.dumps({
        "elapsed_min": round(elapsed / 60, 2),
        "n_files": len(files),
        "ok": results, "skipped": skipped, "failed": failed,
        "survivor_counts": counts,
    }, indent=2), encoding="utf-8")

    print(f"[retest] Report: {report_path}")
    print(f"[retest] JSON:   {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
