"""
Sliding-window edge-miner survey.

Hypothesis: the 18mo-train / 5mo-test default window crosses too much regime
shift (adversarial AUC ~0.88 even after cleaning non-stationary features).
With 4mo / 1mo sliding windows, intra-window drift is small enough that the
miner can complete a real LightGBM + SHAP + walk-forward CV pass — and
features that recur as top-SHAP-ranked predictors across many windows are
the structurally stable ones (vs single-window noise).

Output: tools/sliding_window_report.md + tools/sliding_window_results.json.

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/sliding_window_survey.py
    python tools/sliding_window_survey.py --n-windows 1   # smoke
    python tools/sliding_window_survey.py --sessions ny   # one session only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import edge_miner as miner
from edge_engine import PAIR_PIP_SIZE

CACHE_DIR  = PROJECT_ROOT / "edge_prepared_cache"
LIVE_PATH  = PROJECT_ROOT / "live_measured_spreads.json"

TRAIN_DAYS = 120
TEST_DAYS  = 30
STEP_DAYS  = 30
TOP_N_FEAT_FOR_AGG = 5
ADV_PASS_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--sessions", default="ny,london",
                   help="Comma-separated session filters (default: ny,london)")
    p.add_argument("--n-windows", type=int, default=0,
                   help="Cap the number of windows (0 = all). Use 1 for smoke.")
    p.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    p.add_argument("--test-days",  type=int, default=TEST_DAYS)
    p.add_argument("--step-days",  type=int, default=STEP_DAYS)
    p.add_argument("--report", default=str(PROJECT_ROOT / "tools" / "sliding_window_report.md"))
    p.add_argument("--json",   default=str(PROJECT_ROOT / "tools" / "sliding_window_results.json"))
    return p.parse_args()


def load_prepared() -> dict[str, pd.DataFrame]:
    """Read all cached prepared parquets and trim to cross-pair common date range."""
    full = {}
    for fp in sorted(CACHE_DIR.glob("*_m1.parquet")):
        pair = fp.stem.replace("_m1", "")
        df = pd.read_parquet(fp)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        if "timestamp" in df.columns and "ts" not in df.columns:
            df["ts"] = df["timestamp"]
        full[pair] = df

    if not full:
        raise RuntimeError(f"No parquets in {CACHE_DIR}")

    common_min = max(df["date"].min() for df in full.values())
    common_max = min(df["date"].max() for df in full.values())
    for pair in list(full):
        df = full[pair]
        before = len(df)
        df = df[(df["date"] >= common_min) & (df["date"] <= common_max)].reset_index(drop=True)
        full[pair] = df
        print(f"  [{pair}] {before:,} -> {len(df):,} bars in [{common_min}, {common_max}]")
    return full


def load_live_overrides() -> dict[str, float]:
    if not LIVE_PATH.exists():
        return {}
    try:
        with open(LIVE_PATH) as f:
            data = json.load(f)
        return data.get("spreads", {}) or {}
    except Exception:
        return {}


def apply_live_override(train_dfs: dict, test_dfs: dict,
                         measured_spreads: dict, live_spreads: dict) -> None:
    """Mirror the override+recompute logic added to load_all_data — but at the
    per-slice level. Mutates train_dfs/test_dfs/measured_spreads in place."""
    if not live_spreads:
        return
    try:
        from agent.config import NEWS_SPREAD_MULT, COMMISSION_PIPS
    except Exception:
        NEWS_SPREAD_MULT, COMMISSION_PIPS = 2.0, 0.3
    for pair, live_spd in live_spreads.items():
        if pair not in measured_spreads:
            continue
        old = measured_spreads[pair]
        new = float(live_spd)
        scale = (new / old) if old > 0 else 1.0
        measured_spreads[pair] = new
        pip = PAIR_PIP_SIZE.get(pair, 1e-4)
        commission_p = 2.0 * COMMISSION_PIPS * pip
        for dfs in (train_dfs, test_dfs):
            df = dfs.get(pair)
            if df is None or df.empty:
                continue
            df["spread_mean"] = df["spread_mean"] * scale
            if "spread_median" in df.columns:
                df["spread_median"] = df["spread_median"] * scale
            news_mult = (df["near_news"].map({True: NEWS_SPREAD_MULT, False: 1.0})
                                       .fillna(1.0))
            df["spread_adj"] = df["spread_mean"] * news_mult + commission_p


def build_windows(full_dfs: dict, train_days: int, test_days: int,
                  step_days: int, cap: int) -> list[tuple[date, date, date]]:
    """Return list of (train_start, test_start, test_end) tuples."""
    earliest = max(df["date"].min() for df in full_dfs.values())
    latest   = min(df["date"].max() for df in full_dfs.values())
    windows = []
    train_start = earliest
    while True:
        test_start = train_start + timedelta(days=train_days)
        test_end   = test_start + timedelta(days=test_days)
        if test_end > latest:
            break
        windows.append((train_start, test_start, test_end))
        train_start = train_start + timedelta(days=step_days)
    if cap > 0:
        windows = windows[:cap]
    return windows


def slice_window(full_dfs: dict, train_start: date, test_start: date,
                 test_end: date) -> tuple[dict, dict, dict]:
    train_dfs, test_dfs = {}, {}
    for pair, df in full_dfs.items():
        train_dfs[pair] = df[(df["date"] >= train_start) & (df["date"] < test_start)].reset_index(drop=True)
        test_dfs[pair]  = df[(df["date"] >= test_start)  & (df["date"] < test_end)].reset_index(drop=True)
    measured = {p: float(td["spread_mean"].median())
                for p, td in train_dfs.items() if not td.empty and "spread_mean" in td.columns}
    return train_dfs, test_dfs, measured


def extract_top_features(result: dict, top_n: int) -> list[dict]:
    """Pull top features by |t_stat| from the miner's feature_sweep frame."""
    sweep = result.get("feature_sweep")
    if sweep is None or sweep.empty:
        return []
    df = sweep.copy()
    if "t_stat" in df.columns:
        df["_abs"] = df["t_stat"].abs()
        df = df.sort_values("_abs", ascending=False).drop(columns="_abs")
    return df.head(top_n).to_dict(orient="records")


def main() -> int:
    args = parse_args()
    t0 = time.time()
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]

    print("[sliding] Loading prepared parquets...")
    full_dfs = load_prepared()
    live_spreads = load_live_overrides()
    if live_spreads:
        print(f"[sliding] Live spread overrides available for: {sorted(live_spreads)}")

    windows = build_windows(full_dfs, args.train_days, args.test_days,
                            args.step_days, args.n_windows)
    print(f"[sliding] {len(windows)} window(s) × {len(sessions)} session(s) "
          f"= {len(windows) * len(sessions)} miner runs")
    if not windows:
        print("[sliding] No windows fit — check data range vs train/test days.")
        return 2

    runs = []
    for wi, (ts, te_start, te_end) in enumerate(windows, 1):
        for sess in sessions:
            rs = time.time()
            print(f"\n[sliding] win {wi:2d}/{len(windows)} "
                  f"[{ts} .. {te_start} .. {te_end}]  session={sess}")
            train_dfs, test_dfs, measured = slice_window(full_dfs, ts, te_start, te_end)
            apply_live_override(train_dfs, test_dfs, measured, live_spreads)

            try:
                result = miner.run_miner(
                    pair_dfs       = train_dfs,
                    test_dfs       = test_dfs,
                    session_filter = sess,
                    tp_r           = 2.0,
                    sl_r           = 1.0,
                    horizon_bars   = 20,
                    top_n_patterns = 5,
                    adversarial    = True,
                    append_to_file = False,
                )
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                runs.append({
                    "window": str(ts), "session": sess, "error": str(e),
                    "adversarial_auc": None, "n_patterns": 0, "top_features": [],
                })
                continue

            adv = (result.get("adversarial") or {})
            auc = adv.get("auc")
            verdict = adv.get("verdict")
            patterns = result.get("patterns") or []
            top_feats = extract_top_features(result, TOP_N_FEAT_FOR_AGG)
            elapsed = time.time() - rs
            runs.append({
                "window":            str(ts),
                "test_start":        str(te_start),
                "test_end":          str(te_end),
                "session":           sess,
                "adversarial_auc":   None if auc is None else round(float(auc), 4),
                "adversarial_verdict": verdict,
                "n_patterns":        len(patterns),
                "top_features":      top_feats,
                "elapsed_s":         round(elapsed, 1),
            })
            print(f"  AUC={auc if auc is None else round(float(auc),3)}  "
                  f"verdict={verdict}  patterns={len(patterns)}  "
                  f"({elapsed:.1f}s)")

    # ─── Aggregate ──────────────────────────────────────────────────────────────
    passing = [r for r in runs if r.get("adversarial_auc") is not None
               and r["adversarial_auc"] < ADV_PASS_THRESHOLD]
    feat_counts = Counter()
    feat_directions = defaultdict(Counter)   # feature -> Counter({'>': n, '<': n})
    feat_winrates = defaultdict(list)
    for r in passing:
        for f in r.get("top_features", []):
            name = f.get("feature")
            direction = f.get("direction")
            wr = f.get("win_rate")
            if not name:
                continue
            feat_counts[name] += 1
            if direction:
                feat_directions[name][direction] += 1
            if wr is not None:
                feat_winrates[name].append(float(wr))

    n_passing = len(passing)
    recurring = []
    for feat, count in feat_counts.most_common(15):
        dirs = feat_directions[feat]
        dominant_dir, dom_n = dirs.most_common(1)[0] if dirs else (None, 0)
        consistent = (n_passing > 0 and dom_n / count >= 0.7)
        wr_list = feat_winrates[feat]
        mean_wr = round(sum(wr_list) / len(wr_list), 3) if wr_list else None
        recurring.append({
            "feature": feat, "appearances": count,
            "pct_of_passing": round(count / max(n_passing, 1), 2),
            "dominant_direction": dominant_dir,
            "direction_consistency": consistent,
            "mean_winrate": mean_wr,
        })

    # ─── Report ─────────────────────────────────────────────────────────────────
    lines = []
    lines.append("# Sliding-window mining survey\n")
    lines.append(f"- Train days: **{args.train_days}**, Test days: **{args.test_days}**, "
                 f"Step: **{args.step_days}**")
    lines.append(f"- Windows run: **{len(windows)}**  ×  sessions: **{','.join(sessions)}**  "
                 f"= **{len(runs)}** miner runs")
    lines.append(f"- Passed adversarial (AUC < {ADV_PASS_THRESHOLD}): **{n_passing} / {len(runs)}** "
                 f"({(n_passing/max(len(runs),1))*100:.0f}%)\n")

    aucs = [r["adversarial_auc"] for r in runs if r.get("adversarial_auc") is not None]
    if aucs:
        s = pd.Series(aucs)
        lines.append(f"- AUC distribution: min={s.min():.3f}  median={s.median():.3f}  "
                     f"max={s.max():.3f}\n")

    lines.append("## Per-window summary\n")
    lines.append("| window | session | AUC | verdict | n_patterns | elapsed_s |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        lines.append(f"| {r['window']} | {r['session']} | {r.get('adversarial_auc')} | "
                     f"{r.get('adversarial_verdict')} | {r.get('n_patterns')} | "
                     f"{r.get('elapsed_s')} |")
    lines.append("")

    lines.append("## Recurring features (across passing windows)\n")
    if not passing:
        lines.append("_No windows passed adversarial validation. No features to aggregate._\n")
    else:
        lines.append("| feature | appearances | % passing | direction | consistent? | mean win_rate |")
        lines.append("|---|---|---|---|---|---|")
        for r in recurring:
            lines.append(f"| `{r['feature']}` | {r['appearances']} | "
                         f"{r['pct_of_passing']*100:.0f}% | {r['dominant_direction']} | "
                         f"{'yes' if r['direction_consistency'] else 'no'} | "
                         f"{r['mean_winrate']} |")
        lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    if n_passing == 0:
        lines.append("**Broad mining not viable on this data window.** Even shorter windows "
                     "couldn't pass adversarial validation. Pivot to hand-crafted hypotheses.\n")
    elif n_passing / max(len(runs), 1) < 0.10:
        lines.append("**Marginal**: only a small fraction of windows passed adversarial. "
                     "Probably not enough signal to justify wiring into agent prompts.\n")
    else:
        strong = [r for r in recurring if r["pct_of_passing"] >= 0.50 and r["direction_consistency"]]
        if len(strong) >= 3:
            lines.append(f"**Mineable signal detected.** {len(strong)} features appear in ≥50% "
                         "of passing windows with consistent direction — viable seed list for "
                         "wiring into Claude's hypothesis-generation prompt.\n")
            lines.append("Top stable seeds:")
            for s_ in strong[:5]:
                lines.append(f"- `{s_['feature']}` ({s_['dominant_direction']}) — "
                             f"in {s_['pct_of_passing']*100:.0f}% of passing windows, "
                             f"mean win_rate {s_['mean_winrate']}")
            lines.append("")
        else:
            lines.append("**Adversarial passes but no stable feature recurs.** Mining surfaces "
                         "regime-specific signal, not structural. Lean toward hand-crafted "
                         "hypotheses anchored to mechanical rationale.\n")

    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    Path(args.json).write_text(
        json.dumps({
            "args": vars(args),
            "n_runs": len(runs),
            "n_passing_adversarial": n_passing,
            "runs": runs,
            "recurring_features": recurring,
            "elapsed_min": round((time.time() - t0) / 60, 2),
        }, default=str, indent=2),
        encoding="utf-8",
    )

    print(f"\n[sliding] DONE in {(time.time()-t0)/60:.1f} min")
    print(f"  Report: {args.report}")
    print(f"  JSON:   {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
