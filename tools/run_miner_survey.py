"""
Run edge_miner across NY and London sessions on the user's cached pairs.
Prints a clean summary of: top SHAP-ranked features, top patterns surviving
walk-forward CV, and the adversarial-validation verdict per session.

This is a one-off survey to find out whether the miner actually surfaces
stable predictive signal under live-calibrated costs — before deciding to
wire it into the autonomous agent's hypothesis-generation step.

Usage:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    python tools/run_miner_survey.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_engine import load_all_data
import edge_miner as miner


SESSIONS = ["ny", "london"]
OUT_REPORT = PROJECT_ROOT / "tools" / "miner_survey_report.md"
OUT_JSON   = PROJECT_ROOT / "tools" / "miner_survey.json"


def summarise(result: dict, session: str) -> dict:
    sweep   = result.get("feature_sweep")
    patterns = result.get("patterns") or []
    adv      = result.get("adversarial") or {}
    dist     = result.get("dist_shift") or {}
    meta     = result.get("meta_learner") or {}

    out = {
        "session":           session,
        "adversarial_auc":   adv.get("auc"),
        "adversarial_verdict": adv.get("verdict"),
        "adversarial_message": adv.get("message"),
        "dist_shift_features": (dist.get("shifted_features") or [])[:10],
        "n_patterns_passed":  len(patterns),
        "top_patterns":       patterns[:10],   # already SHAP+CV-ranked
        "meta_learner":       meta,
    }

    # Top features by mean |edge| from the feature sweep, if it ran.
    if sweep is not None and not sweep.empty:
        cols = sweep.columns.tolist()
        # heuristic: rank by abs of any column matching 'edge' or 'lift'
        rank_col = next((c for c in cols
                         if any(k in c.lower() for k in ("edge", "lift", "sharpe"))),
                        None)
        if rank_col:
            top = (sweep.assign(_abs=sweep[rank_col].abs())
                       .sort_values("_abs", ascending=False)
                       .drop(columns="_abs")
                       .head(15))
            out["top_features"] = top.to_dict(orient="records")
        else:
            out["top_features"] = sweep.head(15).to_dict(orient="records")
    else:
        out["top_features"] = []

    return out


def fmt_pattern(p: dict) -> str:
    rule = p.get("rule") or p.get("expression") or p.get("conditions") or p
    score = p.get("composite") or p.get("sharpe") or p.get("lift") or p.get("score")
    return f"score={score} :: {rule}"


def render_report(summaries: list[dict]) -> str:
    lines = ["# Edge-miner survey — NY + London under live-calibrated costs\n"]
    lines.append("Goal: see whether the miner surfaces stable signal before we wire it into the agent loop.\n")
    for s in summaries:
        lines.append(f"## Session: {s['session']}\n")
        if s.get("error"):
            lines.append(f"- ERROR: `{s['error']}`\n")
            continue
        lines.append(f"- Adversarial AUC: **{s.get('adversarial_auc')}** "
                     f"(verdict: {s.get('adversarial_verdict')})")
        if s.get("adversarial_message"):
            lines.append(f"  - {s['adversarial_message']}")
        if s.get("dist_shift_features"):
            lines.append(f"- Distribution shift on features: "
                         f"{', '.join(s['dist_shift_features'])}")
        lines.append(f"- Patterns passing walk-forward CV: **{s.get('n_patterns_passed', 0)}**\n")

        if s.get("top_features"):
            lines.append("### Top features (sweep)\n")
            for f in s["top_features"][:10]:
                lines.append(f"- `{f}`")
            lines.append("")
        if s.get("top_patterns"):
            lines.append("### Top patterns (post walk-forward CV)\n")
            for p in s["top_patterns"][:10]:
                lines.append(f"- {fmt_pattern(p)}")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    print("[miner-survey] Loading data...")
    train_dfs, test_dfs, _ = load_all_data()
    print(f"[miner-survey] Loaded {len(train_dfs)} pairs: {sorted(train_dfs)}")
    # edge_miner.build_label_dataset expects a `ts` column; the prepared
    # cache produces `timestamp`. Alias it once here.
    for _dfs in (train_dfs, test_dfs):
        for _pair, _df in _dfs.items():
            if "ts" not in _df.columns and "timestamp" in _df.columns:
                _df["ts"] = _df["timestamp"]

    summaries = []
    for sess in SESSIONS:
        sess_t0 = time.time()
        print(f"\n[miner-survey] === Session: {sess} ===")
        try:
            result = miner.run_miner(
                pair_dfs       = train_dfs,
                test_dfs       = test_dfs,
                session_filter = sess,
                tp_r           = 2.0,
                sl_r           = 1.0,
                horizon_bars   = 20,
                top_n_patterns = 10,
                adversarial    = True,
                append_to_file = False,    # don't write generated hypotheses
            )
        except Exception as e:
            print(f"[miner-survey] {sess}: FAILED — {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            summaries.append({"session": sess, "error": str(e)})
            continue
        elapsed = time.time() - sess_t0
        s = summarise(result, sess)
        s["elapsed_min"] = round(elapsed / 60, 2)
        summaries.append(s)
        print(f"[miner-survey] {sess}: done in {elapsed/60:.1f} min, "
              f"patterns={s['n_patterns_passed']}, adv={s.get('adversarial_verdict')}")

    total = time.time() - t0
    OUT_REPORT.write_text(render_report(summaries), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps({"elapsed_min": round(total/60, 2), "summaries": summaries}, default=str, indent=2),
        encoding="utf-8",
    )
    print(f"\n[miner-survey] DONE in {total/60:.1f} min")
    print(f"  Report: {OUT_REPORT}")
    print(f"  JSON:   {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
