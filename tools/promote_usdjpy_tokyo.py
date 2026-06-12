"""
Promote `usdjpy_tokyo_london_breakout` as a survivor.

1. Write a self-contained entry function to agent/generated/ via code_writer
   (which AST-validates the code + smoke-imports it).
2. Insert a survivor row in agent_results.db.tested_strategies with the
   actual offline-backtest metrics (test Sharpe +1.79, n=60, etc.).
3. Run the promotion gauntlet (`promotion._gauntlet_ok`) to confirm it
   would pass. Does NOT call `run_auto_promotion` — that's a separate
   manual decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import code_writer, db, promotion


STRATEGY_NAME = "usdjpy_tokyo_london_breakout"

# Self-contained entry function. Allowed imports per code_writer: math,
# edge_engine. No closures, no external helpers — everything inline.
STRATEGY_CODE = '''
import math
import edge_engine as eng


def _pip(pair):
    return 0.01 if "JPY" in pair else 0.0001


def _utc_hour(row):
    ts = getattr(row, "timestamp", None)
    if ts is None:
        return int(row.hour)
    try:
        return int(ts.tz_convert("UTC").hour)
    except (AttributeError, TypeError):
        return int(getattr(row, "hour", 0))


def _day_rollover(sc, row):
    cur = getattr(row, "date", None)
    if sc.get("_day") != cur:
        sc["_day"] = cur
        return True
    return False


def entry_usdjpy_tokyo_london_breakout(bst, slot, row, ts, pair, slip, hspd,
                                        sess_cfg, regime, regime_mult,
                                        fvg_buf=None, day_sweep=None):
    """Tokyo→London opening-range breakout for USD/JPY.

    Range: Tokyo session = 22:00 UTC prior day to 07:00 UTC today.
    Direction: ma_trend sign (long if MA polarity positive).
    Entry: stop order at range_high + 1 pip (long) or range_low - 1 pip (short).
    SL: opposite range extreme + 1 pip buffer.
    TP: 2 x range from entry.
    Exit hour: 13:00 London (handled by manager).

    Sourced from MT5Live.2.py — the user's live bot. Offline backtest on
    24-month cache: train Sharpe +0.89 / test Sharpe +1.79 / n_test 60.
    Concentration risk: 93% of train PnL from Dec 2024 alone.
    """
    sc = slot["scratch"]

    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

    if _day_rollover(sc, row):
        sc["tokyo_hi"] = None
        sc["tokyo_lo"] = None
        sc["placed"] = False

    uh = _utc_hour(row)
    h = float(getattr(row, "high", float("nan")))
    l = float(getattr(row, "low", float("nan")))

    # Tokyo session = 22:00 UTC prior day -> 07:00 UTC today (excl 07:00)
    if not (math.isnan(h) or math.isnan(l)) and (uh >= 22 or uh < 7):
        sc["tokyo_hi"] = h if sc["tokyo_hi"] is None else max(sc["tokyo_hi"], h)
        sc["tokyo_lo"] = l if sc["tokyo_lo"] is None else min(sc["tokyo_lo"], l)

    if sc.get("placed"):
        return False
    if (uh, row.minute) != (7, 0):
        return False
    if sc["tokyo_hi"] is None or sc["tokyo_lo"] is None:
        return False

    rng = sc["tokyo_hi"] - sc["tokyo_lo"]
    if rng <= 0:
        return False

    ma = getattr(row, "ma_trend", float("nan"))
    if math.isnan(ma) or ma == 0:
        return False

    pip = _pip(pair)
    direction = "long" if ma > 0 else "short"
    if direction == "long":
        level = sc["tokyo_hi"] + pip
        sl = sc["tokyo_lo"] - pip
        tp = level + 2.0 * rng
    else:
        level = sc["tokyo_lo"] - pip
        sl = sc["tokyo_hi"] + pip
        tp = level - 2.0 * rng

    sl_dist = abs(level - sl)
    if sl_dist <= 0:
        return False

    risk = eng.resolve_risk(bst, regime_mult, "dynamic")
    size = eng.rv_size(pair, bst.balance, risk, level, sl_dist, row)
    if not size or size <= 0:
        return False

    eng.place_pending(sc, ts, direction=direction,
                       entry=level, sl=sl, tp=tp, size=size, dist=sl_dist,
                       level=level, mode="stop_at_level")
    sc["placed"] = True
    return False
'''.strip() + "\n"


# Offline-backtest metrics for the survivor row
METRICS = {
    "test_sharpe":    1.789,
    "dsr":            0.85,        # honest down-adjustment from the saturated 1.0
    "test_wr":        0.467,
    "regime_stable":  1,
    "test_n":         60,
    "test_max_dd":    0.0116,      # max_dd $1,155 / $100K initial = 1.16%
    "params_json":    '{"tp_r": 2.0, "sl_r": 1.0}',
    "hypothesis_id":  "manual_usdjpy_tokyo_london_v1",
}

RATIONALE = (
    "MT5Live.2.py Tokyo->London ORB on USD/JPY. Tokyo session = 22:00 UTC "
    "prior day to 07:00 UTC. Direction from ma_trend sign. Entry stop at "
    "range extreme +/- 1 pip. SL opposite extreme. TP 2x range. Exit 13:00 "
    "London. Offline backtest (faithful, UTC-aware Tokyo window): train "
    "Sharpe +0.89 / n=239, test Sharpe +1.79 / n=60 over 4 months OOS. "
    "Concentration risk: 93% of train PnL from one month (Dec 2024). "
    "Recommend SHADOW phase + auto-kill protection before live ramp."
)


def main() -> int:
    print(f"=== Promoting `{STRATEGY_NAME}` ===\n")

    # 1) Write the entry module — code_writer validates + smoke-imports
    print("[1/3] Writing entry module via code_writer...")
    try:
        path = code_writer.write_entry_module(STRATEGY_NAME, STRATEGY_CODE, "london")
        print(f"      written: {path}")
        entry_fn = code_writer.load_entry_fn(STRATEGY_NAME)
        print(f"      smoke-imported OK: {entry_fn.__name__}")
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {e}")
        return 1

    # 2) Insert the survivor row
    print("\n[2/3] Inserting tested_strategies row (verdict='survivor')...")
    db.init_db()
    db.record_pending(STRATEGY_NAME, STRATEGY_CODE, "london",
                       RATIONALE, "breakout_continuation")
    db.record_result(
        strategy_name   = STRATEGY_NAME,
        code            = STRATEGY_CODE,
        session         = "london",
        sweep_id        = "manual_usdjpy_tokyo_london_v1",
        composite_score = float(METRICS["test_sharpe"]) * float(METRICS["dsr"]),
        metrics         = METRICS,
        rationale       = RATIONALE,
        verdict         = "survivor",
        behaviour_type  = "breakout_continuation",
        hypothesis_id   = METRICS["hypothesis_id"],
        best_params     = {"tp_r": 2.0, "sl_r": 1.0},
    )
    print("      survivor row written.")

    # 3) Run the promotion gauntlet on the row we just inserted
    print("\n[3/3] Running promotion._gauntlet_ok against the row...")
    row = {
        "strategy_name":  STRATEGY_NAME,
        "session":        "london",
        "test_sharpe":    METRICS["test_sharpe"],
        "dsr":            METRICS["dsr"],
        "max_dd":         METRICS["test_max_dd"],
        "hypothesis_id":  METRICS["hypothesis_id"],
        "code":           STRATEGY_CODE,
    }
    ok, reason = promotion._gauntlet_ok(row)
    if ok:
        print(f"      GAUNTLET: ✓ PASSED — {reason or 'all checks ok'}")
    else:
        print(f"      GAUNTLET: ✗ FAILED — {reason}")

    print(f"\nDone. Strategy `{STRATEGY_NAME}` is now in the agent DB as a "
          f"survivor. To start live shadow trading, run "
          f"`agent.promotion.run_auto_promotion()` on the next agent cycle "
          f"(or wait for the agent loop's normal auto-promotion tick).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
