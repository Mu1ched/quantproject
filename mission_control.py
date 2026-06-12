"""
Mission Control — single-page GUI for the autonomous edge agent.

One screen, one button. Pick pairs → click Start → everything runs:
  download (resumable) → prepare → generate → backtest → score → repeat.

The GUI never imports the engine. It writes selections to gui_config.json,
spawns `python -m agent.main` as a subprocess, then becomes a pure dashboard
reading three runtime files:

  runtime/agent_status.json          — phase, round, counters, current strategy
  runtime/download_status_multi.json — per-pair download progress
  runtime/agent_control.json         — pause/resume/stop signal (GUI writes)

Run:
    streamlit run mission_control.py

The page top is the always-visible flight deck (setup, status, controls,
survivors / rejected). Everything else lives in collapsed expanders below
so you can drill into detail without leaving the page.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from agent import gui_config, runtime_state, data_inventory
from agent.config import (
    AGENT_DB_PATH,
    BUDGET_DAILY_USD as _DEFAULT_BUDGET_DAILY,
    BUDGET_TOTAL_USD as _DEFAULT_BUDGET_TOTAL,
    HYPOTHESES_PER_BATCH as _DEFAULT_HYPS_PER_BATCH,
    BACKTEST_WORKERS as _DEFAULT_WORKERS,
    LOOP_SLEEP_SECONDS as _DEFAULT_LOOP_SLEEP,
    MIN_TEST_SHARPE as _DEFAULT_MIN_SHARPE,
    MIN_DSR as _DEFAULT_MIN_DSR,
    MIN_TEST_TRADES as _DEFAULT_MIN_TRADES,
    MAX_TEST_DRAWDOWN as _DEFAULT_MAX_DD,
)

# Pull canonical pair list straight from the engine — GUI can never drift.
import edge_engine as _eng
ALL_PAIRS_DEFAULT = list(_eng.DUKA_INST.keys())
PAIR_LABELS = dict(getattr(_eng, "PAIR_LABELS", {}))
for _p in ALL_PAIRS_DEFAULT:
    PAIR_LABELS.setdefault(_p, _p.replace("_", "/"))

from agent import session_router as _sr
ALL_SESSIONS_DEFAULT = (
    [name for _s, _e, name, _p, _x in _sr._LIVE_SCHEDULE]
    + [name for name, _p, _x in _sr.SUB_CORNERS]
)
SESSION_LABELS = {s: _sr.session_display_name(s) for s in ALL_SESSIONS_DEFAULT}

RUNTIME_DIR        = PROJECT_DIR / "runtime"
PID_FILE           = RUNTIME_DIR / "agent.pid"
MT5_PID_FILE       = RUNTIME_DIR / "mt5.pid"
# review#4 — watchdog must be auto-started; previously inert (had to be run
# manually). Tied to MT5Live since that's what the heartbeat protects.
WATCHDOG_PID_FILE  = RUNTIME_DIR / "watchdog.pid"
LOG_FILE       = PROJECT_DIR / "agent" / "agent.log"
MT5_LOG_FILE   = PROJECT_DIR / "live_trader.log"
REPORTS_DIR    = PROJECT_DIR / "agent" / "reports"
EDGE_DB_PATH   = PROJECT_DIR / "edge_results.db"
PREPARED_CACHE = PROJECT_DIR / "edge_prepared_cache"
DUKA_CACHE     = PROJECT_DIR / "duka_cache"

st.set_page_config(page_title="Mission Control", layout="wide",
                   initial_sidebar_state="collapsed")


# ── Process management ───────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except Exception:
        return None
    return pid if _pid_alive(pid) else None


def _agent_pid() -> int | None:
    return _read_pid(PID_FILE)


def _mt5_pid() -> int | None:
    return _read_pid(MT5_PID_FILE)


def _watchdog_pid() -> int | None:
    return _read_pid(WATCHDOG_PID_FILE)


def _spawn_detached(args: list[str]) -> int:
    RUNTIME_DIR.mkdir(exist_ok=True)
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
    proc = subprocess.Popen(
        args, cwd=str(PROJECT_DIR), creationflags=creationflags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc.pid


def _start_agent() -> int:
    # review#P3#3 — PID lock. If a live agent is already running, skip the
    # spawn and return the existing PID. Prevents duplicate agent loops on
    # accidental double-click or Streamlit auto-rerun.
    existing = _agent_pid()
    if existing:
        return existing
    pid = _spawn_detached([sys.executable, "-m", "agent.main"])
    PID_FILE.write_text(str(pid))
    return pid


def _start_mt5() -> int | None:
    mt5_path = PROJECT_DIR / "MT5Live.2.py"
    if not mt5_path.exists():
        return None
    # review#P3#3 — PID lock for MT5Live.
    existing_mt5 = _mt5_pid()
    if existing_mt5:
        return existing_mt5
    pid = _spawn_detached([sys.executable, str(mt5_path)])
    MT5_PID_FILE.write_text(str(pid))
    # review#4 — autostart the dead-man watchdog alongside the live trader.
    # The watchdog only protects against stale heartbeats from MT5Live, so
    # there's no point running it without MT5Live; conversely, running MT5
    # without the watchdog is exactly the false-confidence case the review
    # flagged. PID lock (review#P3#3): only spawn if not already live.
    if _watchdog_pid() is None:
        try:
            wpid = _spawn_detached([sys.executable, "-m", "agent.watchdog"])
            WATCHDOG_PID_FILE.write_text(str(wpid))
        except Exception:
            pass
    return pid


def _kill_pid(pid: int | None, pid_file: Path) -> None:
    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               check=False, capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    pid_file.unlink(missing_ok=True)


# ── DB helpers ───────────────────────────────────────────────────────────────

def _q(sql: str, params: tuple = (), db: Path | str = AGENT_DB_PATH) -> pd.DataFrame:
    if not Path(db).exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db)) as con:
            return pd.read_sql_query(sql, con, params=params)
    except Exception as e:
        st.caption(f"DB read failed: {e}")
        return pd.DataFrame()


def _exec(sql: str, params: tuple = (), db: Path | str = AGENT_DB_PATH) -> None:
    if not Path(db).exists():
        return
    try:
        with sqlite3.connect(str(db)) as con:
            con.execute(sql, params)
            con.commit()
    except Exception as e:
        st.error(f"DB write failed: {e}")


def _spend_today_total() -> tuple[float, float]:
    today = datetime.now(timezone.utc).date().isoformat()
    df = _q("SELECT day, SUM(cost_usd) AS s FROM api_spend GROUP BY day")
    if df.empty:
        return 0.0, 0.0
    today_v = float(df.loc[df["day"] == today, "s"].sum()) if "day" in df else 0.0
    total_v = float(df["s"].sum())
    return today_v, total_v


def _survivors() -> pd.DataFrame:
    return _q("""
        SELECT strategy_name, behaviour_type, session, composite_score,
               test_sharpe, dsr, test_wr, n_trades, max_dd, created_at
        FROM tested_strategies
        WHERE verdict = 'survivor'
           OR (verdict IS NULL AND composite_score IS NOT NULL
               AND (rejection_gate IS NULL OR rejection_gate = ''))
        ORDER BY composite_score DESC
        LIMIT 200
    """)


def _rejection_buckets() -> pd.DataFrame:
    return _q("""
        SELECT COALESCE(rejection_gate, 'unknown') AS gate, COUNT(*) AS n
        FROM tested_strategies
        WHERE rejection_gate IS NOT NULL AND rejection_gate != ''
        GROUP BY rejection_gate
        ORDER BY n DESC
    """)


def _rejected_examples(gate: str, limit: int = 50) -> pd.DataFrame:
    return _q(f"""
        SELECT strategy_name, behaviour_type, session, test_sharpe, dsr,
               n_trades, created_at
        FROM tested_strategies
        WHERE rejection_gate = ?
        ORDER BY id DESC LIMIT {int(limit)}
    """, (gate,))


def _recent_rounds(limit: int = 25) -> pd.DataFrame:
    return _q(f"""
        SELECT round_n, session, n_generated, n_skipped_dup, n_swept,
               n_survivors, created_at
        FROM agent_rounds ORDER BY id DESC LIMIT {int(limit)}
    """)


def _strategy_code(name: str) -> str:
    df = _q("SELECT code FROM tested_strategies WHERE strategy_name = ? LIMIT 1",
            (name,))
    return df["code"].iloc[0] if not df.empty else ""


def _strategy_full(name: str) -> dict:
    df = _q("""SELECT * FROM tested_strategies WHERE strategy_name = ? LIMIT 1""",
            (name,))
    return df.iloc[0].to_dict() if not df.empty else {}


def _meta_guidance() -> str:
    df = _q("""SELECT guidance_text FROM meta_knowledge
               ORDER BY id DESC LIMIT 1""")
    return df["guidance_text"].iloc[0] if not df.empty else ""


def _promoted() -> pd.DataFrame:
    return _q("SELECT * FROM live_promoted ORDER BY rowid DESC")


def _live_trades(n: int = 50) -> pd.DataFrame:
    return _q(f"""SELECT * FROM live_trades
                  ORDER BY rowid DESC LIMIT {int(n)}""")


def _daily_pnl_for_strategy(name: str) -> pd.DataFrame:
    """Per-day PnL for survivor (test set), used for equity curve & corr."""
    df = _q("""
        SELECT date(created_at) AS d, SUM(pnl) AS pnl
        FROM tested_hypothesis_trades
        WHERE strategy_name = ? AND split = 'test'
        GROUP BY date(created_at)
        ORDER BY d
    """, (name,))
    return df


def _tail_log(path: Path, n: int = 60) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as e:
        return f"(log read failed: {e})"


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("Mission Control")

status    = runtime_state.read_status()
multi_dl  = data_inventory.read_multi_status()
agent_pid = _agent_pid()
mt5_pid   = _mt5_pid()
phase     = status.get("phase", "idle")
active    = (agent_pid is not None) and phase not in ("stopped",)

if active:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=2000, key="mc_refresh")
    except Exception:
        pass


# ── 1. Setup card ────────────────────────────────────────────────────────────

with st.expander("⚙ Setup", expanded=not active):
    cfg = gui_config.load() or {}

    cols = st.columns(2)
    with cols[0]:
        pairs = st.multiselect(
            "Pairs to trade",
            options=ALL_PAIRS_DEFAULT,
            default=[p for p in (cfg.get("selected_pairs") or
                                 ['EUR_USD', 'GBP_USD'])
                     if p in ALL_PAIRS_DEFAULT],
            format_func=lambda p: PAIR_LABELS.get(p, p),
            help="Only the selected pairs will be downloaded and traded.",
        )
    with cols[1]:
        sessions = st.multiselect(
            "Sessions",
            options=ALL_SESSIONS_DEFAULT,
            default=[s for s in (cfg.get("selected_sessions") or
                                 ['london', 'ny'])
                     if s in ALL_SESSIONS_DEFAULT],
            format_func=lambda s: SESSION_LABELS.get(s, s),
            help=("Live sessions follow the UTC clock. Sub-corner sessions "
                  "rotate the agent through specific edges (cross-yen mean "
                  "reversion, post-news drift, London→NY handoff, etc.)."),
        )

    cols2 = st.columns(3)
    with cols2[0]:
        cost_mult = st.select_slider(
            "Cost multiplier", options=[0.5, 1.0, 1.5],
            value=float(cfg.get("cost_mult") or 1.0),
            help="0.5 = optimistic spreads. 1.0 = realistic. 1.5 = pessimistic.",
        )
    with cols2[1]:
        budget_daily = st.number_input(
            "Daily budget (USD)", min_value=0.0, max_value=100.0,
            value=float(cfg.get("budget_daily_usd") or _DEFAULT_BUDGET_DAILY),
            step=0.25, format="%.2f",
            help="Hard cap on Claude API spend per day. Calls return None once hit.",
        )
    with cols2[2]:
        budget_total = st.number_input(
            "Total budget (USD)", min_value=0.0, max_value=10000.0,
            value=float(cfg.get("budget_total_usd") or _DEFAULT_BUDGET_TOTAL),
            step=1.0, format="%.2f",
            help="Hard cap on Claude API spend across the lifetime of the project.",
        )

    cols3 = st.columns(3)
    with cols3[0]:
        hyps_per_batch = st.number_input(
            "Hypotheses per batch", min_value=1, max_value=10,
            value=int(cfg.get("hypotheses_per_batch") or _DEFAULT_HYPS_PER_BATCH),
            step=1, help="More = faster discovery, more spend.",
        )
    with cols3[1]:
        workers = st.number_input(
            "Backtest workers", min_value=1, max_value=8,
            value=int(cfg.get("backtest_workers") or _DEFAULT_WORKERS),
            step=1, help="Parallel CPU processes. Each ≈ 2-3GB RSS.",
        )
    with cols3[2]:
        loop_sleep = st.number_input(
            "Loop sleep (s)", min_value=10, max_value=600,
            value=int(cfg.get("loop_sleep_seconds") or _DEFAULT_LOOP_SLEEP),
            step=10, help="Pause between rounds. Larger = lower spend rate.",
        )

    with st.expander("Survivor filter thresholds (advanced)", expanded=False):
        cols4 = st.columns(4)
        with cols4[0]:
            min_sharpe = st.number_input(
                "Min test Sharpe", value=float(cfg.get("min_test_sharpe")
                                               or _DEFAULT_MIN_SHARPE),
                step=0.1, format="%.2f")
        with cols4[1]:
            min_dsr = st.number_input(
                "Min DSR", value=float(cfg.get("min_dsr") or _DEFAULT_MIN_DSR),
                step=0.05, format="%.2f")
        with cols4[2]:
            min_trades = st.number_input(
                "Min test trades", min_value=1,
                value=int(cfg.get("min_test_trades") or _DEFAULT_MIN_TRADES),
                step=5)
        with cols4[3]:
            max_dd = st.number_input(
                "Max test drawdown", value=float(cfg.get("max_test_drawdown")
                                                 or _DEFAULT_MAX_DD),
                step=0.05, format="%.2f")

    force_redownload = st.checkbox(
        "Force re-download data on next start",
        value=bool(cfg.get("force_redownload")),
        help="Wipes the prepared parquet cache for selected pairs and re-fetches.")

    save_col, start_col = st.columns([1, 2])
    new_cfg = {
        "selected_pairs":       pairs,
        "selected_sessions":    sessions,
        "miner_overrides":      cfg.get("miner_overrides") or {},
        "cost_mult":            float(cost_mult),
        "hypotheses_per_batch": int(hyps_per_batch),
        "backtest_workers":     int(workers),
        "budget_daily_usd":     float(budget_daily),
        "budget_total_usd":     float(budget_total),
        "loop_sleep_seconds":   int(loop_sleep),
        "min_test_sharpe":      float(min_sharpe),
        "min_dsr":              float(min_dsr),
        "min_test_trades":      int(min_trades),
        "max_test_drawdown":    float(max_dd),
        "force_redownload":     bool(force_redownload),
    }
    with save_col:
        if st.button("💾 Save config", use_container_width=True):
            gui_config.save(new_cfg)
            st.success("Saved. Restart agent for non-pair/session changes.")
    with start_col:
        if st.button("▶ Save & Start everything", type="primary",
                     use_container_width=True,
                     disabled=(active or not pairs)):
            gui_config.save(new_cfg)
            if force_redownload:
                for p in pairs:
                    pp = PREPARED_CACHE / f"{p}_m1.parquet"
                    pp.unlink(missing_ok=True)
            runtime_state.reset_status()
            runtime_state.set_command("run")
            data_inventory.reset_multi_status(pairs)
            new_pid = _start_agent()
            st.success(f"Agent started (pid {new_pid}). Refreshing…")
            st.rerun()


# ── 2. Status card ───────────────────────────────────────────────────────────

st.subheader("Status")

phase_emoji = {
    "idle": "⚪", "downloading": "⬇️", "preparing": "⚙️",
    "generating": "🧬", "backtesting": "📊",
    "paused": "⏸", "stopped": "⏹",
}.get(phase, "•")

spend_today_db, spend_total_db = _spend_today_total()

c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
c1.metric("Phase", f"{phase_emoji} {phase}")
c2.metric("Round", status.get("round", 0))
c3.metric("Tested today", status.get("tested_today", 0))
c4.metric("Survivors today", status.get("survivors_today", 0))
c5.metric("Spend today",
          f"${spend_today_db:.3f}",
          f"of ${budget_daily:.2f}" if 'budget_daily' in dir() else None)
c6.metric("Spend total",
          f"${spend_total_db:.2f}",
          f"of ${budget_total:.2f}" if 'budget_total' in dir() else None)

# Spend gauges
g1, g2 = st.columns(2)
with g1:
    cap = float((gui_config.load() or {}).get("budget_daily_usd")
                or _DEFAULT_BUDGET_DAILY)
    st.progress(min(spend_today_db / cap, 1.0) if cap else 0.0,
                text=f"Daily budget: ${spend_today_db:.3f} / ${cap:.2f}")
with g2:
    cap_t = float((gui_config.load() or {}).get("budget_total_usd")
                  or _DEFAULT_BUDGET_TOTAL)
    st.progress(min(spend_total_db / cap_t, 1.0) if cap_t else 0.0,
                text=f"Total budget: ${spend_total_db:.2f} / ${cap_t:.2f}")

if status.get("current_strategy"):
    st.caption(f"⇒ currently backtesting **{status['current_strategy']}**"
               + (f" ({status.get('current_session')})"
                  if status.get('current_session') else ""))
if status.get("last_error"):
    st.warning(f"Last error: {status['last_error']}")
if not agent_pid:
    st.info("Agent is not running. Click **Start everything** above.")

# Per-pair download bars.
dl_pairs = (multi_dl or {}).get("pairs", {})
if dl_pairs:
    st.markdown("**Data download**")
    for pair, entry in dl_pairs.items():
        ph = entry.get("phase", "?")
        done = int(entry.get("days_done", 0) or 0)
        total = int(entry.get("days_total", 0) or 0)
        pct = (done / total) if total else (1.0 if ph == "done" else 0.0)
        label = f"{PAIR_LABELS.get(pair, pair)} — {ph}"
        if total:
            label += f" ({done}/{total} days)"
        st.progress(min(max(pct, 0.0), 1.0), text=label)


# ── 3. Controls ──────────────────────────────────────────────────────────────

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
cmd = runtime_state.current_command()
with ctrl1:
    if st.button("⏸ Pause", disabled=(not active or cmd == "pause"),
                 use_container_width=True):
        runtime_state.set_command("pause"); st.rerun()
with ctrl2:
    if st.button("▶ Resume", disabled=(cmd != "pause"),
                 use_container_width=True):
        runtime_state.set_command("resume"); st.rerun()
with ctrl3:
    if st.button("⏹ Stop", disabled=(not active),
                 use_container_width=True):
        runtime_state.set_command("stop")
        st.info("Stop requested — finishing current backtest…")
        st.rerun()
with ctrl4:
    if st.button("✖ Force kill", disabled=(not agent_pid),
                 use_container_width=True,
                 help="Last resort. Skips the cooperative stop."):
        _kill_pid(agent_pid, PID_FILE)
        runtime_state.update_status(phase="stopped")
        st.rerun()


# ── 4. Survivors / Rejected ──────────────────────────────────────────────────

st.markdown("---")
left, right = st.columns(2)

with left:
    surv = _survivors()
    st.subheader(f"✅ Survivors ({len(surv)})")
    if surv.empty:
        st.caption("No survivors yet. They appear here once the agent has "
                   "produced a strategy that passes every filter.")
    else:
        show = surv.copy()
        for c in ("composite_score", "test_sharpe", "dsr", "test_wr", "max_dd"):
            if c in show.columns:
                show[c] = show[c].astype(float).round(3)
        st.dataframe(show, use_container_width=True, hide_index=True, height=420)

with right:
    buckets = _rejection_buckets()
    total_rej = int(buckets["n"].sum()) if not buckets.empty else 0
    st.subheader(f"❌ Rejected ({total_rej})")
    if buckets.empty:
        st.caption("No rejections recorded yet.")
    else:
        for _, r in buckets.iterrows():
            with st.expander(f"{r['gate']}  ({int(r['n'])})"):
                ex = _rejected_examples(r['gate'])
                if ex.empty:
                    st.caption("(no examples)")
                else:
                    for c in ("test_sharpe", "dsr"):
                        if c in ex.columns:
                            ex[c] = ex[c].astype(float).round(3)
                    st.dataframe(ex, use_container_width=True,
                                 hide_index=True, height=240)


# ── 5. Drill-down expanders ──────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Detail & Operations")

# 5a. Survivor detail
with st.expander("🔬 Survivor detail — equity curve & generated code"):
    if surv.empty:
        st.caption("No survivors to inspect yet.")
    else:
        names = surv["strategy_name"].tolist()
        chosen = st.selectbox("Pick a survivor", names, key="surv_pick")
        if chosen:
            full = _strategy_full(chosen)
            metric_cols = st.columns(6)
            for i, (k, fmt) in enumerate([
                ("composite_score", "{:.3f}"), ("test_sharpe", "{:.2f}"),
                ("dsr", "{:.2f}"), ("test_wr", "{:.1%}"),
                ("n_trades", "{:.0f}"), ("max_dd", "{:.1%}"),
            ]):
                v = full.get(k)
                metric_cols[i].metric(
                    k, fmt.format(float(v)) if v not in (None, "") else "—")
            pnl_df = _daily_pnl_for_strategy(chosen)
            if not pnl_df.empty:
                pnl_df["equity"] = pnl_df["pnl"].cumsum()
                st.line_chart(pnl_df.set_index("d")["equity"],
                              use_container_width=True, height=240)
            else:
                st.caption("No per-day PnL persisted for this strategy yet.")
            code = _strategy_code(chosen)
            if code:
                st.code(code, language="python")
            if full.get("rationale"):
                st.caption(f"Rationale: {full['rationale']}")

# 5b. Portfolio correlation
with st.expander("📊 Portfolio correlation between survivors"):
    if surv.empty or len(surv) < 2:
        st.caption("Need at least 2 survivors before correlation is meaningful.")
    else:
        if st.button("Compute correlation matrix", key="corr_btn"):
            pnl_by = {}
            for n in surv["strategy_name"].tolist()[:30]:
                d = _daily_pnl_for_strategy(n)
                if not d.empty:
                    pnl_by[n] = d.set_index("d")["pnl"]
            if not pnl_by:
                st.caption("No per-day PnL available yet.")
            else:
                wide = pd.DataFrame(pnl_by).fillna(0.0)
                corr = wide.corr().round(2)
                st.dataframe(corr, use_container_width=True)
                hi = []
                for i, a in enumerate(corr.columns):
                    for b in corr.columns[i+1:]:
                        if abs(corr.loc[a, b]) > 0.70:
                            hi.append((a, b, float(corr.loc[a, b])))
                if hi:
                    st.warning(f"{len(hi)} pair(s) above |0.70|:")
                    st.dataframe(pd.DataFrame(hi, columns=["A", "B", "corr"]),
                                 use_container_width=True, hide_index=True)

# 5c. TCA + live decay
with st.expander("💸 TCA — live cost & per-strategy decay"):
    if st.button("Refresh TCA from MT5 CSVs", key="tca_btn"):
        try:
            from agent import tca as _tca
            _tca.run()
            st.success("TCA refreshed.")
        except Exception as e:
            st.error(f"TCA failed: {e}")
    try:
        from agent import tca as _tca
        per_pair = _tca.per_pair_summary()
        if not per_pair.empty:
            st.markdown("**Per-pair cost summary**")
            st.dataframe(per_pair, use_container_width=True, hide_index=True)
        decay = _tca.per_strategy_decay()
        if not decay.empty:
            st.markdown("**Per-strategy live-vs-backtest decay**")
            st.dataframe(decay, use_container_width=True, hide_index=True)
        if per_pair.empty and decay.empty:
            st.caption("No live trades ingested yet — let MT5live run, then refresh.")
    except Exception as e:
        st.caption(f"TCA module unavailable: {e}")

# 5d. MT5 live trader
with st.expander("📡 MT5 live trader"):
    mt5_running = mt5_pid is not None
    cols_mt5 = st.columns([1, 1, 2])
    with cols_mt5[0]:
        if st.button("▶ Start MT5", disabled=mt5_running,
                     use_container_width=True):
            new_pid = _start_mt5()
            if new_pid:
                st.success(f"MT5live started (pid {new_pid}).")
            else:
                st.error("MT5Live.2.py not found in project root.")
            st.rerun()
    with cols_mt5[1]:
        if st.button("⏹ Stop MT5", disabled=not mt5_running,
                     use_container_width=True):
            _kill_pid(mt5_pid, MT5_PID_FILE)
            # review#4 — tear down the watchdog with MT5Live so we don't
            # leave it polling a heartbeat that will never refresh.
            _kill_pid(_watchdog_pid(), WATCHDOG_PID_FILE)
            st.rerun()
    with cols_mt5[2]:
        st.caption(f"MT5 PID: {mt5_pid or '—'}  ·  "
                   f"requires `MT5_LOGIN/PASSWORD/SERVER` in `.env`")

    promoted = _promoted()
    if not promoted.empty:
        st.markdown("**Promoted strategies (live / shadow)**")
        st.dataframe(promoted, use_container_width=True, hide_index=True,
                     height=200)

    trades = _live_trades()
    if not trades.empty:
        st.markdown("**Recent live trades**")
        st.dataframe(trades, use_container_width=True, hide_index=True,
                     height=200)

    with st.expander("MT5 log tail", expanded=False):
        st.code(_tail_log(MT5_LOG_FILE, 60), language="text")

# 5e. Manual ops
with st.expander("🧬 Manual operations — run sweep, mine, send report"):
    cols_ops = st.columns(3)

    # Manual named sweep
    with cols_ops[0]:
        st.markdown("**Run named sweep**")
        try:
            from edge_hypotheses import SWEEPS
            sweep_keys = list(SWEEPS.keys())
        except Exception:
            sweep_keys = []
        if sweep_keys:
            sweep_pick = st.selectbox("Sweep", sweep_keys, key="sweep_pick")
            if st.button("Run sweep", key="sweep_run", use_container_width=True):
                _spawn_detached([sys.executable, "-c",
                                 f"import edge_engine as e, edge_hypotheses as h; "
                                 f"t,_,_=e.load_all_data(); "
                                 f"e.run_sweep('{sweep_pick}', h.SWEEPS['{sweep_pick}'])"])
                st.success(f"Spawned sweep '{sweep_pick}' (background).")
        else:
            st.caption("No SWEEPS dict found.")

    # Run miner
    with cols_ops[1]:
        st.markdown("**Run pattern miner**")
        if st.button("Run LightGBM miner now", key="mine_run",
                     use_container_width=True):
            _spawn_detached([sys.executable, "-m", "agent.main", "--mine-now"])
            st.success("Miner spawned (background).")

    # Send report
    with cols_ops[2]:
        st.markdown("**Send Telegram report**")
        if st.button("Send daily report now", key="report_run",
                     use_container_width=True):
            _spawn_detached([sys.executable, "-m", "agent.main", "--report-now"])
            st.success("Report spawned (background).")

# 5f. Meta-learner & rounds
with st.expander("🧠 Meta-learner guidance & rounds history"):
    g = _meta_guidance()
    if g:
        st.markdown("**Current meta-learner guidance** (injected into every prompt)")
        st.code(g, language="markdown")
    else:
        st.caption("Meta-learner has not synthesised guidance yet.")

    rr = _recent_rounds(40)
    if not rr.empty:
        st.markdown("**Recent rounds**")
        st.dataframe(rr, use_container_width=True, hide_index=True, height=300)

    if REPORTS_DIR.exists():
        reports = sorted(
            (p for p in REPORTS_DIR.rglob("*.md")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )[:25]
        if reports:
            st.markdown("**Generated reports**")
            for r in reports:
                rel = r.relative_to(PROJECT_DIR)
                st.caption(f"📄 {rel}  ·  "
                           f"{datetime.fromtimestamp(r.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")

# 5g. Logs
with st.expander("📜 Agent log tail (last 80 lines)"):
    st.code(_tail_log(LOG_FILE, 80), language="text")

# 5h. Admin
with st.expander("⚙ Admin — wipe data, reset DBs (destructive)"):
    st.warning("These actions cannot be undone. The agent must be stopped first.")
    cols_adm = st.columns(3)
    with cols_adm[0]:
        confirm_db = st.checkbox("Confirm: wipe agent DB", key="adm_db")
        if st.button("🗑 Wipe agent_results.db", disabled=not confirm_db,
                     use_container_width=True):
            try:
                Path(AGENT_DB_PATH).unlink(missing_ok=True)
                st.success("agent_results.db deleted (will be recreated).")
            except Exception as e:
                st.error(str(e))
    with cols_adm[1]:
        confirm_edge = st.checkbox("Confirm: wipe edge sweep DB", key="adm_edge")
        if st.button("🗑 Wipe edge_results.db", disabled=not confirm_edge,
                     use_container_width=True):
            try:
                EDGE_DB_PATH.unlink(missing_ok=True)
                st.success("edge_results.db deleted.")
            except Exception as e:
                st.error(str(e))
    with cols_adm[2]:
        confirm_data = st.checkbox("Confirm: wipe prepared cache", key="adm_data")
        if st.button("🗑 Wipe edge_prepared_cache",
                     disabled=not confirm_data, use_container_width=True):
            try:
                if PREPARED_CACHE.exists():
                    shutil.rmtree(PREPARED_CACHE)
                st.success("Prepared cache cleared (raw duka_cache kept).")
            except Exception as e:
                st.error(str(e))

    cols_adm2 = st.columns(3)
    with cols_adm2[0]:
        if st.button("Clear meta-learner knowledge", use_container_width=True):
            _exec("DELETE FROM meta_knowledge")
            st.success("Meta knowledge cleared.")
    with cols_adm2[1]:
        if st.button("Clear rejection log", use_container_width=True):
            _exec("DELETE FROM rejection_log")
            _exec("DELETE FROM feature_rejections")
            st.success("Rejection log cleared.")
    with cols_adm2[2]:
        if st.button("Reset live circuit breakers", use_container_width=True):
            _exec("DELETE FROM strategy_breaker")
            st.success("Circuit breakers cleared.")


# ── Footer ───────────────────────────────────────────────────────────────────

st.caption(
    f"agent_status.ts={status.get('ts') or '—'}  ·  "
    f"agent_pid={agent_pid or '—'}  ·  "
    f"mt5_pid={mt5_pid or '—'}  ·  "
    f"db={Path(AGENT_DB_PATH).name}"
)
