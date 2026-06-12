# -*- coding: utf-8 -*-
"""
Run GUI — minimal 2-tab dashboard for the autonomous agent.

  Tab 1 (Run): one big Start button. Live status, recent activity, and the
               equity curve of the most-recently-completed strategy.
  Tab 2 (Results): sortable table of every strategy ever tested; click a
                   row to inspect that strategy's equity curve.

Launch:
    cd C:\\Users\\malac\\Downloads\\Quantproject
    streamlit run run_gui.py
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import queue
import threading

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH = True
except ImportError:
    _AUTOREFRESH = False

PROJECT_DIR  = Path(__file__).resolve().parent
RUNTIME_DIR  = PROJECT_DIR / "runtime"
PID_FILE     = RUNTIME_DIR / "agent.pid"
STATUS_FILE  = RUNTIME_DIR / "agent_status.json"
CONTROL_FILE = RUNTIME_DIR / "agent_control.json"
LOG_PATH     = PROJECT_DIR / "agent" / "agent.log"
AGENT_DB     = PROJECT_DIR / "agent"  / "agent_results.db"
EDGE_DB      = PROJECT_DIR / "edge_results.db"


# ── Process management ───────────────────────────────────────────────────────

def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return None
    # Verify alive
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                              capture_output=True, text=True).stdout
        if str(pid) in out:
            return pid
    else:
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            return None
    return None


def _spawn_agent() -> int:
    existing = _read_pid()
    if existing:
        return existing
    RUNTIME_DIR.mkdir(exist_ok=True)
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent.main"],
        cwd=str(PROJECT_DIR),
        creationflags=creationflags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    PID_FILE.write_text(str(proc.pid))
    # Make sure control file says "run"
    CONTROL_FILE.write_text(json.dumps({"command": "run"}))
    return proc.pid


def _stop_agent(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    PID_FILE.unlink(missing_ok=True)


# ── Status / log readers ────────────────────────────────────────────────────

def _read_status() -> dict:
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {}


def _read_log_tail(n: int = 30) -> list[str]:
    if not LOG_PATH.exists():
        return []
    try:
        # Naive tail — fine at typical log sizes (<10 MB before rotation)
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


# ── DB queries ───────────────────────────────────────────────────────────────

def _latest_tested_strategies(limit: int = 100) -> pd.DataFrame:
    if not AGENT_DB.exists():
        return pd.DataFrame()
    con = sqlite3.connect(AGENT_DB)
    try:
        df = pd.read_sql(
            "SELECT id, strategy_name, session, sweep_id, test_sharpe, dsr, "
            "test_wr, n_trades, max_dd, verdict, best_params_json, created_at "
            "FROM tested_strategies ORDER BY id DESC LIMIT ?",
            con, params=(limit,),
        )
    finally:
        con.close()
    return df


def _equity_curve_for_sweep(sweep_id: str) -> pd.DataFrame:
    """Return an equity curve DataFrame for the best hypothesis of a sweep."""
    if not EDGE_DB.exists() or not sweep_id:
        return pd.DataFrame()
    con = sqlite3.connect(EDGE_DB)
    try:
        # Best hypothesis by test_sharpe
        h = con.execute(
            "SELECT hypothesis_id FROM hypotheses WHERE sweep_id=? "
            "ORDER BY test_sharpe DESC LIMIT 1", (sweep_id,)
        ).fetchone()
        if not h:
            return pd.DataFrame()
        hyp_id = h[0]
        trades = pd.read_sql(
            "SELECT exit_time, pnl, balance FROM trades "
            "WHERE hypothesis_id=? AND split='test' ORDER BY exit_time",
            con, params=(hyp_id,),
        )
    finally:
        con.close()
    return trades


def _latest_completed_sweep_id() -> str | None:
    """Most recent sweep_id seen in tested_strategies (non-NULL)."""
    df = _latest_tested_strategies(limit=10)
    if df.empty:
        return None
    df = df[df["sweep_id"].notna()]
    if df.empty:
        return None
    return str(df.iloc[0]["sweep_id"])


# ── Live backtest tail (written by agent.loop._live_trades_*) ────────────────

LIVE_TRADES_FILE = RUNTIME_DIR / "live_trades.jsonl"


def _read_live_trades_stream() -> dict:
    """Read the running per-trade JSONL written by the agent's pre_screen
    progress_callback. Returns:
      {"header": {...} | None,
       "trades": [{pnl, balance, ...}, ...],
       "done":   {passed, reason, ended_at} | None,
       "stale":  bool}   # True if file hasn't changed in 60s+
    """
    out = {"header": None, "trades": [], "done": None, "stale": False}
    if not LIVE_TRADES_FILE.exists():
        return out
    try:
        # Stale check — pre_screen never runs longer than ~120s, so a file
        # untouched for 5min means the agent crashed or moved on
        age = time.time() - LIVE_TRADES_FILE.stat().st_mtime
        out["stale"] = age > 300
        for ln in LIVE_TRADES_FILE.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            etype = ev.get("type")
            if etype == "start":
                out["header"] = ev
                out["trades"] = []   # truncate when we see a new start
                out["done"]   = None
            elif etype == "trade":
                out["trades"].append(ev)
            elif etype == "done":
                out["done"] = ev
    except Exception:
        pass
    return out


def _recent_outcomes(limit: int = 5) -> pd.DataFrame:
    """Last N strategies with non-pending verdict, newest first.
    Used to show what JUST got accepted/rejected in the main tab."""
    df = _latest_tested_strategies(limit=50)
    if df.empty:
        return df
    df = df[df["verdict"].isin(["rejected", "survivor", "correlated",
                                 "adversarial_reject"])]
    return df.head(limit)


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Edge Agent", layout="wide")

agent_pid = _read_pid()
status    = _read_status()
phase     = status.get("phase", "idle")
active    = (agent_pid is not None) and phase not in ("stopped",)

# Always-visible header banner
phase_emoji = {
    "idle": "⚪", "downloading": "⬇️", "preparing": "⚙️",
    "generating": "🧬", "backtesting": "📊",
    "paused": "⏸", "stopped": "⏹",
}.get(phase, "•")
st.title(f"{phase_emoji}  Edge Agent  ·  {phase}  ·  pid {agent_pid or '—'}")

tab_run, tab_results, tab_live = st.tabs(
    ["▶  Run & Watch", "📋  Results", "📉  Live Backtest"]
)


# ── Tab 1: Run & Watch ──────────────────────────────────────────────────────

with tab_run:
    # Big button
    if not active:
        if st.button("▶  GO  —  Start agent",
                     type="primary", use_container_width=True):
            new_pid = _spawn_agent()
            st.success(f"Agent started (pid {new_pid}). Refreshing…")
            time.sleep(0.5)
            st.rerun()
    else:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("⏹  STOP agent", type="primary",
                         use_container_width=True):
                _stop_agent(agent_pid)
                st.warning("Stop signal sent.")
                time.sleep(0.5)
                st.rerun()
        with b2:
            if st.button("⏸  Pause / ▶ Resume", use_container_width=True):
                cur = "run"
                try:
                    cur = json.loads(CONTROL_FILE.read_text()).get("command", "run")
                except Exception:
                    pass
                new_cmd = "pause" if cur != "pause" else "run"
                CONTROL_FILE.write_text(json.dumps({"command": new_cmd}))
                st.info(f"Sent {new_cmd}.")
                time.sleep(0.3)
                st.rerun()

    st.divider()

    # Status row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round", status.get("round", 0))
    c2.metric("Tested today", status.get("tested_today", 0))
    c3.metric("Survivors today", status.get("survivors_today", 0))
    c4.metric("Spend today", f"${status.get('spend_today_usd', 0):.3f}")

    cur_strat = status.get("current_strategy") or "—"
    cur_sess  = status.get("current_session")  or "—"
    cur_pair  = status.get("current_pair")     or "—"
    activity  = status.get("activity")

    # Elapsed since last status change (proxy for "how long has this been
    # doing X"). status['ts'] gets rewritten on every update_status call,
    # so this is the dwell time in the current activity.
    elapsed_str = ""
    try:
        ts_iso = status.get("ts")
        if ts_iso:
            from datetime import datetime as _dt
            elapsed = (_dt.utcnow() - _dt.fromisoformat(ts_iso.replace("Z","")
                                                          .replace("+00:00",""))
                       ).total_seconds()
            if elapsed >= 0 and elapsed < 86400:
                elapsed_str = f" · {elapsed:.0f}s"
    except Exception:
        pass

    if activity:
        st.info(f"▶ **{activity}**{elapsed_str}")
    else:
        st.info(f"⏸  Idle  ·  phase={phase}{elapsed_str}")
    st.caption(f"strategy={cur_strat}  ·  session={cur_sess}  ·  pair={cur_pair}")

    last_err = status.get("last_error")
    if last_err:
        st.error(f"Last error: {last_err}")

    st.divider()

    # ── Live / Last / Idle equity curve ──────────────────────────────────────
    stream = _read_live_trades_stream()
    live_active = (stream["header"] is not None
                   and stream["done"] is None
                   and not stream["stale"])

    if live_active:
        # ── A strategy is being pre-screened RIGHT NOW ───────────────────────
        hdr = stream["header"]
        trades = stream["trades"]
        st.subheader(f"🔴  LIVE backtest  ·  {hdr.get('strategy', '?')}")
        st.caption(f"pair={hdr.get('pair','?')}  ·  session={hdr.get('session','?')}"
                   f"  ·  stage={hdr.get('stage','?')}  ·  started "
                   f"{hdr.get('started_at','')[:19]}Z")

        if trades:
            pnl_series = [
                sum(float(t.get("pnl", 0.0)) for t in trades[:i + 1])
                for i in range(len(trades))
            ]
            cur_pnl = pnl_series[-1] if pnl_series else 0.0
            ca, cb, cc = st.columns(3)
            ca.metric("Trades so far", len(trades))
            cb.metric("PnL", f"${cur_pnl:+,.2f}")
            cc.metric("Last trade", f"${float(trades[-1].get('pnl',0)):+,.2f}")
            st.line_chart(
                pd.DataFrame({"PnL ($ from start)": pnl_series}),
                height=280,
            )
        else:
            st.caption("Waiting for the first trade to close…")
    elif stream["header"] is not None and stream["done"] is not None:
        # ── Last strategy that ran the live stream — show the final curve ───
        hdr = stream["header"]
        d = stream["done"]
        trades = stream["trades"]
        passed_emoji = "✅" if d.get("passed") else "❌"
        st.subheader(f"{passed_emoji}  Last live backtest  ·  {hdr.get('strategy','?')}")
        st.caption(
            f"pair={hdr.get('pair','?')}  ·  stage={hdr.get('stage','?')}  ·  "
            f"result: {d.get('reason','')[:120]}"
        )
        if trades:
            pnl_series = [
                sum(float(t.get("pnl", 0.0)) for t in trades[:i + 1])
                for i in range(len(trades))
            ]
            st.line_chart(
                pd.DataFrame({"PnL ($ from start)": pnl_series}),
                height=260,
            )
            st.caption(
                f"{len(trades)} trades · "
                f"final PnL **${pnl_series[-1]:+,.2f}**"
            )
    else:
        # ── No live backtest — explain what the agent IS doing ──────────────
        st.subheader("⚪  No strategy currently in backtest")
        # Show the most recent completed sweep's curve as the "last result" view
        latest_sweep_id = _latest_completed_sweep_id()
        if latest_sweep_id:
            st.caption(f"Most recent completed sweep: `{latest_sweep_id}`")
            eq_df = _equity_curve_for_sweep(latest_sweep_id)
            if not eq_df.empty and "balance" in eq_df.columns:
                try:
                    initial = float(eq_df["balance"].iloc[0]) - float(eq_df["pnl"].iloc[0])
                except Exception:
                    initial = float(eq_df["balance"].iloc[0])
                pnl_curve = eq_df["balance"] - initial
                final_pnl = float(pnl_curve.iloc[-1])
                st.line_chart(
                    pd.DataFrame({"PnL ($ from start)": pnl_curve.values}),
                    height=260,
                )
                st.caption(
                    f"{len(eq_df)} trades · final PnL **${final_pnl:+,.2f}**"
                )
        # Always show the explanation block so user knows WHY no curve
        st.markdown(
            f"**Currently:** `{activity or '(no activity reported)'}` ·  "
            f"phase=`{phase}`{elapsed_str}\n\n"
            f"The live equity curve appears here when the agent enters "
            f"**pre-screen** for a new strategy. Outside that window the "
            f"agent is one of: draining a queue (zero cost), calling Claude "
            f"to generate hypotheses (visible in **Spend today**), validating "
            f"mechanics against historical data (no trades), or compiling "
            f"the next entry function. The activity banner above tells you "
            f"which one."
        )

    st.divider()

    # ── Last 5 outcomes (rejections + survivors) ─────────────────────────────
    st.subheader("📋  Last 5 strategy outcomes")
    recent = _recent_outcomes(limit=5)
    if recent.empty:
        st.caption("No completed outcomes yet.")
    else:
        st.dataframe(
            recent[["strategy_name", "session", "verdict",
                    "test_sharpe", "n_trades", "created_at"]],
            use_container_width=True, hide_index=True, height=220,
        )

    st.divider()

    # Live log tail
    st.subheader("📜  Recent activity (last 60 log lines)")
    log_lines = _read_log_tail(60)
    if log_lines:
        st.code("\n".join(log_lines), language="text")
    else:
        st.caption("Log empty — agent hasn't started yet.")

    # Auto-refresh while active — 1.5s for near-live "constant updates" feel.
    # The status JSON is atomically replaced by the agent on every step, so
    # the activity line lags the agent by at most one refresh tick.
    if active and _AUTOREFRESH:
        st_autorefresh(interval=1500, key="run_refresh")


# ── Tab 2: Results ──────────────────────────────────────────────────────────

with tab_results:
    df = _latest_tested_strategies(limit=300)
    if df.empty:
        st.info("No strategies tested yet. Hit GO on the Run tab to start.")
    else:
        # Filter / sort controls
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            verdict_filter = st.multiselect(
                "Verdict filter",
                options=sorted(df["verdict"].dropna().unique().tolist()) or ["survivor", "rejected", "pending"],
                default=[],
                help="Empty = show all",
            )
        with c2:
            sort_col = st.selectbox(
                "Sort by",
                ["test_sharpe", "dsr", "n_trades", "created_at"],
                index=0,
            )
        with c3:
            sort_desc = st.checkbox("Descending", value=True)

        view = df.copy()
        if verdict_filter:
            view = view[view["verdict"].isin(verdict_filter)]
        try:
            view = view.sort_values(sort_col, ascending=not sort_desc,
                                     na_position="last")
        except Exception:
            pass

        st.caption(f"{len(view)} of {len(df)} strategies shown.")

        display_cols = ["strategy_name", "session", "test_sharpe", "dsr",
                        "n_trades", "max_dd", "verdict", "created_at",
                        "best_params_json"]
        st.dataframe(
            view[display_cols],
            use_container_width=True, height=400,
            hide_index=True,
        )

        st.divider()

        # Per-strategy equity curve viewer
        st.subheader("Inspect a strategy")
        names = view["strategy_name"].tolist()
        chosen = st.selectbox("Strategy", names) if names else None
        if chosen:
            row = view[view["strategy_name"] == chosen].iloc[0]

            def _num(key, default=0.0):
                v = row.get(key)
                try:
                    if v is None or pd.isna(v):
                        return default
                    return float(v)
                except (TypeError, ValueError):
                    return default

            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Test Sharpe", f"{_num('test_sharpe'):+.3f}")
            cc2.metric("DSR", f"{_num('dsr'):+.3f}")
            cc3.metric("Trades", int(_num("n_trades")))
            cc4.metric("Verdict", str(row.get("verdict") or "—"))

            params_json = row.get("best_params_json") or ""
            if params_json:
                st.caption(f"Best params: `{params_json}`")
            else:
                st.caption("Best params not recorded for this strategy.")

            sweep_id = row.get("sweep_id")
            if sweep_id:
                eq_df = _equity_curve_for_sweep(sweep_id)
                if not eq_df.empty and "balance" in eq_df.columns:
                    try:
                        initial = float(eq_df["balance"].iloc[0]) - float(eq_df["pnl"].iloc[0])
                    except Exception:
                        initial = float(eq_df["balance"].iloc[0])
                    pnl_curve = eq_df["balance"] - initial
                    chart_df = pd.DataFrame({"PnL ($ from start)": pnl_curve.values})
                    st.line_chart(chart_df, height=320)
                    fp = float(pnl_curve.iloc[-1])
                    st.caption(
                        f"final PnL **${fp:+,.2f}** "
                        f"({(fp/initial*100 if initial else 0):+.2f}%) · "
                        f"{len(eq_df)} trades"
                    )
                else:
                    st.caption("No trade-level equity data for this sweep.")
            else:
                st.caption("No sweep_id linked.")


# ── Tab 3: Live Backtest (in-process, per-trade live updates) ───────────────

with tab_live:
    st.header("📉  Live Backtest")
    st.caption(
        "Inspection tool: pick ONE strategy + ONE pair + ONE param combo and "
        "watch the PnL curve update as each trade closes. Single-threaded, "
        "~10-15s per run. Use this to spot pathological behaviour (all trades "
        "same side, huge drawdown spikes) that aggregated stats hide. "
        "**This runs separately from the autonomous agent** — your running "
        "agent keeps generating in the background."
    )

    # Lazy-import to keep top-of-file imports tight
    try:
        import edge_engine as eng
        from edge_hypotheses import SWEEPS as _SWEEPS
        from agent.session_router import _LIVE_SCHEDULE as _SCHED
        from agent.config import COST_MULT as _LIVE_COST_MULT
        _LIVE_OK = True
    except Exception as _e:
        _LIVE_OK = False
        st.error(f"Live Backtest imports failed: {_e}")

    if _LIVE_OK:
        # Make sure data is loaded (use the agent's cache if available)
        try:
            from agent.loop import _cache as _agent_cache, _load_data as _agent_load_data
            if not _agent_cache:
                with st.spinner("Loading market data (one-time)..."):
                    _agent_load_data()
            _train_dfs = _agent_cache.get("train_dfs", {})
            _test_dfs  = _agent_cache.get("test_dfs",  {})
        except Exception as _e:
            _train_dfs, _test_dfs = {}, {}
            st.warning(f"Couldn't reuse agent cache: {_e}. Live backtest disabled.")

        if _train_dfs:
            # Session state
            if "live_bt" not in st.session_state:
                st.session_state.live_bt = {
                    "queue":   queue.Queue(),
                    "thread":  None,
                    "trades":  [],   # list of {pnl, exit_ts, ...}
                    "balance": eng.INITIAL_BALANCE,
                    "progress": 0.0,
                    "started_at": None,
                    "done": False,
                    "error": None,
                }
            s = st.session_state.live_bt

            # Inputs
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                sweep_key = st.selectbox("Strategy", list(_SWEEPS.keys()),
                                          key="live_bt_sweep")
            with c2:
                avail_pairs = list(_train_dfs.keys())
                chosen_pair = st.selectbox("Pair", avail_pairs, key="live_bt_pair")
            with c3:
                split_choice = st.selectbox("Split", ["train", "test"], key="live_bt_split")

            sweep = _SWEEPS.get(sweep_key)
            param_combos = list(sweep["grid"]) if sweep else []
            if param_combos:
                combo_idx = st.selectbox(
                    f"Param combo (1 of {len(param_combos)})",
                    list(range(len(param_combos))),
                    format_func=lambda i: str(param_combos[i]),
                    key="live_bt_combo",
                )
            else:
                combo_idx = 0

            thread_alive = s["thread"] is not None and s["thread"].is_alive()

            # Run button
            if st.button("▶  Run live backtest",
                         disabled=thread_alive or sweep is None,
                         type="primary"):
                s["queue"]   = queue.Queue()
                s["trades"]  = []
                s["balance"] = eng.INITIAL_BALANCE
                s["progress"] = 0.0
                s["started_at"] = time.time()
                s["done"] = False
                s["error"] = None

                params = param_combos[combo_idx]
                slot_class = f"livebt_{sweep_key[:14]}".replace("-", "_").lower()
                registry = [{
                    "id":               f"livebt_{sweep_key}",
                    "family":           "session_based",
                    "slot_class":       slot_class,
                    "pairs":            [chosen_pair],
                    "session":          sweep["session"],
                    "allow_concurrent": False,
                    "regime_mult":      sweep["regime_mult"],
                    "params":           params,
                }]
                slot_managers = {slot_class: sweep["manager_fn"]}
                slot_entries  = {slot_class: sweep["entry_fn"]}
                data_dict = _train_dfs if split_choice == "train" else _test_dfs
                subset = {chosen_pair: data_dict[chosen_pair]}

                # Map session → hours
                session_hours = next(
                    ((lo, hi - 1) for lo, hi, n, *_ in _SCHED if n == sweep["session"]),
                    None,
                )

                q = s["queue"]
                def _cb(ev):
                    try: q.put_nowait(ev)
                    except Exception: pass

                def _worker():
                    try:
                        eng.run_backtest(
                            subset_dfs        = subset,
                            spread_override   = None,
                            slippage_override = None,
                            registry          = registry,
                            slot_managers     = slot_managers,
                            slot_entries      = slot_entries,
                            cost_mult         = _LIVE_COST_MULT,
                            session_hours     = session_hours,
                            progress_callback = _cb,
                        )
                    except Exception as ex:
                        q.put({"type": "error", "error": f"{type(ex).__name__}: {ex}"})
                    finally:
                        q.put({"type": "done"})

                t = threading.Thread(target=_worker, daemon=True)
                t.start()
                s["thread"] = t

            # Drain queue
            while True:
                try:
                    ev = s["queue"].get_nowait()
                except queue.Empty:
                    break
                etype = ev.get("type")
                if etype == "trade_closed":
                    s["balance"] = ev["balance"]
                    s["trades"].append(ev["trade"])
                elif etype == "tick":
                    total = max(int(ev.get("total_bars", 1)), 1)
                    s["progress"] = ev.get("bar_idx", 0) / total
                elif etype == "error":
                    s["error"] = ev.get("error", "unknown")
                elif etype == "done":
                    s["done"] = True
                    s["progress"] = 1.0

            # Render
            ca, cb, cc, cd = st.columns(4)
            initial = eng.INITIAL_BALANCE
            pnl_now = (s["balance"] or initial) - initial
            ca.metric("Trades", len(s["trades"]))
            cb.metric("PnL", f"${pnl_now:+,.2f}",
                      f"{(pnl_now/initial*100 if initial else 0):+.2f}%")
            cc.metric("Progress", f"{s['progress']*100:.0f}%")
            cd.metric("Status",
                      "Running" if thread_alive else ("Done" if s["done"] else "Idle"))

            st.progress(min(max(s["progress"], 0.0), 1.0))

            if s["error"]:
                st.error(f"Backtest error: {s['error']}")

            # Live PnL curve — start at 0, auto-scales to actual range
            if s["trades"]:
                pnl_series = [
                    sum(float(t.get("pnl", 0.0)) for t in s["trades"][:i + 1])
                    for i in range(len(s["trades"]))
                ]
                chart_df = pd.DataFrame({"PnL ($ from start)": pnl_series})
                st.line_chart(chart_df, height=320)
                # Latest 15 trades table
                st.subheader(f"Latest {min(15, len(s['trades']))} trades")
                td = pd.DataFrame(s["trades"][-15:])
                if "pnl" in td.columns:
                    td["pnl"] = td["pnl"].round(2)
                st.dataframe(td, use_container_width=True, height=300)
            else:
                st.caption(
                    "PnL curve will appear once the first trade closes. "
                    "Strategies with strict entry conditions may take several seconds "
                    "before the first fill."
                )

            if s["started_at"] and thread_alive:
                elapsed = time.time() - s["started_at"]
                st.caption(f"Running for {elapsed:.0f}s — {len(s['trades'])} trades so far")
            elif s["done"] and s["started_at"]:
                total_t = time.time() - s["started_at"]
                st.success(
                    f"Complete in {total_t:.1f}s — "
                    f"{len(s['trades'])} trades, "
                    f"PnL ${pnl_now:+,.2f} "
                    f"({(pnl_now/initial*100 if initial else 0):+.2f}%)"
                )

            # Auto-refresh while running
            if thread_alive and _AUTOREFRESH:
                st_autorefresh(interval=500, key="live_bt_refresh")
            elif thread_alive and not _AUTOREFRESH:
                st.caption("Install `streamlit-autorefresh` for auto-updating curves.")
        else:
            st.warning("No train data loaded yet — start the agent on the Run tab first.")
