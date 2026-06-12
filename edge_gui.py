# -*- coding: utf-8 -*-
"""
Edge Discovery GUI — Streamlit Dashboard

Run:
    streamlit run edge_gui.py

Tabs:
    1. Run         — choose sweep, workers, cost_mult, launch
    2. Results     — sortable hypothesis table with BH significance
    3. Detail      — equity curve + stats for selected hypothesis
    4. Robustness  — walk-forward, regime breakdown, Monte Carlo
    5. Mine        — automated pattern mining + self-improving loop
    6. Portfolio   — pairwise correlation between survivors
    7. TCA         — live transaction-cost analysis (live trades + executions)
    8. Agent       — start/stop the autonomous Claude loop, view recent rounds
"""

import importlib
import os
import subprocess
import sys
import time
import threading
import queue
import json
import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_AVAILABLE = True
except ImportError:
    _AUTOREFRESH_AVAILABLE = False

# ── Path setup so edge_engine resolves from Downloads ────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import edge_engine as eng

try:
    import edge_miner as miner
    _MINER_AVAILABLE = True
except ImportError:
    _MINER_AVAILABLE = False

try:
    from agent import tca as agent_tca
    from agent import live_ingest as agent_live_ingest
    from agent.config import AGENT_DB_PATH, LOG_PATH, GENERATED_DIR
    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False
    AGENT_DB_PATH = _HERE / 'agent' / 'agent_results.db'
    LOG_PATH      = _HERE / 'agent' / 'agent.log'
    GENERATED_DIR = _HERE / 'agent' / 'generated'

try:
    from agent import gui_config, data_inventory
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

_REPORTS_DIR = _HERE / 'agent' / 'reports'

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Edge Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global look & feel ───────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tabs: bigger, more legible, clearer active indicator */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
      height: 44px; padding: 0 18px;
      background: rgba(255,255,255,0.03); border-radius: 8px 8px 0 0;
      font-size: 15px; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
      background: rgba(82, 139, 255, 0.18) !important;
      border-bottom: 2px solid #528bff;
  }
  /* Metric cards: subtle border so they read as cards */
  [data-testid="stMetric"] {
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 8px; padding: 10px 14px;
  }
  /* Section headers tighter */
  h2 { margin-top: 0.4rem !important; }
  h3 { margin-top: 1.2rem !important; }
  /* Status pills used in coverage table & elsewhere */
  .pill { padding: 2px 10px; border-radius: 12px; font-size: 12px;
          font-weight: 600; display: inline-block; }
  .pill-green  { background:#1a7f2e; color:white; }
  .pill-amber  { background:#a67c00; color:white; }
  .pill-grey   { background:#444;    color:#ddd; }
  .pill-red    { background:#7f1a1a; color:white; }
  /* Info callouts inside tabs */
  .helpbox {
      background: rgba(82,139,255,0.08); border-left: 3px solid #528bff;
      padding: 10px 14px; border-radius: 4px; margin-bottom: 14px;
      font-size: 14px; line-height: 1.5;
  }
</style>
""", unsafe_allow_html=True)


def _help_box(text: str) -> None:
    """Render a small explanation banner at the top of a tab/section."""
    st.markdown(f'<div class="helpbox">{text}</div>', unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

def _init_state():
    defaults = {
        'data_loaded':       False,
        'train_dfs':         {},
        'test_dfs':          {},
        'measured_spreads':  {},
        'selected_sweep_id': None,
        'selected_hyp_id':   None,
        'sweep_running':     False,
        'sweep_done':        0,
        'sweep_total':       0,
        'sweep_error':       None,
        'last_sweep_id':     None,
        # Single-run miner state
        'miner_running':       False,
        'miner_results':       None,
        'miner_error':         None,
        'miner_step':          0,
        'miner_step_label':    '',
        # Self-improving loop state
        'loop_running':        False,
        'loop_round':          0,
        'loop_total_rounds':   0,
        'loop_msg':            '',
        'loop_results':        None,
        'loop_error':          None,
        'loop_memory':         None,
        # Agent-loop subprocess state
        'agent_proc_pid':      None,
        'agent_started_at':    None,
        # TCA cache
        'tca_last_run':        None,
        'tca_last_path':       None,
        # Data download tracking
        'download_threads':    {},   # pair -> Thread
        # MT5 live trader subprocess state
        'mt5_proc_pid':        None,
        'mt5_started_at':      None,
        # On-demand miner-now subprocess
        'miner_now_running':   False,
        'miner_now_msg':       '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# =============================================================================
# HELPERS
# =============================================================================

def _load_hypotheses_module():
    """Import edge_hypotheses.py fresh every call so live edits are reflected."""
    mod_name = 'edge_hypotheses'
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / 'edge_hypotheses.py')
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[mod_name] = mod
    return mod


def _get_sweeps() -> dict:
    mod = _load_hypotheses_module()
    if mod is None or not hasattr(mod, 'SWEEPS'):
        return {}
    return mod.SWEEPS


def _color_verdict(val):
    colors = {'VIABLE': 'background-color:#1a7f2e;color:white',
              'MARGINAL': 'background-color:#a67c00;color:white',
              'NO EDGE': 'background-color:#7f1a1a;color:white',
              'pending': ''}
    return colors.get(val, '')


def _color_bh(val):
    return 'color:#00c853;font-weight:bold' if val == 1 else 'color:#9e9e9e'


def _fmt_params(params) -> str:
    if isinstance(params, dict):
        return '  '.join(f"{k}={v}" for k, v in params.items())
    return str(params)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🔍 Edge Discovery")
    st.caption("FX strategy research, mining & live trading control")
    st.markdown("---")

    # ── Quick start guide ───────────────────────────────────────────────────
    with st.expander("👋  New here? Quick start", expanded=False):
        st.markdown(
            "**1.** Open **⚙ Config** → pick the pairs and sessions you care about, "
            "click **Save**.\n\n"
            "**2.** Open **📥 Data** → check what's already downloaded; "
            "click **Download** for any missing pair.\n\n"
            "**3.** Click **Load / Refresh Data** below to load everything "
            "into memory (do this once per session).\n\n"
            "**4.** Either click **Run miner now** in the Config tab "
            "(one-shot) or **Start agent loop** (autonomous, runs forever).\n\n"
            "**5.** Promoted strategies show up in **💹 Live Strategies**. "
            "Start the **MT5 trader** from Config to actually trade them."
        )

    # ── Data loading ─────────────────────────────────────────────────────────
    st.subheader("Load data into memory")
    st.caption("Reads parquet caches into RAM so the Run / Mine / Robust tabs "
               "can use them. Downloads first if needed.")
    force_refresh = st.checkbox("Force re-download from Dukascopy",
                                value=False,
                                help="Skip the local cache and pull fresh tick data. "
                                     "First-time download is 10–30 min per pair.")

    if st.button("⬇  Load / Refresh Data", type="primary",
                 disabled=st.session_state.sweep_running,
                 use_container_width=True):
        with st.spinner("Downloading & preparing all pairs (may take 10–30 min first time)…"):
            try:
                train, test, spreads = eng.load_all_data(force_refresh=force_refresh)
                st.session_state.train_dfs        = train
                st.session_state.test_dfs         = test
                st.session_state.measured_spreads = spreads
                st.session_state.data_loaded      = True
                st.success("Data ready!")
            except Exception as exc:
                st.error(f"Data load failed: {exc}")

    if st.session_state.data_loaded:
        pairs = list(st.session_state.train_dfs.keys())
        st.success(f"✓ {len(pairs)} pair(s) in memory")
        with st.expander("Loaded pairs & measured spreads"):
            st.write(", ".join(pairs))
            spreads = st.session_state.measured_spreads
            if spreads:
                spd_df = pd.DataFrame([
                    {'Pair': eng.PAIR_LABELS.get(p, p),
                     'Med. spread': f"{v:.6f}",
                     'Pips': f"{v / eng.PAIR_PIP_SIZE.get(p, 0.0001):.2f}"}
                    for p, v in spreads.items()
                ])
                st.dataframe(spd_df, hide_index=True, use_container_width=True)
    else:
        st.info("⚠ No data loaded yet. Click the button above.")

    st.markdown("---")

    # ── Sweep selector ───────────────────────────────────────────────────────
    st.subheader("Results browser")
    sweep_list = eng.load_sweep_list()
    if sweep_list:
        sweep_labels = [f"{s['sweep_name']}  [{s['created_at'][:16]}]" for s in sweep_list]
        sweep_idx    = st.selectbox("Select sweep", range(len(sweep_list)),
                                    format_func=lambda i: sweep_labels[i])
        st.session_state.selected_sweep_id = sweep_list[sweep_idx]['sweep_id']
    else:
        st.caption("No sweeps in DB yet. Run one in the Run tab.")

# =============================================================================
# TABS
# =============================================================================

(tab_config, tab_data, tab_run, tab_results, tab_detail, tab_robust, tab_mine,
 tab_portfolio, tab_tca, tab_live, tab_agent, tab_live_bt) = st.tabs(
    ["⚙  Config", "📥  Data",
     "▶  Run", "📋  Results", "📈  Detail", "🛡  Robustness", "🧬  Mine",
     "📊  Portfolio", "💸  TCA", "💹  Live Strategies", "🤖  Agent",
     "📉  Live Backtest"]
)


# ── Process control helpers (shared by Config + Agent + MT5) ────────────────

_RUNTIME_DIR  = _HERE / 'runtime'
_MT5_PID_FILE = _RUNTIME_DIR / 'mt5live.pid'
_MT5_LOG_FILE = _HERE / 'live_trader.log'


def _proc_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _kill_proc(pid: int):
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                           capture_output=True, text=True, timeout=10)
        else:
            os.kill(pid, 15)
    except Exception:
        pass


def _read_mt5_pid() -> int | None:
    if not _MT5_PID_FILE.exists():
        return None
    try:
        return int(_MT5_PID_FILE.read_text().strip())
    except Exception:
        return None


def _write_mt5_pid(pid: int) -> None:
    _RUNTIME_DIR.mkdir(exist_ok=True)
    _MT5_PID_FILE.write_text(str(pid))


def _clear_mt5_pid() -> None:
    try:
        _MT5_PID_FILE.unlink()
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TAB — CONFIG (system-wide control panel)
# ─────────────────────────────────────────────────────────────────────────────
with tab_config:
    st.header("⚙ Configuration & Process Control")
    _help_box(
        "<b>This is the system's control room.</b> Pick which currency pairs and "
        "trading sessions you want the engine to focus on, then save. Your picks "
        "are remembered across restarts and used by both the on-demand miner "
        "(button below) and the autonomous agent loop on its next round. "
        "Leave a field empty to fall back to the built-in UTC-clock defaults."
    )

    if not _CONFIG_AVAILABLE:
        st.error("agent.gui_config / agent.data_inventory not importable. "
                 "Make sure the agent/ folder is on PYTHONPATH.")
    else:
        cur = gui_config.load()
        cur_pairs    = cur.get('selected_pairs')    or []
        cur_sessions = cur.get('selected_sessions') or []
        cur_overrides = cur.get('miner_overrides')  or {}

        # ── Selection: pairs + sessions ─────────────────────────────────────
        st.subheader("1. What to trade")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Currency pairs**")
            st.caption("Pick any subset of the 13 supported pairs. "
                       "Each one needs raw data first — see the **📥 Data** tab.")
            sel_pairs = st.multiselect(
                "Pairs",
                options=list(eng.ALL_PAIRS),
                default=[p for p in cur_pairs if p in eng.ALL_PAIRS],
                label_visibility='collapsed',
                placeholder='Choose pairs…  (empty = use the session default)',
            )
        with cc2:
            st.markdown("**Trading sessions**")
            st.caption("`ny`, `london`, `asian` follow the live clock. "
                       "Sub-corners are research themes the agent rotates through.")
            try:
                from agent.session_router import SUB_CORNERS
                sub_names = [s[0] for s in SUB_CORNERS]
            except Exception:
                sub_names = []
            session_options = ['ny', 'london', 'asian'] + sub_names
            sel_sessions = st.multiselect(
                "Sessions",
                options=session_options,
                default=[s for s in cur_sessions if s in session_options],
                label_visibility='collapsed',
                placeholder='Choose sessions…  (empty = follow UTC clock)',
            )

        # ── Miner knobs ─────────────────────────────────────────────────────
        st.subheader("2. Miner settings")
        st.caption(
            "These shape how the pattern miner labels trades during its "
            "simulation. Defaults are sensible — only change them if you "
            "know what you're doing."
        )
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            ov_tp = st.number_input(
                "Take-profit (R)",
                value=float(cur_overrides.get('tp_r', 2.0)),
                min_value=0.5, max_value=5.0, step=0.5,
                help="Profit target as a multiple of risk. 2.0 = aim for 2× the stop distance.",
            )
        with oc2:
            ov_sl = st.number_input(
                "Stop-loss (R)",
                value=float(cur_overrides.get('sl_r', 1.0)),
                min_value=0.25, max_value=3.0, step=0.25,
                help="Stop distance as a multiple of the OR range. 1.0 = standard.",
            )
        with oc3:
            ov_h  = st.number_input(
                "Forward horizon (bars)",
                value=int(cur_overrides.get('horizon_bars', 20)),
                min_value=5, max_value=100, step=1,
                help="How many minutes after entry to scan for TP/SL hit.",
            )
        with oc4:
            ov_n  = st.number_input(
                "Top N patterns to keep",
                value=int(cur_overrides.get('top_n_patterns', 5)),
                min_value=1, max_value=20, step=1,
                help="The miner emits this many candidate hypotheses per run.",
            )

        # ── Save / reset row ────────────────────────────────────────────────
        sb1, sb2, sb_info = st.columns([1, 1, 4])
        with sb1:
            if st.button("💾  Save settings", type="primary",
                         use_container_width=True):
                gui_config.save({
                    'selected_pairs':    sel_pairs,
                    'selected_sessions': sel_sessions,
                    'miner_overrides':   {
                        'tp_r':           ov_tp,
                        'sl_r':           ov_sl,
                        'horizon_bars':   ov_h,
                        'top_n_patterns': ov_n,
                    },
                })
                st.success("Saved.")
                st.rerun()
        with sb2:
            if st.button("🗑  Clear & use defaults",
                         use_container_width=True):
                gui_config.reset()
                st.success("Cleared.")
                st.rerun()
        with sb_info:
            saved_at = cur.get('updated_at')
            if saved_at:
                st.caption(f"Last saved: {saved_at[:19]} UTC")
            else:
                st.caption("No config saved yet — using built-in defaults.")

        # ── Effective config preview ────────────────────────────────────────
        st.markdown("---")
        st.subheader("3. What the engine will do next")
        st.caption(
            "Live preview of the session, pair list, and exit hour the engine "
            "will use on its very next round, given your saved config."
        )
        try:
            from agent.session_router import get_current_session, session_display_name
            name, pairs, exit_h = get_current_session(0)
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Session",  session_display_name(name))
            ec2.metric("Pair list", ", ".join(pairs))
            ec3.metric("Exit hour", f"{exit_h:02d}:00 UTC")
        except Exception as exc:
            st.warning(f"Effective-config preview failed: {exc}")

        # ── Process control: agent loop + MT5 + on-demand miner ─────────────
        st.markdown("---")
        st.subheader("4. Start / stop processes")
        _help_box(
            "Three independent background processes you can run from this GUI:<br>"
            "&nbsp;&nbsp;• <b>Agent loop</b> — keeps generating &amp; backtesting "
            "new strategy ideas forever.<br>"
            "&nbsp;&nbsp;• <b>MT5 live trader</b> — connects to MetaTrader 5 and "
            "places real orders for promoted strategies.<br>"
            "&nbsp;&nbsp;• <b>Run miner now</b> — single one-shot mining run "
            "against your config selection."
        )

        # MT5 PID may have been written by a previous session
        if not st.session_state.mt5_proc_pid:
            file_pid = _read_mt5_pid()
            if file_pid and _proc_alive(file_pid):
                st.session_state.mt5_proc_pid = file_pid

        agent_running = _proc_alive(st.session_state.agent_proc_pid)
        mt5_running   = _proc_alive(st.session_state.mt5_proc_pid)

        pc1, pc2, pc3 = st.columns(3)

        # --- Agent loop ---
        with pc1:
            st.markdown("#### 🤖 Agent loop")
            pill = ('<span class="pill pill-green">RUNNING</span>'
                    if agent_running else
                    '<span class="pill pill-grey">STOPPED</span>')
            st.markdown(
                f"{pill} &nbsp; PID: `{st.session_state.agent_proc_pid or '—'}`",
                unsafe_allow_html=True,
            )
            st.caption("Autonomous Claude-driven hypothesis generator. "
                       "Runs forever until you stop it (capped by BUDGET_TOTAL_USD).")
            if not agent_running:
                if st.button("▶  Start agent loop", key="cfg_start_agent",
                             type="primary", use_container_width=True):
                    cmd = [sys.executable, '-m', 'agent.main']
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                    try:
                        proc = subprocess.Popen(
                            cmd, cwd=str(_HERE),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, creationflags=creationflags,
                        )
                        st.session_state.agent_proc_pid   = proc.pid
                        st.session_state.agent_started_at = pd.Timestamp.utcnow().strftime('%H:%M UTC')
                        st.success(f"Started PID {proc.pid}")
                        time.sleep(1); st.rerun()
                    except Exception as exc:
                        st.error(f"Start failed: {exc}")
            else:
                if st.button("⏹  Stop agent loop", key="cfg_stop_agent",
                             use_container_width=True):
                    _kill_proc(st.session_state.agent_proc_pid)
                    st.session_state.agent_proc_pid   = None
                    st.session_state.agent_started_at = None
                    st.success("Stopped.")
                    time.sleep(1); st.rerun()

        # --- MT5 live trader ---
        with pc2:
            st.markdown("#### 💹 MT5 live trader")
            pill = ('<span class="pill pill-green">RUNNING</span>'
                    if mt5_running else
                    '<span class="pill pill-grey">STOPPED</span>')
            st.markdown(
                f"{pill} &nbsp; PID: `{st.session_state.mt5_proc_pid or '—'}`",
                unsafe_allow_html=True,
            )
            st.caption("Connects to MetaTrader 5 and places real orders for "
                       "promoted strategies. Make sure MT5 is logged in first.")
            if not mt5_running:
                if st.button("▶  Start MT5 trader", key="cfg_start_mt5",
                             type="primary", use_container_width=True):
                    mt5_path = _HERE / 'MT5Live.2.py'
                    if not mt5_path.exists():
                        st.error(f"Not found: {mt5_path}")
                    else:
                        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                        try:
                            log_fp = open(_MT5_LOG_FILE, 'a', buffering=1)
                            proc = subprocess.Popen(
                                [sys.executable, str(mt5_path)],
                                cwd=str(_HERE),
                                stdout=log_fp, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                creationflags=creationflags,
                            )
                            _write_mt5_pid(proc.pid)
                            st.session_state.mt5_proc_pid   = proc.pid
                            st.session_state.mt5_started_at = pd.Timestamp.utcnow().strftime('%H:%M UTC')
                            st.success(f"Started PID {proc.pid}")
                            time.sleep(1); st.rerun()
                        except Exception as exc:
                            st.error(f"Start failed: {exc}")
            else:
                if st.button("⏹  Stop MT5 trader", key="cfg_stop_mt5",
                             use_container_width=True):
                    _kill_proc(st.session_state.mt5_proc_pid)
                    _clear_mt5_pid()
                    st.session_state.mt5_proc_pid   = None
                    st.session_state.mt5_started_at = None
                    st.success("Stopped.")
                    time.sleep(1); st.rerun()

        # --- On-demand miner ---
        with pc3:
            st.markdown("#### 🧬 Run miner now")
            running_pill = ('<span class="pill pill-amber">WORKING</span>'
                            if st.session_state.miner_now_running else
                            '<span class="pill pill-grey">IDLE</span>')
            st.markdown(running_pill, unsafe_allow_html=True)
            st.caption("Single mining run against your saved pairs/sessions. "
                       "Adds promising patterns to `edge_hypotheses.py`.")
            if not st.session_state.data_loaded:
                st.warning("⚠ Load data from the sidebar first.")
            elif st.session_state.miner_now_running:
                st.info(st.session_state.miner_now_msg or "Running…")
            else:
                if st.button("🧬  Run miner now", key="cfg_run_miner",
                             type="primary", use_container_width=True,
                             disabled=not _MINER_AVAILABLE):
                    pairs_filter = sel_pairs or list(st.session_state.train_dfs.keys())
                    sess_filter  = (sel_sessions[0] if sel_sessions else None)
                    train = {p: df for p, df in st.session_state.train_dfs.items()
                             if p in pairs_filter}
                    test  = {p: df for p, df in st.session_state.test_dfs.items()
                             if p in pairs_filter}
                    if not train:
                        st.error("No matching pairs in loaded data.")
                    else:
                        st.session_state.miner_now_running = True
                        st.session_state.miner_now_msg     = f"Mining {len(train)} pairs…"

                        def _do_miner_now(_train, _test, _sess):
                            try:
                                miner.run_miner(
                                    pair_dfs       = _train,
                                    test_dfs       = _test,
                                    session_filter = _sess,
                                    tp_r           = ov_tp,
                                    sl_r           = ov_sl,
                                    horizon_bars   = int(ov_h),
                                    top_n_patterns = int(ov_n),
                                    append_to_file = True,
                                )
                                st.session_state.miner_now_msg = "✓ Done — check Results tab."
                            except Exception as exc:
                                st.session_state.miner_now_msg = f"Failed: {exc}"
                            finally:
                                st.session_state.miner_now_running = False

                        threading.Thread(
                            target=_do_miner_now,
                            args=(train, test, sess_filter),
                            daemon=True,
                        ).start()
                        st.rerun()

        # MT5 log tail
        if _MT5_LOG_FILE.exists():
            with st.expander("MT5 live log (last 30 lines)", expanded=False):
                try:
                    lines = _MT5_LOG_FILE.read_text(encoding='utf-8',
                                                    errors='replace').splitlines()
                    st.code("\n".join(lines[-30:]), language='text')
                except Exception as exc:
                    st.caption(f"(log read failed: {exc})")


# ─────────────────────────────────────────────────────────────────────────────
# TAB — DATA (inventory + per-pair download)
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.header("📥 Historical Data")
    _help_box(
        "<b>What's on disk?</b> Each row below shows whether a pair has been "
        "downloaded from Dukascopy, the date range covered, and how stale it is. "
        "<br><br>"
        "<b>Status legend:</b> "
        "<span class='pill pill-green'>LOADED</span> ready to mine &amp; trade — "
        "<span class='pill pill-amber'>RAW ONLY</span> ticks downloaded but not yet "
        "prepared into 1-minute bars — "
        "<span class='pill pill-grey'>NONE</span> nothing on disk yet, click "
        "<b>Download</b> below."
    )

    if not _CONFIG_AVAILABLE:
        st.error("agent.data_inventory not importable.")
    else:
        cov = data_inventory.coverage()

        # ── Top-of-tab summary metrics ──────────────────────────────────────
        n_loaded    = sum(1 for v in cov.values() if v['status'] == 'loaded')
        n_raw       = sum(1 for v in cov.values() if v['status'] == 'raw_only')
        n_missing   = sum(1 for v in cov.values() if v['status'] == 'not_downloaded')
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Pairs supported", len(cov))
        sm2.metric("Loaded",          n_loaded)
        sm3.metric("Raw only",        n_raw)
        sm4.metric("Not downloaded",  n_missing)

        # ── Coverage table ──────────────────────────────────────────────────
        st.subheader("Coverage by pair")
        rc1, rc2 = st.columns([1, 8])
        with rc1:
            if st.button("🔄  Refresh", key="data_refresh",
                         use_container_width=True):
                st.rerun()
        with rc2:
            st.caption("Re-scans `duka_cache/` and `edge_prepared_cache/` on click. "
                       "Auto-refreshes while a download is running.")

        rows = []
        for pair, info in cov.items():
            badge = {'loaded':         '🟢 loaded',
                     'raw_only':       '🟡 raw only',
                     'not_downloaded': '⚫ none'}.get(info['status'], info['status'])
            rows.append({
                'Pair':    pair,
                'Status':  badge,
                'First':   str(info['first']) if info['first'] else '—',
                'Last':    str(info['last'])  if info['last']  else '—',
                'Days':    info['n_days'],
                'Age (d)': info['age_days'] if info['age_days'] is not None else '—',
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)

        # ── Live download status ────────────────────────────────────────────
        status = data_inventory.read_status()
        if status:
            st.markdown("---")
            st.subheader("Download progress")
            phase     = status.get('phase', '?')
            pair_in_p = status.get('pair', '?')
            done      = status.get('days_done', 0)
            total     = status.get('days_total', 0) or 0
            if phase == 'error':
                st.error(f"❌ {pair_in_p}: {status.get('error', 'unknown error')}")
            elif phase == 'done':
                st.success(f"✓ {pair_in_p} finished — "
                           f"{status.get('n_bars', 0):,} 1-min bars saved.")
            else:
                pct_str = f"{int(done/total*100)}%" if total else "…"
                st.info(f"⬇  **{pair_in_p}** — {phase} — "
                        f"day {done}/{total} ({pct_str})")
                if total:
                    st.progress(min(done / total, 1.0))

        # ── Per-pair download trigger ───────────────────────────────────────
        st.markdown("---")
        st.subheader("Download a pair")
        st.caption(
            "Pulls every available day of tick data from Dukascopy, then prepares "
            "1-minute OHLC bars. Roughly 5–15 minutes per year of history per pair. "
            "Runs in the background — you can keep navigating the GUI."
        )
        dl1, dl2 = st.columns([2, 3])
        with dl1:
            pick = st.selectbox("Pair to download",
                                options=list(eng.ALL_PAIRS),
                                key="data_pick_pair",
                                help="Picks one pair to fetch right now.")
        with dl2:
            st.write("")  # vertical alignment
            st.write("")
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("📥  Download this pair", type="primary",
                             use_container_width=True):
                    t = st.session_state.download_threads.get(pick)
                    if t and t.is_alive():
                        st.warning(f"{pick} already downloading.")
                    else:
                        t = data_inventory.download_pair_async(pick)
                        st.session_state.download_threads[pick] = t
                        st.success(f"Started download for {pick}")
                        time.sleep(0.5); st.rerun()
            with bc2:
                if st.button("📥  Download all from Config",
                             use_container_width=True,
                             help="Downloads any pair from your Config selection "
                                  "that doesn't already have a raw cache."):
                    cur_sel = (gui_config.selected_pairs() or []) if _CONFIG_AVAILABLE else []
                    missing = data_inventory.missing_pairs(cur_sel) if cur_sel else []
                    if not cur_sel:
                        st.warning("No pairs selected in the ⚙ Config tab yet.")
                    elif not missing:
                        st.info("All selected pairs already have raw cache.")
                    else:
                        for p in missing:
                            t = st.session_state.download_threads.get(p)
                            if not (t and t.is_alive()):
                                st.session_state.download_threads[p] = (
                                    data_inventory.download_pair_async(p)
                                )
                        st.success(f"Started: {', '.join(missing)}")
                        time.sleep(0.5); st.rerun()

        # auto-refresh while any download thread is alive
        any_alive = any(t.is_alive()
                        for t in st.session_state.download_threads.values())
        if any_alive:
            time.sleep(2)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RUN
# ─────────────────────────────────────────────────────────────────────────────
with tab_run:
    st.header("▶ Run a Parameter Sweep")
    _help_box(
        "Pick a <b>sweep</b> (a named family of strategies defined in "
        "<code>edge_hypotheses.py</code>) and the engine will backtest every "
        "parameter combination on your loaded data. Use this when you have a "
        "specific idea you want to grid-search; for open-ended discovery, use "
        "the <b>🧬 Mine</b> tab instead."
    )

    if not st.session_state.data_loaded:
        st.warning("⚠ Load data from the sidebar first.")
    else:
        sweeps = _get_sweeps()
        if not sweeps:
            st.error(
                "No SWEEPS found in `edge_hypotheses.py`. "
                "Create the file or add entries to its SWEEPS dict."
            )
        else:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                sweep_key = st.selectbox("Sweep", list(sweeps.keys()),
                                         help="Defined in edge_hypotheses.py")
            with col2:
                cost_mult = st.select_slider(
                    "Cost multiplier",
                    options=[0.0, 0.25, 0.5, 1.0, 1.5],
                    value=0.5,
                    help="0.0 = no costs (diagnostic only)  0.5 = realistic post-recal  1.0 = pre-recal default  1.5 = pessimistic",
                )
            with col3:
                n_workers = st.slider("Workers", 1, 8, 4,
                                      help="Parallel CPU processes")

            search_mode = st.radio(
                "Search mode",
                ["Grid (exhaustive)", "Bayesian (Optuna)"],
                horizontal=True,
                help="Bayesian uses Optuna TPE — learns good param regions, "
                     "~5× faster for large spaces.",
            )
            use_optuna = (search_mode == "Bayesian (Optuna)")

            sweep_cfg = sweeps[sweep_key]
            grid      = sweep_cfg['grid']

            if use_optuna:
                n_trials  = st.slider("Optuna trials", 20, 500, 100, 10)
                est_secs  = n_trials * 2 / max(n_workers, 1)
                st.caption(
                    f"Bayesian: **{n_trials}** trials   "
                    f"Pairs: {', '.join(sweep_cfg.get('pairs', []))}   "
                    f"Session: `{sweep_cfg.get('session', '?')}`   "
                    f"Est. time ≈ {est_secs/60:.1f} min"
                )
            else:
                n_trials  = None
                n_combos  = len(grid)
                est_secs  = n_combos * 2 / max(n_workers, 1)
                st.caption(
                    f"Grid: **{n_combos}** combinations   "
                    f"Pairs: {', '.join(sweep_cfg.get('pairs', []))}   "
                    f"Session: `{sweep_cfg.get('session', '?')}`   "
                    f"Est. time ≈ {est_secs/60:.1f} min"
                )

            run_btn = st.button(
                "🚀  Run Sweep",
                disabled=st.session_state.sweep_running,
                type="primary",
            )

            prog_bar  = st.progress(0)
            prog_text = st.empty()

            _sweep_total = n_trials if use_optuna else len(grid)

            if run_btn:
                st.session_state.sweep_running = True
                st.session_state.sweep_done    = 0
                st.session_state.sweep_total   = _sweep_total
                st.session_state.sweep_error   = None

            if st.session_state.sweep_running:
                prog_text.text(
                    f"Running… {st.session_state.sweep_done}/{st.session_state.sweep_total}"
                )
                prog_bar.progress(
                    st.session_state.sweep_done / max(st.session_state.sweep_total, 1)
                )

                def _progress(done, total):
                    st.session_state.sweep_done  = done
                    st.session_state.sweep_total = total

                def _do_sweep():
                    try:
                        common_kwargs = dict(
                            sweep_name        = sweep_key,
                            entry_fn          = sweep_cfg['entry_fn'],
                            manager_fn        = sweep_cfg['manager_fn'],
                            pairs             = sweep_cfg.get('pairs', eng.NY_PAIRS),
                            session           = sweep_cfg.get('session', 'ny'),
                            regime_mult       = sweep_cfg.get('regime_mult', {}),
                            train_dfs         = st.session_state.train_dfs,
                            test_dfs          = st.session_state.test_dfs,
                            cost_mult         = cost_mult,
                            n_workers         = n_workers,
                            progress_callback = _progress,
                        )
                        if use_optuna:
                            from edge_engine import OptunaGrid, run_sweep_optuna
                            # Wrap existing ParameterGrid ranges as OptunaGrid if needed
                            raw_grid = sweep_cfg['grid']
                            if isinstance(raw_grid, eng.ParameterGrid):
                                # Convert lists to categorical OptunaGrid
                                opt_grid = OptunaGrid(
                                    {k: v for k, v in raw_grid._ranges.items()},
                                    n_trials=n_trials,
                                )
                            else:
                                opt_grid = raw_grid
                                opt_grid.n_trials = n_trials
                            sid = run_sweep_optuna(grid=opt_grid, **common_kwargs)
                        else:
                            sid = eng.run_sweep(
                                grid   = sweep_cfg['grid'],
                                family = sweep_cfg.get('family', 'session_based'),
                                allow_concurrent = sweep_cfg.get('allow_concurrent', False),
                                measured_spreads = st.session_state.measured_spreads,
                                **common_kwargs,
                            )
                        st.session_state.last_sweep_id     = sid
                        st.session_state.selected_sweep_id = sid
                    except Exception as exc:
                        import traceback
                        st.session_state.sweep_error = f"{exc}\n{traceback.format_exc()}"
                    finally:
                        st.session_state.sweep_running = False

                t = threading.Thread(target=_do_sweep, daemon=True)
                t.start()
                t.join()   # block main thread so Streamlit re-renders on finish

                if st.session_state.sweep_error:
                    st.error(f"Sweep failed: {st.session_state.sweep_error}")
                elif st.session_state.last_sweep_id:
                    prog_bar.progress(1.0)
                    prog_text.success(
                        f"Done! sweep_id = {st.session_state.last_sweep_id}  "
                        f"({st.session_state.sweep_done} hypotheses)"
                    )
                    st.info("Go to the Results tab to explore results.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    st.header("📋 Hypothesis Results")
    _help_box(
        "Browse every backtest from a saved sweep. <b>Verdict</b> tells you which "
        "strategies passed the gauntlet (Sharpe, DSR, BH-significance, trade "
        "count). Click a row in the <b>📈 Detail</b> tab to drill into one. "
        "Pick which sweep to inspect from the sidebar."
    )

    sweep_id = st.session_state.selected_sweep_id
    if sweep_id is None:
        st.info("⚠ Select a sweep from the sidebar's 'Results browser'.")
    else:
        df_all = eng.load_sweep_results(sweep_id)
        if df_all.empty:
            st.warning("No results for this sweep yet.")
        else:
            # Filters
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                only_sig = st.checkbox("BH significant only", value=False)
            with c2:
                only_pos = st.checkbox("Test PnL > 0 only",   value=False)
            with c3:
                min_n    = st.number_input("Min trades (test)", value=0, min_value=0)
            with c4:
                only_real = st.checkbox("Real edges only (DSR>0.5 + CI_low>0)",
                                        value=False,
                                        help="Fake-edge filters: Deflated Sharpe + bootstrap CI")

            dff = df_all.copy()
            if only_sig:
                dff = dff[dff['bh_sig'] == 1]
            if only_pos:
                dff = dff[dff['test_pnl'] > 0]
            if min_n > 0:
                dff = dff[dff['test_n'] >= min_n]
            if only_real:
                if 'dsr' in dff.columns:
                    dff = dff[dff['dsr'].fillna(0) > 0.5]
                if 'sharpe_ci_low' in dff.columns:
                    dff = dff[dff['sharpe_ci_low'].fillna(-1) > 0]

            # Display columns (include fake-edge metrics if present)
            show_cols = ['hypothesis_id', 'params_json',
                         'train_n', 'train_wr', 'train_sharpe',
                         'test_n',  'test_wr',  'test_sharpe', 'test_pnl',
                         'test_sortino', 'test_calmar',
                         'dsr', 'sharpe_ci_low', 'regime_stable',
                         'p_adj', 'bh_sig', 'verdict']
            disp = dff[[c for c in show_cols if c in dff.columns]].copy()
            disp['params_json'] = disp['params_json'].apply(
                lambda x: _fmt_params(json.loads(x)) if isinstance(x, str) else str(x)
            )
            disp = disp.rename(columns={
                'hypothesis_id': 'ID', 'params_json': 'Params',
                'train_n': 'Tr.N', 'train_wr': 'Tr.WR%', 'train_sharpe': 'Tr.Sharpe',
                'test_n':  'Te.N', 'test_wr':  'Te.WR%', 'test_sharpe':  'Te.Sharpe',
                'test_pnl': 'Te.PnL', 'test_sortino': 'Sortino', 'test_calmar': 'Calmar',
                'dsr': 'DSR', 'sharpe_ci_low': 'CI_low', 'regime_stable': 'RegStable',
                'p_adj': 'p(adj)', 'bh_sig': 'BH✓', 'verdict': 'Verdict',
            })
            for col in ['Tr.WR%', 'Te.WR%']:
                if col in disp.columns:
                    disp[col] = disp[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else '—')
            for col in ['Tr.Sharpe', 'Te.Sharpe', 'Sortino', 'Calmar']:
                if col in disp.columns:
                    disp[col] = disp[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else '—')
            if 'p(adj)' in disp.columns:
                disp['p(adj)'] = disp['p(adj)'].apply(lambda v: f"{v:.4f}" if pd.notna(v) else '—')
            if 'Te.PnL' in disp.columns:
                disp['Te.PnL'] = disp['Te.PnL'].apply(lambda v: f"£{v:,.0f}" if pd.notna(v) else '—')
            if 'DSR' in disp.columns:
                disp['DSR'] = disp['DSR'].apply(
                    lambda v: f"{v:.3f}" if pd.notna(v) else '—')
            if 'CI_low' in disp.columns:
                disp['CI_low'] = disp['CI_low'].apply(
                    lambda v: f"{v:.2f}" if pd.notna(v) else '—')
            if 'RegStable' in disp.columns:
                disp['RegStable'] = disp['RegStable'].apply(
                    lambda v: '✓' if v == 1 else '✗' if pd.notna(v) else '—')

            st.caption(f"Showing {len(disp)} / {len(df_all)} hypotheses")

            st.dataframe(
                disp.style
                    .map(_color_verdict, subset=['Verdict'] if 'Verdict' in disp.columns else [])
                    .map(_color_bh,      subset=['BH✓']     if 'BH✓'    in disp.columns else []),
                use_container_width=True,
                height=500,
            )

            # Row selector
            st.markdown("---")
            st.subheader("Select hypothesis for detail view")
            hyp_options = dff['hypothesis_id'].tolist()
            if hyp_options:
                selected = st.selectbox("Hypothesis ID", hyp_options)
                if st.button("View Detail →"):
                    st.session_state.selected_hyp_id = selected
                    st.info("Switch to the Detail tab.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DETAIL
# ─────────────────────────────────────────────────────────────────────────────
with tab_detail:
    st.header("📈 Hypothesis Detail")
    _help_box(
        "Deep dive into a single backtested strategy: equity curve, trade "
        "distribution, drawdowns, and per-regime performance. Pick one from "
        "the table in the <b>📋 Results</b> tab to populate this view."
    )

    hyp_id = st.session_state.selected_hyp_id
    if hyp_id is None:
        st.info("Select a hypothesis in the Results tab.")
    else:
        st.caption(f"hypothesis_id: `{hyp_id}`")

        # Load stats row
        sweep_id = st.session_state.selected_sweep_id
        hyp_row  = None
        if sweep_id:
            df_all = eng.load_sweep_results(sweep_id)
            hits   = df_all[df_all['hypothesis_id'] == hyp_id]
            if not hits.empty:
                hyp_row = hits.iloc[0]

        if hyp_row is not None:
            params = json.loads(hyp_row['params_json']) if isinstance(hyp_row['params_json'], str) else {}
            st.subheader("Parameters")
            param_items = [{'Parameter': k, 'Value': str(v)} for k, v in params.items()]
            st.dataframe(pd.DataFrame(param_items), hide_index=True, use_container_width=False)

            # Stats table
            st.subheader("Performance Summary")
            stats_data = {
                'Metric': ['Trades', 'Win Rate', 'Sharpe', 'Sortino', 'Calmar',
                           'PnL', 'Max DD', 'BH sig', 'Verdict'],
                'Train': [
                    hyp_row.get('train_n', '—'),
                    f"{hyp_row.get('train_wr', 0):.1f}%",
                    f"{hyp_row.get('train_sharpe', 0):.2f}",
                    '—', '—',
                    f"£{hyp_row.get('train_pnl', 0):,.0f}",
                    f"£{hyp_row.get('train_max_dd', 0):,.0f}",
                    '—', '—',
                ],
                'Test': [
                    hyp_row.get('test_n', '—'),
                    f"{hyp_row.get('test_wr', 0):.1f}%",
                    f"{hyp_row.get('test_sharpe', 0):.2f}",
                    f"{hyp_row.get('test_sortino', 0):.2f}",
                    f"{hyp_row.get('test_calmar', 0):.2f}",
                    f"£{hyp_row.get('test_pnl', 0):,.0f}",
                    f"£{hyp_row.get('test_max_dd', 0):,.0f}",
                    '✓' if hyp_row.get('bh_sig') == 1 else '✗',
                    hyp_row.get('verdict', '—'),
                ],
            }
            st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=False)

        # Equity curves
        col_tr, col_te = st.columns(2)

        with col_tr:
            st.subheader("Train — Equity Curve")
            tr_trades = eng.load_hypothesis_trades(hyp_id, split='train')
            if tr_trades.empty:
                st.caption("No train trades.")
            else:
                tr_m = eng._merge_partials(tr_trades)
                fig  = eng.plot_equity_figure(tr_m, label='Train')
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col_te:
            st.subheader("Test — Equity Curve")
            te_trades = eng.load_hypothesis_trades(hyp_id, split='test')
            if te_trades.empty:
                st.caption("No test trades.")
            else:
                te_m = eng._merge_partials(te_trades)
                fig  = eng.plot_equity_figure(te_m, label='Test')
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Exit reason and pair breakdowns (test set)
        if not te_trades.empty:
            te_m = eng._merge_partials(te_trades)
            st.markdown("---")
            cx, cy = st.columns(2)

            with cx:
                st.subheader("Exit reasons (test)")
                er = te_m.groupby('exit_reason')['pnl'].agg(['count', 'sum']).reset_index()
                er.columns = ['Reason', 'Count', 'Total PnL']
                er['Total PnL'] = er['Total PnL'].apply(lambda v: f"£{v:,.0f}")
                st.dataframe(er, hide_index=True, use_container_width=True)

            with cy:
                st.subheader("By pair (test)")
                pp = te_m.groupby('instrument').agg(
                    N=('pnl', 'count'),
                    WR=('pnl', lambda x: (x > 0).mean() * 100),
                    PnL=('pnl', 'sum'),
                ).reset_index()
                pp['WR']  = pp['WR'].apply(lambda v: f"{v:.1f}%")
                pp['PnL'] = pp['PnL'].apply(lambda v: f"£{v:,.0f}")
                st.dataframe(pp, hide_index=True, use_container_width=True)

        # Hypothesis tests (test set)
        if not te_trades.empty:
            te_m = eng._merge_partials(te_trades)
            st.markdown("---")
            st.subheader("Hypothesis tests (test set)")
            ht = eng.ht_summary(te_m)
            if ht:
                ht_display = []
                if 'binom_p' in ht:
                    ht_display.append({'Test': 'Binomial (WR vs break-even)',
                                       'p-value': f"{ht['binom_p']:.4f}",
                                       'Significant': '✓' if ht['binom_sig'] else '✗'})
                if 'ttest_p' in ht:
                    ht_display.append({'Test': 't-test (mean PnL > 0)',
                                       'p-value': f"{ht['ttest_p']:.4f}",
                                       'Significant': '✓' if ht['ttest_sig'] else '✗'})
                if 'perm_p' in ht:
                    ht_display.append({'Test': 'Permutation Sharpe',
                                       'p-value': f"{ht['perm_p']:.4f}",
                                       'Significant': '✓' if ht['perm_sig'] else '✗'})
                st.dataframe(pd.DataFrame(ht_display), hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
with tab_robust:
    st.header("🛡 Robustness Analysis")
    _help_box(
        "A strategy that looks great in-sample often dies live. This tab "
        "stress-tests the selected hypothesis with <b>walk-forward</b> "
        "(out-of-sample re-optimization), <b>regime breakdown</b> (does it work "
        "in trending vs ranging markets?), and <b>Monte Carlo</b> resampling "
        "(is the equity curve a fluke?). Pick a hypothesis in the Detail tab first."
    )

    hyp_id = st.session_state.selected_hyp_id
    if hyp_id is None:
        st.info("Select a hypothesis in the Results tab.")
    else:
        te_trades = eng.load_hypothesis_trades(hyp_id, split='test')
        if te_trades.empty:
            st.warning("No test trades for this hypothesis.")
        else:
            te_m = eng._merge_partials(te_trades)

            # ── Walk-forward ──────────────────────────────────────────────────
            st.subheader("Walk-forward (4-fold test set split)")
            wf = eng.walk_forward_test(te_m, n_folds=4)
            if wf.empty:
                st.caption("Not enough trades for walk-forward.")
            else:
                def _color_wf_row(row):
                    style = [''] * len(row)
                    idx = row.index.tolist()
                    if 'sharpe' in idx:
                        v = row['sharpe']
                        color = '#1a7f2e' if v > 0.5 else ('#a67c00' if v > 0 else '#7f1a1a')
                        style[idx.index('sharpe')] = f'background-color:{color};color:white'
                    return style

                st.dataframe(
                    wf.style.apply(_color_wf_row, axis=1),
                    hide_index=True,
                    use_container_width=True,
                )
                consistent = (wf['sharpe'] > 0).sum()
                total_folds = len(wf)
                label = '✓ Consistent' if consistent >= 3 else ('⚠ Inconsistent' if consistent >= 2 else '✗ Poor')
                st.metric("Folds profitable", f"{consistent}/{total_folds}", label)

            st.markdown("---")

            # ── Regime breakdown ─────────────────────────────────────────────
            st.subheader("Regime breakdown")
            rb = eng.regime_breakdown(te_m)
            if rb.empty:
                st.caption("No regime data.")
            else:
                st.dataframe(rb, hide_index=True, use_container_width=True)

            st.markdown("---")

            # ── Monte Carlo ───────────────────────────────────────────────────
            st.subheader("Monte Carlo (500 paths, 90-day challenge)")
            n_mc = st.slider("Simulations", 100, 2000, 500, 100)
            if st.button("Run Monte Carlo"):
                with st.spinner("Simulating…"):
                    fig = eng.plot_mc_figure(te_m, n_sims=n_mc, challenge_days=90)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    mc_stats = eng.run_monte_carlo(te_m, n_sims=n_mc, challenge_days=90)
                    if mc_stats:
                        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                        mcol1.metric("Pass rate",    f"{mc_stats.get('pass_pct',0):.1f}%")
                        mcol2.metric("Blown rate",   f"{mc_stats.get('blown_pct',0):.1f}%")
                        mcol3.metric("Median days",  str(mc_stats.get('median_days_to_pass','—')))
                        mcol4.metric("MC verdict",   mc_stats.get('verdict','—'))

            st.markdown("---")

            # ── Prop firm pass checklist ──────────────────────────────────────
            st.subheader("Prop firm pass-readiness checklist")
            stats    = eng.calc_stats(te_m)
            ext      = eng.extended_risk_metrics(te_m)
            mc_quick = eng.run_monte_carlo(te_m, n_sims=200, challenge_days=90)

            def _check(label, passed: bool, detail: str = ''):
                icon = '✅' if passed else '❌'
                if detail:
                    st.markdown(f"{icon}  **{label}** — {detail}")
                else:
                    st.markdown(f"{icon}  **{label}**")

            if stats:
                _check("Win rate above break-even",
                       stats['wr'] > stats['be'],
                       f"WR={stats['wr']:.1f}%  BE={stats['be']:.1f}%")
                _check("Sharpe ratio ≥ 1.0",
                       stats['sharpe'] >= 1.0,
                       f"{stats['sharpe']:.2f}")
                _check("Max drawdown < 6% of account",
                       abs(stats['max_dd']) < eng.INITIAL_BALANCE * 0.06,
                       f"£{stats['max_dd']:,.0f}")
                _check("At least 30 test trades",
                       stats['n'] >= 30,
                       f"{stats['n']}")
            if ext:
                _check("Sortino ≥ 1.0",
                       ext['sortino'] >= 1.0,
                       f"{ext['sortino']:.2f}")
                _check("Profit factor ≥ 1.3",
                       ext['profit_factor'] >= 1.3,
                       f"{ext['profit_factor']:.2f}")
                _check("Max consecutive losses ≤ 8",
                       ext['max_consec_loss'] <= 8,
                       f"{ext['max_consec_loss']}")
            if mc_quick:
                _check("MC pass rate ≥ 60%",
                       mc_quick.get('pass_pct', 0) >= 60,
                       f"{mc_quick.get('pass_pct',0):.1f}%")
            if not wf.empty:
                _check("Walk-forward: ≥3/4 folds profitable",
                       (wf['sharpe'] > 0).sum() >= 3)

            # ── Fake-edge filter checks ───────────────────────────────────────
            st.markdown("---")
            st.subheader("Fake-edge filter checks")
            sweep_id_r = st.session_state.selected_sweep_id
            if sweep_id_r:
                df_all_r = eng.load_sweep_results(sweep_id_r)
                hits     = df_all_r[df_all_r['hypothesis_id'] == hyp_id]
                if not hits.empty:
                    hr = hits.iloc[0]

                    dsr_val = hr.get('dsr')
                    if pd.notna(dsr_val) if hasattr(pd, 'notna') else dsr_val is not None:
                        _check(
                            "Deflated Sharpe Ratio > 0.5 (Layer 3)",
                            float(dsr_val) > 0.5,
                            f"DSR={dsr_val:.3f}  (corrects for selection bias over "
                            f"{len(df_all_r)} tested combinations)",
                        )
                    else:
                        st.caption("DSR: not computed (run sweep first)")

                    ci_lo = hr.get('sharpe_ci_low')
                    ci_hi = hr.get('sharpe_ci_high')
                    if pd.notna(ci_lo) if hasattr(pd, 'notna') else ci_lo is not None:
                        _check(
                            "Bootstrap Sharpe CI lower bound > 0 (Layer 3)",
                            float(ci_lo) > 0,
                            f"Sharpe 95% CI = [{ci_lo:.2f}, {ci_hi:.2f}]  "
                            f"(resampled trade log 500×)",
                        )
                    else:
                        st.caption("Bootstrap CI: not computed (run sweep first)")

                    reg_stable = hr.get('regime_stable')
                    if reg_stable is not None:
                        _check(
                            "Regime stability (Layer 4)",
                            int(reg_stable) == 1,
                            "Profitable in ≥2 regimes, or one dominant regime (≥60% of trades)",
                        )
                    else:
                        st.caption("Regime stability: not computed")

            # Portfolio correlation hint
            st.markdown("---")
            st.caption("See the **Portfolio** tab to check correlation with other "
                       "survivors in this sweep before allocating capital.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_mine:
    st.header("🧬 Automated Pattern Mining")
    _help_box(
        "<b>Open-ended discovery.</b> The miner simulates long &amp; short trades "
        "on every bar, then uses LightGBM + SHAP to discover which market "
        "conditions predict profitable trades. Top patterns are auto-converted "
        "to runnable strategy functions and appended to "
        "<code>edge_hypotheses.py</code>. Use <b>▶ Run</b> for hand-crafted ideas; "
        "use this tab to find ones you didn't think of."
    )

    if not _MINER_AVAILABLE:
        st.error(
            "edge_miner.py not found in the same folder as edge_gui.py. "
            "Make sure `edge_miner.py` is in the Downloads folder."
        )
    elif not st.session_state.data_loaded:
        st.warning("⚠ Load data from the sidebar first.")
    else:
        st.markdown(
            "The miner simulates long/short trades on every bar, then uses a "
            "feature sweep + LightGBM/SHAP to discover which market conditions "
            "predict profitable trades. Top patterns are converted to entry "
            "functions and appended to `edge_hypotheses.py`."
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            mine_session = st.selectbox("Session", ['ny', 'london', 'asian', 'all'],
                                        help="Filter bars to this session only")
        with mc2:
            mine_tp_r = st.number_input("Simulation TP (R)", value=2.0,
                                        min_value=0.5, max_value=5.0, step=0.5)
        with mc3:
            mine_sl_r = st.number_input("Simulation SL (R)", value=1.0,
                                        min_value=0.25, max_value=3.0, step=0.25)
        with mc4:
            mine_top_n = st.slider("Top N patterns", 1, 10, 5)

        mine_horizon  = st.slider("Forward horizon (bars)", 5, 50, 20,
                                  help="Max bars to scan for TP/SL hit")
        mine_append   = st.checkbox("Append generated code to edge_hypotheses.py",
                                    value=True)

        cfg_pairs = gui_config.selected_pairs() if _CONFIG_AVAILABLE else None
        use_cfg_pairs = st.checkbox(
            f"Use Config selection ({', '.join(cfg_pairs) if cfg_pairs else 'none set'})",
            value=False,
            disabled=not cfg_pairs,
            help="When ticked, restrict the run to the pairs picked in the Config tab.",
        )

        mine_btn = st.button(
            "🧬  Run Miner",
            disabled=st.session_state.miner_running,
            type="primary",
        )

        mine_prog  = st.progress(0)
        mine_label = st.empty()

        if mine_btn:
            st.session_state.miner_running   = True
            st.session_state.miner_results   = None
            st.session_state.miner_error     = None
            st.session_state.miner_step      = 0
            st.session_state.miner_step_label = ''

        if st.session_state.miner_running:
            mine_label.text(
                st.session_state.miner_step_label or 'Starting…'
            )
            mine_prog.progress(
                st.session_state.miner_step / 5
            )

            def _mine_progress(step, total, label=''):
                st.session_state.miner_step       = step
                st.session_state.miner_step_label = label

            def _do_mine():
                try:
                    sess = None if mine_session == 'all' else mine_session
                    train_dfs = st.session_state.train_dfs
                    test_dfs  = st.session_state.test_dfs
                    if use_cfg_pairs and cfg_pairs:
                        train_dfs = {p: d for p, d in train_dfs.items() if p in cfg_pairs}
                        test_dfs  = {p: d for p, d in test_dfs.items()  if p in cfg_pairs}
                    results = miner.run_miner(
                        pair_dfs          = train_dfs,
                        test_dfs          = test_dfs,
                        session_filter    = sess,
                        tp_r              = mine_tp_r,
                        sl_r              = mine_sl_r,
                        horizon_bars      = mine_horizon,
                        top_n_patterns    = mine_top_n,
                        append_to_file    = mine_append,
                        progress_callback = _mine_progress,
                    )
                    st.session_state.miner_results = results
                except Exception as exc:
                    import traceback
                    st.session_state.miner_error = f"{exc}\n{traceback.format_exc()}"
                finally:
                    st.session_state.miner_running = False

            t = threading.Thread(target=_do_mine, daemon=True)
            t.start()
            t.join()

            if st.session_state.miner_error:
                st.error(f"Miner failed:\n{st.session_state.miner_error}")
            elif st.session_state.miner_results is not None:
                mine_prog.progress(1.0)
                mine_label.success("Mining complete!")

        # ── Display results ───────────────────────────────────────────────────
        results = st.session_state.miner_results
        if results:
            st.markdown("---")

            # ── Distribution shift + Adversarial validation ───────────────────
            adv = results.get('adversarial', {})
            ds  = results.get('dist_shift',  {})

            if adv or ds:
                st.subheader("Data validity checks")
                c_adv, c_ks = st.columns(2)

                with c_adv:
                    if adv:
                        verdict_color = {'OK': '🟢', 'WARN': '🟡', 'ABORT': '🔴'}.get(
                            adv.get('verdict', 'OK'), '⚪')
                        st.metric("Adversarial AUC", f"{adv.get('auc', 0):.3f}",
                                  delta=adv.get('verdict', '—'),
                                  delta_color='normal' if adv.get('verdict') == 'OK'
                                              else 'inverse')
                        st.caption(f"{verdict_color}  {adv.get('message', '')}")
                        if adv.get('drift_features'):
                            st.warning(
                                f"Down-weighted in mining: "
                                f"`{'`, `'.join(adv['drift_features'])}`"
                            )
                    else:
                        st.caption("Adversarial validation: not run (provide test_dfs)")

                with c_ks:
                    if ds:
                        n_shifted = len(ds.get('shifted_features', []))
                        st.metric("Features with distribution shift",
                                  f"{n_shifted} / {len(ds.get('ks_results', []))}",
                                  delta='shift detected' if ds.get('any_shift') else 'OK',
                                  delta_color='inverse' if ds.get('any_shift') else 'normal')
                        if not ds.get('ks_results', pd.DataFrame()).empty:
                            ks_df = ds['ks_results'][['feature', 'ks_stat', 'p_value', 'shifted']]
                            st.dataframe(ks_df, hide_index=True, use_container_width=True,
                                         height=150)
                    else:
                        st.caption("Distribution shift check: not run (provide test_dfs)")

            # ── Meta-learner (if memory available) ────────────────────────────
            meta = results.get('meta_learner', {})
            if meta and meta.get('feature_ranking'):
                st.markdown("---")
                st.subheader("Meta-learner: feature robustness ranking")
                st.caption(meta.get('note', ''))
                rank_df = pd.DataFrame(meta['feature_ranking'],
                                       columns=['Feature', 'Robustness score'])
                rank_df['Robustness score'] = rank_df['Robustness score'].apply(
                    lambda v: f"{v:.4f}")
                st.dataframe(rank_df, hide_index=True, use_container_width=True)

            st.markdown("---")

            # Feature sweep table
            st.subheader("Feature Sweep — all significant conditions")
            fs_df = results.get('feature_sweep', pd.DataFrame())
            if fs_df.empty:
                st.caption("No significant features found at current thresholds.")
            else:
                disp_fs = fs_df[['feature', 'direction', 'threshold', 'pct',
                                  'target', 'n', 'win_rate', 't_stat', 'p_value']].copy()
                disp_fs['win_rate'] = disp_fs['win_rate'].apply(lambda v: f"{v:.2%}")
                disp_fs['t_stat']   = disp_fs['t_stat'].apply(lambda v: f"{v:.2f}")
                disp_fs['p_value']  = disp_fs['p_value'].apply(lambda v: f"{v:.4f}")
                st.dataframe(disp_fs, hide_index=True, use_container_width=True, height=300)

            st.markdown("---")

            # ML patterns table
            st.subheader("ML Patterns (SHAP-ranked)")
            patterns = results.get('patterns', [])
            if not patterns:
                st.caption(
                    "No ML patterns found. LightGBM/SHAP may not be installed "
                    "(pip install lightgbm shap), or not enough data."
                )
            else:
                pat_rows = []
                for p in patterns:
                    conds = f"{p['feature']} {p['direction']} {p['threshold']:.4g}"
                    for c in p.get('conditions', []):
                        conds += f"  AND  {c['feature']} {c['direction']} {c['threshold']:.4g}"
                    pat_rows.append({
                        'Target':     p['target'],
                        'Conditions': conds,
                        'Win rate':   f"{p['win_rate']:.2%}",
                        'N':          p['n_samples'],
                        'SHAP':       f"{p['shap_value']:.4f}",
                    })
                st.dataframe(pd.DataFrame(pat_rows), hide_index=True,
                             use_container_width=True)

            st.markdown("---")

            # Generated hypothesis code
            st.subheader("Generated Hypotheses")
            hypotheses = results.get('hypotheses', [])
            if not hypotheses:
                st.caption("No hypotheses generated.")
            else:
                for h in hypotheses:
                    with st.expander(f"entry_{h['name']}  ({h['pattern']['target']}, "
                                     f"WR={h['pattern']['win_rate']:.2%})"):
                        st.code(h['code'], language='python')

                if mine_append:
                    st.success(
                        f"{len(hypotheses)} hypothesis function(s) appended to "
                        "`edge_hypotheses.py`. Reload the Run tab to see them in the dropdown."
                    )
                else:
                    if st.button("Append to edge_hypotheses.py"):
                        try:
                            miner.append_to_hypotheses_file(hypotheses)
                            st.success("Appended successfully.")
                        except Exception as e:
                            st.error(str(e))

        # ── Self-Improving Loop ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Self-Improving Loop")
        st.markdown(
            "Runs N rounds automatically. Each round: **mine → sweep → learn → mutate**. "
            "The feature leaderboard and proven conditions update after every round, "
            "guiding each subsequent mining run toward what has historically worked."
        )

        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            loop_rounds    = st.slider("Rounds", 2, 20, 5)
        with lc2:
            loop_hyp_round = st.slider("Hypotheses/round", 2, 10, 5)
        with lc3:
            loop_trials    = st.slider("Optuna trials/hyp", 20, 200, 80)
        with lc4:
            loop_workers   = st.slider("Workers (loop)", 1, 8, 2)

        lc5, lc6 = st.columns(2)
        with lc5:
            loop_session  = st.selectbox("Session (loop)", ['ny', 'london', 'asian'],
                                         key='loop_session_select')
        with lc6:
            loop_mutation = st.slider("Mutation rate", 0.05, 0.30, 0.15, 0.05,
                                      help="How much to perturb survivor thresholds ±")

        loop_append = st.checkbox("Append generated code to edge_hypotheses.py",
                                  value=True, key='loop_append_chk')

        loop_btn = st.button(
            "🔁  Run Self-Improving Loop",
            disabled=st.session_state.loop_running or not st.session_state.data_loaded,
            type="primary",
        )

        loop_prog  = st.progress(0)
        loop_label = st.empty()

        if loop_btn:
            st.session_state.loop_running      = True
            st.session_state.loop_round        = 0
            st.session_state.loop_total_rounds = loop_rounds
            st.session_state.loop_msg          = ''
            st.session_state.loop_results      = None
            st.session_state.loop_error        = None

        if st.session_state.loop_running:
            loop_label.text(st.session_state.loop_msg or 'Starting loop…')
            loop_prog.progress(
                st.session_state.loop_round / max(st.session_state.loop_total_rounds, 1)
            )

            def _loop_progress(round_n, total, msg=''):
                st.session_state.loop_round        = round_n
                st.session_state.loop_total_rounds = total
                st.session_state.loop_msg          = msg

            def _do_loop():
                try:
                    mem = st.session_state.loop_memory or miner.PatternMemory()
                    results = miner.run_discovery_loop(
                        pair_dfs             = st.session_state.train_dfs,
                        train_dfs            = st.session_state.train_dfs,
                        test_dfs             = st.session_state.test_dfs,
                        n_rounds             = loop_rounds,
                        session_filter       = loop_session,
                        hypotheses_per_round = loop_hyp_round,
                        n_optuna_trials      = loop_trials,
                        n_workers            = loop_workers,
                        mutation_rate        = loop_mutation,
                        memory               = mem,
                        append_to_file       = loop_append,
                        progress_callback    = _loop_progress,
                    )
                    st.session_state.loop_results = results
                    st.session_state.loop_memory  = results['memory']
                except Exception as exc:
                    import traceback
                    st.session_state.loop_error = f"{exc}\n{traceback.format_exc()}"
                finally:
                    st.session_state.loop_running = False

            t = threading.Thread(target=_do_loop, daemon=True)
            t.start()
            t.join()

            if st.session_state.loop_error:
                st.error(f"Loop failed:\n{st.session_state.loop_error}")
            elif st.session_state.loop_results:
                loop_prog.progress(1.0)
                loop_label.success("Loop complete!")

        # ── Loop results display ──────────────────────────────────────────────
        loop_res = st.session_state.loop_results
        mem      = st.session_state.loop_memory

        if loop_res:
            st.markdown("---")

            # Round-by-round summary
            st.subheader("Round summary")
            rounds_data = loop_res.get('rounds', [])
            if rounds_data:
                rd_df = pd.DataFrame(rounds_data)[['round', 'n_tested', 'n_survivors']]
                rd_df.columns = ['Round', 'Hypotheses tested', 'Survivors (BH sig)']
                st.dataframe(rd_df, hide_index=True, use_container_width=True)

            total_surv = len(loop_res.get('survivors', pd.DataFrame()))
            st.metric("Total BH-significant results", total_surv)

        if mem is not None:
            st.markdown("---")

            tab_lead, tab_proven, tab_history = st.tabs(
                ["🏆 Feature Leaderboard", "✅ Proven Conditions", "📜 Round History"]
            )

            with tab_lead:
                lb = mem.feature_leaderboard_df()
                if lb.empty:
                    st.caption("No data yet — run the loop first.")
                else:
                    # Bar chart of appearances
                    st.bar_chart(lb.set_index('feature')['appearances'])
                    st.dataframe(lb, hide_index=True, use_container_width=True)

            with tab_proven:
                pc = mem.proven_conditions_df()
                if pc.empty:
                    st.caption("No proven conditions yet.")
                else:
                    disp = pc[['feature', 'direction', 'threshold', 'session',
                               'target', 'hit_count', 'avg_wr']].copy()
                    disp['avg_wr'] = disp['avg_wr'].apply(lambda v: f"{v:.2%}")
                    st.dataframe(disp, hide_index=True, use_container_width=True)

            with tab_history:
                rh = mem.rounds_df()
                if rh.empty:
                    st.caption("No rounds recorded yet.")
                else:
                    st.dataframe(rh, hide_index=True, use_container_width=True)

            if st.button("🗑  Clear Memory", key='clear_memory_btn'):
                mem.clear()
                st.session_state.loop_memory  = mem
                st.session_state.loop_results = None
                st.success("Memory cleared.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — PORTFOLIO CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_portfolio:
    st.header("📊 Portfolio Correlation")
    _help_box(
        "Two strategies that win on the same days don't diversify — they double "
        "your risk. This tab computes the pairwise correlation of daily PnL "
        "across the survivors in your selected sweep, so you can pick a "
        "complementary basket. <b>Aim for average correlation &lt; 0.5.</b>"
    )

    sweep_id_p = st.session_state.selected_sweep_id
    if sweep_id_p is None:
        st.info("⚠ Select a sweep from the sidebar's 'Results browser'.")
    else:
        min_sharpe_p = st.slider("Min test Sharpe to include", -2.0, 3.0, 0.0, 0.1)

        if st.button("Compute Portfolio Correlation", type="primary"):
            with st.spinner("Computing…"):
                try:
                    pc_res = eng.portfolio_correlation(sweep_id_p, min_sharpe=min_sharpe_p)
                    n_s    = pc_res['n_survivors']

                    if n_s < 2:
                        st.warning(f"Only {n_s} survivor(s) with Sharpe ≥ {min_sharpe_p} — "
                                   "need ≥2 for correlation.")
                    else:
                        st.metric("Strategies analysed", n_s)
                        st.metric("Average pairwise correlation",
                                  f"{pc_res['avg_corr']:.3f}",
                                  delta="OK" if pc_res['avg_corr'] < 0.5 else "High",
                                  delta_color="normal" if pc_res['avg_corr'] < 0.5
                                              else "inverse")

                        corr_df = pc_res['corr_matrix']
                        if not corr_df.empty:
                            st.subheader("Correlation matrix")
                            # Colour-map: green = low corr, red = high
                            st.dataframe(
                                corr_df.style.background_gradient(
                                    cmap='RdYlGn_r', vmin=-1, vmax=1),
                                use_container_width=True,
                            )

                        high_pairs = pc_res['high_corr_pairs']
                        if high_pairs:
                            st.subheader("High-correlation pairs (> 0.70)")
                            hp_df = pd.DataFrame(high_pairs, columns=['Hypothesis A',
                                                                        'Hypothesis B',
                                                                        'Correlation'])
                            st.dataframe(hp_df, hide_index=True, use_container_width=True)
                            st.warning(
                                f"{len(high_pairs)} pair(s) are highly correlated. "
                                "Running both simultaneously adds risk without "
                                "diversification — choose the one with the higher DSR."
                            )
                        else:
                            st.success("No pairs above 0.70 correlation — "
                                       "survivors are well-diversified.")

                except Exception as exc:
                    import traceback
                    st.error(f"Portfolio correlation failed: {exc}\n{traceback.format_exc()}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — TCA (Transaction Cost Analysis)
# ─────────────────────────────────────────────────────────────────────────────
with tab_tca:
    st.header("💸 Transaction Cost Analysis")
    _help_box(
        "<b>Where your edge actually goes.</b> Spreads, slippage, and commission "
        "can quietly eat half your backtested profit. This tab reads real "
        "executions from your MT5 trade log and shows you, per pair and per "
        "strategy, how much edge survives once costs are paid. If a strategy "
        "looks great here in backtest but bleeds in live TCA, kill it."
    )

    if not _AGENT_AVAILABLE:
        st.error(
            "Agent modules failed to import. TCA needs `agent/tca.py` and "
            "`agent/live_ingest.py`. Make sure dependencies are installed."
        )
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            ingest_btn = st.button("📥  Ingest latest MT5 CSVs",
                                   help="Re-scan execution_quality.csv and live_trade_log.csv")
        with c2:
            tca_btn = st.button("🔄  Refresh TCA + write report", type="primary")

        if ingest_btn:
            with st.spinner("Ingesting live CSVs…"):
                try:
                    res = agent_live_ingest.ingest_all()
                    st.success(f"Ingested: {res}")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

        if tca_btn:
            with st.spinner("Building TCA report…"):
                try:
                    res = agent_tca.run()
                    st.session_state.tca_last_run  = pd.Timestamp.utcnow()
                    st.session_state.tca_last_path = res['report_path']
                    st.success(
                        f"Report → {res['report_path']}  "
                        f"({res['n_pairs']} pairs, {res['n_strategies']} strategies)"
                    )
                except Exception as exc:
                    import traceback
                    st.error(f"TCA failed: {exc}\n{traceback.format_exc()}")

        if st.session_state.tca_last_run:
            st.caption(
                f"Last refresh: {st.session_state.tca_last_run.strftime('%Y-%m-%d %H:%M UTC')}  "
                f"→ `{st.session_state.tca_last_path}`"
            )

        st.markdown("---")

        # ── Per-pair summary ─────────────────────────────────────────────────
        st.subheader("Per-pair cost summary")
        try:
            pp = agent_tca.per_pair_summary()
        except Exception as exc:
            pp = pd.DataFrame()
            st.error(f"per_pair_summary failed: {exc}")

        if pp.empty:
            st.info("No live execution data yet. Trade live with MT5live or wait for "
                    "CSVs to populate, then click Ingest above.")
        else:
            def _color_quality(v):
                return {
                    'good':       'background-color:#1a7f2e;color:white',
                    'mixed':      'background-color:#a67c00;color:white',
                    'tail-heavy': 'background-color:#7f1a1a;color:white',
                }.get(v, '')

            st.dataframe(
                pp.style.map(_color_quality,
                                  subset=['fill_quality'] if 'fill_quality' in pp.columns else []),
                hide_index=True,
                use_container_width=True,
            )

            # Totals
            tcols = st.columns(4)
            tcols[0].metric("Pairs traded",       len(pp))
            tcols[1].metric("Total live trades",  int(pp['n_trades'].sum()))
            tcols[2].metric("Avg cost/trade (pips)",
                            f"{pp['cost_per_trade_pips'].mean():.2f}")
            n_bad = (pp['fill_quality'] == 'tail-heavy').sum()
            tcols[3].metric("Tail-heavy pairs",   int(n_bad),
                            delta="investigate" if n_bad else "OK",
                            delta_color="inverse" if n_bad else "normal")

        st.markdown("---")

        # ── By-hour profile ──────────────────────────────────────────────────
        st.subheader("Cost profile by UTC hour")
        try:
            execs = agent_tca.load_live_executions()
        except Exception:
            execs = pd.DataFrame()

        pair_options = ['(all)'] + (sorted(execs['pair'].dropna().unique().tolist())
                                    if not execs.empty and 'pair' in execs.columns else [])
        sel_pair = st.selectbox("Pair", pair_options, key='tca_hour_pair')
        try:
            hh = agent_tca.by_hour_profile(pair=None if sel_pair == '(all)' else sel_pair)
        except Exception as exc:
            hh = pd.DataFrame()
            st.error(f"by_hour_profile failed: {exc}")

        if hh.empty:
            st.caption("No hourly data yet.")
        else:
            st.dataframe(hh, hide_index=True, use_container_width=True)
            chart_df = hh.set_index('hour')[['avg_spread', 'avg_slip']]
            st.bar_chart(chart_df, height=240)
            worst = hh.sort_values('max_spread', ascending=False).head(3)
            if not worst.empty:
                hours_str = ', '.join(f"{int(h)}h" for h in worst['hour'].tolist())
                st.caption(f"Worst spread hours (UTC): {hours_str} — consider gating "
                           "strategies off in these windows.")

        st.markdown("---")

        # ── Per-strategy decay ───────────────────────────────────────────────
        st.subheader("Per-strategy live-vs-backtest decay")
        try:
            decay = agent_tca.per_strategy_decay()
        except Exception as exc:
            decay = pd.DataFrame()
            st.error(f"per_strategy_decay failed: {exc}")

        if decay.empty:
            st.caption("No decay data — needs live trades tagged with `strategy_name` "
                       "and matching survivors in `tested_strategies`.")
        else:
            def _color_verdict_d(v):
                return {
                    'PROMOTE':      'background-color:#1a7f2e;color:white',
                    'HOLD':         'background-color:#3b6cb0;color:white',
                    'REDUCE':       'background-color:#a67c00;color:white',
                    'KILL':         'background-color:#7f1a1a;color:white',
                    'INSUFFICIENT': 'color:#9e9e9e',
                }.get(v, '')

            st.dataframe(
                decay.style.map(_color_verdict_d,
                                     subset=['verdict'] if 'verdict' in decay.columns else []),
                hide_index=True,
                use_container_width=True,
            )

        # ── Open the markdown report ────────────────────────────────────────
        if st.session_state.tca_last_path and Path(st.session_state.tca_last_path).exists():
            st.markdown("---")
            with st.expander("📄  View full markdown report"):
                st.markdown(Path(st.session_state.tca_last_path).read_text(encoding='utf-8'))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — AGENT (autonomous Claude loop control)
# ─────────────────────────────────────────────────────────────────────────────
def _agent_proc_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _kill_agent(pid: int):
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                           capture_output=True, text=True, timeout=10)
        else:
            os.kill(pid, 15)
    except Exception:
        pass


with tab_live:
    st.header("💹 Live Strategies")
    _help_box(
        "Strategies that survived backtesting and are now risking real (or paper) "
        "money. Each one auto-ramps through four stages as it earns trust:"
        "<br>&nbsp;&nbsp;<b>SHADOW</b> — paper-trades to confirm live behavior matches backtest"
        "<br>&nbsp;&nbsp;<b>LIVE_QUARTER → LIVE_HALF → LIVE_FULL</b> — real money at 25% / 50% / 100% size"
        "<br>&nbsp;&nbsp;<b>KILL</b> — auto-disabled if live decay drops below 0.3 vs backtest baseline."
    )
    try:
        from agent import live_analytics
    except Exception as _e:
        st.error(f"live_analytics unavailable: {_e}")
        live_analytics = None

    if live_analytics is not None:
        names = live_analytics.list_strategies()
        if not names:
            st.info("No strategies promoted yet. The agent loop auto-promotes "
                    "survivors after each sweep — first one will appear here.")
        else:
            sel = st.selectbox("Strategy", names, key='live_strat_select')

            promo = live_analytics.promotion_state(sel)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode",         promo.get('mode')         or '—')
            c2.metric("Live trades",  promo.get('live_n')       or 0)
            c3.metric("Session",      promo.get('session')      or '—')
            c4.metric("Kill verdict", promo.get('kill_verdict') or 'OK')

            metrics = live_analytics.live_metrics(sel)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("n trades",       metrics.get('n_trades') or 0)
            m2.metric("Win rate %",     metrics.get('win_rate')      if metrics.get('win_rate') is not None else '—')
            m3.metric("Expectancy (R)", metrics.get('expectancy_r')  if metrics.get('expectancy_r') is not None else '—')
            m4.metric("Sharpe (ann.)",  metrics.get('sharpe_annualised') if metrics.get('sharpe_annualised') is not None else '—')

            d1, d2, d3 = st.columns(3)
            d1.metric("Gross PnL ($)", metrics.get('gross_pnl_usd') if metrics.get('gross_pnl_usd') is not None else '—')
            d2.metric("Max DD ($)",    metrics.get('max_dd_usd')    if metrics.get('max_dd_usd')    is not None else '—')
            tca = live_analytics.tca_summary(sel)
            d3.metric("Decay vs BT",   tca.get('decay') if tca.get('decay') is not None else '—',
                      delta=tca.get('verdict') or '—')

            st.subheader("By promotion mode")
            by_mode = metrics.get('by_mode') or {}
            if by_mode:
                st.dataframe(pd.DataFrame(by_mode).T, use_container_width=True)
            else:
                st.caption("No closed trades yet.")

            st.subheader("Equity curve")
            try:
                fig = live_analytics.equity_curve(sel)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.caption("No equity figure available.")
            except Exception as _e:
                st.warning(f"Equity-curve render failed: {_e}")

            st.subheader("Recent live trades")
            df = live_analytics.load_live_trades(sel)
            if df.empty:
                st.caption("No live trades recorded yet.")
            else:
                show_cols = [c for c in
                             ('ts_open', 'ts_close', 'pair', 'side', 'mode',
                              'entry', 'exit', 'pnl_usd', 'pnl_r', 'regime')
                             if c in df.columns]
                st.dataframe(df[show_cols].tail(50), use_container_width=True)


with tab_agent:
    st.header("🤖 Autonomous Agent")
    _help_box(
        "Claude generates new strategy ideas, backtests them, filters out the "
        "duds, and records the survivors — round after round, hands-free. "
        "<br><br>"
        "<b>Tip:</b> the easiest way to start the loop is from the <b>⚙ Config</b> "
        "tab — pick your pairs &amp; sessions there first, then click "
        "<i>Start agent loop</i>. This tab gives you finer control plus the "
        "live status of every recent round and daily report exports."
    )

    if not _AGENT_AVAILABLE:
        st.error("Agent modules failed to import. Check `agent/` is on PYTHONPATH "
                 "and the .env file has ANTHROPIC_API_KEY.")
    else:
        # ── Status row ──────────────────────────────────────────────────────
        running = _agent_proc_alive(st.session_state.agent_proc_pid)
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Status", "🟢 Running" if running else "⚫ Stopped")
        sc2.metric("PID",     st.session_state.agent_proc_pid or "—")
        sc3.metric("Started", st.session_state.agent_started_at or "—")

        # ── Controls ────────────────────────────────────────────────────────
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            mode = st.selectbox(
                "Mode",
                ['continuous loop', 'single round', 'dry run (no backtest)'],
                help=("continuous = run forever (BUDGET_TOTAL_USD caps spend); "
                      "single round = one round then exit; "
                      "dry run = generate one hypothesis to verify wiring"),
            )
        with ac2:
            start_btn = st.button("▶  Start agent",
                                  disabled=running, type="primary")
        with ac3:
            stop_btn  = st.button("⏹  Stop agent",
                                  disabled=not running, type="secondary")
        with ac4:
            report_btn = st.button("📊  Send report now",
                                   help="Trigger today's daily report immediately")

        if start_btn:
            cmd = [sys.executable, '-m', 'agent.main']
            if mode == 'single round':
                cmd.append('--round-once')
            elif mode == 'dry run (no backtest)':
                cmd.append('--dry-run')
            try:
                creationflags = 0
                if os.name == 'nt':
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_HERE),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                st.session_state.agent_proc_pid   = proc.pid
                st.session_state.agent_started_at = pd.Timestamp.utcnow().strftime('%H:%M UTC')
                st.success(f"Agent started (PID {proc.pid}, mode={mode})")
                time.sleep(1)
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to start agent: {exc}")

        if stop_btn and st.session_state.agent_proc_pid:
            _kill_agent(st.session_state.agent_proc_pid)
            st.session_state.agent_proc_pid   = None
            st.session_state.agent_started_at = None
            st.success("Agent stopped.")
            time.sleep(1)
            st.rerun()

        if report_btn:
            try:
                subprocess.Popen(
                    [sys.executable, '-m', 'agent.main', '--report-now'],
                    cwd=str(_HERE),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                st.success("Report send triggered (check Telegram).")
            except Exception as exc:
                st.error(f"Report trigger failed: {exc}")

        st.markdown("---")

        # ── Recent results from agent_results.db ────────────────────────────
        st.subheader("Recent strategies tested")
        try:
            con = sqlite3.connect(AGENT_DB_PATH)
            recent = pd.read_sql(
                """
                SELECT created_at, strategy_name, session, behaviour_type,
                       verdict, composite_score, test_sharpe, dsr, n_trades
                FROM tested_strategies
                ORDER BY created_at DESC
                LIMIT 25
                """, con)
            survivor_ct = pd.read_sql(
                "SELECT COUNT(*) AS n FROM tested_strategies WHERE verdict='survivor'", con
            )['n'].iloc[0]
            total_ct = pd.read_sql("SELECT COUNT(*) AS n FROM tested_strategies", con)['n'].iloc[0]
            con.close()

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total tested",   int(total_ct))
            mc2.metric("Survivors",      int(survivor_ct))
            mc3.metric("Survivor rate",
                       f"{(survivor_ct / max(total_ct,1) * 100):.1f}%")

            if recent.empty:
                st.caption("No strategies in agent_results.db yet.")
            else:
                def _color_v(v):
                    return {
                        'survivor':  'background-color:#1a7f2e;color:white',
                        'rejected':  'color:#9e9e9e',
                        'failed':    'background-color:#7f1a1a;color:white',
                    }.get(v, '')

                disp = recent.copy()
                for col in ('composite_score', 'test_sharpe', 'dsr'):
                    if col in disp.columns:
                        disp[col] = disp[col].apply(
                            lambda v: f"{v:.2f}" if pd.notna(v) else '—')
                st.dataframe(
                    disp.style.map(_color_v,
                                        subset=['verdict'] if 'verdict' in disp.columns else []),
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                )
        except Exception as exc:
            st.warning(f"Could not query agent_results.db: {exc}")

        st.markdown("---")

        # ── Recent log tail ─────────────────────────────────────────────────
        st.subheader("Recent log output")
        try:
            log_path = Path(LOG_PATH)
            if log_path.exists():
                lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
                tail = '\n'.join(lines[-80:])
                st.code(tail or '(log empty)', language='text')
            else:
                st.caption(f"Log file not found at {log_path}. Will be created when "
                           "the agent first runs.")
        except Exception as exc:
            st.warning(f"Could not read log: {exc}")

        st.markdown("---")

        # ── Recent reports ──────────────────────────────────────────────────
        st.subheader("Generated reports")
        if not _REPORTS_DIR.exists():
            st.caption(f"No reports yet. They'll appear in `{_REPORTS_DIR}` once "
                       "survivors are found.")
        else:
            report_files = sorted(_REPORTS_DIR.rglob('*.md'),
                                  key=lambda p: p.stat().st_mtime, reverse=True)[:25]
            if not report_files:
                st.caption("No markdown reports yet.")
            else:
                for f in report_files:
                    rel = f.relative_to(_REPORTS_DIR)
                    mtime = pd.Timestamp(f.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')
                    with st.expander(f"📄  {rel}   _({mtime})_"):
                        st.markdown(f.read_text(encoding='utf-8', errors='replace'))


# =============================================================================
# 📉 Live Backtest — single-strategy inspection with real-time equity curve
# =============================================================================
with tab_live_bt:
    st.header("📉  Live Backtest")
    _help_box(
        "Inspection tool: run ONE strategy on ONE pair with ONE param combo "
        "and watch the equity curve update as trades happen. Single-threaded "
        "(no worker pool), ~10-15s per run on 5-month test split. Bit-identical "
        "to the engine's normal run_backtest output. <br>"
        "Use this to spot pathological behaviour (e.g. all trades same side, "
        "huge drawdown spike) that bulk sweep metrics hide."
    )

    if "train_dfs" not in st.session_state or not st.session_state.train_dfs:
        st.warning("Load data first (📥 Data tab).")
    else:
        try:
            from edge_hypotheses import SWEEPS as _SWEEPS_FOR_LIVE
        except Exception as _e:
            _SWEEPS_FOR_LIVE = {}
            st.error(f"Couldn't import SWEEPS: {_e}")

        # ── State ────────────────────────────────────────────────────────────
        if "live_bt" not in st.session_state:
            st.session_state.live_bt = {
                "queue":       queue.Queue(),
                "thread":      None,
                "equity":      [],
                "trades":      [],
                "progress":    0.0,
                "balance":     None,
                "started_at":  None,
                "done":        False,
                "error":       None,
            }
        s = st.session_state.live_bt

        # ── Inputs ───────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sweep_key = st.selectbox("Strategy (SWEEPS key)",
                                      list(_SWEEPS_FOR_LIVE.keys()),
                                      key="live_bt_sweep")
        with c2:
            available_pairs = list(st.session_state.train_dfs.keys())
            chosen_pair = st.selectbox("Pair", available_pairs, key="live_bt_pair")
        with c3:
            split = st.selectbox("Split", ["train", "test"], key="live_bt_split")

        # Param combo selector
        sweep = _SWEEPS_FOR_LIVE.get(sweep_key)
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

        # ── Run button ───────────────────────────────────────────────────────
        thread_alive = s["thread"] is not None and s["thread"].is_alive()
        if st.button("▶  Run live backtest", disabled=thread_alive or sweep is None):
            # Reset state
            s["queue"] = queue.Queue()
            s["equity"] = [eng.INITIAL_BALANCE]
            s["trades"] = []
            s["progress"] = 0.0
            s["balance"] = eng.INITIAL_BALANCE
            s["started_at"] = time.time()
            s["done"] = False
            s["error"] = None

            params = param_combos[combo_idx]
            entry_fn = sweep["entry_fn"]
            manager_fn = sweep["manager_fn"]
            session_name = sweep["session"]
            regime_mult = sweep["regime_mult"]
            slot_class = f"livebt_{sweep_key[:14]}".replace("-", "_").lower()
            registry = [{
                "id": f"livebt_{sweep_key}",
                "family": "session_based",
                "slot_class": slot_class,
                "pairs": [chosen_pair],
                "session": session_name,
                "allow_concurrent": False,
                "regime_mult": regime_mult,
                "params": params,
            }]
            slot_managers = {slot_class: manager_fn}
            slot_entries  = {slot_class: entry_fn}
            data_dict = (st.session_state.train_dfs if split == "train"
                         else st.session_state.test_dfs)
            subset = {chosen_pair: data_dict[chosen_pair]}

            # session_hours from session_router if available
            try:
                from agent.session_router import _LIVE_SCHEDULE as _SCHED
                _hours = next(((lo, hi - 1) for lo, hi, n, *_ in _SCHED
                               if n == session_name), None)
            except Exception:
                _hours = None

            # Get cost_mult
            try:
                from agent.config import COST_MULT as _COST_MULT
            except Exception:
                _COST_MULT = 0.5

            q = s["queue"]
            def _cb(ev):
                try:
                    q.put_nowait(ev)
                except Exception:
                    pass

            def _worker():
                try:
                    eng.run_backtest(
                        subset_dfs        = subset,
                        spread_override   = None,
                        slippage_override = None,
                        registry          = registry,
                        slot_managers     = slot_managers,
                        slot_entries      = slot_entries,
                        cost_mult         = _COST_MULT,
                        session_hours     = _hours,
                        progress_callback = _cb,
                    )
                except Exception as _ex:
                    q.put({"type": "error", "error": f"{type(_ex).__name__}: {_ex}"})
                finally:
                    q.put({"type": "done"})

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            s["thread"] = t

        # ── Drain queue ──────────────────────────────────────────────────────
        while True:
            try:
                ev = s["queue"].get_nowait()
            except queue.Empty:
                break
            etype = ev.get("type")
            if etype == "trade_closed":
                s["balance"] = ev["balance"]
                s["equity"].append(ev["balance"])
                s["trades"].append(ev["trade"])
            elif etype == "tick":
                total = max(int(ev.get("total_bars", 1)), 1)
                s["progress"] = ev.get("bar_idx", 0) / total
            elif etype == "error":
                s["error"] = ev.get("error", "unknown")
            elif etype == "done":
                s["done"] = True
                s["progress"] = 1.0

        # ── Render ───────────────────────────────────────────────────────────
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Trades", len(s["trades"]))
        col_b.metric("Balance",
                     f"${s['balance']:.0f}" if s["balance"] is not None else "—")
        col_c.metric("Progress", f"{s['progress']*100:.0f}%")
        col_d.metric("Status",
                     "Running" if (s["thread"] and s["thread"].is_alive())
                     else ("Done" if s["done"] else "Idle"))

        st.progress(min(max(s["progress"], 0.0), 1.0))

        if s["error"]:
            st.error(f"Backtest error: {s['error']}")

        if s["equity"] and len(s["equity"]) > 1:
            equity_df = pd.DataFrame({"equity": s["equity"]})
            st.line_chart(equity_df, height=300)
        else:
            st.caption("Equity curve will appear once trades start closing.")

        if s["trades"]:
            st.subheader(f"Latest {min(20, len(s['trades']))} trades")
            trades_tail = pd.DataFrame(s["trades"][-20:])
            st.dataframe(trades_tail, use_container_width=True, height=300)

        # Live timer
        if s["started_at"] and (s["thread"] and s["thread"].is_alive()):
            elapsed = time.time() - s["started_at"]
            st.caption(f"Running for {elapsed:.0f}s — {len(s['trades'])} trades so far")
        elif s["done"] and s["started_at"]:
            total_elapsed = time.time() - s["started_at"]
            st.success(f"Complete in {total_elapsed:.1f}s — "
                       f"{len(s['trades'])} trades, "
                       f"final balance ${s['balance']:.2f}, "
                       f"PnL ${s['balance'] - eng.INITIAL_BALANCE:+.2f}")

        # Auto-refresh while backtest is running
        if s["thread"] and s["thread"].is_alive():
            if _AUTOREFRESH_AVAILABLE:
                st_autorefresh(interval=500, key="live_bt_refresh")
            else:
                st.caption("(Install streamlit-autorefresh for auto-updating: "
                           "`pip install streamlit-autorefresh`. Otherwise rerun the page.)")
