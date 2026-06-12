# -*- coding: utf-8 -*-
"""Live Trader Dashboard — Streamlit GUI for MT5live.2.py

Run:  streamlit run live_gui.py
Reads live data from MT5 terminal + CSV logs written by the live trader.
Refreshes automatically every 10 seconds.
"""

import os
import csv
import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv

# ── MT5 connection ────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN", 0))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER", "")

MT5_OK = False
try:
    import MetaTrader5 as mt5
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        MT5_OK = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
except Exception:
    mt5 = None

LONDON_TZ = pytz.timezone("Europe/London")

# ── Constants (mirror MT5live.2.py) ──────────────────────────────────────────
PAIR_PIP_SIZE = {
    "GBPUSD": 0.0001, "GBPJPY": 0.010, "EURUSD": 0.0001,
    "USDJPY": 0.010,  "EURJPY": 0.010, "XAUUSD": 1.00,
}
PAIR_DECIMALS = {
    "GBPUSD": 5, "GBPJPY": 3, "EURUSD": 5,
    "USDJPY": 3, "EURJPY": 3, "XAUUSD": 2,
}
PAIR_SESSION_LABEL = {
    "GBPUSD": "London", "GBPJPY": "London", "EURJPY": "London",
    "EURUSD": "NY",     "USDJPY": "NY",     "XAUUSD": "NY",
}
LOG_FILE      = Path(__file__).parent / "live_trade_log.csv"
EXEC_LOG_FILE = Path(__file__).parent / "execution_quality.csv"
FADE_LOG_FILE = Path(__file__).parent / "fade_trade_log.csv"

PROP_DAILY_LIMIT   = 0.05
PROP_MAX_DRAWDOWN  = 0.06

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_secs = st.slider("Interval (s)", 5, 60, 10)
    st.divider()
    st.caption(f"Last update: {datetime.now(LONDON_TZ).strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Now"):
        st.rerun()

if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()

# ── Data helpers ──────────────────────────────────────────────────────────────
def london_now():
    return datetime.now(pytz.utc).astimezone(LONDON_TZ)

@st.cache_data(ttl=10)
def read_csv_log(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()

def get_account():
    if not MT5_OK:
        return None
    info = mt5.account_info()
    return info

def get_positions():
    if not MT5_OK:
        return []
    pos = mt5.positions_get()
    return list(pos) if pos else []

def get_orders():
    if not MT5_OK:
        return []
    orders = mt5.orders_get()
    return list(orders) if orders else []

def get_tick(symbol):
    if not MT5_OK:
        return None
    return mt5.symbol_info_tick(symbol)

def pnl_color(v):
    if v is None:
        return "gray"
    return "green" if v >= 0 else "red"

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("📈 Live Trader Dashboard")

tab_dash, tab_trades, tab_exec, tab_risk = st.tabs([
    "Dashboard", "Trade History", "Execution Quality", "Risk Monitor"
])

# =============================================================================
# TAB 1 — DASHBOARD
# =============================================================================
with tab_dash:

    # Connection status
    col_conn, col_time = st.columns([3, 1])
    with col_conn:
        if MT5_OK:
            info = mt5.terminal_info()
            connected = info is not None and info.connected
            st.success("MT5 Connected" if connected else "MT5 Terminal Disconnected")
        else:
            st.error("MT5 not available — check credentials in .env")

    with col_time:
        now = london_now()
        st.metric("London Time", now.strftime("%H:%M:%S"))

    st.divider()

    # ── Account metrics ──────────────────────────────────────────────────────
    acct = get_account()
    if acct:
        balance  = acct.balance
        equity   = acct.equity
        floating = equity - balance
        margin   = acct.margin
        free_m   = acct.margin_free

        # Estimate daily loss vs starting balance (use equity low as proxy)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Balance",     f"${balance:,.2f}")
        c2.metric("Equity",      f"${equity:,.2f}",
                  delta=f"{floating:+.2f}",
                  delta_color="normal")
        c3.metric("Floating P&L", f"${floating:+.2f}",
                  delta_color="normal")
        c4.metric("Margin Used", f"${margin:,.2f}")
        c5.metric("Free Margin", f"${free_m:,.2f}")

        # Prop firm guardrails
        st.divider()
        col_dd, col_dl = st.columns(2)
        with col_dd:
            max_dd_level = balance * (1 - PROP_MAX_DRAWDOWN)
            dd_used      = max(0, balance - equity) / balance
            st.caption("Max Drawdown Limit (6%)")
            st.progress(min(dd_used / PROP_MAX_DRAWDOWN, 1.0),
                        text=f"{dd_used:.1%} used  (floor ${max_dd_level:,.0f})")
        with col_dl:
            daily_trades = read_csv_log(LOG_FILE)
            if not daily_trades.empty and "date" in daily_trades.columns:
                today_str    = london_now().strftime("%Y-%m-%d")
                today_trades = daily_trades[daily_trades["date"] == today_str]
                day_pnl      = today_trades["pnl"].astype(float).sum() if not today_trades.empty and "pnl" in today_trades.columns else 0.0
            else:
                day_pnl = 0.0
            dl_used = max(0, -day_pnl) / (balance * PROP_DAILY_LIMIT) if balance > 0 else 0
            st.caption("Daily Loss Limit (5%)")
            st.progress(min(dl_used, 1.0),
                        text=f"{max(0,-day_pnl):.2f} used of {balance*PROP_DAILY_LIMIT:.2f}")
    else:
        st.warning("Account data unavailable")

    st.divider()

    # ── Live pair status ─────────────────────────────────────────────────────
    st.subheader("Pair Status")

    positions = get_positions()
    orders    = get_orders()

    pos_by_sym   = {p.symbol: p for p in positions}
    order_by_sym = {o.symbol: o for o in orders}

    pairs = ["GBPUSD", "GBPJPY", "EURUSD", "USDJPY", "EURJPY", "XAUUSD"]
    rows  = []
    for sym in pairs:
        tick = get_tick(sym)
        bid  = tick.bid if tick else None
        ask  = tick.ask if tick else None
        mid  = (bid + ask) / 2 if bid and ask else None
        spd  = (ask - bid) / PAIR_PIP_SIZE[sym] if bid and ask else None
        dec  = PAIR_DECIMALS[sym]
        pip  = PAIR_PIP_SIZE[sym]

        if sym in pos_by_sym:
            pos       = pos_by_sym[sym]
            direction = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
            entry     = pos.price_open
            sl        = pos.sl
            tp        = pos.tp
            pnl_usd   = pos.profit
            pnl_pips  = ((mid - entry) / pip if direction == "LONG" else (entry - mid) / pip) if mid else None
            phase     = "IN_TRADE"
        elif sym in order_by_sym:
            o         = order_by_sym[sym]
            direction = "LONG" if o.type == mt5.ORDER_TYPE_BUY_STOP else "SHORT"
            entry     = o.price_open
            sl        = o.sl
            tp        = o.tp
            pnl_usd   = None
            pnl_pips  = None
            phase     = "ORDER_PLACED"
        else:
            direction = "—"
            entry = sl = tp = pnl_usd = pnl_pips = None
            phase = "IDLE"

        rows.append({
            "Pair":      sym,
            "Session":   PAIR_SESSION_LABEL[sym],
            "Phase":     phase,
            "Direction": direction,
            "Entry":     f"{entry:.{dec}f}" if entry else "—",
            "Current":   f"{mid:.{dec}f}"   if mid   else "—",
            "P&L $":     f"{pnl_usd:+.2f}"  if pnl_usd is not None else "—",
            "P&L pips":  f"{pnl_pips:+.1f}" if pnl_pips is not None else "—",
            "SL":        f"{sl:.{dec}f}"     if sl    else "—",
            "TP":        f"{tp:.{dec}f}"     if tp    else "—",
            "Spread":    f"{spd:.1f}"        if spd is not None else "—",
        })

    df_status = pd.DataFrame(rows)

    def phase_style(val):
        colors = {"IN_TRADE": "background-color:#1a4a1a;color:#7fff7f",
                  "ORDER_PLACED": "background-color:#3a3a10;color:#ffff80",
                  "IDLE": ""}
        return colors.get(val, "")

    def pnl_style(val):
        if val == "—":
            return ""
        try:
            return "color:#7fff7f" if float(val.replace("+","")) >= 0 else "color:#ff6666"
        except Exception:
            return ""

    styled = (df_status.style
              .applymap(phase_style, subset=["Phase"])
              .applymap(pnl_style,   subset=["P&L $", "P&L pips"]))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Today's trades summary ───────────────────────────────────────────────
    st.divider()
    st.subheader("Today's Trades")
    trades_df = read_csv_log(LOG_FILE)
    if not trades_df.empty and "date" in trades_df.columns:
        today_str = london_now().strftime("%Y-%m-%d")
        today_df  = trades_df[trades_df["date"] == today_str].copy()
        if not today_df.empty:
            today_df["pnl"] = pd.to_numeric(today_df["pnl"], errors="coerce")
            wins   = (today_df["pnl"] > 0).sum()
            losses = (today_df["pnl"] < 0).sum()
            total  = len(today_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades",   total)
            c2.metric("Wins",     wins)
            c3.metric("Losses",   losses)
            c4.metric("Day P&L",  f"${today_df['pnl'].sum():+.2f}",
                      delta_color="normal")
            st.dataframe(today_df[["pair","direction","entry","exit_price","pnl",
                                   "exit_reason","breakeven_set","profit_lock_set"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No trades closed today yet.")
    else:
        st.info("No trade log found.")

# =============================================================================
# TAB 2 — TRADE HISTORY
# =============================================================================
with tab_trades:
    trades_df = read_csv_log(LOG_FILE)

    if trades_df.empty:
        st.info("No trade history found. Trades will appear here after the first close.")
    else:
        trades_df["pnl"]  = pd.to_numeric(trades_df["pnl"],  errors="coerce")
        trades_df["date"] = pd.to_datetime(trades_df["date"], errors="coerce")

        # Filters
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            pairs_filter = st.multiselect("Pair", options=trades_df["pair"].unique().tolist(),
                                          default=trades_df["pair"].unique().tolist())
        with cf2:
            dir_filter = st.multiselect("Direction", options=["long","short"],
                                        default=["long","short"])
        with cf3:
            days_back = st.slider("Days back", 7, 365, 30)

        cutoff = pd.Timestamp(london_now()) - pd.Timedelta(days=days_back)
        mask   = (trades_df["pair"].isin(pairs_filter) &
                  trades_df["direction"].isin(dir_filter) &
                  (trades_df["date"] >= cutoff))
        fdf    = trades_df[mask].copy()

        # Summary stats
        if not fdf.empty:
            n      = len(fdf)
            wins   = (fdf["pnl"] > 0).sum()
            losses = (fdf["pnl"] < 0).sum()
            wr     = wins / n if n > 0 else 0
            avg_w  = fdf.loc[fdf["pnl"] > 0, "pnl"].mean() if wins > 0 else 0
            avg_l  = fdf.loc[fdf["pnl"] < 0, "pnl"].mean() if losses > 0 else 0
            pf     = abs(avg_w * wins / (avg_l * losses)) if losses > 0 and avg_l != 0 else float("inf")
            total  = fdf["pnl"].sum()

            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Trades",         n)
            c2.metric("Win Rate",       f"{wr:.1%}")
            c3.metric("Avg Win",        f"${avg_w:.2f}")
            c4.metric("Avg Loss",       f"${avg_l:.2f}")
            c5.metric("Profit Factor",  f"{pf:.2f}" if pf != float('inf') else "∞")
            c6.metric("Total P&L",      f"${total:+.2f}", delta_color="normal")

            st.divider()

            # Daily P&L bar chart
            daily = fdf.groupby(fdf["date"].dt.date)["pnl"].sum().reset_index()
            daily.columns = ["date", "pnl"]
            daily["color"] = daily["pnl"].apply(lambda x: "green" if x >= 0 else "red")

            st.subheader("Daily P&L")
            st.bar_chart(daily.set_index("date")["pnl"])

            # Rolling win rate
            st.subheader("Rolling 10-Trade Win Rate")
            fdf_sorted = fdf.sort_values("date").reset_index(drop=True)
            fdf_sorted["win"]     = (fdf_sorted["pnl"] > 0).astype(int)
            fdf_sorted["roll_wr"] = fdf_sorted["win"].rolling(10, min_periods=3).mean()
            fdf_sorted["trade_n"] = range(1, len(fdf_sorted)+1)
            st.line_chart(fdf_sorted.set_index("trade_n")["roll_wr"])

            # By pair breakdown
            st.subheader("By Pair")
            by_pair = fdf.groupby("pair").agg(
                trades=("pnl","count"),
                wins=("pnl", lambda x: (x>0).sum()),
                total_pnl=("pnl","sum"),
                avg_pnl=("pnl","mean"),
            ).reset_index()
            by_pair["win_rate"] = (by_pair["wins"] / by_pair["trades"]).map("{:.1%}".format)
            by_pair["total_pnl"] = by_pair["total_pnl"].map("${:+.2f}".format)
            by_pair["avg_pnl"]   = by_pair["avg_pnl"].map("${:+.2f}".format)
            st.dataframe(by_pair, use_container_width=True, hide_index=True)

            # Exit reason breakdown
            st.subheader("By Exit Reason")
            by_exit = fdf.groupby("exit_reason").agg(
                trades=("pnl","count"),
                total_pnl=("pnl","sum"),
                avg_pnl=("pnl","mean"),
            ).reset_index()
            by_exit["total_pnl"] = by_exit["total_pnl"].map("${:+.2f}".format)
            by_exit["avg_pnl"]   = by_exit["avg_pnl"].map("${:+.2f}".format)
            st.dataframe(by_exit, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Full Trade Log")
            st.dataframe(fdf.sort_values("date", ascending=False),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No trades match the current filters.")

# =============================================================================
# TAB 3 — EXECUTION QUALITY
# =============================================================================
with tab_exec:
    exec_df = read_csv_log(EXEC_LOG_FILE)

    if exec_df.empty:
        st.info("No execution quality data yet. Fills are logged here as trades are placed.")
    else:
        exec_df["slippage_pips"] = pd.to_numeric(exec_df["slippage_pips"], errors="coerce")
        exec_df["spread_pips"]   = pd.to_numeric(exec_df["spread_pips"],   errors="coerce")
        exec_df["expected_fill"] = pd.to_numeric(exec_df["expected_fill"], errors="coerce")
        exec_df["actual_fill"]   = pd.to_numeric(exec_df["actual_fill"],   errors="coerce")

        n = len(exec_df)
        avg_slip = exec_df["slippage_pips"].mean()
        avg_spd  = exec_df["spread_pips"].mean()
        max_slip = exec_df["slippage_pips"].max()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Fills",      n)
        c2.metric("Avg Slippage",     f"{avg_slip:.2f} pips")
        c3.metric("Avg Spread @ Fill",f"{avg_spd:.2f} pips")
        c4.metric("Worst Slippage",   f"{max_slip:.2f} pips")

        st.divider()

        # Slippage per pair
        st.subheader("Average Slippage by Pair")
        slip_by_pair = exec_df.groupby("pair")["slippage_pips"].agg(["mean","max","count"]).reset_index()
        slip_by_pair.columns = ["pair","avg_slip","max_slip","fills"]
        st.bar_chart(slip_by_pair.set_index("pair")["avg_slip"])
        st.dataframe(slip_by_pair, use_container_width=True, hide_index=True)

        # Spread at fill vs expected
        st.subheader("Spread at Fill by Pair")
        spd_by_pair = exec_df.groupby("pair")["spread_pips"].agg(["mean","max"]).reset_index()
        spd_by_pair.columns = ["pair","avg_spread","max_spread"]

        # Annotate with normal spread
        normal_spreads = {
            "GBPUSD": 1.5, "GBPJPY": 2.5, "EURUSD": 1.2,
            "USDJPY": 1.0, "EURJPY": 2.0, "XAUUSD": 30.0,
        }
        spd_by_pair["normal_spread"] = spd_by_pair["pair"].map(normal_spreads)
        spd_by_pair["vs_normal"]     = (spd_by_pair["avg_spread"] / spd_by_pair["normal_spread"]).map("{:.2f}×".format)
        st.dataframe(spd_by_pair, use_container_width=True, hide_index=True)

        # Slippage distribution
        st.subheader("Slippage Distribution")
        hist_data = exec_df["slippage_pips"].dropna()
        if not hist_data.empty:
            bins = pd.cut(hist_data, bins=10)
            hist = hist_data.groupby(bins, observed=True).count().reset_index()
            hist.columns = ["range","count"]
            hist["range"] = hist["range"].astype(str)
            st.bar_chart(hist.set_index("range")["count"])

        st.divider()
        st.subheader("Full Execution Log")
        st.dataframe(exec_df.sort_values(["date","time"], ascending=False),
                     use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4 — RISK MONITOR
# =============================================================================
with tab_risk:
    trades_df = read_csv_log(LOG_FILE)

    if trades_df.empty:
        st.info("No trade data available for risk analysis.")
    else:
        trades_df["pnl"]  = pd.to_numeric(trades_df["pnl"],  errors="coerce")
        trades_df["date"] = pd.to_datetime(trades_df["date"], errors="coerce")
        trades_df = trades_df.sort_values("date").reset_index(drop=True)

        # Equity curve (reconstructed from initial balance estimate)
        acct    = get_account()
        balance = acct.balance if acct else None

        if balance:
            cumulative_pnl = trades_df["pnl"].cumsum()
            start_balance  = balance - cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else balance
            trades_df["equity"] = start_balance + cumulative_pnl

            st.subheader("Equity Curve")
            st.line_chart(trades_df.set_index(trades_df.index)["equity"])

            # Drawdown
            peak = trades_df["equity"].cummax()
            dd   = (trades_df["equity"] - peak) / peak
            trades_df["drawdown_pct"] = dd * 100

            st.subheader("Drawdown %")
            st.area_chart(trades_df.set_index(trades_df.index)["drawdown_pct"])

            max_dd = dd.min()
            c1, c2, c3 = st.columns(3)
            c1.metric("Max Drawdown",     f"{max_dd:.2%}")
            c2.metric("Prop DD Limit",    f"{PROP_MAX_DRAWDOWN:.0%}")
            c3.metric("Remaining Buffer", f"{PROP_MAX_DRAWDOWN + max_dd:.2%}",
                      delta_color="inverse")

        st.divider()

        # Daily P&L with loss limit overlay
        st.subheader("Daily P&L vs Loss Limit")
        daily = trades_df.groupby(trades_df["date"].dt.date)["pnl"].sum().reset_index()
        daily.columns = ["date","pnl"]
        st.bar_chart(daily.set_index("date")["pnl"])

        # Consecutive losses
        st.subheader("Streak Analysis")
        trades_df["win"] = (trades_df["pnl"] > 0).astype(int)
        streak = 0
        max_loss_streak = 0
        cur_loss_streak = 0
        for w in trades_df["win"]:
            if w == 0:
                cur_loss_streak += 1
                max_loss_streak = max(max_loss_streak, cur_loss_streak)
            else:
                cur_loss_streak = 0

        # Current streak (from end of trades)
        cur_streak_val = 0
        cur_streak_dir = "—"
        for w in reversed(trades_df["win"].tolist()):
            if cur_streak_val == 0:
                cur_streak_dir = "Win" if w == 1 else "Loss"
            if (w == 1 and cur_streak_dir == "Win") or (w == 0 and cur_streak_dir == "Loss"):
                cur_streak_val += 1
            else:
                break

        c1, c2, c3 = st.columns(3)
        c1.metric("Max Loss Streak",    max_loss_streak)
        c2.metric("Current Streak",     f"{cur_streak_val} {cur_streak_dir}s",
                  delta_color="inverse" if cur_streak_dir == "Loss" else "normal")
        c3.metric("Loss Pause Trigger", "3 losses")

        # Monthly breakdown
        st.divider()
        st.subheader("Monthly Summary")
        trades_df["month"] = trades_df["date"].dt.to_period("M").astype(str)
        monthly = trades_df.groupby("month").agg(
            trades=("pnl","count"),
            wins=("pnl", lambda x: (x>0).sum()),
            total_pnl=("pnl","sum"),
        ).reset_index()
        monthly["win_rate"]  = (monthly["wins"] / monthly["trades"]).map("{:.1%}".format)
        monthly["total_pnl"] = monthly["total_pnl"].map("${:+.2f}".format)
        st.dataframe(monthly, use_container_width=True, hide_index=True)
