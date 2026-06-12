"""
Read-only analytics for promoted survivor strategies. Powers the GUI
"Live Strategies" tab and CLI inspection. All functions take a strategy
name and return either a DataFrame, a metrics dict, or a matplotlib
Figure suitable for direct embedding.

The data flows through here unchanged from `live_trades` — the live bridge
writes once, every consumer reads from this module.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path

import pandas as pd

from agent.config import AGENT_DB_PATH

log = logging.getLogger(__name__)


# ── Loaders ──────────────────────────────────────────────────────────────────

def list_strategies() -> list[str]:
    """All strategies that have at least one row in live_trades OR appear in
    live_promoted. Sorted alphabetically. Empty list if neither table exists."""
    if not Path(AGENT_DB_PATH).exists():
        return []
    names = set()
    con = sqlite3.connect(AGENT_DB_PATH)
    try:
        for row in con.execute(
            "SELECT DISTINCT strategy_name FROM live_promoted"
        ).fetchall():
            if row[0]:
                names.add(row[0])
    except Exception:
        pass
    try:
        for row in con.execute(
            "SELECT DISTINCT strategy_name FROM live_trades"
        ).fetchall():
            if row[0]:
                names.add(row[0])
    except Exception:
        pass
    con.close()
    return sorted(names)


def load_live_trades(strategy_name: str | None = None) -> pd.DataFrame:
    """Return all live_trades rows for one strategy (or all rows if None).
    Includes a `pnl` alias column so the existing equity-curve renderer works."""
    if not Path(AGENT_DB_PATH).exists():
        return pd.DataFrame()
    con = sqlite3.connect(AGENT_DB_PATH)
    try:
        if strategy_name:
            df = pd.read_sql(
                "SELECT * FROM live_trades WHERE strategy_name = ? "
                "ORDER BY ts_open",
                con, params=(strategy_name,),
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM live_trades ORDER BY ts_open", con,
            )
    except Exception as e:
        log.warning("[live_analytics] load_live_trades failed: %s", e)
        df = pd.DataFrame()
    finally:
        con.close()
    if df.empty:
        return df
    # Alias pnl_usd -> pnl so plot_equity_figure / extended_risk_metrics work
    if 'pnl_usd' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['pnl_usd']
    if 'ts_close' in df.columns and 'exit_time' not in df.columns:
        df['exit_time'] = df['ts_close']
    return df


def closed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out positions that haven't closed yet (no ts_close, no pnl)."""
    if df.empty:
        return df
    if 'ts_close' in df.columns:
        df = df[df['ts_close'].notna()]
    if 'pnl' in df.columns:
        df = df[df['pnl'].notna()]
    return df


# ── Metrics ──────────────────────────────────────────────────────────────────

def live_metrics(strategy_name: str) -> dict:
    """Live-only summary: Sharpe (annualised, from daily PnL), win rate,
    expectancy in R, max DD in $, n_trades, by_mode breakdown."""
    df = closed_only(load_live_trades(strategy_name))
    out = {
        'strategy': strategy_name,
        'n_trades': 0,
        'win_rate': None,
        'avg_pnl_usd': None,
        'expectancy_r': None,
        'sharpe_annualised': None,
        'max_dd_usd': None,
        'gross_pnl_usd': None,
        'by_mode': {},
    }
    if df.empty:
        return out

    n = len(df)
    out['n_trades']      = n
    out['win_rate']      = round((df['pnl'] > 0).mean() * 100, 1)
    out['avg_pnl_usd']   = round(df['pnl'].mean(), 2)
    out['gross_pnl_usd'] = round(df['pnl'].sum(), 2)

    # Expectancy in R — average of pnl_r when populated
    if 'pnl_r' in df.columns:
        r_series = df['pnl_r'].dropna()
        if not r_series.empty:
            out['expectancy_r'] = round(r_series.mean(), 3)

    # Annualised Sharpe from daily PnL
    if 'ts_close' in df.columns:
        try:
            ts = pd.to_datetime(df['ts_close'], errors='coerce', utc=True)
            daily = (df.assign(_d=ts.dt.date)
                       .groupby('_d')['pnl'].sum())
            if len(daily) >= 2 and daily.std() > 0:
                out['sharpe_annualised'] = round(
                    daily.mean() / daily.std() * math.sqrt(252), 2)
        except Exception:
            pass

    # Max drawdown in $
    eq        = df['pnl'].cumsum()
    peak      = eq.cummax()
    drawdown  = (eq - peak)
    out['max_dd_usd'] = round(float(drawdown.min()), 2) if len(drawdown) else None

    # By-mode breakdown — separate SHADOW (paper) from real-money tiers.
    if 'mode' in df.columns:
        by_mode = {}
        for m, sub in df.groupby('mode'):
            by_mode[str(m)] = {
                'n':           int(len(sub)),
                'pnl_usd':     round(float(sub['pnl'].sum()), 2),
                'win_rate':    round(float((sub['pnl'] > 0).mean()) * 100, 1),
            }
        out['by_mode'] = by_mode

    return out


# ── TCA summary (decay + slippage) ───────────────────────────────────────────

def tca_summary(strategy_name: str) -> dict:
    """Live-vs-backtest decay + execution-quality summary for one strategy.
    Best-effort: returns whatever's available, missing fields are None."""
    out = {
        'strategy':      strategy_name,
        'decay':         None,
        'verdict':       None,
        'avg_slip_open': None,
        'avg_slip_close': None,
    }
    try:
        from agent.tca import per_strategy_decay
        decay_df = per_strategy_decay()
        if not decay_df.empty:
            sub = decay_df[decay_df['strategy_name'] == strategy_name]
            if not sub.empty:
                row = sub.iloc[0]
                d = row.get('decay')
                v = row.get('verdict')
                out['decay']   = float(d) if pd.notna(d) else None
                out['verdict'] = str(v) if v else None
    except Exception as e:
        log.debug("[live_analytics] decay lookup failed: %s", e)

    df = closed_only(load_live_trades(strategy_name))
    if not df.empty:
        if 'slip_open_pip' in df.columns:
            s = df['slip_open_pip'].dropna()
            if not s.empty:
                out['avg_slip_open'] = round(float(s.mean()), 3)
        if 'slip_close_pip' in df.columns:
            s = df['slip_close_pip'].dropna()
            if not s.empty:
                out['avg_slip_close'] = round(float(s.mean()), 3)
    return out


# ── Equity figure ────────────────────────────────────────────────────────────

def equity_curve(strategy_name: str):
    """Return a matplotlib Figure of the live equity + drawdown for one
    strategy. Wraps edge_engine.plot_equity_figure on the live trades DataFrame."""
    try:
        from edge_engine import plot_equity_figure
    except Exception as e:
        log.error("[live_analytics] could not import plot_equity_figure: %s", e)
        return None
    df = closed_only(load_live_trades(strategy_name))
    return plot_equity_figure(df, label=f"{strategy_name} (LIVE)")


# ── Promotion mode ───────────────────────────────────────────────────────────

def promotion_state(strategy_name: str) -> dict:
    """Pull the full promotion record, plus the latest TCA kill verdict if any."""
    try:
        from agent import db
        promo = db.get_promotion(strategy_name) or {}
        kill  = db.get_live_kill(strategy_name)
    except Exception as e:
        log.warning("[live_analytics] promotion_state failed: %s", e)
        promo, kill = {}, None
    return {
        'mode':          promo.get('mode'),
        'live_n':        promo.get('live_n'),
        'promoted_at':   promo.get('promoted_at'),
        'last_ramp_at':  promo.get('last_ramp_at'),
        'session':       promo.get('session'),
        'pairs':         promo.get('pairs'),
        'kill_verdict':  (kill or {}).get('verdict'),
        'kill_decay':    (kill or {}).get('decay'),
    }
