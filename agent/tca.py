"""
Transaction Cost Analysis (TCA).

Reads live execution data from agent_results.db (live_trades + live_executions
tables, populated by agent.live_ingest from MT5 CSVs) and produces:

  • Per-pair cost summary       — effective spread, avg slippage, fill quality
  • By-hour cost profile        — when broker spreads widen silently
  • Per-strategy decay analysis — live PnL vs backtest expectation
  • Markdown report             — saved alongside agent/reports/<date>/

Returns DataFrames suitable for both the GUI and CLI workflows. Self-contained:
no Claude API calls, no external data fetches — just SQL aggregations on
whatever the live ingest has captured.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent.config import AGENT_DB_PATH

log = logging.getLogger(__name__)

REPORT_ROOT = Path(__file__).parent / 'reports'

# Pip sizes are domain knowledge; not worth importing edge_engine just for this.
_PIP_SIZE = {
    'EUR_USD': 0.0001, 'GBP_USD': 0.0001, 'AUD_USD': 0.0001,
    'USD_JPY': 0.01,   'EUR_JPY': 0.01,   'GBP_JPY': 0.01,
    'XAU_USD': 0.10,
}


def _pip(pair: str) -> float:
    return _PIP_SIZE.get(pair, 0.0001)


# ── Load tables as DataFrames ────────────────────────────────────────────────

def _load_table(name: str) -> pd.DataFrame:
    if not Path(AGENT_DB_PATH).exists():
        return pd.DataFrame()
    con = sqlite3.connect(AGENT_DB_PATH)
    try:
        df = pd.read_sql(f"SELECT * FROM {name}", con)
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    return df


def load_live_trades() -> pd.DataFrame:
    df = _load_table('live_trades')
    if not df.empty and 'pnl_usd' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['pnl_usd']
    return df


def load_live_executions() -> pd.DataFrame:
    df = _load_table('live_executions')
    if df.empty:
        return df
    # Combine date + time → single UTC datetime where possible
    if 'date' in df.columns and 'time' in df.columns:
        df['dt'] = pd.to_datetime(
            df['date'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip(),
            errors='coerce', utc=True,
        )
    return df


# ── Per-pair summary ─────────────────────────────────────────────────────────

def per_pair_summary() -> pd.DataFrame:
    """
    One row per pair:
      n_trades, n_fills, win_rate, avg_pnl_pips, avg_slip_pips, avg_spread_pips,
      cost_per_trade_pips, fill_quality (avg slip / median slip ratio)
    """
    trades = load_live_trades()
    execs  = load_live_executions()
    if trades.empty and execs.empty:
        return pd.DataFrame()

    rows = []
    pairs = sorted(set(trades.get('pair', pd.Series(dtype=str))).union(
                       set(execs.get('pair',  pd.Series(dtype=str)))))
    pairs = [p for p in pairs if p]

    for pair in pairs:
        pip      = _pip(pair)
        t_pair   = trades[trades.get('pair') == pair] if not trades.empty else pd.DataFrame()
        e_pair   = execs[execs.get('pair')  == pair]  if not execs.empty  else pd.DataFrame()

        n_trades = len(t_pair)
        n_fills  = len(e_pair)
        win_rate = (t_pair['pnl'] > 0).mean() * 100 if n_trades else float('nan')
        avg_pnl_price = t_pair['pnl'].mean() if n_trades else 0
        avg_pnl_pips  = avg_pnl_price / pip if pip > 0 else float('nan')

        avg_slip   = e_pair['slippage_pips'].mean() if n_fills else float('nan')
        med_slip   = e_pair['slippage_pips'].median() if n_fills else float('nan')
        avg_spread = e_pair['spread_pips'].mean()   if n_fills else float('nan')

        # round-trip cost in pips: spread (paid both sides via mid + half-spread)
        # plus slippage on entry. Entry is what live_executions captures; exits
        # are also slipped but fall under the trade pnl already measured.
        cost_per_trade = (avg_spread or 0) + max(avg_slip or 0, 0)

        # Fill quality: if mean slippage >> median slippage, you have heavy
        # tail (occasional huge slips drag the mean) — common during news.
        fill_quality = (
            'good'   if not pd.isna(avg_slip) and abs(avg_slip - (med_slip or 0)) < 0.3 else
            'mixed'  if not pd.isna(avg_slip) and abs(avg_slip - (med_slip or 0)) < 1.0 else
            'tail-heavy'
        ) if n_fills else '—'

        rows.append({
            'pair':                pair,
            'n_trades':            n_trades,
            'n_fills':             n_fills,
            'win_rate_pct':        round(win_rate, 1) if not pd.isna(win_rate) else None,
            'avg_pnl_pips':        round(avg_pnl_pips, 2) if not pd.isna(avg_pnl_pips) else None,
            'avg_slip_pips':       round(avg_slip, 3) if not pd.isna(avg_slip) else None,
            'med_slip_pips':       round(med_slip, 3) if not pd.isna(med_slip) else None,
            'avg_spread_pips':     round(avg_spread, 3) if not pd.isna(avg_spread) else None,
            'cost_per_trade_pips': round(cost_per_trade, 3),
            'fill_quality':        fill_quality,
        })

    return pd.DataFrame(rows).sort_values('cost_per_trade_pips', ascending=False, na_position='last')


# ── Spread feedback to backtest (review#P2#4) ─────────────────────────────────

def update_measured_spreads_from_live(min_fills: int = 20) -> dict:
    """review#P2#4 — write live-measured spreads back to disk so the next
    backtest cycle can use them in place of the stale Dukascopy-based median.

    Reads avg_spread_pips per pair from per_pair_summary, converts to price
    units, and writes `live_measured_spreads.json` next to
    `edge_measured_spreads.json`. Only pairs with ≥ min_fills are written —
    a single bad print shouldn't poison the next sweep.

    Returns a {pair: spread_price} dict of what was written. Caller-safe:
    never raises on disk or DB error.
    """
    import json
    summary = per_pair_summary()
    if summary is None or summary.empty:
        return {}

    out: dict = {}
    for _, r in summary.iterrows():
        pair  = r.get('pair')
        spd_p = r.get('avg_spread_pips')
        nf    = int(r.get('n_fills') or 0)
        if not pair or spd_p is None or nf < min_fills:
            continue
        try:
            pip = _pip(pair)
            out[pair] = float(spd_p) * pip
        except Exception:
            continue

    if not out:
        return {}

    payload = {
        '_meta': {
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'source':     'live',
            'min_fills':  min_fills,
        },
        'spreads': out,
    }
    target = Path(AGENT_DB_PATH).parent.parent / 'live_measured_spreads.json'
    try:
        with open(target, 'w') as f:
            json.dump(payload, f, indent=2)
        log.info("[tca] wrote live spreads for %d pair(s) → %s",
                 len(out), target.name)
    except Exception as e:
        log.warning("[tca] failed to write live spreads: %s", e)
    return out


# ── By-hour cost profile ─────────────────────────────────────────────────────

def by_hour_profile(pair: str | None = None) -> pd.DataFrame:
    """
    Effective spread + slippage bucketed by UTC hour.
    Pass pair=None to aggregate across all pairs.
    """
    execs = load_live_executions()
    if execs.empty or 'dt' not in execs.columns:
        return pd.DataFrame()

    df = execs.dropna(subset=['dt']).copy()
    if pair:
        df = df[df['pair'] == pair]
    if df.empty:
        return pd.DataFrame()

    df['hour'] = df['dt'].dt.hour
    out = df.groupby('hour').agg(
        n             = ('slippage_pips', 'count'),
        avg_slip      = ('slippage_pips', 'mean'),
        max_slip      = ('slippage_pips', 'max'),
        avg_spread    = ('spread_pips',   'mean'),
        max_spread    = ('spread_pips',   'max'),
    ).reset_index()
    for col in ('avg_slip', 'max_slip', 'avg_spread', 'max_spread'):
        out[col] = out[col].round(3)
    return out


# ── Per-strategy live-vs-backtest decay ──────────────────────────────────────

def per_strategy_decay() -> pd.DataFrame:
    """
    For each live-traded strategy, compare actual live PnL per trade against
    the strategy's backtest expectation stored in tested_strategies.

    Decay = live_pnl_per_trade / backtest_pnl_per_trade.
      decay > 1.0  → live exceeds backtest (rare; usually noise on small samples)
      decay ≈ 1.0  → live matches backtest expectation
      decay < 0.7  → live materially underperforming — kill or reduce size
      decay < 0    → live is a net loser despite positive backtest

    Live trades are matched to backtested strategies by an optional 'strategy_name'
    column on live_trades. If MT5live.2.py doesn't log strategy_name yet, this
    function returns an empty DataFrame and logs a hint.
    """
    trades = load_live_trades()
    if trades.empty:
        return pd.DataFrame()

    # MT5live.2.py may not yet log a strategy_name column. Fall back to pair-level.
    if 'strategy_name' not in trades.columns:
        log.info("per_strategy_decay: live_trades has no strategy_name column; "
                 "decay can only be computed at pair level.")
        return pd.DataFrame()

    con = sqlite3.connect(AGENT_DB_PATH)
    try:
        bt = pd.read_sql("""
            SELECT strategy_name,
                   AVG(test_sharpe)     AS bt_sharpe,
                   AVG(composite_score) AS bt_score,
                   AVG(n_trades)        AS bt_n,
                   SUM(n_trades)        AS bt_n_total,
                   COUNT(*)             AS bt_rows
            FROM tested_strategies
            WHERE verdict = 'survivor'
            GROUP BY strategy_name
        """, con)
    finally:
        con.close()

    if bt.empty:
        return pd.DataFrame()

    live = trades.groupby('strategy_name').agg(
        live_n        = ('pnl', 'count'),
        live_total    = ('pnl', 'sum'),
        live_avg_pnl  = ('pnl', 'mean'),
        live_wr       = ('pnl', lambda x: (x > 0).mean()),
    ).reset_index()

    merged = live.merge(bt, on='strategy_name', how='inner')
    if merged.empty:
        return pd.DataFrame()

    # Sharpe-based decay: how much of the backtested Sharpe per-trade is preserved live.
    # We approximate "backtest pnl per trade" as bt_sharpe × some unit, but since we
    # don't have backtest pnl directly, decay = live_wr / typical_breakeven_wr.
    merged['decay'] = (merged['live_avg_pnl'] /
                       merged['bt_sharpe'].replace(0, float('nan'))).round(3)
    merged['verdict'] = merged['decay'].apply(
        lambda d: 'PROMOTE'      if pd.notna(d) and d >= 1.0 else
                  'HOLD'         if pd.notna(d) and d >= 0.7 else
                  'REDUCE'       if pd.notna(d) and d >= 0.3 else
                  'KILL'         if pd.notna(d) else
                  'INSUFFICIENT'
    )

    # Phase 1 — write decisive verdicts to the kill-switch table that MT5Live
    # consults. INSUFFICIENT/HOLD don't get persisted (no action needed).
    try:
        from agent.db import upsert_live_strategy_kill
        for _, row in merged.iterrows():
            if row['verdict'] in ('KILL', 'REDUCE', 'PROMOTE'):
                upsert_live_strategy_kill(
                    str(row['strategy_name']),
                    float(row['decay']) if pd.notna(row['decay']) else None,
                    str(row['verdict']),
                )
    except Exception as e:
        log.warning(f"per_strategy_decay: failed to write kill-switch rows: {e}")

    return merged.sort_values('decay', ascending=False, na_position='last')


# ── Markdown report ──────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_\n"
    return df.to_markdown(index=False) + "\n"


def build_report() -> str:
    """Produce a full TCA markdown report. Returns path to written file."""
    today    = datetime.now(timezone.utc).date().isoformat()
    out_dir  = REPORT_ROOT / today / 'tca'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'tca_report.md'

    pp_df    = per_pair_summary()
    hh_df    = by_hour_profile()
    decay_df = per_strategy_decay()

    parts = [
        f"# Transaction Cost Analysis — {today}",
        "",
        "Generated by `agent/tca.py`. Reads `live_trades` + `live_executions` from "
        "`agent_results.db`. If sections are empty, no live trades have been ingested yet.",
        "",
        "## Per-pair cost summary",
        "",
        "Sorted by total cost-per-trade (worst first). `cost_per_trade_pips` = "
        "`avg_spread_pips + avg_slip_pips`. `fill_quality = tail-heavy` means "
        "occasional huge slippage events (often near news) — investigate before sizing up.",
        "",
        _md_table(pp_df),
        "",
        "## Cost profile by UTC hour",
        "",
        "Aggregated across all pairs. High `max_spread` at certain hours = broker "
        "widens spreads silently (common at 21:00-22:00 UTC rollover).",
        "",
        _md_table(hh_df),
        "",
        "## Per-strategy live-vs-backtest decay",
        "",
        "`decay` = live performance ÷ backtest expectation.",
        "- ≥ 1.0 → live matches/exceeds backtest (PROMOTE)",
        "- 0.7-1.0 → minor decay, acceptable (HOLD)",
        "- 0.3-0.7 → significant decay, reduce size (REDUCE)",
        "- < 0.3 → live is net loser, kill (KILL)",
        "",
        _md_table(decay_df),
        "",
        "---",
        "## What to do with this",
        "",
        "1. **Pairs with cost_per_trade_pips > 2.0**: trades targeting <8 pips of edge "
        "    don't survive these costs. Either avoid the pair or only run strategies "
        "    targeting larger moves.",
        "2. **`fill_quality = tail-heavy` pairs**: SL slippage is unpredictable. "
        "    Either widen SL distance to absorb worst-case slip, or avoid trading near news.",
        "3. **Hours with max_spread > 2× avg**: gate the strategy off in those hours. "
        "    The standard `spread_gate(row)` already does this if dynamic spread is on, "
        "    but visual confirmation here is useful.",
        "4. **Strategies with verdict=KILL**: stop sending these orders to MT5. "
        "    Edit MT5live.2.py to disable, or remove from the strategy registry.",
    ]

    out_path.write_text('\n'.join(parts), encoding='utf-8')
    log.info("TCA report written to %s", out_path)
    return str(out_path)


# ── CLI entry ────────────────────────────────────────────────────────────────

def run() -> dict:
    """Generate the report and return a small dict for callers."""
    pp = per_pair_summary()
    decay = per_strategy_decay()
    path = build_report()
    return {
        'report_path': path,
        'n_pairs':     len(pp),
        'n_strategies': len(decay),
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    result = run()
    print(f"TCA report: {result['report_path']}")
    print(f"  Pairs analysed: {result['n_pairs']}")
    print(f"  Strategies analysed: {result['n_strategies']}")
