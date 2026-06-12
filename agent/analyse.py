"""
Deep-dive analyser for surviving strategies.

Produces a per-strategy markdown report covering:
  • Top-line metrics + verdict tier
  • Original Claude rationale + behaviour_type
  • Equity curve (PNG) and PnL distribution (PNG)
  • Per-pair, per-year, per-regime breakdowns
  • Trade dependency check (does removing top N trades kill the edge?)
  • Bootstrap Sharpe CI re-check + interpretation
  • Walk-forward consistency re-check
  • Monte Carlo prop-challenge re-simulation
  • Parameter sensitivity heat-check (neighbourhood of best params)
  • Portfolio fit (correlation against existing survivors)
  • PROMOTE / INVESTIGATE / DISCARD verdict with reasoning
  • Verbatim strategy code for review

Usage:
    python -m agent.analyse                       # report on all survivors from the last 7 days
    python -m agent.analyse --days 30             # widen window
    python -m agent.analyse --top 10              # only the top N by composite_score
    python -m agent.analyse --strategy NAME       # one specific strategy
    python -m agent.analyse --out reports/        # custom output directory
    python -m agent.analyse --html                # also render HTML alongside markdown

Reports land in agent/reports/<YYYY-MM-DD>/<strategy_name>.md (+ PNGs).
An index.md is generated listing every report with the headline verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_proj = str(Path(__file__).parent.parent)
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np
import pandas as pd

import agent.db as adb
import agent.robustness as robustness
from agent.config import (
    AGENT_DB_PATH, EDGE_DB_PATH, AGENT_DIR,
    PORTFOLIO_MAX_CORR, MIN_SHARPE_CI_LOW,
)

log = logging.getLogger(__name__)

REPORTS_DIR = AGENT_DIR / "reports"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_survivors(days: int, top: Optional[int],
                    strategy_name: Optional[str]) -> List[dict]:
    con = sqlite3.connect(AGENT_DB_PATH)
    if strategy_name:
        sql = """
            SELECT strategy_name, session, behaviour_type, composite_score,
                   test_sharpe, dsr, test_wr, regime_stable, n_trades, max_dd,
                   rationale, hypothesis_id, code, created_at, sweep_id
              FROM tested_strategies
             WHERE strategy_name = ?
        """
        rows = con.execute(sql, (strategy_name,)).fetchall()
    else:
        sql = f"""
            SELECT strategy_name, session, behaviour_type, composite_score,
                   test_sharpe, dsr, test_wr, regime_stable, n_trades, max_dd,
                   rationale, hypothesis_id, code, created_at, sweep_id
              FROM tested_strategies
             WHERE verdict = 'survivor'
               AND composite_score IS NOT NULL
               AND created_at >= datetime('now', '-{int(days)} days')
             ORDER BY composite_score DESC
        """
        rows = con.execute(sql).fetchall()
        if top:
            rows = rows[:top]
    con.close()
    cols = ['strategy_name', 'session', 'behaviour_type', 'composite_score',
            'test_sharpe', 'dsr', 'test_wr', 'regime_stable', 'n_trades',
            'max_dd', 'rationale', 'hypothesis_id', 'code', 'created_at',
            'sweep_id']
    return [dict(zip(cols, r)) for r in rows]


def _load_hypothesis_row(hypothesis_id: str) -> Optional[dict]:
    if not hypothesis_id:
        return None
    try:
        con = sqlite3.connect(EDGE_DB_PATH)
        row = con.execute("""
            SELECT hypothesis_id, sweep_id, params_json,
                   train_n, train_sharpe, train_pnl,
                   test_n,  test_sharpe,  test_pnl,  test_wr, test_max_dd,
                   dsr, sharpe_ci_low, sharpe_ci_high, regime_stable, bh_sig
              FROM hypotheses WHERE hypothesis_id = ?
        """, (hypothesis_id,)).fetchone()
        con.close()
    except Exception as e:
        log.warning("Could not load hypothesis row: %s", e)
        return None
    if not row:
        return None
    cols = ['hypothesis_id', 'sweep_id', 'params_json',
            'train_n', 'train_sharpe', 'train_pnl',
            'test_n', 'test_sharpe', 'test_pnl', 'test_wr', 'test_max_dd',
            'dsr', 'sharpe_ci_low', 'sharpe_ci_high', 'regime_stable', 'bh_sig']
    return dict(zip(cols, row))


def _load_trades(hypothesis_id: str) -> pd.DataFrame:
    if not hypothesis_id:
        return pd.DataFrame()
    try:
        con = sqlite3.connect(EDGE_DB_PATH)
        df = pd.read_sql(
            "SELECT * FROM trades WHERE hypothesis_id = ? AND split = 'test'",
            con, params=(hypothesis_id,),
        )
        con.close()
    except Exception as e:
        log.warning("Could not load trades: %s", e)
        return pd.DataFrame()
    if df.empty:
        return df
    for c in ('entry_time', 'exit_time'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', utc=True)
    return df


def _load_sweep_rows(sweep_id: str) -> List[dict]:
    if not sweep_id:
        return []
    try:
        return adb.load_sweep_results(sweep_id)
    except Exception:
        return []


def _load_other_survivors(self_name: str) -> List[dict]:
    con = sqlite3.connect(AGENT_DB_PATH)
    rows = con.execute("""
        SELECT strategy_name, hypothesis_id
          FROM tested_strategies
         WHERE verdict = 'survivor'
           AND strategy_name != ?
           AND hypothesis_id IS NOT NULL
    """, (self_name,)).fetchall()
    con.close()
    return [{'strategy_name': r[0], 'hypothesis_id': r[1]} for r in rows]


# ── Plotting (matplotlib optional) ────────────────────────────────────────────

def _plot_equity_curve(trades: pd.DataFrame, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if trades.empty or 'pnl' not in trades.columns:
        return False
    df = trades.sort_values('exit_time').reset_index(drop=True).copy()
    df['cum_pnl'] = df['pnl'].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['exit_time'], df['cum_pnl'], linewidth=1.4)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_title("Cumulative PnL — test split")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def _plot_pnl_histogram(trades: pd.DataFrame, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if trades.empty or 'pnl' not in trades.columns:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trades['pnl'].dropna(), bins=40, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='red', linewidth=0.7)
    mean = trades['pnl'].mean()
    ax.axvline(mean, color='green', linewidth=0.9, linestyle='--',
               label=f'mean = {mean:.2f}')
    ax.set_title("Per-trade PnL distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


# ── Analysis blocks ───────────────────────────────────────────────────────────

def _trade_dependency(trades: pd.DataFrame) -> dict:
    """If we drop the top-N positive trades, does the strategy still make money?"""
    if trades.empty or 'pnl' not in trades.columns:
        return {'skipped': True}
    pnl = trades['pnl'].dropna().values
    total = float(pnl.sum())
    sorted_pnl = np.sort(pnl)[::-1]
    top1 = float(sorted_pnl[0]) if len(sorted_pnl) >= 1 else 0.0
    top3 = float(sorted_pnl[:3].sum()) if len(sorted_pnl) >= 3 else float(sorted_pnl.sum())
    top5 = float(sorted_pnl[:5].sum()) if len(sorted_pnl) >= 5 else float(sorted_pnl.sum())
    return {
        'total_pnl':      round(total, 2),
        'top1_pnl':       round(top1, 2),
        'top3_pnl':       round(top3, 2),
        'top5_pnl':       round(top5, 2),
        'pnl_minus_top1': round(total - top1, 2),
        'pnl_minus_top3': round(total - top3, 2),
        'pnl_minus_top5': round(total - top5, 2),
        'top3_share_pct': round(100 * top3 / total, 1) if total > 0 else None,
    }


def _per_pair_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or 'instrument' not in trades.columns:
        return pd.DataFrame()
    grp = trades.groupby('instrument')['pnl']
    df = pd.DataFrame({
        'n':        grp.count(),
        'total':    grp.sum().round(2),
        'mean':     grp.mean().round(3),
        'win_rate': (grp.apply(lambda s: (s > 0).mean()) * 100).round(1),
    }).sort_values('total', ascending=False)
    return df


def _per_year_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or 'exit_time' not in trades.columns:
        return pd.DataFrame()
    df = trades.dropna(subset=['exit_time']).copy()
    if df.empty:
        return pd.DataFrame()
    df['year'] = df['exit_time'].dt.year
    grp = df.groupby('year')['pnl']
    out = pd.DataFrame({
        'n':       grp.count(),
        'total':   grp.sum().round(2),
        'mean':    grp.mean().round(3),
    }).sort_index()
    return out


def _per_regime_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or 'regime' not in trades.columns:
        return pd.DataFrame()
    grp = trades.groupby('regime')['pnl']
    return pd.DataFrame({
        'n':        grp.count(),
        'total':    grp.sum().round(2),
        'win_rate': (grp.apply(lambda s: (s > 0).mean()) * 100).round(1),
    }).sort_values('total', ascending=False)


def _portfolio_correlation(trades: pd.DataFrame, others: List[dict]) -> List[dict]:
    """Pearson corr of daily PnL vs each other survivor's daily PnL."""
    if trades.empty or 'exit_time' not in trades.columns or 'pnl' not in trades.columns:
        return []
    self_daily = (
        trades.dropna(subset=['exit_time'])
              .assign(date=lambda d: d['exit_time'].dt.date)
              .groupby('date')['pnl'].sum()
    )
    out = []
    for o in others:
        other_trades = _load_trades(o['hypothesis_id'])
        if other_trades.empty:
            continue
        other_daily = (
            other_trades.dropna(subset=['exit_time'])
                        .assign(date=lambda d: d['exit_time'].dt.date)
                        .groupby('date')['pnl'].sum()
        )
        joined = pd.concat([self_daily, other_daily], axis=1, join='inner')
        if len(joined) < 10:
            continue
        joined.columns = ['self', 'other']
        corr = float(joined['self'].corr(joined['other']) or 0.0)
        out.append({'strategy_name': o['strategy_name'], 'corr': round(corr, 3),
                    'overlap_days': len(joined)})
    out.sort(key=lambda d: abs(d['corr']), reverse=True)
    return out[:10]


# ── Verdict synthesis ─────────────────────────────────────────────────────────

def _synthesise_verdict(
    composite: float,
    dsr: float,
    ci_low: Optional[float],
    dependency: dict,
    per_year: pd.DataFrame,
    per_pair: pd.DataFrame,
    portfolio: List[dict],
    n_trades: int,
) -> tuple:
    """Return (tier, headline, reasons[]) where tier is PROMOTE/INVESTIGATE/DISCARD."""
    reasons_pro: List[str] = []
    reasons_con: List[str] = []

    if dsr is not None and dsr >= 0.20:
        reasons_pro.append(f"DSR {dsr:.2f} clears the 0.20 'real edge' bar")
    elif dsr is not None and dsr >= 0.10:
        reasons_pro.append(f"DSR {dsr:.2f} positive but marginal")
    else:
        reasons_con.append(f"DSR {dsr or 0:.2f} below the 0.10 floor")

    if ci_low is not None:
        if ci_low > 0.20:
            reasons_pro.append(f"Bootstrap Sharpe CI low {ci_low:.2f} indicates robust edge")
        elif ci_low > MIN_SHARPE_CI_LOW:
            reasons_con.append(
                f"Bootstrap Sharpe CI low {ci_low:.2f} is barely positive — fragile"
            )
        else:
            reasons_con.append(f"Bootstrap Sharpe CI low {ci_low:.2f} ≤ 0 — edge could vanish")

    if not dependency.get('skipped') and dependency.get('top3_share_pct') is not None:
        share = dependency['top3_share_pct']
        if share > 80:
            reasons_con.append(
                f"Top-3 trades produce {share:.0f}% of total PnL — lottery, not edge"
            )
        elif share > 50:
            reasons_con.append(
                f"Top-3 trades produce {share:.0f}% of total PnL — concentrated"
            )
        else:
            reasons_pro.append(
                f"Top-3 trades only {share:.0f}% of total PnL — distributed edge"
            )

    if not per_year.empty:
        n_years    = len(per_year)
        profitable = int((per_year['total'] > 0).sum())
        if n_years >= 2:
            if profitable == n_years:
                reasons_pro.append(f"All {n_years} years profitable")
            elif profitable >= n_years - 1:
                reasons_pro.append(f"{profitable}/{n_years} years profitable")
            elif profitable >= max(2, n_years // 2):
                reasons_con.append(
                    f"Only {profitable}/{n_years} years profitable — regime-dependent"
                )
            else:
                reasons_con.append(
                    f"Only {profitable}/{n_years} years profitable — likely dead regime"
                )

    if not per_pair.empty and per_pair['total'].sum() != 0:
        top_pair_share = (per_pair['total'].iloc[0] / per_pair['total'].sum()) * 100
        if top_pair_share > 80:
            reasons_con.append(
                f"{per_pair.index[0]} delivers {top_pair_share:.0f}% of PnL — restrict to that pair only"
            )

    if portfolio:
        max_corr = max(abs(p['corr']) for p in portfolio)
        if max_corr > PORTFOLIO_MAX_CORR:
            top = next(p for p in portfolio if abs(p['corr']) == max_corr)
            reasons_con.append(
                f"Daily-PnL correlation {top['corr']:+.2f} with {top['strategy_name']} "
                f"breaches portfolio cap ({PORTFOLIO_MAX_CORR})"
            )
        else:
            reasons_pro.append(
                f"Max correlation with existing survivors only {max_corr:.2f} — additive"
            )

    if n_trades < 50:
        reasons_con.append(f"Only {n_trades} trades — statistically thin")
    elif n_trades > 800:
        reasons_con.append(f"{n_trades} trades — possibly overtrading micro-noise")

    has_critical = any(
        kw in r for r in reasons_con
        for kw in ('lottery', 'CI low', 'likely dead regime', 'breaches portfolio cap')
    )
    strong_pro = sum(
        1 for r in reasons_pro
        if any(kw in r for kw in ('clears the 0.20', 'robust edge', 'All', 'distributed'))
    )

    if has_critical:
        tier     = "DISCARD"
        headline = "Critical robustness flag — do not promote"
    elif strong_pro >= 3 and len(reasons_con) <= 1:
        tier     = "PROMOTE"
        headline = "Paper-trade candidate — strong on multiple axes"
    elif strong_pro >= 1 and len(reasons_con) <= 2:
        tier     = "INVESTIGATE"
        headline = "Worth deeper review — promising but caveats apply"
    else:
        tier     = "DISCARD"
        headline = "Insufficient strength of evidence"

    return tier, headline, reasons_pro, reasons_con


# ── Markdown rendering ────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame, index_label: str = "") -> str:
    if df.empty:
        return "_(no data)_"
    cols = [index_label] + list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for idx, row in df.iterrows():
        out.append("| " + str(idx) + " | "
                   + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(out)


def _build_markdown(
    survivor:    dict,
    hyp_row:     Optional[dict],
    trades:      pd.DataFrame,
    sweep_rows:  list,
    eq_png:      Optional[str],
    hist_png:    Optional[str],
    dependency:  dict,
    per_pair:    pd.DataFrame,
    per_year:    pd.DataFrame,
    per_regime:  pd.DataFrame,
    portfolio:   list,
    robustness_report: dict,
    tier:        str,
    headline:    str,
    reasons_pro: list,
    reasons_con: list,
) -> str:
    name = survivor.get('strategy_name', 'unknown')
    parts: List[str] = []

    parts.append(f"# {name}\n")
    parts.append(f"**VERDICT — {tier}: {headline}**\n")

    parts.append("## Top-line metrics\n")
    parts.append(
        f"- Session: `{survivor.get('session','?')}`  \n"
        f"- Behaviour type: `{survivor.get('behaviour_type','-')}`  \n"
        f"- Composite score: **{survivor.get('composite_score',0):.3f}**  \n"
        f"- Test Sharpe: **{survivor.get('test_sharpe',0):.2f}**  \n"
        f"- DSR: **{survivor.get('dsr',0):.3f}**  \n"
        f"- Win rate: {survivor.get('test_wr',0)*100:.1f}%  \n"
        f"- Trades: {survivor.get('n_trades',0)}  \n"
        f"- Max drawdown: {survivor.get('max_dd',0)*100:.1f}%  \n"
        f"- Regime stable: {'yes' if survivor.get('regime_stable') else 'no'}  \n"
        f"- Discovered: {survivor.get('created_at','-')}  \n"
    )

    parts.append("\n## Why this verdict\n")
    if reasons_pro:
        parts.append("**Strengths:**")
        for r in reasons_pro:
            parts.append(f"- {r}")
        parts.append("")
    if reasons_con:
        parts.append("**Concerns:**")
        for r in reasons_con:
            parts.append(f"- {r}")
        parts.append("")

    parts.append("## Original Claude rationale\n")
    parts.append(f"> {survivor.get('rationale','(none)')}\n")

    if eq_png:
        parts.append("## Equity curve\n")
        parts.append(f"![equity curve]({eq_png})\n")

    if hist_png:
        parts.append("## PnL distribution\n")
        parts.append(f"![pnl histogram]({hist_png})\n")

    parts.append("## Trade dependency\n")
    if dependency.get('skipped'):
        parts.append("_(no trades data)_\n")
    else:
        parts.append(
            f"- Total PnL: **{dependency['total_pnl']}**  \n"
            f"- Top-1 trade: {dependency['top1_pnl']} → without it: {dependency['pnl_minus_top1']}  \n"
            f"- Top-3 trades: {dependency['top3_pnl']} → without them: {dependency['pnl_minus_top3']}  \n"
            f"- Top-5 trades: {dependency['top5_pnl']} → without them: {dependency['pnl_minus_top5']}  \n"
            f"- Top-3 trade share: **{dependency['top3_share_pct']}%**\n"
        )

    parts.append("\n## Per-pair breakdown\n")
    parts.append(_md_table(per_pair, "pair"))
    parts.append("\n\n## Per-year breakdown\n")
    parts.append(_md_table(per_year, "year"))
    parts.append("\n\n## Per-regime breakdown\n")
    parts.append(_md_table(per_regime, "regime"))

    parts.append("\n\n## Portfolio fit (correlation vs other survivors)\n")
    if portfolio:
        rows = []
        for p in portfolio:
            rows.append([p['strategy_name'], f"{p['corr']:+.3f}", p['overlap_days']])
        df_corr = pd.DataFrame(rows, columns=['strategy', 'daily_pnl_corr', 'overlap_days']
                              ).set_index('strategy')
        parts.append(_md_table(df_corr, "strategy"))
    else:
        parts.append("_(no other survivors with overlapping data)_")

    parts.append("\n\n## Robustness re-check\n")
    rb = robustness_report
    parts.append(
        f"- **Monte Carlo:** {rb.get('mc',{}).get('detail','-')}  \n"
        f"- **Walk-forward:** {rb.get('walk_forward',{}).get('detail','-')}  \n"
        f"- **Bootstrap CI:** {rb.get('bootstrap_ci',{}).get('detail','-')}  \n"
        f"- **Param sensitivity:** {rb.get('param_sensitivity',{}).get('detail','-')}\n"
    )

    if hyp_row:
        try:
            params = json.loads(hyp_row.get('params_json') or '{}')
        except Exception:
            params = {}
        parts.append("\n## Best parameter combo\n")
        parts.append("```json\n" + json.dumps(params, indent=2) + "\n```\n")
        parts.append(
            f"- Train Sharpe: {hyp_row.get('train_sharpe',0):.2f} "
            f"(n={hyp_row.get('train_n',0)})  \n"
            f"- Test Sharpe: {hyp_row.get('test_sharpe',0):.2f} "
            f"(n={hyp_row.get('test_n',0)})  \n"
            f"- Train→Test decay: "
            f"{(1 - (hyp_row.get('test_sharpe') or 0)/max(hyp_row.get('train_sharpe') or 1e-9, 1e-9))*100:.0f}%\n"
        )

    parts.append("\n## Strategy code\n")
    parts.append("```python")
    parts.append(survivor.get('code', '# (code not stored — older survivor)'))
    parts.append("```\n")

    parts.append("\n## Promotion checklist\n")
    parts.append(
        "- [ ] Rationale matches what the code actually does  \n"
        "- [ ] Equity curve broadly monotonic (not driven by a few spikes)  \n"
        "- [ ] At least 3 of 4 walk-forward folds profitable  \n"
        "- [ ] Bootstrap Sharpe CI lower bound > 0.20  \n"
        "- [ ] Top-3 trade share < 50% of total PnL  \n"
        "- [ ] Pair concentration acceptable (or restrict to dominant pair)  \n"
        "- [ ] Daily-PnL correlation with existing live strategies < 0.65  \n"
        "- [ ] You can explain the edge in one sentence to a sceptic\n"
    )

    return "\n".join(parts)


# ── Per-strategy pipeline ─────────────────────────────────────────────────────

def analyse_one(survivor: dict, output_dir: Path,
                others: List[dict]) -> dict:
    name = survivor['strategy_name']
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    hyp_row    = _load_hypothesis_row(survivor.get('hypothesis_id', ''))
    trades     = _load_trades(survivor.get('hypothesis_id', ''))
    sweep_rows = _load_sweep_rows(survivor.get('sweep_id', ''))

    eq_png_path   = out_dir / "equity_curve.png"
    hist_png_path = out_dir / "pnl_histogram.png"
    eq_png   = "equity_curve.png"   if _plot_equity_curve(trades, eq_png_path) else None
    hist_png = "pnl_histogram.png"  if _plot_pnl_histogram(trades, hist_png_path) else None

    dependency = _trade_dependency(trades)
    per_pair   = _per_pair_breakdown(trades)
    per_year   = _per_year_breakdown(trades)
    per_regime = _per_regime_breakdown(trades)
    portfolio  = _portfolio_correlation(trades, others)

    if hyp_row:
        _, rb_report = robustness.run_all_checks(hyp_row, sweep_rows, trades)
    else:
        rb_report = {
            'mc':                {'detail': 'hypothesis row missing'},
            'walk_forward':      {'detail': 'hypothesis row missing'},
            'bootstrap_ci':      {'detail': 'hypothesis row missing'},
            'param_sensitivity': {'detail': 'hypothesis row missing'},
        }

    ci_low = (hyp_row or {}).get('sharpe_ci_low')
    tier, headline, reasons_pro, reasons_con = _synthesise_verdict(
        composite  = survivor.get('composite_score', 0),
        dsr        = survivor.get('dsr', 0),
        ci_low     = ci_low,
        dependency = dependency,
        per_year   = per_year,
        per_pair   = per_pair,
        portfolio  = portfolio,
        n_trades   = survivor.get('n_trades', 0),
    )

    md = _build_markdown(
        survivor, hyp_row, trades, sweep_rows,
        eq_png, hist_png, dependency, per_pair, per_year, per_regime,
        portfolio, rb_report, tier, headline, reasons_pro, reasons_con,
    )
    (out_dir / "report.md").write_text(md, encoding='utf-8')
    log.info("  Wrote report: %s", out_dir / "report.md")

    return {
        'strategy_name':   name,
        'tier':            tier,
        'headline':        headline,
        'composite_score': survivor.get('composite_score', 0),
        'dsr':             survivor.get('dsr', 0),
        'sharpe':          survivor.get('test_sharpe', 0),
        'session':         survivor.get('session', '-'),
        'behaviour_type':  survivor.get('behaviour_type', '-'),
        'report_path':     str((out_dir / "report.md").relative_to(output_dir)),
    }


# ── Index ─────────────────────────────────────────────────────────────────────

def _write_index(output_dir: Path, summaries: List[dict]):
    promote     = [s for s in summaries if s['tier'] == 'PROMOTE']
    investigate = [s for s in summaries if s['tier'] == 'INVESTIGATE']
    discard     = [s for s in summaries if s['tier'] == 'DISCARD']

    lines: List[str] = []
    lines.append(f"# Survivor analysis — {datetime.now(timezone.utc).date()}\n")
    lines.append(f"Total reports: **{len(summaries)}** "
                 f"(promote: {len(promote)}, investigate: {len(investigate)}, "
                 f"discard: {len(discard)})\n")

    def _section(title: str, items: List[dict]):
        if not items:
            return
        lines.append(f"\n## {title}\n")
        lines.append("| Strategy | Session | Behaviour | Composite | DSR | Sharpe | Headline |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for s in items:
            lines.append(
                f"| [{s['strategy_name']}]({s['report_path']}) "
                f"| {s['session']} | {s['behaviour_type']} "
                f"| {s['composite_score']:.3f} | {s['dsr']:.3f} | {s['sharpe']:.2f} "
                f"| {s['headline']} |"
            )

    _section("PROMOTE — paper-trade candidates", promote)
    _section("INVESTIGATE — review further", investigate)
    _section("DISCARD — do not promote", discard)

    (output_dir / "index.md").write_text("\n".join(lines), encoding='utf-8')


# ── HTML rendering (optional) ─────────────────────────────────────────────────

def _md_to_html(md_text: str, title: str) -> str:
    try:
        import markdown as md
        body = md.markdown(md_text, extensions=['tables', 'fenced_code'])
    except ImportError:
        body = "<pre>" + md_text.replace("<", "&lt;").replace(">", "&gt;") + "</pre>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1000px;"
        "margin:auto;padding:24px;line-height:1.5}"
        "table{border-collapse:collapse;margin:1em 0}"
        "th,td{border:1px solid #ddd;padding:6px 10px}"
        "img{max-width:100%}code,pre{background:#f5f5f5;padding:2px 6px;"
        "border-radius:4px}pre{padding:12px;overflow-x:auto}"
        "</style></head><body>" + body + "</body></html>"
    )


def _emit_html(output_dir: Path):
    for md_file in output_dir.rglob("*.md"):
        try:
            text = md_file.read_text(encoding='utf-8')
            (md_file.with_suffix(".html")).write_text(
                _md_to_html(text, md_file.stem), encoding='utf-8',
            )
        except Exception as e:
            log.warning("HTML render failed for %s: %s", md_file, e)


# ── CLI ───────────────────────────────────────────────────────────────────────

def run(days: int = 7, top: Optional[int] = None,
        strategy_name: Optional[str] = None,
        out_root: Optional[Path] = None,
        emit_html: bool = False) -> int:
    out_root = Path(out_root) if out_root else REPORTS_DIR
    today    = datetime.now(timezone.utc).date().isoformat()
    out_dir  = out_root / today
    out_dir.mkdir(parents=True, exist_ok=True)

    survivors = _load_survivors(days, top, strategy_name)
    if not survivors:
        log.info("No survivors matched (days=%d, top=%s, name=%s).",
                 days, top, strategy_name)
        return 0

    log.info("Analysing %d survivor(s) → %s", len(survivors), out_dir)
    summaries: List[dict] = []
    for s in survivors:
        try:
            others = _load_other_survivors(s['strategy_name'])
            summaries.append(analyse_one(s, out_dir, others))
        except Exception as e:
            log.exception("Analysis failed for %s: %s", s.get('strategy_name'), e)

    _write_index(out_dir, summaries)
    if emit_html:
        _emit_html(out_dir)

    log.info("Index written to %s", out_dir / "index.md")
    return len(summaries)


def _cli():
    parser = argparse.ArgumentParser(
        description="Deep-dive analyser for surviving strategies."
    )
    parser.add_argument('--days',     type=int, default=7,
                        help="Look-back window in days (default 7)")
    parser.add_argument('--top',      type=int, default=None,
                        help="Only analyse the top-N survivors by composite_score")
    parser.add_argument('--strategy', default=None,
                        help="Analyse one specific strategy by name")
    parser.add_argument('--out',      default=None,
                        help="Output root (default agent/reports/)")
    parser.add_argument('--html',     action='store_true',
                        help="Also render HTML alongside markdown")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    n = run(days=args.days, top=args.top, strategy_name=args.strategy,
            out_root=Path(args.out) if args.out else None,
            emit_html=args.html)
    print(f"\nAnalysed {n} survivor(s).")
    if n:
        out_root = Path(args.out) if args.out else REPORTS_DIR
        today    = datetime.now(timezone.utc).date().isoformat()
        print(f"Open {out_root / today / 'index.md'} to review.")
    return 0


if __name__ == '__main__':
    sys.exit(_cli())
