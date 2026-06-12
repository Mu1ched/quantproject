"""
Edge hunt — H1 FX, real engine + full gauntlet, anti-p-hacking by construction.

Pipeline per strategy:
  M1 cache --resample--> H1 (UTC) --> [train | test | LOCKED HOLDOUT]
  run_sweep(train, test) on a tp/sl(+extra) grid, dynamic spread + commission
  -> scorer.pick_best_from_sweep -> robustness.run_all_checks (MC + WF enforced)
  -> concentration check (>35% of PnL from any month/pair/trade = reject)
  -> log one block per strategy to tools/edge_hunt_log.md

Success = is_survivor AND robust_passed (MC+WF) AND concentration_ok on train/test,
THEN confirmed once on the locked holdout. Holdout is touched only by --confirm.

Usage:
    python tools/edge_hunt.py                 # run all registered strategies once
    python tools/edge_hunt.py --only NAME     # single strategy
    python tools/edge_hunt.py --confirm NAME --params '{"tp_r":2.5,...}'  # holdout
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import edge_engine as eng
from agent import db, robustness, scorer
from agent.scorer import rejection_reason

import edge_hunt_strategies as strat

LOG_PATH = PROJECT_ROOT / "tools" / "edge_hunt_log.md"
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY", "AUD_USD"]
CONCENTRATION_MAX = 0.35   # max share of total PnL from any one month / pair / trade


# ── Feature recompute on H1 (UTC) ───────────────────────────────────────────
def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _adx(high, low, close, n=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up.clip(lower=0)
    minus_dm = ((dn > up) & (dn > 0)) * dn.clip(lower=0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    plus_di = 100 * _wilder(plus_dm, n) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, n) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx.fillna(0), n)


def _rsi(close, n=14):
    d = close.diff()
    gain = _wilder(d.clip(lower=0), n)
    loss = _wilder((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _join_asof(out: pd.DataFrame, src, value_cols: list) -> pd.DataFrame:
    """Lookahead-safe join: each bar gets the latest already-released record
    (src must have an 'available_date' column)."""
    if src is None or not len(src):
        return out
    s = src.sort_values('available_date')[['available_date'] + value_cols].copy()
    s['available_date'] = pd.to_datetime(s['available_date'])
    left = out[['timestamp']].copy()
    left['bar_dt'] = out['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    left = left.sort_values('bar_dt')
    merged = pd.merge_asof(left, s, left_on='bar_dt',
                           right_on='available_date', direction='backward').sort_index()
    for c in value_cols:
        out[c] = merged[c].values
    return out


def _join_cot(out: pd.DataFrame, cot_pair) -> pd.DataFrame:
    return _join_asof(out, cot_pair, ['net_spec_pct', 'cot_index'])


PAIR_CCY = {"EUR_USD": ("EUR", "USD"), "GBP_USD": ("GBP", "USD"),
            "USD_JPY": ("USD", "JPY"), "EUR_JPY": ("EUR", "JPY"),
            "AUD_USD": ("AUD", "USD")}
_USD_QUOTED = {"EUR_USD", "GBP_USD", "AUD_USD"}


def add_carry(df: pd.DataFrame, pair: str, rates_long) -> pd.DataFrame:
    """Add 'carry_diff' (annual %, base_rate - quote_rate), lookahead-safe."""
    if rates_long is None or pair not in PAIR_CCY:
        df["carry_diff"] = float("nan"); return df
    base, quote = PAIR_CCY[pair]

    def _rate_for(ccy):
        s = rates_long[rates_long.ccy == ccy].rename(columns={"rate": f"r_{ccy}"})
        return _join_asof(df[["timestamp"]].copy(), s, [f"r_{ccy}"])[f"r_{ccy}"].values
    df["carry_diff"] = _rate_for(base) - _rate_for(quote)
    return df


def resample_d1(m1: pd.DataFrame, cot_pair=None, vix_df=None) -> pd.DataFrame:
    """Daily-bar path for slow signals (COT). hour=0 + exit_hour=23 in the
    registry means NO intraday force-exit, so positions hold for multiple days
    until SL/TP — matching the weekly COT horizon, where one round-trip cost is
    a tiny fraction of daily ATR."""
    df = m1.copy()
    ts = df['timestamp']
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('Europe/London')
    df['timestamp'] = ts.dt.tz_convert('UTC')
    df = df.set_index('timestamp').sort_index()

    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    for c in ('spread_mean', 'spread_median', 'spread_adj'):
        if c in df.columns:
            agg[c] = 'median' if c == 'spread_median' else 'mean'
    if 'near_news' in df.columns:
        agg['near_news'] = 'max'

    d = df.resample('1D').agg(agg).dropna(subset=['open', 'high', 'low', 'close'])
    c, hi, lo = d['close'], d['high'], d['low']
    prev_c = c.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    d['atr'] = _wilder(tr, 14)
    d['ema_fast'] = c.ewm(span=20, adjust=False).mean()
    d['ema_slow'] = c.ewm(span=50, adjust=False).mean()
    d['ema_200'] = c.ewm(span=100, adjust=False).mean()   # ~100d trend on daily
    d['ma_trend'] = d['ema_fast'] - d['ema_slow']
    d['rsi_14'] = _rsi(c, 14)

    d = d.reset_index()
    d['hour'] = 0
    d['minute'] = 0
    d['date'] = d['timestamp'].dt.date
    d['regime'] = 'UNDEFINED'
    for col in ('spread_mean', 'spread_median', 'spread_adj'):
        if col not in d.columns:
            d[col] = 0.0
    if 'near_news' not in d.columns:
        d['near_news'] = False
    d['near_news'] = d['near_news'].astype(bool)
    d = d.dropna(subset=['atr', 'ma_trend']).reset_index(drop=True)
    d = _join_cot(d, cot_pair)
    d = _join_asof(d, vix_df, ['vix_close', 'vix_chg', 'vix_z'])
    return d


def resample_h1(m1: pd.DataFrame, cot_pair=None, vix_df=None) -> pd.DataFrame:
    df = m1.copy()
    # Normalise to UTC so 'hour' aligns with run_sweep's session-hour filter
    ts = df['timestamp']
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('Europe/London')
    df['timestamp'] = ts.dt.tz_convert('UTC')
    df = df.set_index('timestamp').sort_index()

    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    for c in ('spread_mean', 'spread_median', 'spread_adj'):
        if c in df.columns:
            agg[c] = 'mean' if c != 'spread_median' else 'median'
    if 'near_news' in df.columns:
        agg['near_news'] = 'max'
    # Batch 4: order-flow + positioning aggregation (economically distinct from price)
    _flow_agg = {'delta': 'sum', 'tick_imbalance': 'mean', 'vol_imbalance': 'mean',
                 'aggressive_buy_ratio': 'mean', 'persistent_imbalance': 'mean',
                 'cumulative_delta': 'last', 'retail_long_pct': 'last',
                 'cot_net_position_pct': 'last', 'cot_extreme_flag': 'max'}
    for c, how in _flow_agg.items():
        if c in df.columns:
            agg[c] = how

    h1 = df.resample('1h', label='left', closed='left').agg(agg)
    h1 = h1.dropna(subset=['open', 'high', 'low', 'close'])

    c, hi, lo = h1['close'], h1['high'], h1['low']
    prev_c = c.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    h1['atr'] = _wilder(tr, 14)
    h1['ema_fast'] = c.ewm(span=20, adjust=False).mean()
    h1['ema_slow'] = c.ewm(span=50, adjust=False).mean()
    h1['ema_200'] = c.ewm(span=200, adjust=False).mean()   # higher-TF trend proxy
    h1['ma_trend'] = h1['ema_fast'] - h1['ema_slow']
    # Donchian(20) excluding current bar (no lookahead)
    h1['donchian_hi'] = hi.rolling(20).max().shift(1)
    h1['donchian_lo'] = lo.rolling(20).min().shift(1)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    h1['bb_mid'] = bb_mid
    h1['bb_up'] = bb_mid + 2 * bb_std
    h1['bb_lo'] = bb_mid - 2 * bb_std
    h1['adx'] = _adx(hi, lo, c, 14)
    h1['rsi_14'] = _rsi(c, 14)

    # Batch 4 derived order-flow features
    if 'delta' in h1.columns:
        dm = h1['delta'].rolling(50).mean()
        ds = h1['delta'].rolling(50).std()
        h1['delta_z'] = ((h1['delta'] - dm) / ds.replace(0, np.nan)).fillna(0.0)
    if 'cumulative_delta' in h1.columns:
        h1['cd_slope6'] = h1['cumulative_delta'].diff(6)   # 6h net flow direction
    if 'retail_long_pct' in h1.columns:
        h1['retail_long_pct'] = h1['retail_long_pct'].ffill()
    if 'cot_net_position_pct' in h1.columns:
        h1['cot_net_position_pct'] = h1['cot_net_position_pct'].ffill()

    h1 = h1.reset_index()
    h1['hour'] = h1['timestamp'].dt.hour          # UTC hour
    h1['minute'] = 0
    h1['date'] = h1['timestamp'].dt.date
    h1['regime'] = 'UNDEFINED'

    # Asian range (00:00–07:00 UTC) of the SAME day, applied to later bars only
    asia = h1[h1['hour'] < 7].groupby('date').agg(asian_high=('high', 'max'),
                                                  asian_low=('low', 'min'))
    h1 = h1.merge(asia, on='date', how='left')

    # Prior-day high/low (shifted — no lookahead)
    daily = h1.groupby('date').agg(dh=('high', 'max'), dl=('low', 'min')).reset_index()
    daily['prev_day_high'] = daily['dh'].shift(1)
    daily['prev_day_low'] = daily['dl'].shift(1)
    h1 = h1.merge(daily[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')

    for col in ('spread_mean', 'spread_median', 'spread_adj'):
        if col not in h1.columns:
            h1[col] = 0.0
    if 'near_news' not in h1.columns:
        h1['near_news'] = False
    h1['near_news'] = h1['near_news'].astype(bool)

    h1 = h1.dropna(subset=['atr', 'ma_trend']).reset_index(drop=True)
    h1 = _join_cot(h1, cot_pair)                                  # batch 6
    h1 = _join_asof(h1, vix_df, ['vix_close', 'vix_chg', 'vix_z'])  # batch 8
    return h1


# ── Concentration / consistency check ───────────────────────────────────────
def concentration_check(trades: pd.DataFrame) -> tuple[bool, dict]:
    if trades is None or trades.empty:
        return False, {"reason": "no trades"}
    pnl = trades['pnl'].astype(float)
    total = pnl.sum()
    if total <= 0:
        return False, {"reason": "non-positive total PnL"}
    # by month
    t = pd.to_datetime(trades['entry_time'])
    by_month = pnl.groupby(t.dt.to_period('M')).sum()
    month_share = (by_month.max() / total) if len(by_month) else 1.0
    # by pair
    pair_share = 1.0
    if 'instrument' in trades.columns:
        by_pair = pnl.groupby(trades['instrument']).sum()
        pair_share = (by_pair.max() / total) if len(by_pair) else 1.0
    # by single trade
    trade_share = pnl.max() / total
    worst = max(month_share, pair_share, trade_share)
    ok = worst <= CONCENTRATION_MAX
    return ok, {"month": round(float(month_share), 3),
                "pair": round(float(pair_share), 3),
                "trade": round(float(trade_share), 3),
                "worst": round(float(worst), 3)}


# ── One strategy through the gauntlet ───────────────────────────────────────
def _grid(extra: dict):
    from agent.config import GRID_TP_VALUES, GRID_SL_VALUES
    g = {"tp_r": GRID_TP_VALUES, "sl_r": GRID_SL_VALUES}
    g.update(extra or {})
    return eng.ParameterGrid(g)


def run_strategy(s: dict, train_dfs, test_dfs, spreads, label="train/test") -> dict:
    name = s["name"]
    grid = _grid(s.get("extra_grid"))
    manager = eng.make_manager(exit_hour=s["exit_hour"], use_breakeven=True)
    regime_mult = {k: 1.0 for k in
                   ("TRENDING", "RANGING", "TRANSITIONING", "VOLATILE", "UNDEFINED")}
    sweep_id = eng.run_sweep(
        sweep_name=f"hunt_{name}",
        entry_fn=s["fn"], manager_fn=manager, grid=grid,
        pairs=PAIRS, session=s["session"], regime_mult=regime_mult,
        train_dfs=train_dfs, test_dfs=test_dfs, measured_spreads=spreads,
        family=s["family"], n_workers=2, cost_mult=1.0, use_dynamic_spread=True,
    )
    rows = db.load_sweep_results(sweep_id)
    if not rows:
        return {"name": name, "verdict": "NO ROWS", "sweep_id": sweep_id}

    # Custom selection: scorer.is_survivor PLUS a train-positivity gate.
    # Learning from batch 1: is_survivor only checks train->test decay when
    # train_sharpe>0, so a negative-train / positive-test fluke (regime luck)
    # can slip to robustness. Require BOTH splits positive — strictly stronger.
    def _qualifies(r):
        if not scorer.is_survivor(r):
            return False
        if float(r.get("train_sharpe") or -9) <= 0:
            return False
        if float(r.get("test_sharpe") or -9) <= 0:
            return False
        return True

    quals = [r for r in rows if _qualifies(r)]
    if not quals:
        top = sorted(rows, key=lambda r: float(r.get("test_sharpe") or 0), reverse=True)[0]
        # Distinguish "lost on train" from a normal gauntlet rejection.
        if scorer.is_survivor(top) and float(top.get("train_sharpe") or -9) <= 0:
            gate = f"negative train Sharpe ({float(top.get('train_sharpe') or 0):.2f})"
        else:
            gate = rejection_reason(top)
        return {"name": name, "verdict": "REJECTED", "sweep_id": sweep_id,
                "gate": gate, "best": top, "score": 0.0}
    best = max(quals, key=scorer.composite_score)
    score = scorer.composite_score(best)

    hyp_id = best.get("hypothesis_id", "")
    trades = db.load_test_trades(hyp_id) if hyp_id else None
    robust_passed, report = robustness.run_all_checks(best, rows, trades)
    conc_ok, conc = concentration_check(trades)

    success = robust_passed and conc_ok
    return {"name": name, "verdict": "SURVIVOR" if success else "REJECTED",
            "sweep_id": sweep_id, "best": best, "score": score,
            "robust_passed": robust_passed, "report": report,
            "conc_ok": conc_ok, "conc": conc, "params": best.get("params_json"),
            "gate": None if success else
                    ("concentration" if not conc_ok else "robustness")}


# ── Logging ─────────────────────────────────────────────────────────────────
def log_block(res: dict, s: dict):
    b = res.get("best") or {}
    rep = res.get("report") or {}
    mc = rep.get("mc", {})
    wf = rep.get("walk_forward", {})
    conc = res.get("conc", {})

    def g(k, d="—"):
        v = b.get(k)
        return f"{v:.2f}" if isinstance(v, (int, float)) else d

    line = (
        f"\n### {res['name']}  —  **{res['verdict']}**  "
        f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}Z)\n"
        f"- rationale: {s.get('rationale','')}\n"
        f"- family: {s.get('family','')} | sweep: `{res.get('sweep_id','')}`\n"
        f"- n train/test: {b.get('train_n','—')}/{b.get('test_n','—')} | "
        f"train_sh/test_sh: {g('train_sharpe')}/{g('test_sharpe')}\n"
        f"- DSR {g('dsr')} | PSR {g('psr')} | PBO {g('pbo_score')} | "
        f"CI_low {g('sharpe_ci_low')} | by_sig {b.get('by_sig', b.get('bh_sig','—'))}\n"
        f"- MC pass {mc.get('pass_pct','—')}% / blown {mc.get('blown_pct','—')}% | "
        f"WF folds {wf.get('profitable_folds','—')}/{wf.get('total_folds','—')}\n"
        f"- concentration worst {conc.get('worst','—')} "
        f"(month {conc.get('month','—')}, pair {conc.get('pair','—')}, trade {conc.get('trade','—')})\n"
        f"- params: `{res.get('params','—')}`\n"
        f"- killed by: {res.get('gate','—')}\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    print(line)


def learn_from_data() -> str:
    """Aggregate every hunt sweep's trades to date and surface what the data
    says — exit-reason economics by family, gap-stop burden, win-rate vs RR.
    Written to the log as a meta-reflection so each batch builds on the last."""
    import sqlite3
    con = sqlite3.connect(eng.DB_PATH)
    sw = pd.read_sql("SELECT sweep_id, sweep_name FROM sweeps "
                     "WHERE sweep_name LIKE 'hunt_%'", con)
    if sw.empty:
        con.close(); return ""
    hyp = pd.read_sql("SELECT hypothesis_id, sweep_id, train_sharpe, test_sharpe "
                      "FROM hypotheses", con)
    hyp = hyp.merge(sw, on="sweep_id", how="inner")
    tr = pd.read_sql(
        f"SELECT t.hypothesis_id, t.exit_reason, t.pnl, t.family "
        f"FROM trades t WHERE t.split='train'", con)
    con.close()
    if tr.empty:
        return ""
    lines = ["\n---\n#### Meta-reflection (learned from all hunt data so far)\n"]
    # Exit-reason economics across the whole hunt
    er = tr.groupby("exit_reason")["pnl"].agg(["count", "mean", "sum"]).round(1)
    er = er.sort_values("sum")
    lines.append("Exit-reason economics (train, all hunt strategies pooled):")
    for reason, row in er.iterrows():
        lines.append(f"  - {reason}: n={int(row['count'])}, "
                     f"mean=${row['mean']:.0f}, total=${row['sum']:,.0f}")
    gap = tr[tr["exit_reason"].isin(["gap_stop"])]["pnl"]
    tot = tr["pnl"].sum()
    if len(gap):
        lines.append(f"\nGap-stop burden: {len(gap)} trades, ${gap.sum():,.0f} "
                     f"({gap.sum()/tot*100:.0f}% of total PnL was lost to gaps)"
                     if tot else f"\nGap-stop total ${gap.sum():,.0f}")
    # Best families by train Sharpe
    fam = tr.groupby("family")["pnl"].agg(["count", "mean"]).round(2)
    lines.append("\nTrain PnL/trade by family (positive = worth pursuing):")
    for f_, r in fam.sort_values("mean", ascending=False).iterrows():
        lines.append(f"  - {f_}: {int(r['count'])} trades, ${r['mean']:.0f}/trade")
    text = "\n".join(lines) + "\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return text


REP_PRICE = {"EUR_USD": 1.08, "GBP_USD": 1.27, "USD_JPY": 150.0,
             "EUR_JPY": 160.0, "AUD_USD": 0.66}


def _make_real_swap_fn(pair_carry: dict):
    """Replacement for eng._compute_swap_pips that returns swap in pips from the
    REAL time-varying rate differential (gross interbank carry, no broker skim —
    an optimistic upper bound). Engine applies it with the actual position size."""
    def _swap(pair, position, entry_time, exit_time):
        if entry_time is None or exit_time is None or pair not in pair_carry:
            return 0.0
        cds = pair_carry[pair]
        if cds is None or not len(cds):
            return 0.0
        idx = cds.index.searchsorted(pd.Timestamp(entry_time))
        cd = float(cds.iloc[min(idx, len(cds) - 1)]) if idx >= 0 else 0.0
        nights = (pd.Timestamp(exit_time).normalize() - pd.Timestamp(entry_time).normalize()).days
        if nights <= 0:
            return 0.0
        pip = 0.01 if "JPY" in pair else 0.0001
        per_night = REP_PRICE.get(pair, 1.0) * (cd / 100.0) / 365.0 / pip  # long earns +cd
        sign = 1.0 if position == "long" else -1.0
        return per_night * sign * nights
    return _swap


def carry_eval(rates_long, cot_all, vix_df, params):
    """Carry test across train/test/locked-holdout, with swap driven by REAL
    OECD rate differentials (gross). Compares engine-default vs real-rate swap."""
    trm, tem, sp = eng.load_all_data(pairs=PAIRS)
    def cotp(p):
        if cot_all is None: return None
        s = cot_all[cot_all.pair == p]; return s if len(s) else None
    splits = {"TRAIN": {}, "TEST": {}, "HOLD": {}}
    pair_carry = {}
    for p in PAIRS:
        tr = add_carry(resample_h1(trm[p], cotp(p), vix_df), p, rates_long)
        te = add_carry(resample_h1(tem[p], cotp(p), vix_df), p, rates_long)
        mid = len(te) // 2
        splits["TRAIN"][p] = tr
        splits["TEST"][p] = te.iloc[:mid].reset_index(drop=True)
        splits["HOLD"][p] = te.iloc[mid:].reset_index(drop=True)
        both = pd.concat([tr[["timestamp", "carry_diff"]], te[["timestamp", "carry_diff"]]])
        pair_carry[p] = both.dropna().set_index("timestamp")["carry_diff"].sort_index()

    mgr = eng.make_manager(exit_hour=99, use_breakeven=True)
    rmult = {k: 1.0 for k in ("TRENDING", "RANGING", "TRANSITIONING", "VOLATILE", "UNDEFINED")}
    reg = [{"id": "carry", "family": "carry", "slot_class": "carry", "pairs": PAIRS,
            "session": "all", "allow_concurrent": False, "regime_mult": rmult, "params": params}]

    orig_swap = eng._compute_swap_pips
    eng._compute_swap_pips = _make_real_swap_fn(pair_carry)   # real-rate gross swap
    try:
        print(f"\nCARRY eval (swap = REAL OECD rate differential, gross) | params={params}")
        print(f"{'split':6s} {'n':>4s} {'Sharpe':>7s} {'PnL':>11s} {'maxDD':>9s} {'WR':>4s}")
        out = {}
        for name, dfs in splits.items():
            tr, _, _ = eng.run_backtest(dfs, None, None, reg, {"carry": mgr},
                                        {"carry": strat.entry_carry}, cost_mult=1.0)
            n = 0 if tr is None or tr.empty else len(tr)
            if not n:
                print(f"{name:6s} {0:>4d}  (no trades)"); continue
            st = eng.calc_stats(tr)
            sh = float(st.get("sharpe", 0) or 0); dd = float(st.get("max_dd", 0) or 0)
            pnl = float(tr["pnl"].sum()); wr = float((tr["pnl"] > 0).mean()) * 100
            print(f"{name:6s} {n:>4d} {sh:>+7.2f} {pnl:>+11,.0f} {dd:>+9,.0f} {wr:>3.0f}%")
            out[name] = sh
        return out
    finally:
        eng._compute_swap_pips = orig_swap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--daily", action="store_true", help="resample to D1 (slow signals)")
    ap.add_argument("--carry", action="store_true", help="run swap-aware carry eval")
    ap.add_argument("--confirm")
    ap.add_argument("--params")
    args = ap.parse_args()

    if args.carry:
        rp = PROJECT_ROOT / "tools" / "rates_data.parquet"
        cp = PROJECT_ROOT / "tools" / "cot_data.parquet"
        vp = PROJECT_ROOT / "tools" / "vix_data.parquet"
        rates = pd.read_parquet(rp) if rp.exists() else None
        cot = pd.read_parquet(cp) if cp.exists() else None
        vix = pd.read_parquet(vp) if vp.exists() else None
        if rates is None:
            print("No rates_data.parquet — run tools/fetch_rates.py first."); return 1
        for params in (
            {"sl_r": 3.0, "tp_r": 5.0, "carry_th": 1.0, "sign": 1.0},
            {"sl_r": 5.0, "tp_r": 8.0, "carry_th": 2.0, "sign": 1.0},
        ):
            carry_eval(rates, cot, vix, params)
        return 0

    print("Loading M1 cache + resampling to H1 (UTC)...")
    train_m1, test_m1, spreads = eng.load_all_data(pairs=PAIRS)

    cot_path = PROJECT_ROOT / "tools" / "cot_data.parquet"
    cot_all = pd.read_parquet(cot_path) if cot_path.exists() else None
    def _cot(p):
        if cot_all is None:
            return None
        sub = cot_all[cot_all["pair"] == p]
        return sub if len(sub) else None
    vix_path = PROJECT_ROOT / "tools" / "vix_data.parquet"
    vix_df = pd.read_parquet(vix_path) if vix_path.exists() else None

    _resample = resample_d1 if args.daily else resample_h1
    tf = "D1" if args.daily else "H1"
    train_h1, test_h1, hold_h1 = {}, {}, {}
    for p in PAIRS:
        tr = _resample(train_m1[p], _cot(p), vix_df)
        te_full = _resample(test_m1[p], _cot(p), vix_df)
        mid = len(te_full) // 2
        train_h1[p] = tr
        test_h1[p] = te_full.iloc[:mid].reset_index(drop=True)     # search test
        hold_h1[p] = te_full.iloc[mid:].reset_index(drop=True)     # LOCKED holdout
        print(f"  {p}: train {len(tr):>6} | test {len(test_h1[p]):>5} | "
              f"holdout {len(hold_h1[p]):>5} {tf} bars | "
              f"ATR~{tr['atr'].median()/strat._pip(p):.1f}p")

    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Edge Hunt Log\n\n_H1 FX, real engine + full gauntlet "
                            "(MC+WF enforced) + concentration check._\n", encoding="utf-8")

    todo = strat.STRATEGIES
    if args.batch:
        todo = [s for s in todo if s.get("batch") == args.batch]
    if args.only:
        todo = [s for s in todo if s["name"] == args.only]

    survivors = []
    for s in todo:
        print(f"\n{'='*70}\n[RUN] {s['name']}\n{'='*70}")
        try:
            res = run_strategy(s, train_h1, test_h1, spreads)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"name": s["name"], "verdict": "ERROR", "gate": f"{type(e).__name__}: {e}"}
        log_block(res, s)
        if res.get("verdict") == "SURVIVOR":
            survivors.append(res)
            print(f"\n*** {s['name']} PASSED train/test gauntlet — needs holdout "
                  f"confirmation: python tools/edge_hunt.py --confirm {s['name']} "
                  f"--params '{res.get('params')}' ***")

    if not args.only:
        learn_from_data()

    print(f"\n{'='*70}\nDONE. {len(survivors)} train/test survivor(s) "
          f"awaiting holdout confirmation.\n{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
