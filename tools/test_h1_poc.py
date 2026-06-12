"""
H1 timeframe pivot — single-pair (EUR/USD) proof-of-concept.

81 designed M1 FX strategies → 1 candidate. Hypothesis: on M1, cost is
12-30% of ATR (any edge IS the cost); on H1, cost is 1.5-4% of ATR, so
modest edges survive. This script resamples EUR/USD M1 → H1 in-memory,
recomputes a minimal H1 feature set, and runs 4 H1-native strategies
through the SAME engine + cost model + acceptance gate as the M1 batteries.

If ≥1 strategy clears the bar (test Sharpe ≥ 0.5, n ≥ 30, train+test same
sign) on EUR/USD H1 — where EUR/USD failed every M1 battery — the pivot
is validated.

Usage:
    python tools/test_h1_poc.py [--strategy NAME]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import edge_engine as eng

PAIR = "EUR_USD"
REPORT_PATH = PROJECT_ROOT / "tools" / "h1_poc_report.md"
EXIT_HOUR = 23   # end-of-day backstop; SL/TP do the real work


def _pip(pair):
    return 0.01 if "JPY" in pair else 0.0001


# ═══════════════════════════════════════════════════════════════════════════
# Resample M1 -> H1 + recompute minimal feature set
# ═══════════════════════════════════════════════════════════════════════════

def resample_h1(m1: pd.DataFrame) -> pd.DataFrame:
    """Resample an M1 prepared DataFrame to H1, carrying cost-relevant columns
    and recomputing the minimal feature set the engine + strategies need.
    """
    df = m1.copy()
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/London')
    df = df.set_index('timestamp').sort_index()

    agg = {
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
    }
    # Cost-relevant columns carried forward (a trade filling within the H1 bar
    # pays ~the average M1 spread during that hour — one round-trip, same as M1)
    if 'spread_adj' in df.columns:    agg['spread_adj']    = 'mean'
    if 'spread_mean' in df.columns:   agg['spread_mean']   = 'mean'
    if 'spread_median' in df.columns: agg['spread_median'] = 'median'
    if 'near_news' in df.columns:     agg['near_news']     = 'max'   # any() over the hour

    h1 = df.resample('1h', label='left', closed='left').agg(agg)
    h1 = h1.dropna(subset=['open', 'high', 'low', 'close'])   # drop empty (weekend) bins

    # ── Recompute features on H1 bars ─────────────────────────────────────
    close, high, low = h1['close'], h1['high'], h1['low']

    # ATR (Wilder, 14) on H1
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    h1['atr'] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Trend EMAs
    h1['ema_fast'] = close.ewm(span=20, adjust=False).mean()
    h1['ema_slow'] = close.ewm(span=50, adjust=False).mean()
    # ma_trend must be non-NaN or the engine skips the bar; doubles as direction
    h1['ma_trend'] = h1['ema_fast'] - h1['ema_slow']

    # Bollinger (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    h1['bb_mid']   = bb_mid
    h1['bb_up']    = bb_up
    h1['bb_lo']    = bb_lo
    h1['bb_pct']   = (close - bb_lo) / (bb_up - bb_lo)
    h1['bb_width'] = (bb_up - bb_lo) / bb_mid

    # Donchian (20), shifted to exclude current bar
    h1['donchian_hi'] = high.rolling(20).max().shift(1)
    h1['donchian_lo'] = low.rolling(20).min().shift(1)

    # Prior-bar range for inside-bar detection
    h1['prev_high'] = high.shift(1)
    h1['prev_low']  = low.shift(1)

    # Rebuild engine time columns from the H1 index
    h1 = h1.reset_index()           # 'timestamp' back as a column
    h1['hour']   = h1['timestamp'].dt.hour
    h1['minute'] = 0
    h1['date']   = h1['timestamp'].dt.date
    # regime omitted → engine defaults to 'UNDEFINED'

    # Backfill spread cols if missing (defensive)
    for c in ('spread_mean', 'spread_median', 'spread_adj'):
        if c not in h1.columns:
            h1[c] = 0.0
    if 'near_news' not in h1.columns:
        h1['near_news'] = False
    h1['near_news'] = h1['near_news'].astype(bool)

    # Drop early rows with NaN in any required feature
    h1 = h1.dropna(subset=['atr', 'ma_trend']).reset_index(drop=True)
    return h1


def _size(bst, regime_mult, pair, balance, entry, sl_dist, row):
    risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
    return eng.rv_size(pair, balance, risk, entry, sl_dist, row)


def _bar_idx(sc):
    sc['_n'] = sc.get('_n', 0) + 1
    return sc['_n']


# ═══════════════════════════════════════════════════════════════════════════
# H1 strategies
# ═══════════════════════════════════════════════════════════════════════════

# ─── S1: H1 trend pullback ──────────────────────────────────────────────────
def entry_h1_trend_pullback(bst, slot, row, ts, pair, slip, hspd,
                             sess_cfg, regime, regime_mult,
                             fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if n - sc.get('last', -1000) < 4:
        return False

    ema_f = getattr(row, 'ema_fast', float('nan'))
    ema_s = getattr(row, 'ema_slow', float('nan'))
    atr   = float(getattr(row, 'atr', 0) or 0)
    close = float(getattr(row, 'close', 0))
    if math.isnan(ema_f) or math.isnan(ema_s) or atr <= 0 or close <= 0:
        return False

    dist_to_ema = abs(close - ema_f)
    if dist_to_ema > 0.25 * atr:
        return False    # not pulled back to the fast EMA

    if ema_f > ema_s:
        direction = 'long'
        entry = close; sl = close - 1.0 * atr; tp = close + 2.5 * atr
    elif ema_f < ema_s:
        direction = 'short'
        entry = close; sl = close + 1.0 * atr; tp = close - 2.5 * atr
    else:
        return False

    sl_dist = abs(entry - sl)
    size = _size(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
    if not size or size <= 0:
        return False
    eng.place_pending(sc, ts, direction=direction, entry=entry, sl=sl, tp=tp,
                      size=size, dist=sl_dist, mode='market_next_open')
    sc['last'] = n
    return False


# ─── S2: H1 Donchian breakout (Turtle-style trend follow) ───────────────────
def entry_h1_donchian_breakout(bst, slot, row, ts, pair, slip, hspd,
                                sess_cfg, regime, regime_mult,
                                fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if n - sc.get('last', -1000) < 4:
        return False

    dhi = getattr(row, 'donchian_hi', float('nan'))
    dlo = getattr(row, 'donchian_lo', float('nan'))
    atr = float(getattr(row, 'atr', 0) or 0)
    close = float(getattr(row, 'close', 0))
    if math.isnan(dhi) or math.isnan(dlo) or atr <= 0 or close <= 0:
        return False

    if close > dhi:
        direction = 'long'
        entry = close; sl = close - 1.5 * atr; tp = close + 3.0 * atr
    elif close < dlo:
        direction = 'short'
        entry = close; sl = close + 1.5 * atr; tp = close - 3.0 * atr
    else:
        return False

    sl_dist = abs(entry - sl)
    size = _size(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
    if not size or size <= 0:
        return False
    eng.place_pending(sc, ts, direction=direction, entry=entry, sl=sl, tp=tp,
                      size=size, dist=sl_dist, mode='market_next_open')
    sc['last'] = n
    return False


# ─── S3: H1 BB squeeze → expansion breakout ─────────────────────────────────
def make_h1_bb_squeeze(width_p20: float):
    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        n = _bar_idx(sc)
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
        if n - sc.get('last', -1000) < 4:
            return False

        bw    = getattr(row, 'bb_width', float('nan'))
        bb_up = getattr(row, 'bb_up', float('nan'))
        bb_lo = getattr(row, 'bb_lo', float('nan'))
        atr   = float(getattr(row, 'atr', 0) or 0)
        close = float(getattr(row, 'close', 0))
        if (math.isnan(bw) or math.isnan(bb_up) or math.isnan(bb_lo)
                or atr <= 0 or close <= 0):
            return False

        # Count consecutive squeezed bars
        if bw <= width_p20:
            sc['squeeze'] = sc.get('squeeze', 0) + 1
        else:
            sq = sc.get('squeeze', 0)
            sc['squeeze'] = 0
            # On the bar that EXITS the squeeze, check for band breakout
            if sq >= 6:
                if close > bb_up:
                    direction = 'long'
                    entry = close; sl = close - 1.0 * atr; tp = close + 2.0 * atr
                elif close < bb_lo:
                    direction = 'short'
                    entry = close; sl = close + 1.0 * atr; tp = close - 2.0 * atr
                else:
                    return False
                sl_dist = abs(entry - sl)
                size = _size(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
                if not size or size <= 0:
                    return False
                eng.place_pending(sc, ts, direction=direction, entry=entry,
                                  sl=sl, tp=tp, size=size, dist=sl_dist,
                                  mode='market_next_open')
                sc['last'] = n
        return False
    return entry_fn


# ─── S4: H1 inside-bar breakout (OCO) ───────────────────────────────────────
def entry_h1_inside_bar_breakout(bst, slot, row, ts, pair, slip, hspd,
                                  sess_cfg, regime, regime_mult,
                                  fvg_buf=None, day_sweep=None):
    sc = slot['scratch']
    n = _bar_idx(sc)
    if eng.spread_gate(row):
        return False
    if eng.has_pending(sc):
        return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if n - sc.get('last', -1000) < 4:
        return False

    high = float(getattr(row, 'high', 0))
    low  = float(getattr(row, 'low', 0))
    ph   = getattr(row, 'prev_high', float('nan'))
    pl   = getattr(row, 'prev_low', float('nan'))
    atr  = float(getattr(row, 'atr', 0) or 0)
    if math.isnan(ph) or math.isnan(pl) or atr <= 0 or high <= 0:
        return False

    # Inside bar: current range within prior bar's range
    if not (high <= ph and low >= pl):
        return False

    pip = _pip(pair)
    rng = ph - pl
    if rng <= 0:
        return False
    risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
    long_level  = ph + pip
    short_level = pl - pip
    long_sl     = pl - pip
    short_sl    = ph + pip
    long_tp     = long_level  + 2.0 * rng
    short_tp    = short_level - 2.0 * rng
    long_size   = eng.rv_size(pair, bst.balance, risk, long_level,  rng, row)
    short_size  = eng.rv_size(pair, bst.balance, risk, short_level, rng, row)

    eng.place_oco_pending(sc, ts,
                          long_level=long_level, long_sl=long_sl, long_tp=long_tp,
                          long_size=long_size, long_dist=rng,
                          short_level=short_level, short_sl=short_sl, short_tp=short_tp,
                          short_size=short_size, short_dist=rng)
    sc['last'] = n
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_one(name, entry_fn, train_h1, test_h1, cost_mult=1.0):
    manager_fn = eng.make_manager(exit_hour=EXIT_HOUR, use_breakeven=False)
    slot_class = f"h1_{name[:12]}".replace("-", "_").lower()
    registry = [{
        "id": f"h1_{name}", "family": "battery", "slot_class": slot_class,
        "pairs": [PAIR], "session": "ny", "allow_concurrent": False,
        "regime_mult": {"TRENDING": 1.0, "TRANSITIONING": 1.0, "RANGING": 1.0,
                        "VOLATILE": 1.0, "UNDEFINED": 1.0},
        "params": {},
    }]
    slot_managers = {slot_class: manager_fn}
    slot_entries  = {slot_class: entry_fn}

    out = {}
    for split, df in (("train", train_h1), ("test", test_h1)):
        try:
            trades, _, _ = eng.run_backtest(
                {PAIR: df}, None, None,
                registry, slot_managers, slot_entries,
                cost_mult=cost_mult,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            out[split] = {"n": 0, "sharpe": 0, "pnl": 0, "wr": 0,
                          "error": f"{type(e).__name__}: {e}"}
            continue
        nt = 0 if trades is None or trades.empty else len(trades)
        stats = eng.calc_stats(trades) if nt > 0 else {}
        pnl   = float(trades['pnl'].sum()) if nt > 0 else 0.0
        wr    = float((trades['pnl'] > 0).mean()) if nt > 0 else 0.0
        out[split] = {"n": nt, "sharpe": float(stats.get("sharpe", 0) or 0),
                      "pnl": pnl, "wr": wr,
                      "max_dd": float(stats.get("max_dd", 0) or 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", help="run only this strategy")
    args = ap.parse_args()

    print("Loading M1 cache (need >=2 pairs for cross-pair builder)...")
    train_dfs, test_dfs, _ = eng.load_all_data(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"])

    print(f"\nResampling {PAIR} M1 -> H1...")
    train_h1 = resample_h1(train_dfs[PAIR])
    test_h1  = resample_h1(test_dfs[PAIR])

    # ── Resample sanity header ──────────────────────────────────────────────
    pip = _pip(PAIR)
    atr_med_pips = float(train_h1['atr'].median()) / pip
    print(f"  train H1 bars: {len(train_h1):,}  test H1 bars: {len(test_h1):,}")
    print(f"  H1 ATR median: {atr_med_pips:.1f} pips  (M1 was ~2-5 pips)")
    print(f"  ma_trend sign flips (train): "
          f"{int((np.sign(train_h1['ma_trend']).diff() != 0).sum())} "
          f"over {len(train_h1)} bars")
    req = ['timestamp', 'open', 'high', 'low', 'close', 'hour', 'minute',
           'ma_trend', 'spread_mean', 'spread_median', 'atr']
    missing = [c for c in req if c not in train_h1.columns]
    print(f"  required cols present: {'YES' if not missing else 'MISSING ' + str(missing)}")
    print()

    width_p20 = float(np.percentile(train_h1['bb_width'].dropna(), 20))

    strategies = [
        ("h1_trend_pullback",      entry_h1_trend_pullback),
        ("h1_donchian_breakout",   entry_h1_donchian_breakout),
        ("h1_bb_squeeze_expansion", make_h1_bb_squeeze(width_p20)),
        ("h1_inside_bar_breakout", entry_h1_inside_bar_breakout),
    ]
    if args.strategy:
        strategies = [s for s in strategies if s[0] == args.strategy]
        if not strategies:
            print(f"unknown strategy. choices: "
                  f"{[s[0] for s in [('h1_trend_pullback',0),('h1_donchian_breakout',0),('h1_bb_squeeze_expansion',0),('h1_inside_bar_breakout',0)]]}")
            return 2

    rows = []
    for name, fn in strategies:
        print(f"=== {name} ===")
        r = run_one(name, fn, train_h1, test_h1)
        rows.append((name, r))
        t, te = r['train'], r['test']
        print(f"  train: n={t['n']:>4d} sh={t['sharpe']:+6.2f} wr={t['wr']*100:>4.1f}% "
              f"${t['pnl']:>+9,.0f}")
        print(f"  test:  n={te['n']:>4d} sh={te['sharpe']:+6.2f} wr={te['wr']*100:>4.1f}% "
              f"${te['pnl']:>+9,.0f}")
        is_cand = (te['sharpe'] >= 0.5 and te['n'] >= 30
                   and t['pnl'] > 0 and te['pnl'] > 0)
        print(f"  CANDIDATE: {'YES ✓' if is_cand else 'no —'}")
        print()

    # ── Summary + report ────────────────────────────────────────────────────
    print("=" * 90)
    print(f"{'STRATEGY':28s} {'TR_N':>5s} {'TR_SH':>7s} {'TR_PNL':>10s} "
          f"{'TE_N':>5s} {'TE_SH':>7s} {'TE_PNL':>10s} {'CAND':>5s}")
    print("-" * 90)
    cands = []
    lines = ["# H1 POC — EUR/USD results\n"]
    from datetime import datetime, timezone
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} UTC_\n")
    lines.append(f"H1 ATR median: {atr_med_pips:.1f} pips. "
                 f"Gate: test_sharpe ≥ 0.5, n_test ≥ 30, train+test PnL same sign.\n")
    lines.append("| strategy | tr_n | tr_sh | tr_pnl | te_n | te_sh | te_pnl | cand |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, r in rows:
        t, te = r['train'], r['test']
        is_cand = (te['sharpe'] >= 0.5 and te['n'] >= 30
                   and t['pnl'] > 0 and te['pnl'] > 0)
        mark = "✓" if is_cand else "—"
        print(f"{name:28s} {t['n']:>5d} {t['sharpe']:>+7.2f} ${t['pnl']:>+8,.0f} "
              f"{te['n']:>5d} {te['sharpe']:>+7.2f} ${te['pnl']:>+8,.0f} {mark:>5s}")
        lines.append(f"| `{name}` | {t['n']} | {t['sharpe']:+.2f} | "
                     f"${t['pnl']:+,.0f} | {te['n']} | {te['sharpe']:+.2f} | "
                     f"${te['pnl']:+,.0f} | {mark} |")
        if is_cand:
            cands.append((name, te['sharpe'], te['n'], te['pnl']))
    print("=" * 90)
    print()
    lines.append("\n## Decision\n")
    if cands:
        print(f"PROMOTION CANDIDATES ({len(cands)}) — H1 PIVOT VALIDATED:")
        lines.append(f"**{len(cands)} candidate(s) — H1 pivot validated.** "
                     f"Proceed to full multi-pair H1 cache build.\n")
        for n, sh, nn, pnl in sorted(cands, key=lambda x: -x[1]):
            print(f"  ✓ {n}: test_sharpe=+{sh:.2f} n={nn} pnl=${pnl:+,.0f}")
            lines.append(f"- **{n}**: test_sharpe=+{sh:.2f}, n={nn}, "
                         f"test_pnl=${pnl:+,.0f}")
    else:
        print("No H1 strategy cleared the bar on EUR/USD.")
        lines.append("_No candidate._ Timeframe alone did not unlock edge on "
                     "EUR/USD. The cost-math hypothesis is weakened; reconsider "
                     "(market pivot, or accept the single USD/JPY M1 candidate).")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
