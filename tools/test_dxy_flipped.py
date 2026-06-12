"""
Quick test of two flipped DXY-divergence variants.

V2 of the battery had a DXY-divergence fade that lost catastrophically
(test Sharpe -14.23). The signal exists — but in the opposite direction.
Test two flipped versions to see which (if any) has real edge.

Variant A: trend-CONTINUATION with bb_pct confirmation.
  usd_score == +3 AND bb_pct <= 0.15 → SHORT EUR/USD (USD strong + EUR/USD already falling)
  usd_score == -3 AND bb_pct >= 0.85 → LONG  EUR/USD (USD weak + EUR/USD already rising)

Variant B: pure trend-follow, no bb gate (fires more, less selective).
  usd_score == +3 → SHORT EUR/USD
  usd_score == -3 → LONG  EUR/USD
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import edge_engine as eng


def _bar_idx(sc):
    sc['_bar_n'] = sc.get('_bar_n', 0) + 1
    return sc['_bar_n']


def _size_for(bst, regime_mult, pair, balance, entry, sl_dist, row):
    risk = eng.resolve_risk(bst, regime_mult, 'dynamic')
    return eng.rv_size(pair, balance, risk, entry, sl_dist, row)


def make_dxy_continuation_entry(other_pair_dfs: dict, use_bb_gate: bool):
    """Factory: produces an entry_fn that trend-FOLLOWS the USD-score consensus.

    use_bb_gate=True  → Variant A (with bb_pct confirmation; rarer, more selective)
    use_bb_gate=False → Variant B (no bb gate; fires more often)
    """
    other_lookups = {p: (df.set_index('timestamp') if 'timestamp' in df.columns else df)
                     for p, df in other_pair_dfs.items()}

    def entry_fn(bst, slot, row, ts, pair, slip, hspd,
                  sess_cfg, regime, regime_mult,
                  fvg_buf=None, day_sweep=None):
        sc = slot['scratch']
        cur_n = _bar_idx(sc)
        if eng.spread_gate(row):
            return False
        if eng.has_pending(sc):
            return eng.check_and_fill(sc, row, slot, ts, regime, hspd, slip)

        if cur_n - sc.get('last_trade_bar', -10_000) < 60:
            return False

        try:
            usd_jpy = other_lookups['USD_JPY'].loc[ts]
            gbp_usd = other_lookups['GBP_USD'].loc[ts]
        except (KeyError, TypeError):
            return False

        m_eu = getattr(row,     'momentum_10', float('nan'))
        m_uj = getattr(usd_jpy, 'momentum_10', float('nan'))
        m_gu = getattr(gbp_usd, 'momentum_10', float('nan'))
        if any(math.isnan(x) for x in (m_eu, m_uj, m_gu)):
            return False

        def sgn(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)
        usd_score = sgn(m_uj) - sgn(m_eu) - sgn(m_gu)
        if abs(usd_score) < 3:
            return False

        atr = float(getattr(row, 'atr', 0) or 0)
        close = float(getattr(row, 'close', 0))
        if atr <= 0 or close <= 0:
            return False

        bb = getattr(row, 'bb_pct', float('nan'))
        if use_bb_gate and math.isnan(bb):
            return False

        if usd_score == 3:
            # USD strong → SHORT EUR/USD (trend-follow)
            if use_bb_gate and not (bb <= 0.15):
                return False
            direction = 'short'
            entry = close
            sl    = close + 1.0 * atr
            tp    = close - 1.0 * atr
        else:   # usd_score == -3
            # USD weak → LONG EUR/USD (trend-follow)
            if use_bb_gate and not (bb >= 0.85):
                return False
            direction = 'long'
            entry = close
            sl    = close - 1.0 * atr
            tp    = close + 1.0 * atr

        sl_dist = abs(entry - sl)
        size = _size_for(bst, regime_mult, pair, bst.balance, entry, sl_dist, row)
        if not size or size <= 0:
            return False

        eng.place_pending(sc, ts, direction=direction,
                           entry=entry, sl=sl, tp=tp,
                           size=size, dist=sl_dist,
                           mode='market_next_open')
        sc['last_trade_bar'] = cur_n
        return False

    return entry_fn


def run_variant(name: str, train_dfs, test_dfs, use_bb_gate: bool):
    manager_fn = eng.make_manager(exit_hour=21, use_breakeven=False)
    slot_class = f"dxy_{name[:10]}".replace("-", "_").lower()
    pair = "EUR_USD"
    registry = [{
        "id": f"dxy_{name}", "family": "battery", "slot_class": slot_class,
        "pairs": [pair], "session": "ny",
        "allow_concurrent": False,
        "regime_mult": {"TRENDING": 1.0, "TRANSITIONING": 1.0, "RANGING": 1.0,
                        "VOLATILE": 0.5, "UNDEFINED": 0.5},
        "params": {},
    }]
    out = {}
    for split, dfs in (("train", train_dfs), ("test", test_dfs)):
        others = {p: df for p, df in dfs.items() if p != pair}
        entry_fn = make_dxy_continuation_entry(others, use_bb_gate)
        slot_managers = {slot_class: manager_fn}
        slot_entries  = {slot_class: entry_fn}
        trades, _, _ = eng.run_backtest(
            {pair: dfs[pair]}, None, None,
            registry, slot_managers, slot_entries,
            cost_mult=1.0,
        )
        n = 0 if trades is None or trades.empty else len(trades)
        stats = eng.calc_stats(trades) if n > 0 else {}
        pnl   = float(trades['pnl'].sum()) if n > 0 else 0.0
        wr    = float((trades['pnl'] > 0).mean()) if n > 0 else 0.0
        out[split] = {"n": n, "sharpe": float(stats.get("sharpe", 0) or 0),
                      "pnl": pnl, "wr": wr,
                      "max_dd": float(stats.get("max_dd", 0) or 0)}
    return out


def main():
    print("Loading cached data...")
    train_dfs, test_dfs, _ = eng.load_all_data(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"],
    )
    print()

    for variant_name, use_bb_gate in (("A_continuation", True),
                                      ("B_pure_trend",   False)):
        print(f"=== Variant {variant_name} (bb_gate={use_bb_gate}) ===")
        r = run_variant(variant_name, train_dfs, test_dfs, use_bb_gate)
        t, te = r["train"], r["test"]
        print(f"  train: n={t['n']:>4d} sh={t['sharpe']:+7.2f} wr={t['wr']*100:>4.1f}% "
              f"pnl=${t['pnl']:+8,.0f} dd=${t['max_dd']:+7,.0f}")
        print(f"  test:  n={te['n']:>4d} sh={te['sharpe']:+7.2f} wr={te['wr']*100:>4.1f}% "
              f"pnl=${te['pnl']:+8,.0f} dd=${te['max_dd']:+7,.0f}")
        is_cand = (te['sharpe'] >= 0.5 and te['n'] >= 30
                   and t['pnl'] > 0 and te['pnl'] > 0)
        print(f"  CANDIDATE: {'✓ YES' if is_cand else '— no'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
