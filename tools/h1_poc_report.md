# H1 POC — EUR/USD results

_Generated 2026-06-07T20:44:26.150031+00:00 UTC_

H1 ATR median: 12.5 pips. Gate: test_sharpe ≥ 0.5, n_test ≥ 30, train+test PnL same sign.

| strategy | tr_n | tr_sh | tr_pnl | te_n | te_sh | te_pnl | cand |
|---|---|---|---|---|---|---|---|
| `h1_trend_pullback` | 369 | -2.06 | $-28,935 | 84 | +0.02 | $+5 | — |
| `h1_donchian_breakout` | 355 | -0.67 | $-9,886 | 83 | +0.28 | $-2,183 | — |
| `h1_bb_squeeze_expansion` | 78 | -0.67 | $-5,172 | 10 | -0.94 | $-1,005 | — |
| `h1_inside_bar_breakout` | 380 | -3.52 | $-39,060 | 87 | -2.81 | $-9,511 | — |

## Decision

_No candidate._ Timeframe alone did not unlock edge on EUR/USD. The cost-math hypothesis is weakened; reconsider (market pivot, or accept the single USD/JPY M1 candidate).