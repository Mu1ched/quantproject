# Strategy Battery V2 — Results

_Generated 2026-06-07T10:52:15.807004+00:00 UTC_

Acceptance gate: `test_sharpe ≥ 0.5 AND n_test ≥ 30 AND train_pnl > 0 AND test_pnl > 0`.

## Per-strategy results

| strategy | pair | tr_n | tr_sh | tr_pnl | te_n | te_sh | te_pnl | candidate |
|---|---|---|---|---|---|---|---|---|
| `tokyo_london_breakout` | EUR_USD | 268 | -0.62 | $-3,016 | 66 | -0.64 | $-717 | — |
| `tokyo_london_breakout` | GBP_USD | 274 | +0.15 | $+675 | 68 | -1.89 | $-2,089 | — |
| `tokyo_london_breakout` | USD_JPY | 239 | +0.89 | $+3,329 | 60 | +1.79 | $+1,236 | ✓ |
| `tokyo_london_breakout` | EUR_JPY | 258 | +0.10 | $+415 | 63 | -1.00 | $-1,165 | — |
| `ny_1330_news_momentum` | EUR_USD | 372 | -2.44 | $-32,279 | 83 | -3.20 | $-9,402 | — |
| `ny_1330_news_momentum` | GBP_USD | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `ny_1330_news_momentum` | USD_JPY | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `ny_1330_news_momentum` | EUR_JPY | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `friday_afternoon_revert` | EUR_USD | 83 | -3.90 | $-25,834 | 20 | -3.90 | $-12,916 | — |
| `friday_afternoon_revert` | GBP_USD | 84 | -4.62 | $-28,374 | 20 | -5.22 | $-8,881 | — |
| `friday_afternoon_revert` | USD_JPY | 83 | -2.39 | $-21,869 | 20 | -5.15 | $-9,793 | — |
| `friday_afternoon_revert` | EUR_JPY | 82 | -3.56 | $-30,697 | 20 | -5.64 | $-10,980 | — |
| `lowvol_bb_extreme_revert` | EUR_USD | 498 | -4.04 | $-99,999 | 119 | -11.28 | $-94,948 | — |
| `lowvol_bb_extreme_revert` | GBP_USD | 497 | -4.01 | $-100,000 | 118 | -6.71 | $-98,756 | — |
| `lowvol_bb_extreme_revert` | USD_JPY | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `lowvol_bb_extreme_revert` | EUR_JPY | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `high_adx_trend_pullback` | EUR_USD | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `high_adx_trend_pullback` | GBP_USD | 4 | -1.02 | $-3,535 | 0 | +0.00 | $+0 | — |
| `high_adx_trend_pullback` | USD_JPY | 133 | -3.30 | $-35,750 | 28 | -3.51 | $-10,299 | — |
| `high_adx_trend_pullback` | EUR_JPY | 137 | -4.62 | $-60,603 | 30 | -5.63 | $-19,415 | — |
| `h1_trend_orb` | EUR_USD | 306 | -0.26 | $-1,258 | 71 | -2.86 | $-3,222 | — |
| `h1_trend_orb` | GBP_USD | 317 | +0.37 | $+2,175 | 75 | -2.59 | $-2,973 | — |
| `h1_trend_orb` | USD_JPY | 290 | +1.37 | $+7,259 | 69 | -1.30 | $-1,270 | — |
| `h1_trend_orb` | EUR_JPY | 303 | +0.27 | $+1,316 | 76 | -2.06 | $-2,452 | — |
| `news_drift_15min` | EUR_USD | 289 | -4.28 | $-29,246 | 66 | -4.76 | $-8,226 | — |
| `news_drift_15min` | GBP_USD | 256 | -5.98 | $-69,351 | 76 | -8.32 | $-21,387 | — |
| `news_drift_15min` | USD_JPY | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `news_drift_15min` | EUR_JPY | 0 | +0.00 | $+0 | 0 | +0.00 | $+0 | — |
| `dxy_proxy_divergence_eurusd` | EUR_USD | 445 | -8.38 | $-94,881 | 106 | -14.23 | $-45,460 | — |
| `h4_low_vol_breakout` | EUR_USD | 403 | -9.41 | $-84,665 | 98 | -12.01 | $-35,454 | — |
| `h4_low_vol_breakout` | GBP_USD | 408 | -8.64 | $-92,757 | 94 | -13.68 | $-40,730 | — |
| `h4_low_vol_breakout` | USD_JPY | 402 | -6.14 | $-58,112 | 97 | -8.55 | $-27,854 | — |
| `h4_low_vol_breakout` | EUR_JPY | 400 | -9.02 | $-80,829 | 90 | -9.33 | $-30,411 | — |

## Promotion candidates

- **tokyo_london_breakout** on **USD_JPY**: test_sharpe=+1.79, n=60, test_pnl=$+1,236