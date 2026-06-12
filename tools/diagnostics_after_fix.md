# Quantproject diagnostics — after 2026-06-02 cost recalibration

Pairs: `['EUR_USD', 'USD_JPY']`. Seeds per null sweep: `30`. Default cost_mult tested: `0.5`.

Engine knobs at runtime: `SLIP_RATIO_STATIC=0.1`, `NEWS_SLIPPAGE_MULT=2.0`.

## Dynamic mode

`use_dynamic_spread = True`, `cost_mult = 0.5`. Null sweep id: `null_ny_after_dynamic_20260605_151229`.

**Null test-Sharpe distribution:**

| n | mean | median | std | p25 | p75 | max | frac_pos | mean_trades |
|---|---|---|---|---|---|---|---|---|
| 30 | -0.367 | -0.665 | 1.079 | -0.961 | 0.262 | 2.707 | 0.3 | 106.6 |

**ORB performance under dynamic spreads:**

| n | mean | median | best | frac_pos |
|---|---|---|---|---|
| 8 | -0.035 | 0.059 | 0.318 | 0.75 |

ORB dynamic sweep id: `orb_ny_after_dynamic_20260605_152101`.

## Verdict

Dynamic null mean = **-0.367**. No pre-recal baseline (the first diagnostic ran static-only), but a value in [−0.6, +0.2] is expected; large negatives suggest SLIPPAGE_PROFILE or news multipliers still bite. Compare against your live MT5 PnL once you have trades.
