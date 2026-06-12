# Quantproject diagnostics — cost stack vs edge

Pairs used: `['EUR_USD', 'USD_JPY']` (intersection of NY_PAIRS and cached pairs).

## Part 1 — Null strategy (ORB-shaped, random direction)

Under a no-edge null with calibrated costs, test Sharpe should be roughly bell-shaped around zero. A materially negative mean indicates the cost stack is over-modeled.

Sweep ID: `null_ny_diag_20260602_104839`

**Test-Sharpe distribution across seeds:**

| metric | value |
|---|---|
| n | 200 |
| mean | -0.967 |
| median | -0.94 |
| std | 1.092 |
| min | -4.09 |
| p25 | -1.709 |
| p75 | -0.188 |
| max | 1.878 |
| frac_pos | 0.195 |

Mean trades per hypothesis — train: 349.7, test: 109.5

## Part 2 — Cost-sensitivity (SWEEP_ORB_NY)

Higher cost_mult should monotonically reduce test Sharpe. Strong slope = results dominated by cost assumptions. cost_mult=0.0 is not truly zero (news_mult and swap aren't scaled), but the floor is small for intraday ORB.

| cost_mult | n | mean | median | std | min | p25 | p75 | max | frac_pos | mean_trades |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 8 | 0.483 | 0.54 | 0.274 | 0.082 | 0.364 | 0.66 | 0.77 | 1.0 | 152.0 |
| 0.25 | 8 | 0.263 | 0.338 | 0.278 | -0.16 | 0.171 | 0.43 | 0.537 | 0.75 | 152.0 |
| 0.5 | 8 | 0.124 | 0.205 | 0.276 | -0.297 | 0.033 | 0.296 | 0.385 | 0.75 | 152.0 |
| 1.0 | 8 | -0.26 | -0.164 | 0.299 | -0.721 | -0.351 | -0.073 | 0.011 | 0.25 | 152.0 |
| 1.5 | 8 | -0.427 | -0.334 | 0.308 | -0.901 | -0.519 | -0.242 | -0.138 | 0.0 | 152.0 |

Sweep IDs:
- `cost_mult=0.0` -> `cost_sens_orb_ny_cm0_20260602_115143`
- `cost_mult=0.25` -> `cost_sens_orb_ny_cm0.25_20260602_115355`
- `cost_mult=0.5` -> `cost_sens_orb_ny_cm0.5_20260602_115546`
- `cost_mult=1.0` -> `cost_sens_orb_ny_cm1_20260602_115742`
- `cost_mult=1.5` -> `cost_sens_orb_ny_cm1.5_20260602_115941`

## Interpretation

- Null mean test Sharpe: **-0.967**
- ORB mean test Sharpe at cost_mult=0.0: **0.483**
- ORB mean test Sharpe at cost_mult=1.0: **-0.26**

**Null is materially negative — cost stack appears over-modeled.** Pure noise under your engine's costs loses on average, which means every real strategy starts with a cost handicap. Recalibrate against live TCA before drawing edge conclusions.


ORB has raw edge that vanishes under realistic costs — either find a stronger signal or reduce per-trade cost via wider stops, fewer trades, or better timing.
