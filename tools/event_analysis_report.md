# Event-conditional return analysis

- Event types tested: **7**
- Pairs tested: ['AUD_USD', 'EUR_JPY', 'EUR_USD', 'GBP_USD', 'USD_JPY']
- Total event occurrences in log: 11,073
- Rare events dropped (< 100): ['atr_regime_shift', 'round_number_touch']

## Validated events (passed stability + cross-pair)

_NONE PASSED._

**Phase 2 stop condition triggered**: no event type produced statistically reliable forward-return bias across multiple windows and pairs. Document and pivot.


## Per-(event,pair) details

| event_type | pair | horizon | n | mean_norm | t_vs_baseline | stability |
|---|---|---|---|---|---|---|
| `first_m15_sweep_of_prior_session` | AUD_USD | 5 | 387 | +0.146 | +1.28 | ✓ |
| `first_m15_sweep_of_prior_session` | AUD_USD | 15 | 387 | +0.154 | +0.76 | ✗ |
| `first_m15_sweep_of_prior_session` | AUD_USD | 30 | 387 | +0.358 | +1.30 | ✗ |
| `first_m15_sweep_of_prior_session` | AUD_USD | 60 | 387 | +0.433 | +1.12 | ✓ |
| `first_m15_sweep_of_prior_session` | EUR_JPY | 5 | 318 | +0.079 | +0.61 | ✗ |
| `first_m15_sweep_of_prior_session` | EUR_JPY | 15 | 318 | +0.079 | +0.28 | ✗ |
| `first_m15_sweep_of_prior_session` | EUR_JPY | 30 | 318 | +0.067 | +0.07 | ✗ |
| `first_m15_sweep_of_prior_session` | EUR_JPY | 60 | 318 | +0.028 | -0.18 | ✗ |
| `first_m15_sweep_of_prior_session` | EUR_USD | 5 | 354 | +0.058 | +0.45 | ✗ |
| `first_m15_sweep_of_prior_session` | EUR_USD | 15 | 354 | +0.185 | +0.87 | ✗ |
| `first_m15_sweep_of_prior_session` | EUR_USD | 30 | 354 | +0.060 | +0.14 | ✗ |
| `first_m15_sweep_of_prior_session` | EUR_USD | 60 | 354 | -0.057 | -0.20 | ✗ |
| `first_m15_sweep_of_prior_session` | GBP_USD | 5 | 361 | -0.159 | -1.20 | ✓ |
| `first_m15_sweep_of_prior_session` | GBP_USD | 15 | 361 | -0.179 | -0.90 | ✗ |
| `first_m15_sweep_of_prior_session` | GBP_USD | 30 | 361 | -0.072 | -0.29 | ✗ |
| `first_m15_sweep_of_prior_session` | GBP_USD | 60 | 361 | -0.017 | -0.11 | ✗ |
| `first_m15_sweep_of_prior_session` | USD_JPY | 5 | 311 | +0.027 | +0.19 | ✗ |
| `first_m15_sweep_of_prior_session` | USD_JPY | 15 | 311 | +0.035 | +0.12 | ✗ |
| `first_m15_sweep_of_prior_session` | USD_JPY | 30 | 311 | +0.090 | +0.25 | ✗ |
| `first_m15_sweep_of_prior_session` | USD_JPY | 60 | 311 | +0.242 | +0.51 | ✗ |
| `gap_open` | AUD_USD | 5 | 131 | +0.147 | +0.80 | ✗ |
| `gap_open` | AUD_USD | 15 | 131 | +0.161 | +0.65 | ✗ |
| `gap_open` | AUD_USD | 30 | 131 | +0.280 | +0.85 | ✗ |
| `gap_open` | AUD_USD | 60 | 131 | +1.041 | +2.02 | ✗ |
| `gap_open` | EUR_JPY | 5 | 130 | -0.015 | -0.13 | ✗ |
| `gap_open` | EUR_JPY | 15 | 130 | +0.170 | +0.47 | ✗ |
| `gap_open` | EUR_JPY | 30 | 130 | +0.136 | +0.26 | ✗ |
| `gap_open` | EUR_JPY | 60 | 130 | +1.027 | +1.88 | ✗ |
| `gap_open` | EUR_USD | 5 | 116 | -0.131 | -0.99 | ✗ |
| `gap_open` | EUR_USD | 15 | 116 | +0.001 | -0.05 | ✗ |
| `gap_open` | EUR_USD | 30 | 116 | +0.092 | +0.28 | ✗ |
| `gap_open` | EUR_USD | 60 | 116 | +0.330 | +0.69 | ✗ |
| `gap_open` | GBP_USD | 5 | 122 | +0.180 | +0.97 | ✗ |
| `gap_open` | GBP_USD | 15 | 122 | -0.040 | -0.21 | ✗ |
| `gap_open` | GBP_USD | 30 | 122 | -0.067 | -0.31 | ✗ |
| `gap_open` | GBP_USD | 60 | 122 | +0.285 | +0.58 | ✗ |
| `gap_open` | USD_JPY | 5 | 126 | -0.089 | -0.60 | ✗ |
| `gap_open` | USD_JPY | 15 | 126 | +0.074 | +0.20 | ✗ |
| `gap_open` | USD_JPY | 30 | 126 | +0.270 | +0.68 | ✗ |
| `gap_open` | USD_JPY | 60 | 126 | +1.202 | +2.27 | ✗ |
| `month_end_london_close` | AUD_USD | 5 | 47 | -0.399 | -1.70 | ✗ |
| `month_end_london_close` | AUD_USD | 15 | 47 | +0.458 | +1.20 | ✗ |
| `month_end_london_close` | AUD_USD | 30 | 47 | +0.189 | +0.29 | ✗ |
| `month_end_london_close` | AUD_USD | 60 | 47 | +1.437 | +1.69 | ✗ |
| `month_end_london_close` | EUR_JPY | 5 | 47 | -0.181 | -0.68 | ✗ |
| `month_end_london_close` | EUR_JPY | 15 | 47 | -0.257 | -0.65 | ✗ |
| `month_end_london_close` | EUR_JPY | 30 | 47 | -0.146 | -0.35 | ✗ |
| `month_end_london_close` | EUR_JPY | 60 | 47 | -0.764 | -0.64 | ✗ |
| `month_end_london_close` | EUR_USD | 5 | 48 | -0.681 | -2.12 | ✗ |
| `month_end_london_close` | EUR_USD | 15 | 48 | -0.093 | -0.24 | ✗ |
| `month_end_london_close` | EUR_USD | 30 | 48 | -0.446 | -0.71 | ✗ |
| `month_end_london_close` | EUR_USD | 60 | 48 | +0.335 | +0.30 | ✗ |
| `month_end_london_close` | GBP_USD | 5 | 50 | -0.690 | -2.55 | ✗ |
| `month_end_london_close` | GBP_USD | 15 | 50 | +0.053 | +0.09 | ✗ |
| `month_end_london_close` | GBP_USD | 30 | 50 | -0.115 | -0.20 | ✗ |
| `month_end_london_close` | GBP_USD | 60 | 50 | +0.570 | +0.48 | ✗ |
| `month_end_london_close` | USD_JPY | 5 | 46 | +0.250 | +0.96 | ✗ |
| `month_end_london_close` | USD_JPY | 15 | 46 | -0.323 | -0.89 | ✗ |
| `month_end_london_close` | USD_JPY | 30 | 46 | +0.103 | +0.16 | ✗ |
| `month_end_london_close` | USD_JPY | 60 | 46 | -0.194 | -0.26 | ✗ |
| `nfp_wednesday_close` | AUD_USD | 5 | 23 | +0.245 | +0.57 | ✗ |
| `nfp_wednesday_close` | AUD_USD | 15 | 23 | +1.671 | +1.36 | ✗ |
| `nfp_wednesday_close` | AUD_USD | 30 | 23 | +0.209 | +0.17 | ✗ |
| `nfp_wednesday_close` | AUD_USD | 60 | 23 | -0.579 | -0.65 | ✗ |
| `nfp_wednesday_close` | EUR_JPY | 5 | 23 | -0.043 | -0.12 | ✗ |
| `nfp_wednesday_close` | EUR_JPY | 15 | 23 | -0.290 | -0.18 | ✗ |
| `nfp_wednesday_close` | EUR_JPY | 30 | 23 | -2.633 | -1.11 | ✗ |
| `nfp_wednesday_close` | EUR_JPY | 60 | 23 | -5.072 | -1.76 | ✗ |
| `nfp_wednesday_close` | EUR_USD | 5 | 23 | +0.110 | +0.25 | ✗ |
| `nfp_wednesday_close` | EUR_USD | 15 | 23 | +1.006 | +0.86 | ✗ |
| `nfp_wednesday_close` | EUR_USD | 30 | 23 | -0.202 | -0.23 | ✗ |
| `nfp_wednesday_close` | EUR_USD | 60 | 23 | +0.733 | +0.81 | ✗ |
| `nfp_wednesday_close` | GBP_USD | 5 | 23 | +0.261 | +0.59 | ✗ |
| `nfp_wednesday_close` | GBP_USD | 15 | 23 | +1.488 | +1.43 | ✗ |
| `nfp_wednesday_close` | GBP_USD | 30 | 23 | +1.246 | +1.50 | ✗ |
| `nfp_wednesday_close` | GBP_USD | 60 | 23 | +2.497 | +1.96 | ✗ |
| `nfp_wednesday_close` | USD_JPY | 5 | 22 | +0.065 | +0.17 | ✗ |
| `nfp_wednesday_close` | USD_JPY | 15 | 22 | -1.307 | -0.66 | ✗ |
| `nfp_wednesday_close` | USD_JPY | 30 | 22 | -3.642 | -1.06 | ✗ |
| `nfp_wednesday_close` | USD_JPY | 60 | 22 | -7.523 | -1.85 | ✗ |
| `range_sweep_close_inside_15m` | AUD_USD | 5 | 785 | +0.076 | +1.11 | ✗ |
| `range_sweep_close_inside_15m` | AUD_USD | 15 | 785 | +0.085 | +0.71 | ✗ |
| `range_sweep_close_inside_15m` | AUD_USD | 30 | 785 | -0.067 | -0.66 | ✗ |
| `range_sweep_close_inside_15m` | AUD_USD | 60 | 785 | +0.019 | -0.11 | ✗ |
| `range_sweep_close_inside_15m` | EUR_JPY | 5 | 563 | -0.037 | -0.70 | ✗ |
| `range_sweep_close_inside_15m` | EUR_JPY | 15 | 563 | +0.024 | +0.00 | ✗ |
| `range_sweep_close_inside_15m` | EUR_JPY | 30 | 563 | -0.046 | -0.62 | ✗ |
| `range_sweep_close_inside_15m` | EUR_JPY | 60 | 563 | -0.004 | -0.53 | ✗ |
| `range_sweep_close_inside_15m` | EUR_USD | 5 | 1151 | -0.032 | -0.69 | ✗ |
| `range_sweep_close_inside_15m` | EUR_USD | 15 | 1151 | -0.032 | -0.49 | ✗ |
| `range_sweep_close_inside_15m` | EUR_USD | 30 | 1151 | -0.179 | -1.52 | ✓ |
| `range_sweep_close_inside_15m` | EUR_USD | 60 | 1151 | -0.105 | -0.70 | ✓ |
| `range_sweep_close_inside_15m` | GBP_USD | 5 | 1329 | -0.033 | -0.81 | ✗ |
| `range_sweep_close_inside_15m` | GBP_USD | 15 | 1329 | -0.136 | -1.81 | ✓ |
| `range_sweep_close_inside_15m` | GBP_USD | 30 | 1329 | +0.062 | +0.39 | ✗ |
| `range_sweep_close_inside_15m` | GBP_USD | 60 | 1329 | +0.194 | +0.97 | ✗ |
| `range_sweep_close_inside_15m` | USD_JPY | 5 | 422 | -0.023 | -0.33 | ✗ |
| `range_sweep_close_inside_15m` | USD_JPY | 15 | 422 | +0.152 | +0.98 | ✗ |
| `range_sweep_close_inside_15m` | USD_JPY | 30 | 422 | +0.211 | +0.93 | ✗ |
| `range_sweep_close_inside_15m` | USD_JPY | 60 | 422 | +0.175 | +0.55 | ✗ |
| `spread_spike_then_calm` | EUR_JPY | 5 | 16 | +0.019 | +0.04 | ✗ |
| `spread_spike_then_calm` | EUR_JPY | 15 | 16 | -0.463 | -0.89 | ✗ |
| `spread_spike_then_calm` | EUR_JPY | 30 | 16 | -0.530 | -0.54 | ✗ |
| `spread_spike_then_calm` | EUR_JPY | 60 | 16 | -4.091 | -2.62 | ✗ |
| `spread_spike_then_calm` | EUR_USD | 5 | 53 | +0.293 | +1.25 | ✗ |
| `spread_spike_then_calm` | EUR_USD | 15 | 53 | +0.225 | +0.53 | ✗ |
| `spread_spike_then_calm` | EUR_USD | 30 | 53 | +0.171 | +0.30 | ✗ |
| `spread_spike_then_calm` | EUR_USD | 60 | 53 | -0.371 | -0.27 | ✗ |
| `spread_spike_then_calm` | GBP_USD | 5 | 42 | -0.108 | -0.49 | ✗ |
| `spread_spike_then_calm` | GBP_USD | 15 | 42 | -0.293 | -1.11 | ✗ |
| `spread_spike_then_calm` | GBP_USD | 30 | 42 | -0.069 | -0.22 | ✗ |
| `spread_spike_then_calm` | GBP_USD | 60 | 42 | -0.646 | -1.22 | ✗ |
| `spread_spike_then_calm` | USD_JPY | 5 | 68 | -0.165 | -0.64 | ✗ |
| `spread_spike_then_calm` | USD_JPY | 15 | 68 | -0.374 | -0.68 | ✗ |
| `spread_spike_then_calm` | USD_JPY | 30 | 68 | +0.170 | +0.26 | ✗ |
| `spread_spike_then_calm` | USD_JPY | 60 | 68 | +0.825 | +0.70 | ✗ |
| `tick_imb_streak_3` | AUD_USD | 5 | 770 | +0.052 | +0.65 | ✗ |
| `tick_imb_streak_3` | AUD_USD | 15 | 770 | +0.082 | +0.57 | ✗ |
| `tick_imb_streak_3` | AUD_USD | 30 | 770 | +0.023 | -0.01 | ✗ |
| `tick_imb_streak_3` | AUD_USD | 60 | 770 | +0.327 | +1.20 | ✗ |
| `tick_imb_streak_3` | EUR_JPY | 5 | 570 | +0.148 | +1.70 | ✗ |
| `tick_imb_streak_3` | EUR_JPY | 15 | 570 | +0.009 | -0.11 | ✗ |
| `tick_imb_streak_3` | EUR_JPY | 30 | 570 | +0.166 | +0.67 | ✗ |
| `tick_imb_streak_3` | EUR_JPY | 60 | 570 | +0.637 | +1.76 | ✗ |
| `tick_imb_streak_3` | EUR_USD | 5 | 579 | -0.133 | -1.48 | ✗ |
| `tick_imb_streak_3` | EUR_USD | 15 | 579 | -0.198 | -1.01 | ✗ |
| `tick_imb_streak_3` | EUR_USD | 30 | 579 | +0.297 | +0.81 | ✗ |
| `tick_imb_streak_3` | EUR_USD | 60 | 579 | +0.744 | +1.76 | ✗ |
| `tick_imb_streak_3` | GBP_USD | 5 | 994 | +0.005 | -0.00 | ✗ |
| `tick_imb_streak_3` | GBP_USD | 15 | 994 | +0.157 | +1.26 | ✓ |
| `tick_imb_streak_3` | GBP_USD | 30 | 994 | +0.262 | +1.55 | ✗ |
| `tick_imb_streak_3` | GBP_USD | 60 | 994 | +0.300 | +1.24 | ✗ |
| `tick_imb_streak_3` | USD_JPY | 5 | 1017 | -0.095 | -1.22 | ✗ |
| `tick_imb_streak_3` | USD_JPY | 15 | 1017 | -0.038 | -0.35 | ✗ |
| `tick_imb_streak_3` | USD_JPY | 30 | 1017 | +0.170 | +0.65 | ✗ |
| `tick_imb_streak_3` | USD_JPY | 60 | 1017 | +0.752 | +1.72 | ✗ |