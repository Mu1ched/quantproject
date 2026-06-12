# Deterministic mechanic discovery report

- Generated: 2026-06-06T22:01:19.984994+00:00
- Train window: 2024-05-03 → 2025-12-05
- Pairs: ['EUR_USD', 'GBP_USD']
- Tuples tested: 1,968
- Keys with ≥1 passing pair (pre cross-pair): 78
- Surviving mechanics (post cross-pair + validator round-trip): **8**

## Surviving mechanics

| rank | mechanic_id | session | feature | op | horizon | dir | pairs | n_total | t_stat | mean_atr |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `disc_ny_ma_dist_lt_p25_h30_long` | ny | `ma_dist` | < | 30 | long | EUR_USD,GBP_USD | 133011 | +15.09 | +0.305 |
| 2 | `disc_ny_ma_dist_lt_p25_h60_long` | ny | `ma_dist` | < | 60 | long | EUR_USD,GBP_USD | 133011 | +16.17 | +0.441 |
| 3 | `disc_ny_hurst_gt_p75_h60_long` | ny | `hurst` | > | 60 | long | EUR_USD,GBP_USD | 134076 | +12.99 | +0.361 |
| 4 | `disc_london_ma_dist_gt_p90_h30_short` | london | `ma_dist` | > | 30 | short | EUR_USD,GBP_USD | 22011 | -9.87 | -0.387 |
| 5 | `disc_london_ma_dist_gt_p90_h15_short` | london | `ma_dist` | > | 15 | short | EUR_USD,GBP_USD | 22011 | -8.50 | -0.246 |
| 6 | `disc_ny_ma_dist_lt_p10_h30_long` | ny | `ma_dist` | < | 30 | long | EUR_USD,GBP_USD | 56489 | +11.74 | +0.388 |
| 7 | `disc_ny_hurst_gt_p90_h60_long` | ny | `hurst` | > | 60 | long | EUR_USD,GBP_USD | 53405 | +10.14 | +0.448 |
| 8 | `disc_ny_ma_dist_lt_p10_h60_long` | ny | `ma_dist` | < | 60 | long | EUR_USD,GBP_USD | 56489 | +10.37 | +0.466 |