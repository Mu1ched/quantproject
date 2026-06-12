# Asian-range sweep-and-reverse — baseline result

- Sweep ID: `asian_sweep_reverse_baseline_20260604_130453`
- Pairs: `['GBP_USD', 'EUR_JPY', 'AUD_USD']`
- Combos: 36
- Wall time: 5.5 min

## Summary statistics

| metric | value |
|---|---|
| Mean test Sharpe | -1.306 |
| Median test Sharpe | -1.877 |
| Best variant test Sharpe | 2.592 |
| Mean test trades | 9.0 |
| Frac variants > 0 | 19.4% |
| Frac variants > 0.3 | 19.4% |
| Frac variants > 0.5 | 19.4% |
| Delta vs null baseline (−0.7) | -0.61 |

## Top 10 variants

| sharpe | trades | win_rate | max_dd | params |
|---|---|---|---|---|
| +2.592 | 3 | 6666.7% | +0.000 | `{"tp_r": 1.5, "sl_buffer": 0.3, "min_sweep": 0.03, "max_close_dist": 0.4}` |
| +2.443 | 3 | 6666.7% | +0.000 | `{"tp_r": 1.5, "sl_buffer": 0.5, "min_sweep": 0.03, "max_close_dist": 0.4}` |
| +2.054 | 3 | 6666.7% | +0.000 | `{"tp_r": 1.0, "sl_buffer": 0.3, "min_sweep": 0.03, "max_close_dist": 0.4}` |
| +1.813 | 3 | 6666.7% | +0.000 | `{"tp_r": 1.0, "sl_buffer": 0.5, "min_sweep": 0.03, "max_close_dist": 0.4}` |
| +1.324 | 3 | 3333.3% | -4.522 | `{"tp_r": 0.7, "sl_buffer": 0.3, "min_sweep": 0.03, "max_close_dist": 0.4}` |
| +0.991 | 3 | 3333.3% | -3.674 | `{"tp_r": 0.7, "sl_buffer": 0.5, "min_sweep": 0.03, "max_close_dist": 0.4}` |
| +0.527 | 20 | 3500.0% | -887.945 | `{"tp_r": 1.5, "sl_buffer": 0.5, "min_sweep": 0.03, "max_close_dist": 0.2}` |
| -0.055 | 11 | 3636.4% | -633.667 | `{"tp_r": 1.5, "sl_buffer": 0.5, "min_sweep": 0.07, "max_close_dist": 0.2}` |
| -0.114 | 2 | 5000.0% | +0.000 | `{"tp_r": 1.5, "sl_buffer": 0.2, "min_sweep": 0.07, "max_close_dist": 0.4}` |
| -0.316 | 20 | 3000.0% | -1144.738 | `{"tp_r": 1.5, "sl_buffer": 0.3, "min_sweep": 0.03, "max_close_dist": 0.2}` |