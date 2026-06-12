# Edge-miner survey — NY + London under live-calibrated costs

Goal: see whether the miner surfaces stable signal before we wire it into the agent loop.

## Session: ny

- Adversarial AUC: **0.8564933231049027** (verdict: ABORT)
  - AUC=0.856 — train/test distributions highly divergent. Mined edges are likely regime artifacts, not structural. Main drift axes: spread_median, hawkes_intensity, spread_mean
- Distribution shift on features: spread_median, hawkes_intensity, spread_mean, vol_imb_lag1, vol_imbalance, spread_ratio, delta, delta_momentum, delta_accel, hurst
- Patterns passing walk-forward CV: **0**

## Session: london

- Adversarial AUC: **0.8826539848766006** (verdict: ABORT)
  - AUC=0.883 — train/test distributions highly divergent. Mined edges are likely regime artifacts, not structural. Main drift axes: spread_median, atr, rv_median
- Distribution shift on features: spread_median, spread_mean, delta, rv_median, vol_imb_lag1, vol_imbalance, delta_momentum, hawkes_intensity, delta_accel, spread_ratio
- Patterns passing walk-forward CV: **0**
