# Sliding-window mining survey

- Train days: **120**, Test days: **30**, Step: **30**
- Windows run: **1**  ×  sessions: **ny**  = **1** miner runs
- Passed adversarial (AUC < 0.75): **0 / 1** (0%)

- AUC distribution: min=0.932  median=0.932  max=0.932

## Per-window summary

| window | session | AUC | verdict | n_patterns | elapsed_s |
|---|---|---|---|---|---|
| 2024-05-03 | ny | 0.9318 | ABORT | 0 | 17.2 |

## Recurring features (across passing windows)

_No windows passed adversarial validation. No features to aggregate._

## Verdict

**Broad mining not viable on this data window.** Even shorter windows couldn't pass adversarial validation. Pivot to hand-crafted hypotheses.
