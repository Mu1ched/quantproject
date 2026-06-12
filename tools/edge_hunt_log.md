# Edge Hunt Log

_H1 FX, real engine + full gauntlet (MC+WF enforced) + concentration check._

### rsi_range_reversion  —  **REJECTED**  (2026-06-07 21:27Z)
- rationale: Fade RSI extremes only when ADX shows no trend; overextension reverts in ranges.
- family: mean_reversion | sweep: `hunt_rsi_range_reversion_20260607_212654`
- n train/test: 136/12 | train_sh/test_sh: -1.26/0.33
- DSR 0.00 | PSR 0.83 | PBO — | CI_low -6.63 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: too few trades (12 < 20)

### rsi_range_reversion  —  **REJECTED**  (2026-06-07 21:27Z)
- rationale: Fade RSI extremes only when ADX shows no trend; overextension reverts in ranges.
- family: mean_reversion | sweep: `hunt_rsi_range_reversion_20260607_212746`
- n train/test: 136/12 | train_sh/test_sh: -1.26/0.33
- DSR 0.00 | PSR 0.83 | PBO — | CI_low -6.63 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: too few trades (12 < 20)

### bb_fade_range  —  **REJECTED**  (2026-06-07 21:28Z)
- rationale: Fade 2-sigma band breaks in non-trending regimes; statistical reversion to mean.
- family: mean_reversion | sweep: `hunt_bb_fade_range_20260607_212756`
- n train/test: 649/71 | train_sh/test_sh: -2.49/0.11
- DSR 0.00 | PSR 0.75 | PBO 0.20 | CI_low -2.60 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### trend_pullback_ema  —  **REJECTED**  (2026-06-07 21:28Z)
- rationale: Buy pullbacks to EMA20 in strong trends; better RR-vs-cost than chasing breakouts.
- family: trend_pullback | sweep: `hunt_trend_pullback_ema_20260607_212818`
- n train/test: 1121/134 | train_sh/test_sh: -2.83/3.56
- DSR 1.00 | PSR 1.00 | PBO 0.31 | CI_low -1.69 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### asian_sweep_fade  —  **REJECTED**  (2026-06-07 21:28Z)
- rationale: Fade London-open stop-runs beyond the Asian range that close back inside.
- family: liquidity_sweep | sweep: `hunt_asian_sweep_fade_20260607_212837`
- n train/test: 983/123 | train_sh/test_sh: -5.02/-1.42
- DSR 0.00 | PSR 0.00 | PBO 0.69 | CI_low -6.67 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### prevday_level_fade  —  **REJECTED**  (2026-06-07 21:29Z)
- rationale: Fade first-touch rejections of prior-day high/low where liquidity clusters.
- family: level_reversion | sweep: `hunt_prevday_level_fade_20260607_212854`
- n train/test: 1444/169 | train_sh/test_sh: -4.39/-2.94
- DSR 0.00 | PSR 0.00 | PBO 0.61 | CI_low -5.93 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### trend_pullback_confirmed  —  **REJECTED**  (2026-06-07 21:39Z)
- rationale: Pullback to EMA in trend WITH a confirmation close in trend direction; filters reversals that gap-stopped batch 1.
- family: trend_pullback | sweep: `hunt_trend_pullback_confirmed_20260607_213933`
- n train/test: 330/42 | train_sh/test_sh: -1.24/2.53
- DSR 0.16 | PSR 1.00 | PBO 0.36 | CI_low 0.21 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### htf_rsi_pullback  —  **REJECTED**  (2026-06-07 21:40Z)
- rationale: RSI dip in the direction of the EMA200 trend — discount entry that resumes WITH momentum (opposite of the fade that failed).
- family: trend_pullback | sweep: `hunt_htf_rsi_pullback_20260607_213953`
- n train/test: 264/33 | train_sh/test_sh: -0.89/1.47
- DSR 0.00 | PSR 1.00 | PBO 0.08 | CI_low -1.90 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### donchian_trend_follow  —  **REJECTED**  (2026-06-07 21:40Z)
- rationale: 20-bar breakout taken only in the EMA200 trend direction; momentum continuation, not naive two-sided breakout.
- family: trend_continuation | sweep: `hunt_donchian_trend_follow_20260607_214013`
- n train/test: 1359/160 | train_sh/test_sh: -1.64/1.54
- DSR 0.00 | PSR 1.00 | PBO 0.02 | CI_low -0.30 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### momentum_persistence  —  **REJECTED**  (2026-06-07 21:40Z)
- rationale: Runs of same-direction H1 closes with rising ADX tend to extend one more leg (short-run autocorrelation).
- family: trend_continuation | sweep: `hunt_momentum_persistence_20260607_214030`
- n train/test: 891/111 | train_sh/test_sh: -0.06/2.61
- DSR 0.08 | PSR 1.00 | PBO 0.29 | CI_low -3.83 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=61376, mean=$-247, total=$-15,143,224
  - gap_stop: n=18589, mean=$-408, total=$-7,579,923
  - weekend_flatten: n=8286, mean=$-44, total=$-366,383
  - family_blown: n=6, mean=$-94, total=$-564
  - session_exit: n=30591, mean=$10, total=$318,729
  - take_profit: n=35678, mean=$385, total=$13,728,408

Gap-stop burden: 18589 trades, $-7,579,923 (84% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - trend_continuation: 33258 trades, $-30/trade
  - trend_pullback: 37459 trades, $-42/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade

### momentum_persistence_intraday  —  **REJECTED**  (2026-06-07 21:43Z)
- rationale: Momentum persistence, but enter 8-16 UTC and flat by 20:00 — removes overnight/illiquid gaps (84% of batch-1/2 losses).
- family: trend_continuation | sweep: `hunt_momentum_persistence_intraday_20260607_214319`
- n train/test: 309/34 | train_sh/test_sh: -2.01/0.40
- DSR 0.00 | PSR 0.99 | PBO 0.24 | CI_low -4.82 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### donchian_trend_intraday  —  **REJECTED**  (2026-06-07 21:43Z)
- rationale: EMA200-filtered Donchian continuation, intraday liquid-hours only, flat before rollover.
- family: trend_continuation | sweep: `hunt_donchian_trend_intraday_20260607_214341`
- n train/test: 879/92 | train_sh/test_sh: -1.41/2.44
- DSR 0.01 | PSR 1.00 | PBO 0.09 | CI_low -0.30 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### htf_rsi_pullback_intraday  —  **REJECTED**  (2026-06-07 21:44Z)
- rationale: EMA200-trend RSI-pullback, intraday liquid-hours only — momentum-aligned entry with gaps removed.
- family: trend_pullback | sweep: `hunt_htf_rsi_pullback_intraday_20260607_214358`
- n train/test: 168/13 | train_sh/test_sh: -0.89/2.12
- DSR 0.07 | PSR 1.00 | PBO 0.11 | CI_low 0.48 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: too few trades (13 < 20)

### trend_pullback_confirmed_intraday  —  **REJECTED**  (2026-06-07 21:44Z)
- rationale: Confirmed trend-pullback, intraday liquid-hours only, flat before rollover.
- family: trend_pullback | sweep: `hunt_trend_pullback_confirmed_intraday_20260607_214417`
- n train/test: 143/25 | train_sh/test_sh: -1.45/1.37
- DSR 0.00 | PSR 1.00 | PBO 0.40 | CI_low -1.22 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=67221, mean=$-262, total=$-17,602,284
  - gap_stop: n=19456, mean=$-420, total=$-8,168,680
  - weekend_flatten: n=8306, mean=$-41, total=$-341,632
  - family_blown: n=6, mean=$-94, total=$-564
  - session_exit: n=48685, mean=$2, total=$118,216
  - take_profit: n=41029, mean=$385, total=$15,792,144

Gap-stop burden: 19456 trades, $-8,168,680 (80% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - trend_continuation: 50835 trades, $-35/trade
  - trend_pullback: 50059 trades, $-39/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade

### order_flow_momentum  —  **REJECTED**  (2026-06-07 21:50Z)
- rationale: Trade WITH a strong hourly aggressive-volume imbalance (delta z-score); informed flow continues.
- family: order_flow | sweep: `hunt_order_flow_momentum_20260607_214950`
- n train/test: 1181/141 | train_sh/test_sh: -2.01/-2.23
- DSR 0.00 | PSR 0.00 | PBO 0.44 | CI_low -4.64 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### delta_divergence  —  **REJECTED**  (2026-06-07 21:50Z)
- rationale: New price extreme made on opposing net delta = exhaustion; fade the unsupported move.
- family: order_flow | sweep: `hunt_delta_divergence_20260607_215012`
- n train/test: 450/41 | train_sh/test_sh: -0.85/-0.62
- DSR 0.00 | PSR 0.00 | PBO — | CI_low -3.23 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### retail_contrarian  —  **REJECTED**  (2026-06-07 21:50Z)
- rationale: Fade extreme retail long/short positioning (EUR_USD/GBP_USD) — crowd is contrarian.
- family: positioning | sweep: `hunt_retail_contrarian_20260607_215023`
- n train/test: 0/0 | train_sh/test_sh: 0.00/0.00
- DSR 0.50 | PSR — | PBO — | CI_low — | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: too few trades (0 < 20)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=70584, mean=$-270, total=$-19,088,075
  - gap_stop: n=19969, mean=$-425, total=$-8,489,032
  - weekend_flatten: n=8314, mean=$-40, total=$-332,905
  - session_exit: n=61889, mean=$-1, total=$-69,762
  - family_blown: n=6, mean=$-94, total=$-564
  - take_profit: n=45052, mean=$386, total=$17,373,831

Gap-stop burden: 19969 trades, $-8,489,032 (80% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - order_flow: 21111 trades, $-19/trade
  - trend_continuation: 50835 trades, $-35/trade
  - trend_pullback: 50059 trades, $-39/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade

### flow_trend_align  —  **REJECTED**  (2026-06-07 21:57Z)
- rationale: Enter only when hourly delta AND EMA20>EMA50 AND price>EMA200 all agree — flow confirms trend.
- family: order_flow | sweep: `hunt_flow_trend_align_20260607_215737`
- n train/test: 630/68 | train_sh/test_sh: -0.98/-2.97
- DSR 0.00 | PSR 0.00 | PBO 0.82 | CI_low -6.11 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### flow_persistence  —  **REJECTED**  (2026-06-07 21:58Z)
- rationale: N consecutive same-sign delta bars in the EMA200 trend direction — sustained informed flow.
- family: order_flow | sweep: `hunt_flow_persistence_20260607_215758`
- n train/test: 577/70 | train_sh/test_sh: -2.66/-0.95
- DSR 0.00 | PSR 0.00 | PBO 0.31 | CI_low -4.16 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### cum_delta_momentum  —  **REJECTED**  (2026-06-07 21:58Z)
- rationale: 6-hour cumulative-delta slope agreeing with the EMA200 trend — sustained directional flow.
- family: order_flow | sweep: `hunt_cum_delta_momentum_20260607_215818`
- n train/test: 1559/180 | train_sh/test_sh: -1.46/-2.69
- DSR 0.00 | PSR 0.00 | PBO 0.17 | CI_low -4.77 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=79704, mean=$-286, total=$-22,755,295
  - gap_stop: n=21160, mean=$-437, total=$-9,253,208
  - weekend_flatten: n=8365, mean=$-37, total=$-312,800
  - session_exit: n=86081, mean=$-3, total=$-271,220
  - family_blown: n=6, mean=$-94, total=$-564
  - take_profit: n=53833, mean=$385, total=$20,718,038

Gap-stop burden: 21160 trades, $-9,253,208 (78% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - order_flow: 64446 trades, $-26/trade
  - trend_continuation: 50835 trades, $-35/trade
  - trend_pullback: 50059 trades, $-39/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade

### cot_contrarian  —  **REJECTED**  (2026-06-07 22:05Z)
- rationale: Fade speculator-positioning extremes (COT index >=80 short / <=20 long) — crowd caught at extremes.
- family: positioning | sweep: `hunt_cot_contrarian_20260607_220456`
- n train/test: 961/133 | train_sh/test_sh: 0.73/-0.86
- DSR 0.00 | PSR 0.00 | PBO 0.58 | CI_low -2.72 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### cot_follow  —  **REJECTED**  (2026-06-07 22:05Z)
- rationale: Follow speculator positioning (COT index >=80 long / <=20 short) — test trend-persistence direction.
- family: positioning | sweep: `hunt_cot_follow_20260607_220511`
- n train/test: 960/133 | train_sh/test_sh: -2.89/-1.52
- DSR 0.00 | PSR 0.00 | PBO 0.39 | CI_low -4.79 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=82195, mean=$-289, total=$-23,730,912
  - gap_stop: n=21196, mean=$-437, total=$-9,267,778
  - weekend_flatten: n=8365, mean=$-37, total=$-312,800
  - session_exit: n=89587, mean=$-3, total=$-247,189
  - family_blown: n=6, mean=$-94, total=$-564
  - take_profit: n=55486, mean=$388, total=$21,533,248

Gap-stop burden: 21196 trades, $-9,267,778 (77% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - positioning: 7686 trades, $-20/trade
  - order_flow: 64446 trades, $-26/trade
  - trend_continuation: 50835 trades, $-35/trade
  - trend_pullback: 50059 trades, $-39/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade

### cot_daily_contrarian  —  **REJECTED**  (2026-06-07 22:11Z)
- rationale: Fade COT speculator extremes on daily bars, holding multiple days to SL/TP — matches the weekly signal horizon; cost is a tiny fraction of daily ATR.
- family: positioning | sweep: `hunt_cot_daily_contrarian_20260607_221133`
- n train/test: 0/0 | train_sh/test_sh: 0.00/0.00
- DSR 0.50 | PSR — | PBO — | CI_low — | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: too few trades (0 < 20)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=82195, mean=$-289, total=$-23,730,912
  - gap_stop: n=21196, mean=$-437, total=$-9,267,778
  - weekend_flatten: n=8365, mean=$-37, total=$-312,800
  - session_exit: n=89587, mean=$-3, total=$-247,189
  - family_blown: n=6, mean=$-94, total=$-564
  - take_profit: n=55486, mean=$388, total=$21,533,248

Gap-stop burden: 21196 trades, $-9,267,778 (77% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - positioning: 7686 trades, $-20/trade
  - order_flow: 64446 trades, $-26/trade
  - trend_continuation: 50835 trades, $-35/trade
  - trend_pullback: 50059 trades, $-39/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade

### cot_swing_contrarian  —  **REJECTED**  (2026-06-07 22:22Z)
- rationale: Fade COT speculator extremes (one entry/day at 08:00), holding multi-day to SL/TP with no intraday force-exit — matches the weekly COT horizon.
- family: positioning | sweep: `hunt_cot_swing_contrarian_20260607_222143`
- n train/test: 745/95 | train_sh/test_sh: 0.87/0.50
- DSR 0.00 | PSR 1.00 | PBO 0.41 | CI_low -2.92 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=88405, mean=$-292, total=$-25,772,570
  - gap_stop: n=21258, mean=$-437, total=$-9,294,247
  - weekend_flatten: n=9853, mean=$-30, total=$-291,638
  - session_exit: n=89587, mean=$-3, total=$-247,189
  - family_blown: n=6, mean=$-94, total=$-564
  - take_profit: n=59114, mean=$401, total=$23,692,888

Gap-stop burden: 21258 trades, $-9,294,247 (78% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - positioning: 19074 trades, $-2/trade
  - order_flow: 64446 trades, $-26/trade
  - trend_continuation: 50835 trades, $-35/trade
  - trend_pullback: 50059 trades, $-39/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade

---
#### HOLDOUT CONFIRMATION — cot_swing_contrarian (best lead) — FAILED
Best combo {tp_r:3.0, sl_r:3.5, ci_hi:70, ci_lo:30, contrarian:1}:
- TRAIN   n=745 Sharpe +0.87 PnL +$8,178 WR 55%
- TEST    n=95  Sharpe +0.50 PnL -$478   WR 53%
- HOLDOUT n=98  Sharpe -0.47 PnL -$906   WR 52%  <-- edge decays out-of-sample
Verdict: real in-sample COT-contrarian effect (sign confirmed: follow = -2.89),
but does NOT generalise to unseen 2026 data. Burned per holdout-once rule.
Not re-tuned on holdout. The locked holdout did its job — caught a decaying edge
that train+test alone (+0.87/+0.50) would have wrongly endorsed.

### vix_risk_contrarian  —  **REJECTED**  (2026-06-07 22:42Z)
- rationale: Fade VIX extremes — high VIX = fear overshoot, bet on risk recovery (long risk pairs); low VIX = complacency (short).
- family: risk_sentiment | sweep: `hunt_vix_risk_contrarian_20260607_224200`
- n train/test: 394/47 | train_sh/test_sh: -0.88/0.92
- DSR 0.00 | PSR 1.00 | PBO — | CI_low -2.59 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

### vix_risk_momentum  —  **REJECTED**  (2026-06-07 22:42Z)
- rationale: Follow VIX — rising fear (high VIX) = risk-off continues (short risk pairs); falling VIX = risk-on (long).
- family: risk_sentiment | sweep: `hunt_vix_risk_momentum_20260607_224218`
- n train/test: 529/62 | train_sh/test_sh: -0.24/0.42
- DSR 0.00 | PSR 0.96 | PBO — | CI_low -3.22 | by_sig 0
- MC pass —% / blown —% | WF folds —/—
- concentration worst — (month —, pair —, trade —)
- params: `—`
- killed by: failed BH correction (p_adj=1.000)

---
#### Meta-reflection (learned from all hunt data so far)

Exit-reason economics (train, all hunt strategies pooled):
  - stop_loss: n=93048, mean=$-292, total=$-27,213,551
  - gap_stop: n=21304, mean=$-437, total=$-9,312,118
  - weekend_flatten: n=10966, mean=$-27, total=$-296,926
  - session_exit: n=89587, mean=$-3, total=$-247,189
  - family_blown: n=6, mean=$-94, total=$-564
  - take_profit: n=61464, mean=$406, total=$24,962,420

Gap-stop burden: 21304 trades, $-9,312,118 (77% of total PnL was lost to gaps)

Train PnL/trade by family (positive = worth pursuing):
  - positioning: 19074 trades, $-2/trade
  - risk_sentiment: 8152 trades, $-24/trade
  - order_flow: 64446 trades, $-26/trade
  - trend_continuation: 50835 trades, $-35/trade
  - trend_pullback: 50059 trades, $-39/trade
  - level_reversion: 26565 trades, $-60/trade
  - mean_reversion: 27862 trades, $-74/trade
  - liquidity_sweep: 17574 trades, $-80/trade
  - session_based: 11808 trades, $-121/trade
