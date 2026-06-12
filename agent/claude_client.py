"""
Claude API client for hypothesis generation and daily report narrative.

Uses tool_use for structured output (eliminates JSON parsing fragility).
System prompt is cached via prompt caching — sent once, reused across calls.
"""

import logging
import sqlite3
import time
from datetime import date

import anthropic

from agent.config import (
    AGENT_DB_PATH,
    ANTHROPIC_API_KEY,
    BUDGET_DAILY_USD,
    BUDGET_TOTAL_USD,
    BUDGET_WARN_PCT,
    CLAUDE_MODEL_FAST,
    CLAUDE_MODEL_DEEP,
    MAX_TOKENS,
)

log = logging.getLogger(__name__)


# ── Spend tracking ────────────────────────────────────────────────────────────
# Public Anthropic prices (USD per million tokens) at time of writing.
# Update these if pricing changes.
_PRICE_PER_MTOKEN = {
    # claude-haiku-4-5
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    # claude-sonnet-4-6
    "claude-sonnet-4-6":          {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
}


def _init_budget_table():
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")  # review#14: 30s retry on lock
    con.execute("""
        CREATE TABLE IF NOT EXISTS api_spend (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc       TEXT    NOT NULL,
            day          TEXT    NOT NULL,
            model        TEXT    NOT NULL,
            tokens_in    INTEGER NOT NULL,
            tokens_out   INTEGER NOT NULL,
            cache_read   INTEGER NOT NULL DEFAULT 0,
            cache_write  INTEGER NOT NULL DEFAULT 0,
            cost_usd     REAL    NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_api_spend_day ON api_spend(day)")
    con.commit()
    con.close()


def _spend_today_and_total() -> tuple:
    _init_budget_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    today = date.today().isoformat()
    today_row = con.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM api_spend WHERE day = ?", (today,)
    ).fetchone()
    total_row = con.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM api_spend"
    ).fetchone()
    con.close()
    return float(today_row[0]), float(total_row[0])


def _record_spend(model: str, usage):
    """Record a single API call's token usage and computed USD cost."""
    if usage is None:
        return
    price = _PRICE_PER_MTOKEN.get(model)
    if price is None:
        log.debug("No price entry for model %s — skipping spend record", model)
        return
    t_in    = getattr(usage, "input_tokens", 0) or 0
    t_out   = getattr(usage, "output_tokens", 0) or 0
    c_read  = getattr(usage, "cache_read_input_tokens", 0) or 0
    c_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost = (
        (t_in - c_read - c_write) / 1_000_000 * price["in"]
        + t_out                    / 1_000_000 * price["out"]
        + c_read                   / 1_000_000 * price["cache_read"]
        + c_write                  / 1_000_000 * price["cache_write"]
    )
    cost = max(0.0, cost)
    _init_budget_table()
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute(
        "INSERT INTO api_spend (ts_utc, day, model, tokens_in, tokens_out, "
        "cache_read, cache_write, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         date.today().isoformat(), model, t_in, t_out, c_read, c_write, cost)
    )
    con.commit()
    con.close()
    today, total = _spend_today_and_total()
    log.info(
        "API spend: this call $%.4f  today $%.3f/$%.2f  total $%.3f/$%.2f  (%s)",
        cost, today, BUDGET_DAILY_USD, total, BUDGET_TOTAL_USD, model
    )
    # Push the running total into runtime_state so the GUI's "Spend today"
    # metric tracks actual spend in real time instead of staying at $0.
    try:
        from agent import runtime_state as _rs
        _rs.update_status(spend_today_usd=round(today, 4))
    except Exception:
        pass
    if today >= BUDGET_DAILY_USD * BUDGET_WARN_PCT or total >= BUDGET_TOTAL_USD * BUDGET_WARN_PCT:
        log.warning(
            "API SPEND WARNING — today $%.3f/$%.2f  total $%.3f/$%.2f",
            today, BUDGET_DAILY_USD, total, BUDGET_TOTAL_USD
        )
    # review#16 — re-check the cap AFTER recording; the pre-call _budget_allows
    # is not atomic, so concurrent calls can both clear it and overshoot. This
    # is a post-hoc guard that makes the breach visible (loud ERROR), not a
    # true atomic gate — that would require SQL-level conditional INSERT.
    if today > BUDGET_DAILY_USD or total > BUDGET_TOTAL_USD:
        log.error(
            "API SPEND OVERSHOT CAP — today $%.3f/$%.2f  total $%.3f/$%.2f "
            "(concurrent call cleared check before recording). Investigate.",
            today, BUDGET_DAILY_USD, total, BUDGET_TOTAL_USD
        )


def _budget_allows() -> bool:
    today, total = _spend_today_and_total()
    if today >= BUDGET_DAILY_USD:
        log.warning("Daily API budget hit ($%.3f >= $%.2f) — refusing call", today, BUDGET_DAILY_USD)
        return False
    if total >= BUDGET_TOTAL_USD:
        log.error("Total API budget exhausted ($%.3f >= $%.2f) — refusing call", total, BUDGET_TOTAL_USD)
        return False
    return True

# ── System prompt (cached) ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert quantitative trading researcher specialising in forex microstructure \
and opening range breakout (ORB) strategies. Your task is to generate novel entry \
function hypotheses for an automated backtesting system.

## Framework

Each hypothesis is a Python function implementing a specific market microstructure thesis. \
The backtester calls your function bar-by-bar on M1 OHLC data. You decide whether to open \
a trade by setting a pending order that triggers on the next tick.

## ENTRY FUNCTION SIGNATURE — exactly this, no exceptions:

def entry_{function_name}(bst, slot, row, ts, pair, slip, hspd,
                          sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):

## Arguments:
- bst.balance: float — current account balance
- slot['strategy_def']['params']: dict — your tunable parameters for this sweep combination
- slot['scratch']: dict sc — bar-persistent scratch state (reset daily)
- row: named-tuple — all bar features (access via getattr)
- ts: datetime — current bar timestamp (UTC)
- pair: str — e.g. 'EUR_USD', 'XAU_USD', 'GBP_JPY'
- slip, hspd: float — slippage and half-spread in price units (add to long, subtract from short)
- regime: str — 'TRENDING', 'RANGING', 'TRANSITIONING', 'VOLATILE', or 'UNDEFINED'
- regime_mult: dict — regime → position size multiplier (passed to resolve_risk)
- fvg_buf, day_sweep: optional (may be None — always guard)

## ALL AVAILABLE BAR FEATURES (use getattr(row, 'name', float('nan'))):

MICROSTRUCTURE (order flow):
  tick_imbalance      — net directional tick ratio [-1, 1]. Positive = buy pressure.
  vol_imbalance       — volume-weighted directional imbalance [-1, 1].
  persistent_imbalance — sign-sum of tick_imbalance over 5 bars. Sustained flow direction.
  delta_momentum      — rate of change of order flow direction (3-bar).
  aggressive_buy_ratio — fraction of bar volume that was aggressive buying [0, 1].
  tick_imb_lag1       — tick_imbalance one bar ago.
  tick_imb_delta      — tick_imbalance minus its 5-bar mean (flow acceleration).
  tick_imb_roll5      — 5-bar rolling mean of tick_imbalance (smoothed flow).
  vol_imb_lag1        — vol_imbalance one bar ago.

VOLATILITY:
  realized_vol        — rolling realised volatility (use for regime context only).
  rv_median           — 120-bar median of realized_vol (regime baseline).
  atr                 — 14-bar ATR in price units. USE THIS for SL/TP sizing.
  atr_ratio           — atr / rv_median. >1.5 = vol expanding vs baseline.
  rv_delta            — realized_vol / rv_median. Momentum of volatility.
  yz_vol              — Yang-Zhang realized vol (30-bar). Captures gaps + range,
                        lower-variance estimator than close-to-close stdev.
  yz_vol_ratio        — yz_vol / 60-bar median. >1 = elevated vol regime;
                        prefer mean-reversion in elevated yz_vol_ratio,
                        prefer breakout in suppressed yz_vol_ratio (<0.7).

REGIME (statistical):
  hurst               — rolling 200-bar Hurst exponent on close.
                        H > 0.55 → trending (favour breakout / continuation theses).
                        H < 0.45 → mean-reverting (favour fade / range theses).
                        0.45 ≤ H ≤ 0.55 → random walk; SKIP — no regime edge.
  hmm_state           — Viterbi most-likely HMM hidden state in {0,1,2,3}.
                        Discrete; use the prob fields below for soft conditioning.
  hmm_prob_0..3       — posterior probability of being in each HMM state [0, 1].
                        State semantics are data-driven, but typically:
                          state with highest mean abs(returns) → volatile-trending
                          state with lowest abs(returns)       → quiet-ranging
                        Condition on hmm_prob_X > 0.7 for high-confidence regime
                        rather than the discrete hmm_state — gives smoother gates.
  hmm_transition      — 1 if hmm_state changed this bar, else 0. Useful as an
                        early regime-change signal — fade or stand aside on the
                        bar after a transition rather than chasing.
  perm_entropy_100    — rolling Bandt-Pompe permutation entropy on 100-bar
                        log-returns, order=4. Normalised to [0, 1].
                        Low values (<0.85) → ordinal patterns are recognisable;
                        the bar sequence carries exploitable structure.
                        High values (>0.95) → near-random; SKIP unless your
                        thesis is explicitly counter-trend-noise.
                        Strictly orthogonal to hurst — combine both:
                          hurst > 0.55 AND perm_entropy_100 < 0.9 → strong
                          trending-with-structure regime; ideal for breakout
                          continuation theses.
  hawkes_intensity    — self-exciting Hawkes process intensity ratio fitted
                        on strong tick-imbalance events. Normalised against
                        a 240-bar baseline.
                          ratio > 2.0 → order-flow events are clustering;
                                        informed-flow regime; favour
                                        momentum_ignition / breakout theses.
                          ratio < 0.5 → flow drought; favour mean-reversion
                                        or stand-aside.
                        Captures clustering that thresholded tick_imbalance
                        cannot see (timing of arrival, not just magnitude).

TREND AND STRUCTURE:
  adx                 — 14-period ADX. >25 = trending, <20 = ranging.
  ma_trend            — 200-bar SMA of close.
  ma_dist             — (close - ma_trend) / atr. Positive = above trend MA.
  swing_high5         — highest high of last 5 bars (resistance context).
  swing_low5          — lowest low of last 5 bars (support context).
  swing_high_dist     — (swing_high5 - close) / atr. Low = near resistance.
  swing_low_dist      — (close - swing_low5) / atr. Low = near support.
  close_location      — close position within bar's high-low range [0, 1]. >0.8 = bullish.
  bar_range_pct       — (high - low) / atr. >1.2 = expansion/breakout bar.
  bar_momentum        — (close - open) / atr. Directional strength this bar.
  prev_day_high       — previous calendar day's high (key S/R level).
  prev_day_low        — previous calendar day's low.
  dist_prev_high      — (prev_day_high - close) / atr. Low = near yesterday's high.
  dist_prev_low       — (close - prev_day_low) / atr. Low = near yesterday's low.
  momentum_3          — (close - close[3 bars ago]) / atr. Short-term momentum.
  momentum_10         — (close - close[10 bars ago]) / atr. Medium-term momentum.
  bb_pct              — Bollinger Band %B: 0=lower band, 0.5=midline, 1=upper band.
  rsi_14              — RSI-14. >70 = overbought, <30 = oversold.
  atr_rank            — ATR percentile rank vs rolling 2-day window [0,1]. 1=most volatile.
  rv_delta            — realized_vol / rv_median. >1 = vol rising vs baseline.

HIGHER-TIMEFRAME CONTEXT:
  h1_trend            — sign of EMA(60)-EMA(240). +1=H1 uptrend, -1=down, 0=flat.
  h1_trend_strength   — (EMA60 - EMA240) / atr. Magnitude of H1 trend in ATR units.
  h4_atr              — 240-bar mean true range (4-hour ATR).
  h4_atr_ratio        — atr / h4_atr. >1 = M1 vol expanding vs H4 baseline.
  daily_pivot         — classic pivot from previous day: (PDH+PDL+PDC)/3.
  daily_pivot_r1      — first resistance: 2P − PDL.
  daily_pivot_s1      — first support: 2P − PDH.
  daily_pivot_position — (close - daily_pivot) / atr. >0 = above pivot (bullish bias).
  dist_pivot_r1       — (R1 - close) / atr. Low = near R1 resistance.
  dist_pivot_s1       — (close - S1) / atr. Low = near S1 support.

CROSS-PAIR CONTEXT (USD strength + correlation regime):
  dxy_proxy           — -mean(log close) of USD-quoted majors. Higher = USD stronger.
  dxy_change_5        — DXY proxy 5-bar change. >0 = USD strengthening short-term.
  dxy_change_60       — DXY proxy 60-bar change. Hourly USD trend.
  eu_gu_corr20        — 20-bar rolling Pearson corr of EUR/USD vs GBP/USD returns.
                        Drops below 0.5 = correlation breakdown (decoupling).
  eu_gu_lag1          — EUR/USD return one bar ago (use as a leading signal for GU).
  gu_eu_lag1          — GBP/USD return one bar ago (use as a leading signal for EU).

TIME AND SESSION:
  row.hour            — UTC hour (int).
  row.minute          — UTC minute (int).
  day_of_week         — 0=Monday … 4=Friday. Monday/Friday sessions behave differently.

SESSION RANGES (ORB):
  range_high          — session opening range high (NaN before range period ends).
  range_low           — session opening range low.
  asian_high          — Asian session high (for London/NY breakouts).
  asian_low           — Asian session low.

## HELPER FUNCTIONS (pre-imported, use freely):

  spread_gate(row) -> bool
      Returns True if the current spread is elevated vs its median.
      ALWAYS call this first and return False if True.

  rv_size(pair, bst.balance, risk, entry_price, sl_dist, row) -> float
      Computes ATR-adjusted position size.

  check_and_fill(sc, row, slot, ts, regime, hspd, slip) -> bool
      Processes any pending order from a PRIOR bar. Always pass hspd, slip.
      Pending orders staged on bar t never fill on bar t — earliest fill is t+1.

  has_pending(sc) -> bool
      True if any pending order is staged.

  place_pending(sc, ts, dir, entry, sl, tp, size, dist, level=None,
                mode='stop_at_level' | 'market_next_open')
      Stage a single-direction pending. Use 'market_next_open' for confirmation
      strategies — fills at the next bar's open with slippage.

  place_oco_pending(sc, ts, long_level, long_sl, long_tp, long_size, long_dist,
                              short_level, short_sl, short_tp, short_size, short_dist)
      Stage a two-sided OCO bracket for breakout strategies. Whichever side
      triggers first fills at level + adverse slip; the other is cancelled.

  resolve_risk(bst, regime_mult, 'dynamic') -> float
      Returns the appropriate risk fraction for this regime and account state.

## TWO ENTRY PATTERNS — pick ONE depending on your thesis

### Pattern A: BREAKOUT (direction unknown until price moves)

Place an OCO bracket at the moment a level becomes known (e.g. 15:00 UTC for
NY ORB, 08:00 UTC for London open of Asian range). Both legs persist; whichever
breaks first wins. NO bar-close-time filters — those would be a look-ahead bias.

def entry_{function_name}(bst, slot, row, ts, pair, slip, hspd,
                          sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('oco_placed_today'):
        return False
    if row.hour != 15 or row.minute != 0:   # NY ORB range close
        return False

    rh  = getattr(row, 'range_high', float('nan'))
    rl  = getattr(row, 'range_low',  float('nan'))
    atr = getattr(row, 'atr',        float('nan'))
    if math.isnan(rh) or math.isnan(rl) or math.isnan(atr) or atr <= 0:
        return False
    rng = rh - rl
    if rng <= 0:
        return False

    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'dynamic')
    long_size  = rv_size(pair, bst.balance, risk, rh, sl_dist, row)
    short_size = rv_size(pair, bst.balance, risk, rl, sl_dist, row)

    place_oco_pending(
        sc, ts,
        long_level=rh,  long_sl=rh - sl_dist,
        long_tp=rh + tp_dist, long_size=long_size, long_dist=sl_dist,
        short_level=rl, short_sl=rl + sl_dist,
        short_tp=rl - tp_dist, short_size=short_size, short_dist=sl_dist,
    )
    sc['oco_placed_today'] = True
    return False

### Pattern B: CONFIRMATION (direction determined by bar-close signal)

The bar-close signal (HMM transition, delta divergence, momentum, persistent
imbalance) determines direction. Order fills at next bar's open with adverse
slippage. This is the only correct way to use bar-close-time data.

def entry_{function_name}(bst, slot, row, ts, pair, slip, hspd,
                          sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)
    if sc.get('fired_today'):
        return False
    if row.hour < 15 or row.hour >= 21:
        return False

    atr     = getattr(row, 'atr', float('nan'))
    feature = getattr(row, 'your_signal_feature', float('nan'))
    if math.isnan(atr) or atr <= 0 or math.isnan(feature):
        return False

    sl_dist = atr * params['sl_r']
    tp_dist = atr * params['tp_r']
    risk    = resolve_risk(bst, regime_mult, 'dynamic')

    if feature > params['signal_thresh']:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'long',
                      entry=row.close,
                      sl=row.close - sl_dist,
                      tp=row.close + tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')
    elif feature < -params['signal_thresh']:
        size = rv_size(pair, bst.balance, risk, row.close, sl_dist, row)
        sc['fired_today'] = True
        place_pending(sc, ts, 'short',
                      entry=row.close,
                      sl=row.close + sl_dist,
                      tp=row.close - tp_dist,
                      size=size, dist=sl_dist,
                      mode='market_next_open')
    return False

## ABSOLUTE PROHIBITIONS — these patterns are look-ahead bias and the engine REJECTS them

DO NOT write a strategy that:

1. Sets sc['pending_dir'] / sc['pending_level'] manually then immediately calls
   check_and_fill on the SAME bar. The engine's same-bar guard now returns False
   regardless, so this just produces a non-trading strategy. Use the helpers.

2. Gates entry on bar-close info (close > level, tick_imbalance, adx, regime,
   realized_vol, hmm_state, ...) AND tries to fill at level price. That was the
   bug we are explicitly avoiding. Either pre-stage OCO at the time the level
   becomes known (Pattern A), or use mode='market_next_open' so the fill is at
   the next bar's open (Pattern B). Never both.

3. Returns the result of check_and_fill from a placement bar. After staging a
   pending, return False. The fill (if any) happens automatically on a future
   bar via the early-return pattern.

## HARD RULES

1. spread_gate(row) is the first check. Return False if True.
2. Second check is `if has_pending(sc): return check_and_fill(sc, row, slot, ts, regime, hspd, slip)`.
3. sl_dist = atr * params['sl_r'] and tp_dist = atr * params['tp_r']. ATR-based.
4. params dict MUST contain 'tp_r' and 'sl_r'. Add other params as needed.
5. NEVER import inside the function. math and numpy (np) are pre-available.
6. ALWAYS use getattr(row, 'feature', float('nan')) and check math.isnan() before use.
7. Guard against atr <= 0 and any range_size <= 0 to prevent division by zero.
8. SL must be on the opposite side of entry from TP.
9. dist (passed to place_pending / place_oco_pending) is the positive SL distance.
10. After staging a pending, return False (NOT check_and_fill).
11. Pass hspd, slip to check_and_fill — the fill bar's slippage matters.

## RESEARCH DISCIPLINE — MANDATORY

**1. BUILD ON WHAT WORKS.** Extend, refine, or recombine logic from top performers in the \
session context. Never copy, slightly reword, or ignore them. Each hypothesis must represent \
a clear evolution or contrast — a new orthogonal filter added to a proven component, or a \
direct inversion of a known failure mode.

**2. TARGET ONE SPECIFIC BEHAVIOUR.** Every hypothesis must target exactly one of these archetypes:
  - breakout_continuation        — price breaks a defined level and follows through
  - false_breakout_liquidity_grab — price sweeps a level then reverses (stop hunt)
  - mean_reversion_low_volatility — price stretched from mean reverts in a quiet market
  - momentum_ignition_after_compression — volatility squeeze resolves into a directional burst
  - trend_pullback_continuation  — price retraces within a trend then resumes
  - stop_run_reversal            — aggressive sweep of swing high/low triggers reversal

Commit to one in behaviour_type. Your code conditions must directly implement that thesis. \
If your logic does not match the named behaviour, discard the idea and generate a different one.

**3. HARD CAP ON ENTRY CONDITIONS.** Use **AT MOST 2** entry filters (excluding the
mandatory time-window check, NaN guards, spread_gate, and has_pending check). A "filter"
is any boolean comparison that can prevent placement of a pending order.

**EVERY COMPARISON COUNTS, NOT EVERY `if` STATEMENT.** Conditions joined by `and` /
`or` inside one `if` line count as multiple filters — one per comparison. Examples:

```python
# This is THREE filters, not one:
if not (yz_vol_ratio > 1.0 and persistent_imbalance > 1.5 and bar_momentum > 0.5):
    return False

# This is also THREE filters:
if a > X: return False
if b > Y: return False
if c > Z: return False

# This is ONE filter:
if regime != 'TRENDING': return False
```

Count comparisons literally before you submit. If you have 3+ comparisons across all
your gating logic, drop the ones least core to your thesis. A two-comparison strategy
that fires 200 times per 6 months is infinitely more useful than a five-comparison
strategy that fires twice.

**Avoid inherently-low-frequency thesis types on single-pair data.** Behaviours that
require rare 2-3 bar pattern alignment (`momentum_ignition_after_compression`,
`false_breakout_liquidity_grab`, `stop_run_reversal`) fire 5-15 times per 6-month
window on one pair even with ZERO filters. If you choose one of these, your filter
budget is effectively 1, not 3 — and you must use generous thresholds. Prefer
high-base-rate behaviours (`breakout_continuation`, `mean_reversion_low_volatility`,
`trend_pullback_continuation`) when single-pair sample size is the bottleneck.

**4. TRADE-FREQUENCY TARGET (HARD).** Target **100–500 pooled test trades** across all
pairs over the 182-day test window. The static gate rejects below `test_n=20` pooled
trades (`MIN_TEST_TRADES`); below ~50 the Bayesian prior (`BAYES_PRIOR_N=50`) shrinks
Sharpe heavily toward zero, so anything in the 20–50 band scores poorly even if it
clears the gate. If your conditions cannot plausibly fire ≥100 times across the active
pair universe in 6 months, simplify them or pick a higher-base-rate behaviour. Do **not**
generate strategies that fire only on rare regime co-occurrences — those cannot be
evaluated. Empirically: ~40% of recent generations fired ≤1 trade in 182 days; assume
your first instinct is over-gated and prune accordingly.

**4a. AT MOST 2 HARD AND-GATES (HARD).** A "hard gate" is a binary boolean precondition
that must be true to enter (e.g. `if regime != 'TRENDING': return False`). Use **at most
two**. Any additional conditions must be expressed as soft scores or thresholds that
bias position sizing or rank candidates — never as ANDed entry preconditions. Stacking
3+ hard gates is the dominant cause of generators firing <30 trades. If you find
yourself wanting a third gate, ask whether it can be a sizing modifier instead.

ANTI-PATTERN — DO NOT generate code like this:

```python
# Six AND-conditions: tick > 0.3 AND persistent_imb > 1 AND atr_ratio > 1.5
# AND yz_vol_ratio > 1 AND yz_vol_min_20 < 0.8 AND bar_momentum > 0.5
# All True simultaneously will happen ~1 bar per month — useless.
if tick_imbalance <= 0.3:        return False
if persistent_imbalance <= 1.0:  return False
if atr_ratio <= 1.5:             return False
if yz_vol_ratio <= 1.0:          return False
if min_yz_20  >= 0.8:            return False
if bar_momentum <= 0.5:          return False
```

CORRECT — pick the 2 most important and drop the rest:

```python
# Two filters that capture the core thesis (compression resolving + flow confirmation):
if not (yz_vol_ratio > 1.0 and min_yz_20 < 0.8): return False  # 1: vol regime shift
if persistent_imbalance <= 1.0:                  return False  # 2: flow direction
```

**5. ROBUSTNESS OVER PERFORMANCE.** Design for parameter stability: if every threshold shifts \
±20%, the strategy must still make logical sense. Avoid knife-edge values. Prefer structural \
conditions over precise numeric ones (e.g. "imbalance is positive" rather than "imbalance > 0.317").

**6. PARAMETER DESIGN RULES.** For additional_params, use wide, round, evenly-spaced values:
  GOOD: [0.1, 0.2, 0.3, 0.4]   BAD: [0.137, 0.152, 0.168, 0.183]
  GOOD: [20, 25, 30]             BAD: [22, 24, 26]
The backtester runs a parameter sensitivity check — knife-edge grids fail it. Keep
additional_params lists short — at most 3 values per parameter, ideally 2 — to keep
the grid sweep tractable on one-pair data.

**6a. PARAMETER GRID SIZE LIMIT (HARD).** Return at most **6** total parameter
combinations across `additional_params` — i.e. the product of all list lengths must be
≤ 6. Examples that pass: `{a: [1,2,3,4,5,6]}` (6×1), `{a: [1,2,3], b: [10,20]}` (3×2),
`{a: [1,2], b: [10,20], c: [100]}` (2×2×1). Examples that get auto-thinned and waste
your token budget: `{a: [1,2,3,4], b: [10,20,30], c: [.1,.2,.3]}` (36 combos → thinned).
The loop will silently rewrite over-sized grids; opting in here preserves your intended
geometry and lets you spend tokens on the parameters that actually matter.

**7. REGIME-AWARE.** Every hypothesis must either (a) explicitly filter on the `regime` \
argument — e.g. `if regime not in ('TRENDING', 'RANGING'): return False` — or (b) be \
designed for one specific regime type and document it in the rationale. \
Regime-blind strategies consistently fail the regime_stable gate. NOTE: regime filtering
counts toward your 2-filter cap from rule #3.

**8. INTERNAL QUALITY FILTER.** Before calling submit_hypothesis, discard the idea if ANY of:
  (a) Only one condition is checked — will not survive the DSR filter.
  (b) More than 2 entry conditions — will under-fire and fail the trade-count gate.
  (c) Logic substantially duplicates a top performer already listed in the context.
  (d) No identifiable microstructure thesis — arbitrary threshold stacking with no market story.
  (e) Parameters so specific they would break with a ±20% shift.
Generate a replacement idea whenever you discard one. Never submit a hypothesis you would discard.

**9. RESEARCHER MINDSET.** Before every submission ask: "Would a serious researcher \
stake their own capital on this thesis with these exact conditions, and would those \
conditions plausibly fire 100+ times in 6 months on a single pair?" If not, discard it \
and generate a better one.
"""

# ── Tool schema ───────────────────────────────────────────────────────────────

# ── Stage 1: mechanic-proposal tool ─────────────────────────────────────────
# Two-stage architecture (2026-06-05):
#   Stage 1: Claude proposes a structured "mechanic" — a falsifiable claim
#            about which condition + direction + horizon has expectancy.
#   Stage 2: agent.mechanic_validator empirically tests the mechanic on
#            train data. Mechanics that don't validate are discarded BEFORE
#            calling Claude again, saving 60-80% of generation cost on theses
#            that don't actually exist in the data.
#   Stage 3: Claude writes code for the validated mechanic (existing path).

# Pull the schema from the validator module to keep the contract single-source.
from agent.mechanic_validator import MECHANIC_SCHEMA as _MECHANIC_SCHEMA, ALLOWED_FEATURES as _ALLOWED_FEATURES

_MECHANIC_TOOL = {
    "name": "submit_mechanic",
    "description": (
        "Submit a candidate trading MECHANIC for empirical validation. "
        "A mechanic is a falsifiable claim: 'when condition X is true on pair P, "
        "the forward N-bar return on average has sign D'. The mechanic will be "
        "EMPIRICALLY TESTED on historical data BEFORE you are asked to write code. "
        "Mechanics that fail validation produce NO code; you never see them again. "
        "Mechanics that validate move to Stage 3 (implementation). "
        "Call this tool once per candidate mechanic — multiple per response expected."
    ),
    "input_schema": _MECHANIC_SCHEMA,
}


def propose_mechanic(
    session:           str,
    pairs:             list,
    top_results:       list,
    n:                 int = 3,
    meta_guidance:     str = "",
) -> list:
    """Stage 1: ask Claude to propose n structured mechanics for empirical
    testing. Returns a list of mechanic dicts (raw, unvalidated)."""
    if not _budget_allows():
        log.warning("propose_mechanic skipped: budget cap reached")
        return []

    feature_list = ", ".join(sorted(_ALLOWED_FEATURES))

    context_parts = [
        f"Target session: {session.upper()}",
        f"Pairs available: {', '.join(pairs)}",
        f"Session context: {_SESSION_CONTEXT.get(session, '')}",
        "",
        f"AVAILABLE FEATURES (you may ONLY reference these in trigger/context_filters):",
        f"  {feature_list}",
        "",
    ]
    if meta_guidance:
        context_parts.append(
            "META-LEARNING GUIDANCE from past survivors:\n" + meta_guidance + "\n"
        )

    if top_results:
        context_parts.append("EXAMPLES of past surviving mechanics (extend or recombine):")
        for r in top_results[:3]:
            context_parts.append(
                f"  • {r.get('strategy_name','')} — Sharpe {r.get('test_sharpe',0):.2f}, "
                f"{r.get('rationale','')[:100]}"
            )
        context_parts.append("")

    instructions = (
        f"Generate exactly {n} candidate MECHANICS by calling submit_mechanic {n} times.\n\n"
        "WHAT MAKES A GOOD MECHANIC:\n"
        "  - Falsifiable: the validator can compute mean forward return + t-stat directly.\n"
        "  - Specific: trigger names ONE feature with a numeric threshold. No vague conditions.\n"
        "  - Mechanically reasoned: rationale must answer WHO is on the other side AND WHY.\n"
        "  - Frequent enough to validate: trigger must fire ≥50 times per pair on train data.\n"
        "  - Directionally committed: pick long or short before you propose, not after.\n\n"
        "AVOID:\n"
        "  - Vague rationale like 'momentum continuation' — be specific about the participant flow.\n"
        "  - Triggers that are clearly cherry-picked (atr_ratio > 2.71, etc.) — use round values.\n"
        "  - Triggers that fire on every bar (e.g. atr > 0.0001) or almost no bars (atr_ratio > 5).\n"
        "  - Inventing feature names not in the AVAILABLE FEATURES list above.\n\n"
        "Each mechanic you propose will be tested empirically. If it validates, you'll be asked "
        "to write code for it. If it doesn't, it costs you nothing further. So propose 3 "
        "STRUCTURALLY DIFFERENT mechanics rather than 3 variants of the same idea."
    )

    user_msg = "\n".join(context_parts) + "\n" + instructions

    system_prompt = (
        "You are a quantitative trading researcher. Your job RIGHT NOW is to "
        "propose candidate market mechanics — falsifiable claims about "
        "directional bias in forward returns conditional on a specific feature "
        "trigger. You will NOT write code in this step. A deterministic Python "
        "validator will test each mechanic's claim against historical data and "
        "you will only be asked to implement those that validate."
    )

    def _call():
        return _client().messages.create(
            model=CLAUDE_MODEL_FAST,
            max_tokens=2048,
            system=[{"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[_MECHANIC_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )

    response = _retry(_call)
    if response is None:
        log.error("propose_mechanic failed after retries")
        return []

    out = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_mechanic":
            out.append(dict(block.input))
    log.info("Claude proposed %d mechanic(s) for validation", len(out))
    return out


_HYPOTHESIS_TOOL = {
    "name": "submit_hypothesis",
    "description": (
        "Submit one trading strategy hypothesis targeting exactly one behaviour_type. "
        "Call this tool once per hypothesis — multiple tool calls per response are expected."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "function_name": {
                "type": "string",
                "description": (
                    "Python identifier for the entry function (without the 'entry_' prefix). "
                    "Lowercase letters, digits, underscores only. Max 60 chars. "
                    "Must be descriptive and unique. Example: 'vol_compression_flow_long'. "
                    "The full function will be named entry_{function_name}."
                ),
            },
            "code": {
                "type": "string",
                "description": (
                    "Complete Python function code defining entry_{function_name}. "
                    "Must follow the MANDATORY PATTERN exactly. "
                    "Do NOT include import statements (math, np, edge_engine functions are pre-imported). "
                    "Do NOT include SWEEP_DEF, ParameterGrid, or any code outside the function."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "1-2 sentences stating the behaviour_type being targeted and the "
                    "microstructure reason it should produce an edge."
                ),
            },
            "behaviour_type": {
                "type": "string",
                "enum": [
                    "breakout_continuation",
                    "false_breakout_liquidity_grab",
                    "mean_reversion_low_volatility",
                    "momentum_ignition_after_compression",
                    "trend_pullback_continuation",
                    "stop_run_reversal",
                ],
                "description": (
                    "The single market microstructure behaviour this hypothesis targets. "
                    "Your code conditions must directly implement this thesis. "
                    "Commit to one before writing code."
                ),
            },
            "additional_params": {
                "type": "object",
                "description": (
                    "Extra parameters beyond tp_r and sl_r that your function reads from params. "
                    "Key = param name, value = list of float values to grid search. "
                    "Use wide, round, evenly-spaced values: "
                    "{\"imb_thresh\": [0.1, 0.2, 0.3], \"min_adx\": [20, 25, 30]}. "
                    "NOT {\"imb_thresh\": [0.137, 0.152, 0.168]}. "
                    "Leave empty ({}) if only tp_r and sl_r are needed."
                ),
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
        },
        "required": ["function_name", "code", "rationale", "behaviour_type"],
    },
}

# ── Session context ───────────────────────────────────────────────────────────

_SESSION_CONTEXT = {
    'asian': (
        "Asian session (00:00-08:00 UTC). Pairs: USD/JPY, EUR/JPY, AUD/USD, XAU/USD. "
        "Typically range-bound with lower volatility. Mean-reversion and range breakout "
        "strategies at the session close (07:00-08:00) can work well."
    ),
    'london': (
        "London session (08:00-13:00 UTC). Pairs: GBP/USD, GBP/JPY, EUR/JPY, AUD/USD. "
        "High volatility at open (08:00-09:00). ORB on the first 30 minutes is the core thesis. "
        "Strong directional moves, news at 09:00-09:30 UK time."
    ),
    'ny': (
        "New York session (14:00-21:00 UTC). Pairs: EUR/USD, USD/JPY, XAU/USD. "
        "Strong momentum, US economic news at 13:30 UTC. ORB on the 14:30-15:00 range. "
        "Delta flows tend to be more persistent here than in Asian session."
    ),

    # ── Sub-corner sessions: less-mined regions of retail-quant hypothesis space ──

    'asian_yen': (
        "ASIAN CROSS-YEN MEAN REVERSION sub-corner (22:00-04:00 UTC). "
        "Pairs: USD/JPY, EUR/JPY, GBP/JPY. "
        "THESIS: cross-yen pairs in the deep Asian window are dominated by Tokyo "
        "interbank flow with thin liquidity. Strategies should target small mean-reverting "
        "moves toward the Asian range midpoint (use asian_high/asian_low/range_high/range_low). "
        "Edge per trade is small but consistent. AVOID breakout/momentum theses here — "
        "those belong in London/NY. Tight ATR-based SL, modest TP (tp_r 1.0-2.0). "
        "Most retail bots ignore this window because volatility looks unappealing — "
        "that lack of competition IS the edge. Behaviour types best suited: "
        "mean_reversion_low_volatility, false_breakout_liquidity_grab, stop_run_reversal."
    ),
    'post_news_drift': (
        "POST-NEWS DRIFT sub-corner (15-90 minutes AFTER high-impact news). "
        "Pairs: EUR/USD, GBP/USD, XAU/USD. "
        "THESIS: most automated systems disable trading around news events. The actual "
        "edge is NOT the news spike (over in seconds, brutal slippage) but the directional "
        "DRIFT in the 15-90 minutes that follow as institutional flow rebalances. "
        "Use near_news=False (the spike has passed) AND require persistent_imbalance and "
        "vol_imbalance confirmation in the drift direction. The h1_trend feature is "
        "particularly useful here — drift typically resolves toward the H1 trend regime. "
        "Only enter if regime is TRENDING. Behaviour types best suited: "
        "trend_pullback_continuation, momentum_ignition_after_compression, breakout_continuation."
    ),
    'london_close_handoff': (
        "LONDON→NY HANDOFF sub-corner (15:00-17:00 UTC). "
        "Pairs: EUR/USD, GBP/USD, USD/JPY. "
        "THESIS: London traders close positions 15:30-16:30 UTC while NY is fully active. "
        "This handoff creates a brief window of momentum continuation OR reversal as one "
        "side's flow exits and the other amplifies. Use bar_phase, dist_pivot_r1/s1, and "
        "prev_day_high/low to identify whether NY is defending or breaking London-built levels. "
        "Most ORB bots are entry-only at session open and miss this handoff entirely. "
        "Behaviour types best suited: trend_pullback_continuation, breakout_continuation, "
        "stop_run_reversal."
    ),
    'friday_squaring': (
        "FRIDAY POSITION-SQUARING sub-corner (Friday 13:00-17:00 UTC only). "
        "Pairs: EUR/USD, GBP/USD, EUR/JPY. "
        "THESIS: Friday-afternoon flow is dominated by institutional position squaring "
        "ahead of the weekend gap risk. This produces a) reversion toward weekly VWAP, "
        "b) trend exhaustion in the week's dominant direction, c) reduced overnight risk-on/off. "
        "MUST gate on row.day_of_week == 4 (Friday). Use h1_trend and momentum_10 to "
        "identify exhausted moves; mean-revert against them with tight stops. Most automated "
        "systems treat all weekdays identically — that's the edge. Behaviour types best "
        "suited: mean_reversion_low_volatility, false_breakout_liquidity_grab, stop_run_reversal."
    ),
}


# ── Client factory ────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _retry(fn, retries: int = 3):
    """Retry with exponential backoff on rate limit / connection errors.

    Pre-call: refuse if daily/total spend cap hit.
    Post-call: record token usage and accumulated USD cost.
    """
    if not _budget_allows():
        return None
    for attempt in range(retries):
        try:
            response = fn()
            try:
                model = getattr(response, "model", "") or ""
                _record_spend(model, getattr(response, "usage", None))
            except Exception as e:
                log.debug("spend recording failed: %s", e)
            return response
        except anthropic.RateLimitError:
            wait = 60 * (2 ** attempt)
            log.warning("Claude rate limit — waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
            time.sleep(wait)
        except anthropic.APIConnectionError:
            log.warning("Claude connection error (attempt %d/%d)", attempt + 1, retries)
            time.sleep(30)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                log.warning("Claude server error %s (attempt %d)", e.status_code, attempt + 1)
                time.sleep(30)
            else:
                # review#15 — 4xx from Claude (bad request, context overflow,
                # invalid model, schema mismatch) used to re-raise and crash
                # the agent loop. Treat as soft-fail so the loop survives.
                log.error("Claude 4xx %s — returning None (was: raise): %s",
                          e.status_code, e)
                return None
        except Exception as e:
            log.error("Claude unexpected error: %s — returning None", e)
            return None
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def generate_hypotheses(
    session:           str,
    pairs:             list,
    proven_conditions: list,
    top_results:       list,
    n:                 int = 3,
    meta_guidance:     str = "",
    bandit_weights:    dict | None = None,
    saturated_features: list | None = None,
) -> list:
    """
    Ask Claude to generate n novel entry function hypotheses.

    meta_guidance: synthesised learning from previous backtest results.
    When present, it steers generation toward productive hypothesis space.

    Returns list of dicts with keys:
        function_name, code, rationale, additional_params
    """
    context_parts = [
        f"Target session: {session.upper()}",
        f"Pairs: {', '.join(pairs)}",
        f"Session context: {_SESSION_CONTEXT.get(session, '')}",
        "",
    ]

    if top_results:
        existing_behaviour_types = sorted({
            r.get('behaviour_type', '') for r in top_results if r.get('behaviour_type')
        })
        context_parts.append(
            "TOP PERFORMING STRATEGIES — you MUST extend or recombine logic from these. "
            "Add one new orthogonal filter to a proven component rather than inventing from scratch:"
        )
        for r in top_results[:5]:
            bt = r.get('behaviour_type', '')
            context_parts.append(
                f"  • [{bt}] {r['strategy_name']} — "
                f"Sharpe: {r.get('test_sharpe', 0):.2f}, DSR: {r.get('dsr', 0):.2f} — "
                f"{r.get('rationale', '')[:100]}"
            )
        if existing_behaviour_types:
            context_parts.append(
                f"  BEHAVIOUR TYPES ALREADY IN TOP PERFORMERS: {', '.join(existing_behaviour_types)}. "
                f"Prioritise types not yet represented."
            )
        context_parts.append("")

    if proven_conditions:
        context_parts.append("KNOWN PRODUCTIVE CONDITIONS (add complementary filters, not duplicates):")
        for c in proven_conditions[:6]:
            context_parts.append(f"  • {c[:120]}")
        context_parts.append("")

    # Phase 5 — Thompson-bandit posterior weights tell the model which
    # *families* are paying off in live/recent backtests. Higher-weighted
    # families should drive new variants; near-zero ones probably aren't
    # worth re-exploring without a fresh angle.
    if bandit_weights:
        ranked = sorted(bandit_weights.items(), key=lambda kv: -kv[1])[:5]
        context_parts.append(
            "BANDIT POSTERIOR — capital-weight ranking of recent survivors "
            "(higher = posterior favours this family right now):"
        )
        for name, w in ranked:
            context_parts.append(f"  • {name}: weight {w:.3f}")
        context_parts.append(
            "  Bias new variants toward the top families. Bottom families need "
            "a genuinely new mechanism (not another parameter twist) to be worth a slot."
        )
        context_parts.append("")

    # Phase 5 — saturated features. Features that show up disproportionately
    # in REJECTED strategies are signals of over-use without edge. Tell the
    # model to avoid them or combine them with something orthogonal.
    if saturated_features:
        context_parts.append(
            "SATURATED FEATURES (used heavily in failed strategies — avoid as primary "
            "signal; only use as a secondary filter on something orthogonal):"
        )
        for feat, pct in saturated_features[:8]:
            context_parts.append(f"  • {feat}: {pct:.0f}% of rejections involved this feature")
        context_parts.append("")

    if meta_guidance:
        context_parts.append(
            "META-LEARNING GUIDANCE (synthesised from all previous backtest results — "
            "follow this closely, it reflects what actually survives our 11-gate filter):"
        )
        context_parts.append(meta_guidance)
        context_parts.append("")

    # Phase 4 — global FDR budget injection. Tells the model how much
    # cumulative false-discovery slack is left across all sweeps. As budget
    # shrinks the model should propose more parsimonious / genuinely-novel
    # hypotheses, not minor variants of existing survivors.
    try:
        from agent.db import fdr_budget_remaining
        b = fdr_budget_remaining()
        if b['cumulative'] > 0:
            context_parts.append(
                f"GLOBAL FDR BUDGET: tested {b['cumulative']} hypotheses cumulatively, "
                f"{b['n_significant']} survived global Benjamini-Yekutieli at α=0.05. "
                f"Budget remaining: {b['budget_remaining']*100:.1f}%. "
                f"As this number drops, every new hypothesis must be more genuinely "
                f"novel (different behavioural mechanism, not a minor parameter twist) "
                f"to clear the global gate."
            )
            context_parts.append("")
    except Exception:
        pass

    # 2026-06-05 — recent-rejection feedback. Show Claude what's been
    # failing so it stops repeating the same failure shapes.
    try:
        import sqlite3
        from agent.config import AGENT_DB_PATH
        _con = sqlite3.connect(AGENT_DB_PATH)
        _rows = _con.execute("""
            SELECT gate, COUNT(*) as n FROM rejection_log
            WHERE created_at > datetime('now', '-7 days')
            GROUP BY gate ORDER BY n DESC LIMIT 6
        """).fetchall()
        _con.close()
        if _rows:
            _summary = ", ".join(f"{n}× {gate}" for gate, n in _rows)
            context_parts.append(
                f"RECENT REJECTION FEEDBACK (last 7 days):\n"
                f"  Failure breakdown: {_summary}\n"
                f"  Avoid generating strategies with the failure shapes above. "
                f"If 'too_few_trades' or 'pre_screen' dominates, your last "
                f"batches were OVER-GATING — relax conditions until the "
                f"strategy fires at least 3 trades per week on average. "
                f"If 'catastrophic_sharpe' dominates, your entry direction "
                f"is wrong on the chosen condition — invert or rethink."
            )
            context_parts.append("")
    except Exception:
        pass

    # 2026-06-05 — explicit trade-frequency constraint. Strategies firing
    # <10 trades total are auto-rejected by pre_screen; Claude must target
    # ≥3 trades/week per pair (≈120/year) to survive.
    context_parts.append(
        "TRADE FREQUENCY CONSTRAINT:\n"
        "  Your strategy MUST fire ≥3 trades per week per pair on average "
        "across train data (≈120 trades/year per pair).\n"
        "  Strategies producing <10 trades total are AUTO-REJECTED by pre_screen "
        "BEFORE any robustness or Sharpe evaluation — they waste backtest "
        "compute and produce statistically meaningless results.\n"
        "  Before writing code, mentally walk through your entry conditions "
        "and estimate how often they fire. If your entry requires ALL of "
        "A AND B AND C AND D AND E to be true, you will fire too rarely.\n"
        "  PREFER:\n"
        "    - Single dominant condition with one filter:  `if A and B:`\n"
        "    - OR-of-pairs:                                `if (A and B) or (C and D):`\n"
        "  AVOID:\n"
        "    - Long AND chains:                            `if A and B and C and D and E:`\n"
        "    - Rare-event triggers with no widening (e.g. require atr_ratio > 2.5 AND adx > 30 AND tick_imb > 0.8 simultaneously)\n"
    )
    context_parts.append("")

    # 2026-06-06 — research mode. Web search is enabled on this call.
    # Ground each hypothesis in a real retail-FX strategy from the open
    # literature rather than brainstorming from the feature whitelist.
    context_parts.append(
        "RESEARCH MODE — use the `web_search` tool (up to 5 queries) before submitting.\n"
        f"  Pull a concrete intraday entry rule from retail-FX strategy sources for the {session.upper()} session and pairs {', '.join(pairs)}.\n"
        "  Good sources: trader blogs, forums (Forex Factory, Babypips, Reddit /r/Forex), "
        "academic working papers, broker education pages, trader ebooks.\n"
        "  Suggested queries:\n"
        f"    - '{session} session forex breakout strategy M1'\n"
        "    - 'FX intraday mean reversion entry rule'\n"
        "    - 'retail forex opening range breakout'\n"
        "    - 'forex pullback entry strategy 5 minute'\n"
        "  REJECT sources that give only vague advice — you need a concrete numeric trigger "
        "you can encode against the AVAILABLE FEATURES below. Map the source's concepts onto "
        "the feature list (do not invent feature names).\n"
        "  In each submit_hypothesis `rationale` field, include the source URL or domain "
        "you found the strategy in — so we can audit whether searches were actually consulted.\n"
    )
    context_parts.append("")

    # 2026-06-06 — TP/SL grid hint. Discovery-driven mechanics with 30/60-bar
    # holds were pre-screen-rejecting under the default tp_r∈[1.0,1.5,2.0],
    # sl_r∈[0.75,1.0,1.25] grid because TP hit before the bias played out.
    # Widen the default grid so mean-reversion theses have room.
    context_parts.append(
        "TP/SL GRID HINT:\n"
        "  When you populate `additional_params`, prefer WIDER TP grids with the new defaults:\n"
        "    tp_r = [1.0, 2.0, 3.0, 4.0]\n"
        "    sl_r = [1.0, 1.5, 2.0]\n"
        "  Tight TPs (≤2.0 ATR) choke any 30–60-bar mean-reversion or drift play before it completes.\n"
    )
    context_parts.append("")

    user_msg = (
        "\n".join(context_parts) +
        f"Generate exactly {n} hypotheses by calling submit_hypothesis {n} times. "
        f"Each must target a different behaviour_type. "
        f"Apply the internal quality filter before each submission — "
        f"if a hypothesis fails any of the four discard criteria, discard it and generate a better one. "
        + ("If no survivors exist yet, invent freely but apply all research discipline rules. " if not top_results else "")
        + ("Follow the META-LEARNING GUIDANCE above — it tells you what actually works. " if meta_guidance else "")
    )

    def _call():
        return _client().messages.create(
            model=CLAUDE_MODEL_FAST,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[
                # Server-side web search (2026-06-06): grounds hypotheses in
                # real retail-FX strategy literature instead of pure
                # brainstorming. Anthropic handles the search round-trips
                # inline; response.content will contain server_tool_use /
                # web_search_tool_result blocks that the parser below skips.
                # allowed_callers=["direct"] disables programmatic tool
                # calling (PTC), which Haiku 4.5 doesn't support — without
                # this the API 400s.
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": 5,
                    "allowed_callers": ["direct"],
                },
                _HYPOTHESIS_TOOL,
            ],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )

    response = _retry(_call)
    if response is None:
        log.error("Claude hypothesis generation failed after all retries")
        return []

    results = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_hypothesis":
            inp = block.input
            results.append({
                "function_name":    inp.get("function_name", ""),
                "code":             inp.get("code", ""),
                "rationale":        inp.get("rationale", ""),
                "behaviour_type":   inp.get("behaviour_type", ""),
                "additional_params": inp.get("additional_params") or {},
            })

    log.info("Claude returned %d hypotheses (requested %d)", len(results), n)
    return results


def bulk_generate_hypotheses_for_session(
    session: str,
    pairs:   list,
    n:       int = 10,
    max_tokens: int = 32000,
) -> list:
    """Bulk one-shot hypothesis generation for a single session via web search.

    Designed to be called offline (cron / manual refill) to populate a queue
    the agent loop drains from across many rounds. One call covers up to
    `max_uses=5` web searches and produces `n` diverse hypotheses, amortising
    the search surcharge across all of them — typically ~5× cheaper per
    strategy than calling `generate_hypotheses` once per round.

    Differs from `generate_hypotheses`:
      - Cold call: no top_results / meta_guidance / bandit / saturated /
        FDR / rejection-log context. Strategies are queued for future
        rounds, so per-round runtime state would be wrong anyway.
      - Higher max_tokens (32K default vs MAX_TOKENS=3000) because N=10
        full code blocks + rationales easily exceeds 16K tokens.
      - Prompt asks for STRUCTURAL diversity across the N strategies
        (different behaviour_types, different feature combinations) so
        the queue isn't full of near-duplicates.

    Returns list of dicts: function_name, code, rationale, behaviour_type,
    additional_params. Same shape as `generate_hypotheses`.
    """
    if not _budget_allows():
        log.warning("bulk_generate_hypotheses_for_session skipped: budget cap reached")
        return []

    user_msg = (
        f"Target session: {session.upper()}\n"
        f"Pairs available: {', '.join(pairs)}\n"
        f"Session context: {_SESSION_CONTEXT.get(session, '')}\n"
        "\n"
        "BULK GENERATION MODE — produce a diverse batch of strategies in one call.\n"
        f"  Generate exactly {n} hypotheses by calling submit_hypothesis {n} times.\n"
        "  Each must target a DIFFERENT behaviour_type or use a STRUCTURALLY "
        "different feature combination. Reject near-duplicates of strategies "
        "you've already submitted in this call.\n"
        "\n"
        "RESEARCH PHASE — start by using `web_search` up to 5 times.\n"
        f"  Pull concrete intraday entry rules for the {session.upper()} session "
        f"on {', '.join(pairs)} from retail-FX strategy sources (trader blogs, "
        "Forex Factory / Babypips / Reddit, academic working papers, broker "
        "education pages, trader ebooks).\n"
        "  Suggested searches (vary them — don't run the same query 5 times):\n"
        f"    - '{session} session forex breakout strategy M1'\n"
        "    - 'FX intraday mean reversion entry rule'\n"
        "    - 'retail forex opening range breakout 5 minute'\n"
        "    - 'forex pullback continuation strategy'\n"
        "    - 'forex liquidity grab stop run strategy'\n"
        "  Reject vague sources — you need a concrete numeric trigger you can "
        "encode against the AVAILABLE FEATURES.\n"
        "\n"
        "IMPLEMENTATION PHASE — submit each strategy via submit_hypothesis.\n"
        "  Map each source's concepts onto the AVAILABLE FEATURES list (do not "
        "invent feature names). Reuse search results across multiple "
        "hypotheses — one good source can seed 2-3 variants with different "
        "behaviour_types or filter angles.\n"
        "  In each rationale, cite the source URL or domain you found the "
        "strategy in. This lets us audit whether the search was actually "
        "consulted.\n"
        "\n"
        "TRADE FREQUENCY CONSTRAINT:\n"
        "  Strategies firing <10 trades total on train data are auto-rejected. "
        "Target ≥3 trades/week per pair (≈120/year). Avoid long AND chains. "
        "Prefer single-dominant-condition + one filter, or OR-of-pairs.\n"
        "\n"
        "TP/SL GRID HINT:\n"
        "  When populating additional_params, prefer wider TP grids: "
        "tp_r=[1.0, 2.0, 3.0, 4.0], sl_r=[1.0, 1.5, 2.0]. Tight TPs choke "
        "30-60-bar mean-reversion plays before they complete.\n"
    )

    def _call():
        # Streaming is required for max_tokens > ~16K (SDK refuses
        # non-streaming calls that may exceed the 10-minute HTTP timeout).
        # get_final_message() collects the complete Message object — same
        # shape the parser below already handles.
        with _client().messages.stream(
            model=CLAUDE_MODEL_FAST,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": 5,
                    "allowed_callers": ["direct"],
                },
                _HYPOTHESIS_TOOL,
            ],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            return stream.get_final_message()

    response = _retry(_call)
    if response is None:
        log.error("bulk_generate_hypotheses_for_session failed for %s", session)
        return []

    results = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_hypothesis":
            inp = block.input
            fn = (inp.get("function_name") or "").strip()
            code = (inp.get("code") or "").strip()
            if not fn or not code:
                continue  # skip Claude's empty submissions (intermittent)
            results.append({
                "function_name":     fn,
                "code":              code,
                "rationale":         inp.get("rationale", ""),
                "behaviour_type":    inp.get("behaviour_type", ""),
                "additional_params": inp.get("additional_params") or {},
            })

    log.info("[bulk] %s: returned %d/%d non-empty hypotheses",
             session, len(results), n)
    return results


def implement_validated_mechanic(
    mechanic:          dict,
    validation_result: dict,
    session:           str,
    pairs:             list,
) -> list:
    """Stage 3: ask Claude to write code for ONE validated mechanic.

    Returns at most 1 hypothesis dict (function_name, code, rationale,
    additional_params). Returns [] if Claude declines or budget is exhausted.

    The prompt makes it clear that the mechanic ALREADY VALIDATED — Claude's
    job is just to implement it cleanly, not to invent new conditions.
    """
    if not _budget_allows():
        log.warning("implement_validated_mechanic skipped: budget cap reached")
        return []

    trig = mechanic["trigger"]
    ctx_filters = mechanic.get("context_filters") or []
    ctx_str = "; ".join(
        f"{c['feature']} {c['comparison']} {c.get('threshold', c.get('values'))}"
        for c in ctx_filters
    ) or "(none)"

    user_msg = (
        f"Implement the following EMPIRICALLY VALIDATED mechanic as an "
        f"entry_{{function_name}} function. The mechanic's directional bias has "
        f"already been verified on historical data — you do NOT need to add "
        f"extra filters or change the trigger logic. Just implement what's here.\n\n"
        f"=== VALIDATED MECHANIC ===\n"
        f"  mechanic_id:           {mechanic['mechanic_id']}\n"
        f"  rationale:             {mechanic['rationale']}\n"
        f"  trigger:               {trig['feature']} {trig['comparison']} {trig['threshold']}\n"
        f"  context filters:       {ctx_str}\n"
        f"  direction:             {mechanic['direction_hypothesis']} (forward {mechanic['forward_horizon_bars']} bars)\n"
        f"  pair_universe:         {mechanic.get('pair_universe') or pairs}\n"
        f"\n"
        f"=== VALIDATION EVIDENCE ===\n"
        f"  n_events:              {validation_result.get('n_events', 0)}\n"
        f"  mean fwd return (ATR): {validation_result.get('mean_fwd_return', 0):+.4f}\n"
        f"  t-stat:                {validation_result.get('t_stat', 0):+.3f}\n"
        f"  per-pair breakdown:    {validation_result.get('per_pair', {})}\n"
        f"\n"
        f"=== YOUR TASK ===\n"
        f"  Session: {session.upper()}.\n"
        f"  Pairs: {', '.join(pairs)}.\n"
        f"  Write ONE entry function that fires on the trigger + context filters above. "
        f"Direction: {mechanic['direction_hypothesis']}. SL/TP scaled to ATR. "
        f"Use place_pending(mode='market_next_open') to fire the order on the "
        f"NEXT bar's open (not the trigger bar itself — would be look-ahead).\n"
        f"  Provide tp_r and sl_r as additional_params with WIDE grids: "
        f"tp_r=[1.0, 2.0, 3.0, 4.0], sl_r=[1.0, 1.5, 2.0]. The validated bias "
        f"plays out over {mechanic['forward_horizon_bars']} bars; tight TPs "
        f"(≤2.0 ATR) would exit before the move completes. No other extra params — "
        f"the mechanic is already specified.\n"
        f"  rationale: copy the mechanic_id and rationale into your "
        f"submit_hypothesis call's rationale field.\n"
    )

    def _call():
        return _client().messages.create(
            model=CLAUDE_MODEL_FAST,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text", "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[_HYPOTHESIS_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )

    response = _retry(_call)
    if response is None:
        log.error("implement_validated_mechanic failed after retries: %s",
                  mechanic.get("mechanic_id"))
        return []

    out = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_hypothesis":
            inp = block.input
            out.append({
                "function_name":     inp.get("function_name", ""),
                "code":              inp.get("code", ""),
                "rationale":         inp.get("rationale", "") or mechanic.get("rationale", ""),
                "behaviour_type":    inp.get("behaviour_type", ""),
                "additional_params": inp.get("additional_params") or {},
                "mechanic_id":       mechanic.get("mechanic_id"),
                "validation":        validation_result,
            })
            break  # Only take the first implementation per validated mechanic

    if out:
        log.info("Implemented mechanic '%s' as '%s'",
                 mechanic.get("mechanic_id"), out[0]["function_name"])
    return out


def fix_syntax_error(original_code: str, error_msg: str, function_name: str) -> list:
    """
    Ask Claude to fix a syntax error in a previously generated function.
    Returns list with one fixed hypothesis dict, or empty list on failure.
    """
    user_msg = (
        f"The following entry function has a syntax error:\n\n"
        f"Error: {error_msg}\n\n"
        f"```python\n{original_code}\n```\n\n"
        f"Fix the syntax error and resubmit via submit_hypothesis. "
        f"Keep the function name entry_{function_name} and preserve the original logic."
    )

    def _call():
        return _client().messages.create(
            model=CLAUDE_MODEL_DEEP,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[_HYPOTHESIS_TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user_msg}],
        )

    try:
        response = _retry(_call)
        if response is None:
            return []
        results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_hypothesis":
                inp = block.input
                results.append({
                    "function_name":    inp.get("function_name", function_name),
                    "code":             inp.get("code", ""),
                    "rationale":        inp.get("rationale", ""),
                    "additional_params": inp.get("additional_params") or {},
                })
        return results
    except Exception as e:
        log.error("Syntax fix attempt failed: %s", e)
        return []


# ── Adversarial review (Phase 5.3) ────────────────────────────────────────────

_ADVERSARIAL_TOOL = {
    "name": "submit_review",
    "description": (
        "Submit an adversarial review of a candidate strategy. Be skeptical: the "
        "default assumption is that the strategy has a hidden flaw. Only PASS if "
        "you can find no concrete look-ahead, peeking, or overfitting risk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type":  "string",
                "enum":  ["PASS", "REJECT"],
                "description": "PASS only if no concrete bias is identified.",
            },
            "category": {
                "type":  "string",
                "enum":  ["look_ahead", "peeking", "overfitting", "data_snooping",
                          "survivorship", "regime_dependence", "none"],
                "description": "Primary failure mode if REJECT, else 'none'.",
            },
            "reason": {
                "type":  "string",
                "description": "One-paragraph explanation citing specific lines or features.",
            },
        },
        "required": ["verdict", "category", "reason"],
    },
}

_ADVERSARIAL_SYSTEM = (
    "You are an adversarial code reviewer for quantitative trading strategies. "
    "Your job is to find look-ahead bias, peeking, data snooping, or overfitting "
    "risk in the candidate code. You are deliberately skeptical: assume there IS a "
    "flaw and search for it. Common failure modes:\n"
    "  1. Look-ahead: rolling/expanding stats that include the current bar without .shift(1)\n"
    "  2. Peeking: comparing to future bars (e.g. row.next_high, df.shift(-1))\n"
    "  3. Data snooping: hyperparameters that look suspiciously fitted to a regime\n"
    "  4. Overfitting: many narrowly-conjunctive conditions ('and X and Y and Z and W'),\n"
    "     each with a specific magic threshold; PASS only if every condition has a clear\n"
    "     mechanical reason\n"
    "  5. Survivorship-implicit features (e.g. references to instruments selected by\n"
    "     historical performance)\n"
    "  6. Regime dependence: signal that only fires in one volatility regime\n"
    "Submit verdict via the submit_review tool. Be concise."
)


def review_strategy_for_bias(strategy_name: str, code: str, rationale: str = "") -> dict:
    """
    Adversarial Claude review of a candidate strategy. Called immediately
    before promotion to survivor. Strategies that fail this review should
    be rejected and (optionally) sent back to the evolver as raw material.

    Returns a dict: {'verdict': 'PASS'|'REJECT', 'category': str, 'reason': str}.
    On API failure or budget exhaustion returns {'verdict': 'PASS', ...}
    so the call never blocks promotion if the reviewer is unavailable —
    primary statistical filters remain authoritative.
    """
    if not _budget_allows():
        return {'verdict': 'PASS', 'category': 'none',
                'reason': 'budget exhausted — review skipped'}

    user_msg = (
        f"Strategy name: {strategy_name}\n"
        f"Author rationale: {rationale or '(none)'}\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        f"Review for look-ahead bias, peeking, data snooping, overfitting, "
        f"survivorship bias, or regime dependence. Submit your verdict via "
        f"submit_review."
    )

    def _call():
        return _client().messages.create(
            model      = CLAUDE_MODEL_FAST,
            max_tokens = 800,
            system     = _ADVERSARIAL_SYSTEM,
            tools      = [_ADVERSARIAL_TOOL],
            tool_choice = {"type": "tool", "name": "submit_review"},
            messages   = [{"role": "user", "content": user_msg}],
        )

    try:
        response = _retry(_call)
        if response is None:
            return {'verdict': 'PASS', 'category': 'none',
                    'reason': 'reviewer unreachable — defaulting PASS'}
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_review":
                inp = block.input or {}
                return {
                    'verdict':  (inp.get('verdict') or 'PASS').upper(),
                    'category': inp.get('category') or 'none',
                    'reason':   (inp.get('reason') or '').strip(),
                }
        return {'verdict': 'PASS', 'category': 'none',
                'reason': 'no tool_use block returned — defaulting PASS'}
    except Exception as e:
        log.warning("Adversarial review failed for '%s': %s", strategy_name, e)
        return {'verdict': 'PASS', 'category': 'none', 'reason': f'reviewer exception: {e}'}


def generate_daily_report(top_results: list, stats: dict) -> str:
    """
    Ask Claude to write a 2-3 sentence narrative summary of top results.
    Returns plain text string, or empty string on failure.
    """
    if not top_results:
        return ""

    results_text = "\n".join(
        f"{i}. {r['strategy_name']} ({r.get('session','?').upper()}): "
        f"Sharpe={r.get('test_sharpe', 0):.2f}, DSR={r.get('dsr', 0):.2f}, "
        f"WR={r.get('test_wr', 0) * 100:.1f}%, N={r.get('n_trades', 0)} — "
        f"{r.get('rationale', 'no rationale')}"
        for i, r in enumerate(top_results, 1)
    )

    def _call():
        return _client().messages.create(
            model=CLAUDE_MODEL_DEEP,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a 2-3 sentence narrative for a quant trader reviewing today's "
                    f"automated edge discovery results. Identify what the top strategies share "
                    f"in common, what market conditions they target, and any pattern worth noting. "
                    f"Tone: direct and analytical. No filler.\n\nResults:\n{results_text}"
                ),
            }],
        )

    try:
        response = _retry(_call)
        if response is None:
            return ""
        # review#15 — guard against empty content list (IndexError used to
        # bubble out and crash the report path).
        blocks = getattr(response, 'content', None) or []
        for blk in blocks:
            text = getattr(blk, 'text', None)
            if text:
                return text.strip()
        return ""
    except Exception as e:
        log.warning("Daily narrative generation failed: %s", e)
        return ""
