"""Static constraints document — prop-firm rules, cost model, gauntlet thresholds."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import paths


def _read_gauntlet_thresholds() -> dict:
    """Pull current threshold values from agent.config (read-only)."""
    try:
        from agent import config as ac  # type: ignore
        return {
            'MIN_TEST_TRADES':    getattr(ac, 'MIN_TEST_TRADES', 20),
            'MIN_TEST_SHARPE':    getattr(ac, 'MIN_TEST_SHARPE', 0.50),
            'MIN_DSR':            getattr(ac, 'MIN_DSR', 0.10),
            'MAX_TEST_DRAWDOWN':  getattr(ac, 'MAX_TEST_DRAWDOWN', 0.20),
            'MAX_TRAIN_TEST_DECAY': getattr(ac, 'MAX_TRAIN_TEST_DECAY', 0.50),
            'MC_MIN_PASS_PCT':    getattr(ac, 'MC_MIN_PASS_PCT', 40.0),
            'MC_MAX_BLOWN_PCT':   getattr(ac, 'MC_MAX_BLOWN_PCT', 15.0),
            'REQUIRE_BH_SIG':     getattr(ac, 'REQUIRE_BH_SIG', True),
            'REQUIRE_REGIME_STABLE': getattr(ac, 'REQUIRE_REGIME_STABLE', True),
            'USE_MC_GATE':        getattr(ac, 'USE_MC_GATE', True),
            'MIN_SHARPE_CI_LOW':  getattr(ac, 'MIN_SHARPE_CI_LOW', 0.0),
        }
    except Exception:
        return {}


def generate(out_path: Optional[Path] = None) -> str:
    """Build constraints.md content. Optionally write to disk."""
    t = _read_gauntlet_thresholds()

    body = f"""# Constraints

Market: **{paths.MARKET_NAME}**
Universe: 8 USDT-margined perpetuals on Bybit (BTCUSDT, ETHUSDT,
SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT). Trades 24/7.
Bar resolution: M5 (5-minute OHLCV + funding/OI/basis side tables).

Cost model:
- Round-trip spread + commission, slippage profile per session.
- Dynamic spread per bar (median spread varies by symbol + time of day).
- No optimism baked in.

Prop firm rules (HyroTrader-style):
- Profit target: 6%
- Max drawdown: 6% (absolute peak-to-trough on equity)
- Daily loss limit: 4% (single-day floor below previous equity)
- **No time limit.** Strategies may hold for hours, days, or weeks.
  This is a major degree of freedom — most retail strategies cluster
  at 1h-24h hold. You are explicitly free to propose multi-day and
  multi-week ideas.

Gauntlet (your proposal will be tested against ALL of these):

| Gate | Threshold |
|---|---|
| Min test trades | {t.get('MIN_TEST_TRADES', '?')} |
| Min test Sharpe (Bayesian-shrunk) | {t.get('MIN_TEST_SHARPE', '?')} |
| Min DSR (Deflated Sharpe) | {t.get('MIN_DSR', '?')} |
| Max test drawdown | {t.get('MAX_TEST_DRAWDOWN', '?')} |
| Max train→test decay | {t.get('MAX_TRAIN_TEST_DECAY', '?')} |
| Benjamini-Hochberg FDR significance | required = {t.get('REQUIRE_BH_SIG', '?')} |
| Regime stability | required = {t.get('REQUIRE_REGIME_STABLE', '?')} |
| MC eval pass rate (prop-firm sim) | ≥ {t.get('MC_MIN_PASS_PCT', '?')}% |
| MC blown rate | ≤ {t.get('MC_MAX_BLOWN_PCT', '?')}% |
| Bootstrap Sharpe CI lower bound | > {t.get('MIN_SHARPE_CI_LOW', '?')} |

Cumulatively across these gates, ~1,800 hypotheses have produced ZERO
viable strategies. The gauntlet is unforgiving. Design proposals to
survive it: enough trades, strong per-trade expectancy, robust across
regimes, not overfit (low degree-of-freedom count).
"""

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding='utf-8')

    return body
