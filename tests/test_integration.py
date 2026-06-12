"""
Phase 8 — integration test scaffold.

Hits the cross-module wiring landed in Phases 1-7 so a future refactor
that breaks the gate / risk / scorer / macro contracts trips a red light
in CI before it reaches live trading.

Each test exercises a code path; none requires network, MT5, or the
Anthropic API. Heavy imports (edge_engine) are lazy so a single broken
module doesn't cascade-fail the rest of the suite.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Phase 7 — survivor scorer gates
# =============================================================================

class TestScorerGates:
    def _base_metrics(self):
        return {
            'test_n': 100, 'test_sharpe': 1.0, 'dsr': 0.5,
            'train_sharpe': 0.8, 'test_max_dd': -0.05,
            'regime_stable': 1, 'bh_sig': 1, 'by_sig': 1,
            'sharpe_ci_low': 0.2,
        }

    def test_grandfathered_row_passes(self):
        from agent.scorer import is_survivor
        # No PBO / PSR populated — older sweep rows must still pass.
        assert is_survivor(self._base_metrics()) is True

    def test_pbo_above_threshold_rejects(self):
        from agent.scorer import is_survivor, rejection_reason
        m = dict(self._base_metrics(), pbo_score=0.7)
        assert is_survivor(m) is False
        assert 'PBO' in rejection_reason(m)

    def test_psr_below_threshold_rejects(self):
        from agent.scorer import is_survivor, rejection_reason
        m = dict(self._base_metrics(), psr=0.5)
        assert is_survivor(m) is False
        assert 'PSR' in rejection_reason(m)

    def test_both_pbo_and_psr_pass(self):
        from agent.scorer import is_survivor
        m = dict(self._base_metrics(), pbo_score=0.3, psr=0.99)
        assert is_survivor(m) is True

    def test_low_trade_count_still_dominant(self):
        # New gates must not relax existing gates.
        from agent.scorer import is_survivor
        m = dict(self._base_metrics(), test_n=5, pbo_score=0.1, psr=0.99)
        assert is_survivor(m) is False


# =============================================================================
# Phase 7 — statistical primitives
# =============================================================================

class TestStatPrimitives:
    def test_psr_monotonic_in_sharpe(self):
        from edge_engine import probabilistic_sharpe_ratio
        # Use small n so the standard error is large enough that PSR(low)
        # doesn't saturate at 1.0 — that's the regime where the gate matters.
        a = probabilistic_sharpe_ratio(0.05, 30)
        b = probabilistic_sharpe_ratio(0.50, 30)
        assert b > a

    def test_psr_in_unit_interval(self):
        from edge_engine import probabilistic_sharpe_ratio
        for s in (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0):
            v = probabilistic_sharpe_ratio(s, 100)
            assert 0.0 <= v <= 1.0

    def test_pbo_random_walk_is_noisy_around_half(self):
        # Bailey-LdP: random strategies should give PBO near 0.5; the function
        # is allowed to drift in [0.2, 0.8] on small samples — this is a smoke
        # test that the routine doesn't return obviously broken numbers.
        import numpy as np
        import pandas as pd
        from edge_engine import pbo_score
        rng = np.random.default_rng(0)
        mat = pd.DataFrame(rng.standard_normal((400, 16)) * 0.01,
                           columns=[f's{i}' for i in range(16)])
        v = pbo_score(mat)
        assert 0.0 <= v <= 1.0

    def test_winsorize_returns_finite_for_normal_pnl(self):
        import numpy as np
        import pandas as pd
        from edge_engine import winsorize_sharpe
        rng = np.random.default_rng(1)
        pnl = pd.Series(rng.standard_normal(400) * 0.005)
        out = winsorize_sharpe(pnl)
        assert math.isfinite(out['sharpe_raw'])
        assert math.isfinite(out['sharpe_winsor'])

    def test_hac_smoke(self):
        import numpy as np
        import pandas as pd
        from edge_engine import hac_sharpe
        rng = np.random.default_rng(2)
        pnl = pd.Series(rng.standard_normal(200) * 0.01)
        v = hac_sharpe(pnl)
        assert math.isfinite(v)


# =============================================================================
# Phase 6 — risk manager combined gate
# =============================================================================

class TestRiskGate:
    def test_dd_size_multiplier_halves_above_threshold(self):
        from agent.risk_manager import dd_size_multiplier
        # Below halve threshold (ratio = 0.01/0.05 = 0.20 < 0.50)
        assert dd_size_multiplier(0.01, 0.05) == 1.0
        # Above halve threshold (ratio = 0.03/0.05 = 0.60 ≥ 0.50)
        assert dd_size_multiplier(0.03, 0.05) == 0.5

    def test_currency_exposure_blocks_overconcentration(self):
        from agent.risk_manager import currency_exposure_ok
        # Already 5% on USD via GBPUSD; another 2% would push past the 6% cap
        open_book = [{'pair': 'GBPUSD', 'risk_usd': 50.0}]
        ok, _reason, _exp = currency_exposure_ok(
            'EURUSD', new_risk=20.0, open_book=open_book, equity=1000.0)
        assert ok is False

    def test_currency_exposure_allows_under_cap(self):
        from agent.risk_manager import currency_exposure_ok
        open_book = [{'pair': 'GBPUSD', 'risk_usd': 20.0}]
        ok, _reason, _exp = currency_exposure_ok(
            'AUDJPY', new_risk=15.0, open_book=open_book, equity=1000.0)
        assert ok is True


# =============================================================================
# Phase 6.3 — impact-aware news buffer
# =============================================================================

class TestNewsBuffer:
    def test_news_buffer_widths_present(self):
        from edge_engine import NEWS_BUFFER_BY_IMPACT
        assert NEWS_BUFFER_BY_IMPACT['high']   >= NEWS_BUFFER_BY_IMPACT['medium']
        assert NEWS_BUFFER_BY_IMPACT['medium'] >= NEWS_BUFFER_BY_IMPACT['low']


# =============================================================================
# Phase 9 — macro_data fetchers must degrade silently offline
# =============================================================================

class TestMacroDataDegradesGracefully:
    def test_all_fetchers_return_dataframe(self):
        import pandas as pd
        from agent import macro_data
        for fn in (macro_data.fetch_forexfactory_week,
                   macro_data.fetch_cot_weekly,
                   macro_data.fetch_macro_daily,
                   macro_data.fetch_oanda_positioning):
            assert isinstance(fn(), pd.DataFrame)

    def test_merge_macro_features_idempotent_on_empty(self):
        import pandas as pd
        from agent import macro_data
        out = macro_data.merge_macro_features(pd.DataFrame(), 'EURUSD')
        assert out.empty

    def test_merge_macro_features_canonical_input_preserves_shape(self, monkeypatch):
        """review#P2#7 — feed a canonical non-empty df and assert (a) row
        count preserved, (b) every input column survives untouched (same
        values, no NaNs introduced), (c) merge succeeds with both fresh
        fetchers and degraded-empty fetchers. Mock the three macro fetchers
        so the test is hermetic (no network)."""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        from agent import macro_data

        # 5 daily bars (one row per date) — minimum useful canonical shape.
        dates = [datetime(2025, 1, d).date() for d in range(2, 7)]
        df_in = pd.DataFrame({
            'date':  dates,
            'close': [1.10, 1.11, 1.09, 1.12, 1.13],
            'atr':   [0.001, 0.0012, 0.0009, 0.0011, 0.0013],
        })

        # Mock the macro daily fetcher to return canned non-empty data
        # covering the input date range.
        fake_macro = pd.DataFrame({
            'DXY_chg':   [0.001, -0.002, 0.0,    0.0015, -0.001],
            'US10Y_chg': [0.005, -0.003, 0.001, -0.002, 0.004],
            'SPX_chg':   [0.002, 0.001, -0.001, 0.003, 0.0],
            'GOLD_chg':  [-0.001, 0.002, 0.0,   0.001, -0.002],
            'VIX_chg':   [0.01, -0.005, 0.0,    0.008, -0.003],
            'VIX':       [15.0, 16.0, 17.0,    18.0, 19.0],
        }, index=pd.to_datetime(dates))
        monkeypatch.setattr(macro_data, 'fetch_macro_daily', lambda: fake_macro)
        monkeypatch.setattr(macro_data, 'cot_net_position_pct',
                            lambda pair: (0.55, 0))
        monkeypatch.setattr(macro_data, 'retail_long_pct', lambda pair: 0.42)

        out = macro_data.merge_macro_features(df_in.copy(), 'EURUSD')

        # (a) row count preserved
        assert len(out) == len(df_in)
        # (b) every original column intact
        for col in df_in.columns:
            assert col in out.columns, f"merge dropped input column {col!r}"
            assert (out[col].to_list() == df_in[col].to_list()), \
                   f"merge mutated input column {col!r}"
            assert not out[col].isna().any(), \
                   f"merge introduced NaN into input column {col!r}"
        # (c) the canonical macro columns landed and are numeric
        for col in ('dxy_chg', 'us10y_chg', 'spx_chg', 'gold_chg',
                    'vix_chg', 'vix_level',
                    'cot_net_position_pct', 'cot_extreme_flag',
                    'retail_long_pct'):
            assert col in out.columns, f"missing macro column {col!r}"
        # The daily-macro columns must align by date (no NaN since the
        # fake_macro index covers exactly df_in['date']).
        assert not out['dxy_chg'].isna().any()
        assert not out['vix_level'].isna().any()

    def test_merge_macro_features_degrades_on_fetcher_failure(self, monkeypatch):
        """review#P2#7 — when every macro source raises, the input df must
        return unchanged in shape (no rows dropped, no input cols dropped).
        This is the hot path on a VPS with no internet."""
        import pandas as pd
        from datetime import datetime
        from agent import macro_data

        df_in = pd.DataFrame({
            'date':  [datetime(2025, 1, d).date() for d in range(2, 5)],
            'close': [1.10, 1.11, 1.09],
        })

        def _boom(*a, **kw):
            raise RuntimeError("network unreachable")
        monkeypatch.setattr(macro_data, 'fetch_macro_daily',
                            lambda: pd.DataFrame())
        monkeypatch.setattr(macro_data, 'cot_net_position_pct', _boom)
        monkeypatch.setattr(macro_data, 'retail_long_pct', _boom)

        out = macro_data.merge_macro_features(df_in.copy(), 'EURUSD')

        assert len(out) == len(df_in)
        for col in df_in.columns:
            assert col in out.columns
            assert (out[col].to_list() == df_in[col].to_list())

    def test_cot_returns_neutral_default_on_no_match(self):
        from agent.macro_data import cot_net_position_pct
        pct, flag = cot_net_position_pct('XYZUSD')   # nonsense pair
        assert 0.0 <= pct <= 1.0
        assert flag in (-1, 0, 1)


# =============================================================================
# Phase 5.2 — saturated-feature query degrades on empty DB
# =============================================================================

class TestMetaLearnerSaturatedFeatures:
    def test_saturated_features_returns_list(self):
        from agent import meta_learner
        out = meta_learner.get_saturated_features()
        assert isinstance(out, list)


# =============================================================================
# Phase 8 — secret redaction filter
# =============================================================================

class TestSecretRedaction:
    def test_redacts_anthropic_key(self):
        import logging
        from agent.log_filter import SecretRedactionFilter
        filt = SecretRedactionFilter()
        rec = logging.LogRecord(
            'test', logging.INFO, __file__, 1,
            'leaked sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA in trace', (), None,
        )
        filt.filter(rec)
        msg = rec.getMessage()
        assert 'sk-ant' not in msg

    def test_redacts_telegram_token(self):
        import logging
        from agent.log_filter import SecretRedactionFilter
        filt = SecretRedactionFilter()
        rec = logging.LogRecord(
            'test', logging.INFO, __file__, 1,
            'bot 123456789:AAH9aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa fetched', (), None,
        )
        filt.filter(rec)
        msg = rec.getMessage()
        assert 'AAH9' not in msg


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
