"""TCA live-data feedback tests (review#P2#5).

Covers `agent.tca.update_measured_spreads_from_live` (review#P2#4): given a
DataFrame of per-pair summaries, the function must (a) skip pairs with too
few fills, (b) convert pip-quoted spreads to price units correctly, and (c)
write a JSON file with `_meta.updated_at`.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import agent.tca as tca


def _summary_df(rows):
    return pd.DataFrame(rows)


def test_update_measured_spreads_skips_thin_data(tmp_path, monkeypatch):
    monkeypatch.setattr(tca, 'AGENT_DB_PATH', str(tmp_path / 'agent' / 'x.db'))
    (tmp_path / 'agent').mkdir()

    df = _summary_df([
        {'pair': 'EUR_USD', 'avg_spread_pips': 0.6, 'n_fills': 5},   # < min_fills
        {'pair': 'GBP_USD', 'avg_spread_pips': 0.9, 'n_fills': 50},  # ok
    ])
    with patch.object(tca, 'per_pair_summary', return_value=df):
        out = tca.update_measured_spreads_from_live(min_fills=20)

    assert 'EUR_USD' not in out
    assert 'GBP_USD' in out
    # 0.9 pips × 0.0001 pip-size = 9e-5 in price units
    assert abs(out['GBP_USD'] - 0.9 * 0.0001) < 1e-9


def test_update_measured_spreads_writes_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(tca, 'AGENT_DB_PATH', str(tmp_path / 'agent' / 'x.db'))
    (tmp_path / 'agent').mkdir()

    df = _summary_df([
        {'pair': 'EUR_USD', 'avg_spread_pips': 0.55, 'n_fills': 30},
    ])
    with patch.object(tca, 'per_pair_summary', return_value=df):
        out = tca.update_measured_spreads_from_live(min_fills=20)

    assert out  # non-empty
    target = tmp_path / 'live_measured_spreads.json'
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload['_meta']['source'] == 'live'
    assert 'updated_at' in payload['_meta']
    assert 'EUR_USD' in payload['spreads']


def test_update_measured_spreads_jpy_pip_conversion(tmp_path, monkeypatch):
    """USD_JPY uses 0.01 pip-size, not 0.0001 — verify the conversion."""
    monkeypatch.setattr(tca, 'AGENT_DB_PATH', str(tmp_path / 'agent' / 'x.db'))
    (tmp_path / 'agent').mkdir()

    df = _summary_df([
        {'pair': 'USD_JPY', 'avg_spread_pips': 1.2, 'n_fills': 40},
    ])
    with patch.object(tca, 'per_pair_summary', return_value=df):
        out = tca.update_measured_spreads_from_live(min_fills=20)

    # 1.2 pips × 0.01 (JPY pip) = 0.012 in price units
    assert abs(out['USD_JPY'] - 1.2 * 0.01) < 1e-9


def test_update_measured_spreads_empty_summary_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(tca, 'AGENT_DB_PATH', str(tmp_path / 'agent' / 'x.db'))
    (tmp_path / 'agent').mkdir()

    with patch.object(tca, 'per_pair_summary', return_value=pd.DataFrame()):
        out = tca.update_measured_spreads_from_live()
    assert out == {}
    # No file should be written when there's nothing to write
    assert not (tmp_path / 'live_measured_spreads.json').exists()
