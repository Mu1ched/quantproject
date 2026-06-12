"""
Persistent GUI configuration shared between edge_gui.py (writer) and the
agent loop / Mine tab (readers). Single JSON file at the project root —
gitignored so user selections don't leak into commits.

Empty / missing file = no override, callers fall back to today's hardcoded
defaults. This module is intentionally tiny: load(), save(), plus two
convenience accessors so callers never hand-code field names.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "gui_config.json"

_VALID_KEYS = {
    "selected_pairs", "selected_sessions", "miner_overrides",
    "cost_mult", "hypotheses_per_batch", "backtest_workers",
    "budget_daily_usd", "budget_total_usd",
    "loop_sleep_seconds", "force_redownload",
    "min_test_sharpe", "min_dsr", "max_test_drawdown",
    "min_test_trades",
    "updated_at",
}


def load() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        log.warning("[gui_config] failed to read %s: %s", CONFIG_PATH, e)
        return {}


def save(cfg: dict) -> None:
    payload = {k: v for k, v in cfg.items() if k in _VALID_KEYS}
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    fd, tmp = tempfile.mkstemp(prefix=".gui_config_", suffix=".json",
                               dir=str(CONFIG_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def reset() -> None:
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()


def selected_pairs() -> list[str] | None:
    v = load().get("selected_pairs")
    return list(v) if v else None


def selected_sessions() -> list[str] | None:
    v = load().get("selected_sessions")
    return list(v) if v else None


def miner_overrides() -> dict:
    v = load().get("miner_overrides")
    return dict(v) if isinstance(v, dict) else {}
