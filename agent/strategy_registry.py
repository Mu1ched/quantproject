"""Single source of truth for strategies tradable by both the backtester
and the live MT5 path.

Discovers two sources at startup:
  1. SWEEPS dict in edge_hypotheses.py (the manually-curated baseline)
  2. agent/generated/entry_*.py (Claude-generated survivors)

Each registry entry exposes the same shape:
  {
    'name':        str,
    'entry_fn':    callable(bst, slot, row, ts, pair, slip, hspd, sess_cfg,
                            regime, regime_mult, fvg_buf=None, day_sweep=None),
    'manager_fn':  callable | None,   # None for generated strategies (they reuse the default manager)
    'pairs':       list[str],
    'session':     'london' | 'ny' | 'asian',
    'regime_mult': dict[str, float],
    'source':      'sweep' | 'generated',
  }
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_REGISTRY: dict[str, dict] | None = None


def _load_sweeps_registry() -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        import edge_hypotheses
    except Exception as e:
        log.warning(f"strategy_registry: edge_hypotheses import failed ({e})")
        return out

    sweeps = getattr(edge_hypotheses, 'SWEEPS', {}) or {}
    for sweep_id, sweep in sweeps.items():
        entry_fn   = sweep.get('entry_fn')
        if entry_fn is None:
            continue
        out[sweep_id] = {
            'name':        sweep_id,
            'entry_fn':    entry_fn,
            'manager_fn':  sweep.get('manager_fn'),
            'pairs':       list(sweep.get('pairs', [])),
            'session':     sweep.get('session', 'ny'),
            'regime_mult': dict(sweep.get('regime_mult', {})),
            'grid':        sweep.get('grid'),
            'source':      'sweep',
        }
    return out


def _load_generated_registry() -> dict[str, dict]:
    out: dict[str, dict] = {}
    gen_dir = Path(__file__).parent / 'generated'
    if not gen_dir.exists():
        return out
    for path in sorted(gen_dir.glob('entry_*.py')):
        modname = f"agent.generated.{path.stem}"
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            log.warning(f"strategy_registry: failed to import {modname}: {e}")
            continue
        # Convention: each generated module exposes one entry function whose
        # name matches the file stem.
        entry_fn = getattr(mod, path.stem, None)
        if entry_fn is None:
            continue
        # Generated files don't currently embed pair/session metadata in code,
        # so default to ALL_PAIRS gated by the regime; the meta is recovered
        # from agent_results.db at registration time by the live router.
        try:
            from edge_engine import ALL_PAIRS, DEFAULT_REGIME_MULT  # type: ignore
        except Exception:
            ALL_PAIRS = []
            DEFAULT_REGIME_MULT = {}
        # Filename convention: entry_<session>_<descriptor>.py
        parts   = path.stem.split('_', 2)
        session = parts[1] if len(parts) > 1 and parts[1] in ('ny', 'london', 'asian') else 'ny'
        out[path.stem] = {
            'name':        path.stem,
            'entry_fn':    entry_fn,
            'manager_fn':  None,
            'pairs':       list(ALL_PAIRS),
            'session':     session,
            'regime_mult': dict(DEFAULT_REGIME_MULT),
            'source':      'generated',
        }
    return out


def load_registry(force_reload: bool = False) -> dict[str, dict]:
    """Return the full strategy registry, building it on first call.

    Pass force_reload=True after a new strategy file has been written.
    """
    global _REGISTRY
    if _REGISTRY is not None and not force_reload:
        return _REGISTRY
    out: dict[str, dict] = {}
    out.update(_load_sweeps_registry())
    out.update(_load_generated_registry())
    _REGISTRY = out
    log.info(f"strategy_registry: loaded {len(out)} strategies "
             f"({sum(1 for s in out.values() if s['source']=='sweep')} sweep, "
             f"{sum(1 for s in out.values() if s['source']=='generated')} generated)")
    return out


def get_strategy(name: str) -> dict | None:
    return load_registry().get(name)


def strategies_for(pair: str, session: str | None = None) -> list[dict]:
    """All strategies that include the given pair (and optionally session)."""
    out = []
    for s in load_registry().values():
        if pair not in s['pairs']:
            continue
        if session is not None and s['session'] != session:
            continue
        out.append(s)
    return out
