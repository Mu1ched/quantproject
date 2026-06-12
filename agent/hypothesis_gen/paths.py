"""Project-root resolution and standard paths (Quantproject / FX flavour)."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

EDGE_DB:         Path = PROJECT_ROOT / 'edge_results.db'
AGENT_DB:        Path = PROJECT_ROOT / 'agent' / 'agent_results.db'
PREPARED_CACHE:  Path = PROJECT_ROOT / 'edge_prepared_cache'
HYPOTHESES_PY:   Path = PROJECT_ROOT / 'edge_hypotheses.py'
ROUNDS_DIR:      Path = PROJECT_ROOT / 'rounds'
RUNTIME_DIR:     Path = PROJECT_ROOT / 'runtime'
HYPGEN_PID:      Path = RUNTIME_DIR / 'hypgen.pid'
HYPGEN_LOG:      Path = RUNTIME_DIR / 'hypgen.log'

THIS_PROJECT_NAME: str = 'fx'
MARKET_NAME:       str = 'FX majors (prop-firm)'

DEFAULT_PAIRS_SYMBOL:  str = 'NY_PAIRS'
DEFAULT_SESSION:       str = 'ny'
DEFAULT_REGIME_MULT:   dict = {
    'TRENDING':      1.0,
    'RANGING':       0.5,
    'TRANSITIONING': 0.5,
    'VOLATILE':      0.0,
    'UNDEFINED':     0.3,
}
