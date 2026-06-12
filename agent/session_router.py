"""
Determines which trading session/sub-corner to focus on.

Strategy: rotate across both the standard 3 sessions AND 4 less-mined sub-corners
of the retail-quant hypothesis space. This 3× the addressable surface without
changing engine code — every "session" is just a (pair list, time window, thesis)
tuple consumed by the existing pipeline.

Sub-corners added:
  asian_yen           — cross-yen mean reversion in the quietest hours
  post_news_drift     — the 15–90min drift after high-impact news (most bots avoid)
  london_close_handoff — London close → NY open momentum continuation
  friday_squaring     — Friday 14:00–16:00 UTC position squaring effects

The current "live" session (asian/london/ny) is still picked by UTC hour for
strategies that should match live conditions. Sub-corner sessions are layered
in by round number so generation explores them on a rotation.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

_proj = str(Path(__file__).parent.parent)
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from edge_engine import NY_PAIRS, LONDON_PAIRS, ASIAN_PAIRS
from agent.config import DEFAULT_REGIME_MULT

# ── Standard session windows (live-aligned, picked by UTC hour) ──────────────
# (utc_start_inclusive, utc_end_exclusive, session_name, pairs, exit_hour)
_LIVE_SCHEDULE = [
    (0,  8,  'asian',  ASIAN_PAIRS,  8),
    (8,  13, 'london', LONDON_PAIRS, 13),
    (13, 24, 'ny',     NY_PAIRS,     21),
]

# ── Sub-corner sessions (rotated regardless of clock) ────────────────────────
# Each sub-corner is the same shape: (name, pairs, exit_hour). They reuse the
# entire pipeline; the differentiator is the THESIS, surfaced via SUB_CORNER_CONTEXT.
_CROSS_YEN_PAIRS = ['USD_JPY', 'EUR_JPY', 'GBP_JPY']

SUB_CORNERS = [
    ('asian_yen',            _CROSS_YEN_PAIRS,         8),
    ('post_news_drift',      ['EUR_USD', 'GBP_USD', 'XAU_USD'], 21),
    ('london_close_handoff', ['EUR_USD', 'GBP_USD', 'USD_JPY'], 17),
    ('friday_squaring',      ['EUR_USD', 'GBP_USD', 'EUR_JPY'], 17),
    ('risk_proxy',           ['AUD_JPY', 'NZD_USD', 'XAU_USD'], 21),
    ('eur_complex',          ['EUR_USD', 'EUR_GBP', 'EUR_JPY'], 13),
    ('chf_safe_haven',       ['USD_CHF', 'CHF_JPY', 'XAU_USD'], 21),
]

# Round-robin: every Nth round picks a sub-corner instead of the live session.
SUB_CORNER_EVERY_N_ROUNDS = 3   # 1 in 3 rounds explores a sub-corner


def get_current_session(round_n: int = 0) -> tuple:
    """
    Return (session_name, pairs, exit_hour).

    round_n drives sub-corner rotation. When round_n is divisible by
    SUB_CORNER_EVERY_N_ROUNDS, returns one of the sub-corners (round-robined).
    Otherwise returns the live UTC-aligned session.

    GUI override: if gui_config.json declares selected_sessions and/or
    selected_pairs, those take precedence over UTC clock + sub-corner rotation.
    Sessions cycle round-robin by round_n; pairs replace the default pair list
    for the chosen session. Either field alone is honored independently.
    """
    try:
        from agent import gui_config
        sel_sessions = gui_config.selected_sessions()
        sel_pairs    = gui_config.selected_pairs()
    except Exception:
        sel_sessions, sel_pairs = None, None

    if sel_sessions:
        name = sel_sessions[round_n % len(sel_sessions)]
        default_pairs, exit_h = _resolve_session(name)
        return name, (sel_pairs or default_pairs), exit_h

    if round_n and SUB_CORNERS and round_n % SUB_CORNER_EVERY_N_ROUNDS == 0:
        name, pairs, exit_h = SUB_CORNERS[
            (round_n // SUB_CORNER_EVERY_N_ROUNDS - 1) % len(SUB_CORNERS)
        ]
        return name, (sel_pairs or pairs), exit_h

    hour = datetime.now(timezone.utc).hour
    for start, end, name, pairs, exit_h in _LIVE_SCHEDULE:
        if start <= hour < end:
            return name, (sel_pairs or pairs), exit_h
    return 'ny', (sel_pairs or NY_PAIRS), 21


def _resolve_session(name: str) -> tuple[list[str], int]:
    """Look up the default pair list + exit hour for a session name. Falls back
    to NY defaults if the name isn't recognized."""
    for start, end, n, pairs, exit_h in _LIVE_SCHEDULE:
        if n == name:
            return pairs, exit_h
    for n, pairs, exit_h in SUB_CORNERS:
        if n == name:
            return pairs, exit_h
    return NY_PAIRS, 21


def session_regime_mult(session: str) -> dict:
    return dict(DEFAULT_REGIME_MULT)


def session_display_name(session: str) -> str:
    return {
        'asian':                'Asian',
        'london':               'London',
        'ny':                   'New York',
        'asian_yen':            'Asian Cross-Yen',
        'post_news_drift':      'Post-News Drift',
        'london_close_handoff': 'London→NY Handoff',
        'friday_squaring':      'Friday Squaring',
        'risk_proxy':           'Risk Proxy (AUD/NZD/Gold)',
        'eur_complex':          'EUR Complex',
        'chf_safe_haven':       'CHF Safe Haven',
    }.get(session, session.upper())
