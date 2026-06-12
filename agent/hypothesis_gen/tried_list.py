"""Tried-list generator — queries edge_results.db and parses the hypotheses
Python file for thesis comments.

Output: a markdown document listing every sweep family already tested,
grouped by category, with per-family stats and a one-line thesis (extracted
from comments in the hypotheses .py file when available, or just the
family name when not).
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

from . import paths, config


_SWEEP_DEF_RE = re.compile(r'^SWEEP_([A-Z0-9_]+)\s*=', re.MULTILINE)
_THESIS_LINE_RE = re.compile(r'^# (.+)$')


def _parse_theses_from_py(py_path: Path) -> dict[str, str]:
    """Walk the hypotheses .py file. For each `SWEEP_NAME = {` block, find
    the immediately-preceding header comment block as the thesis."""
    out: dict[str, str] = {}
    if not py_path.exists():
        return out
    try:
        lines = py_path.read_text(encoding='utf-8').splitlines()
    except Exception:
        return out

    for i, line in enumerate(lines):
        m = _SWEEP_DEF_RE.match(line.strip())
        if not m:
            continue
        slug = m.group(1).lower()
        # Walk backwards collecting consecutive comment lines (after skipping
        # blank lines and section divider lines like "# === ... ===").
        thesis_lines: list[str] = []
        j = i - 1
        while j >= 0:
            ln = lines[j].strip()
            if not ln:
                if thesis_lines:
                    break
                j -= 1
                continue
            if ln.startswith('# ===') or ln.startswith('# ─'):
                if thesis_lines:
                    break
                j -= 1
                continue
            if ln.startswith('#'):
                stripped = ln.lstrip('#').strip()
                if stripped:
                    thesis_lines.insert(0, stripped)
                j -= 1
                continue
            break
        if thesis_lines:
            out[slug] = ' '.join(thesis_lines)[:300]

    return out


def _family_from_sweep_id(sweep_id: str) -> str:
    """Strip trailing _YYYYMMDD_HHMMSS from sweep_id to get the family name."""
    # Pattern: name_YYYYMMDD_HHMMSS (date+time at end, 8 digits then _ then 6 digits)
    return re.sub(r'_\d{8}_\d{6}$', '', sweep_id)


def _query_families(db_path: Path) -> list[dict]:
    """Group hypotheses by extracted family name. Return per-family stats."""
    if not db_path.exists():
        return []
    try:
        import pandas as pd
        con = sqlite3.connect(str(db_path))
        df = pd.read_sql_query("""
            SELECT sweep_id, test_n, test_sharpe, verdict
            FROM hypotheses
            WHERE sweep_id IS NOT NULL
        """, con)
        con.close()
    except Exception:
        return []
    if df.empty:
        return []
    df['family'] = df['sweep_id'].astype(str).map(_family_from_sweep_id)
    grouped = df.groupby('family').agg(
        n=('sweep_id', 'count'),
        med_sharpe=('test_sharpe', 'median'),
        med_n=('test_n', 'median'),
        n_viable=('verdict', lambda v: (v == 'VIABLE').sum()),
    ).reset_index().sort_values('n', ascending=False)
    return grouped.to_dict(orient='records')


def _query_agent_strategies(agent_db: Path, limit: int = 200) -> list[dict]:
    """Pull agent-generated theses from agent_results.db if present."""
    if not agent_db.exists():
        return []
    try:
        con = sqlite3.connect(str(agent_db))
        try:
            rows = con.execute("""
                SELECT strategy_name, rationale, behaviour_type, created_at
                FROM tested_strategies
                WHERE rationale IS NOT NULL AND TRIM(rationale) != ''
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        except sqlite3.Error:
            rows = []
        con.close()
        return [{'name': r[0], 'rationale': r[1],
                 'behaviour': r[2], 'created_at': r[3]} for r in rows]
    except Exception:
        return []


def generate(out_path: Optional[Path] = None) -> str:
    """Build tried.md content."""
    families = _query_families(paths.EDGE_DB)
    theses   = _parse_theses_from_py(paths.HYPOTHESES_PY)
    agent_strategies = _query_agent_strategies(paths.AGENT_DB)

    if len(families) > config.TRIED_LIST_MAX_FAMILIES:
        families = families[:config.TRIED_LIST_MAX_FAMILIES]

    n_total = sum(f['n'] for f in families)
    n_viable = sum(f['n_viable'] for f in families)

    body = [
        '# Tried list — strategies already tested (DO NOT PROPOSE THESE)',
        '',
        f'**Hand-coded sweep families: {len(families)}. Total backtested '
        f'hypotheses: {n_total}. VIABLE: {n_viable}.**',
        '',
        ('Each row below has been tested through the full gauntlet. Treat '
         'this list as a forbidden zone. Anything appearing here, or with '
         '>50% conceptual overlap, is off-limits. Propose orthogonal '
         'directions.'),
        '',
        '## Hand-coded sweep families',
        '',
        '| family | n | median sharpe | median trades | viable | thesis |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for f in families:
        med_sh = f['med_sharpe']
        med_sh_str = f'{float(med_sh):.2f}' if med_sh is not None and med_sh == med_sh else '—'
        med_n = f['med_n']
        med_n_str = f'{int(med_n)}' if med_n is not None and med_n == med_n else '—'
        thesis = theses.get(f['family'].lower(), '(no comment)')
        body.append(
            f"| `{f['family']}` | {f['n']} | {med_sh_str} | {med_n_str} | "
            f"{f['n_viable']} | {thesis} |"
        )
    body.append('')

    if agent_strategies:
        body.append('## Agent-generated strategies (rationales)')
        body.append('')
        body.append('Earlier LLM-generated proposals — these themes are also '
                    'exhausted. Avoid repeating their core ideas.')
        body.append('')
        for s in agent_strategies[:80]:
            r = (s['rationale'] or '').strip().replace('\n', ' ')
            if len(r) > 250:
                r = r[:247] + '…'
            body.append(f"- **{s['name']}** ({s.get('behaviour') or '?'}): {r}")
        body.append('')

    body.append('---')
    body.append('Reminder: a proposal that just reshuffles parameters '
                'within a family above is NOT novel. Novel = different '
                'mechanism, different feature category, different horizon, '
                'or different event anchor.')

    out = '\n'.join(body)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding='utf-8')
    return out
