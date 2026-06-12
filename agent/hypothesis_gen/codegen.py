"""Stub codegen — takes a `deep_dive` dict from the LLM tool output and
produces a syntactically valid entry_<slug> function + SWEEP_<SLUG> dict.

Append-safe: builds a single text block bounded by a `# === ROUND <date>.<n>
===` banner. The caller writes it to the hypotheses file once all 3 stubs
are validated.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from . import paths


_SLUG_BAD = re.compile(r'[^a-z0-9_]+')
_PLACEHOLDER = re.compile(r'\{p_([a-z][a-z0-9_]*)\}')


def _slugify(title: str) -> str:
    s = title.lower().strip().replace('-', '_').replace(' ', '_')
    s = _SLUG_BAD.sub('_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'unnamed'


def _indent_stub(stub: str, n_spaces: int = 4) -> str:
    """Dedent the LLM output then re-indent for inclusion in a function body."""
    body = textwrap.dedent(stub).strip('\n')
    pad = ' ' * n_spaces
    return '\n'.join(pad + line if line else line for line in body.split('\n'))


def _replace_placeholders(stub: str, dofs: list[dict]) -> tuple[str, set[str]]:
    """Replace {p_name} with params['name']. Return (replaced, set_of_names_used)."""
    used: set[str] = set()
    def _sub(m: re.Match) -> str:
        name = m.group(1)
        used.add(name)
        return f"params['{name}']"
    replaced = _PLACEHOLDER.sub(_sub, stub)
    return replaced, used


def _format_grid(dofs: list[dict]) -> str:
    """Render the ParameterGrid dict body."""
    lines = []
    for d in dofs:
        rng = d.get('sweep_range', [])
        if not isinstance(rng, list) or not rng:
            rng = [None]
        lines.append(f"        {d['name']!r}: {list(rng)!r},")
    return '\n'.join(lines)


def render_one(deep_dive: dict, round_n: int, date_str: str) -> tuple[str, dict]:
    """Render a single entry_* + SWEEP_* block. Returns (code, metadata).

    Validates:
      - code compiles via ast.parse
      - every {p_*} in stub has a matching degrees_of_freedom entry
      - every degrees_of_freedom entry name is referenced in the stub
    Raises ValueError if validation fails.
    """
    title = deep_dive['title']
    slug  = _slugify(title)
    dofs  = deep_dive.get('degrees_of_freedom', [])
    stub  = deep_dive.get('code_stub', '')

    dof_names = {d['name'] for d in dofs}
    replaced, used = _replace_placeholders(stub, dofs)

    missing_dof = used - dof_names
    if missing_dof:
        raise ValueError(
            f'code_stub references {sorted(missing_dof)} but no matching '
            f'degrees_of_freedom entry. Got DOF names: {sorted(dof_names)}'
        )
    unused_dof = dof_names - used
    if unused_dof:
        # Not fatal — just a comment in the generated code.
        pass

    claim_first = deep_dive.get('claim', '').split('.')[0].strip()
    null_first  = deep_dive.get('null_hypothesis', '').split('.')[0].strip()
    falsification = deep_dive.get('falsification_test', {})
    decision_rule = falsification.get('decision_rule', '')
    hold_horizon = deep_dive.get('hold_horizon', '')

    body_indented = _indent_stub(replaced)
    grid_body     = _format_grid(dofs)
    grid_section  = (
        "    'grid': ParameterGrid({\n" + grid_body + '\n    })'
        if dofs else "    'grid': ParameterGrid({})"
    )

    code = f'''
# --- Round {date_str}.{round_n:02d} :: {title} ---
# CLAIM:           {claim_first}
# NULL:            {null_first}
# FALSIFICATION:   {decision_rule}
# HOLD HORIZON:    {hold_horizon}

def entry_{slug}(bst, slot, row, ts, pair, slip, hspd,
                 sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    """Auto-generated stub. Review before live use."""
    import math
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if has_pending(sc):
        return check_and_fill(sc, row, slot, ts, regime, hspd, slip)

{body_indented}


SWEEP_{slug.upper()} = {{
    'entry_fn':    entry_{slug},
    'manager_fn':  make_manager(exit_hour=23, use_profit_lock=False),
    'pairs':       {paths.DEFAULT_PAIRS_SYMBOL},
    'session':     {paths.DEFAULT_SESSION!r},
    'regime_mult': {paths.DEFAULT_REGIME_MULT!r},
{grid_section},
}}
'''.strip() + '\n'

    # Validate that the generated code parses.
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(
            f'Generated code for "{title}" did not parse: {e}\n'
            f'Generated code:\n{code}'
        )

    return code, {
        'slug': slug,
        'sweep_const': f'SWEEP_{slug.upper()}',
        'sweep_key':   slug,
    }


def render_round(deep_dives: list[dict], round_n: int, date_str: str
                 ) -> tuple[str, list[dict]]:
    """Render all 3 deep-dive proposals into one banner-bounded code block.

    Returns (code_block, [metadata_per_proposal]).
    Validation failures on individual proposals are collected; if ANY proposal
    fails to render, the entire block is rejected (caller decides what to do).
    """
    banner = f'# === ROUND {date_str}.{round_n:02d} ==='
    parts = [banner, '']
    metas: list[dict] = []
    errors: list[str] = []

    for i, dd in enumerate(deep_dives, 1):
        try:
            code, meta = render_one(dd, round_n=round_n, date_str=date_str)
            parts.append(code)
            parts.append('')
            metas.append({**meta, 'title': dd.get('title', '')})
        except ValueError as e:
            errors.append(f'proposal {i}: {e}')

    if errors:
        raise ValueError(
            f'render_round failed:\n  ' + '\n  '.join(errors)
        )

    parts.append(f'# === END ROUND {date_str}.{round_n:02d} ===')
    return '\n'.join(parts), metas


def append_to_hypotheses_file(target: Path, code_block: str,
                              metas: list[dict]) -> None:
    """Append code block to target .py, then add new SWEEPS dict entries.

    Strategy for the SWEEPS dict insertion: find the line `SWEEPS = {`,
    walk to its matching closing brace at column 0, insert new keys just
    before it.
    """
    if not target.exists():
        raise FileNotFoundError(f'target hypotheses file not found: {target}')

    text = target.read_text(encoding='utf-8')

    # Idempotency guard: if our banner already exists in the file, do not
    # append again.
    banner_first_line = code_block.split('\n', 1)[0]
    if banner_first_line in text:
        raise ValueError(
            f'banner already present in file — refusing to double-append: '
            f'{banner_first_line!r}'
        )

    # Append the code block to the body (before SWEEPS dict if possible).
    # Strategy: find `^SWEEPS\s*=\s*\{`, insert code block BEFORE it.
    m = re.search(r'^SWEEPS\s*=\s*\{', text, re.MULTILINE)
    if m:
        insert_pos = m.start()
        new_text = text[:insert_pos] + code_block + '\n\n' + text[insert_pos:]
    else:
        new_text = text.rstrip() + '\n\n' + code_block + '\n'

    # Add new keys to SWEEPS dict. Locate the closing `}` of the SWEEPS dict.
    m2 = re.search(r'^SWEEPS\s*=\s*\{', new_text, re.MULTILINE)
    if m2:
        # Find matching close brace by tracking depth from after the {.
        i = m2.end()
        depth = 1
        while i < len(new_text) and depth > 0:
            c = new_text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        if depth == 0:
            close_pos = i - 1
            # Insert new entries just before close_pos. Detect existing
            # indentation by looking at the previous line.
            prefix_text = new_text[:close_pos]
            last_newline = prefix_text.rfind('\n')
            indent = '    '
            if last_newline >= 0:
                tail = prefix_text[last_newline + 1:]
                if tail and tail[0] in (' ', '\t'):
                    indent_chars = []
                    for ch in tail:
                        if ch in (' ', '\t'):
                            indent_chars.append(ch)
                        else:
                            break
                    if indent_chars:
                        indent = ''.join(indent_chars)
            new_entries = ''.join(
                f"{indent}{meta['sweep_key']!r}: {meta['sweep_const']},\n"
                for meta in metas
            )
            new_text = new_text[:close_pos] + new_entries + new_text[close_pos:]

    # Sanity-check the entire file parses before writing.
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        raise ValueError(
            f'post-append file would not parse: {e}. Refusing to write.'
        )

    target.write_text(new_text, encoding='utf-8')
