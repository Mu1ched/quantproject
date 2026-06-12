"""Manual ingestion path — for the claude.ai web workflow.

User flow:
  1. `python -m agent.hypothesis_gen.driver --max-rounds 1 --dry-run`
     -> builds rounds/<date>_round_NN/ with tried/rejections/features/constraints
        + a paste-ready prompt.md.
  2. User attaches the 4 .md files to a fresh claude.ai chat, pastes prompt.md.
  3. Claude responds with a fenced ```json block.
  4. User saves response into the round folder as:
       - manual_response.json     (pure JSON)
       - manual_response.md       (full response with ```json fence)
       - manual_response.txt      (any format above)
  5. `python -m agent.hypothesis_gen.manual_ingest --round <date>_round_NN`
     -> parses JSON, validates, codegens stubs, appends to hypotheses file,
        writes stats.json with status='ok_manual' and cost_usd=0.

CLI:
  python -m agent.hypothesis_gen.manual_ingest --round <ROUND_NAME>
                                                [--response-file PATH]
                                                [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import codegen, paths


log = logging.getLogger(__name__)


_JSON_FENCE_RE = re.compile(r'```(?:json|JSON)?\s*\n(.*?)\n```', re.DOTALL)


def _extract_json(text: str) -> str:
    """Extract JSON from a markdown ```json fence, or return text as-is."""
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # If text already looks like JSON (starts with {), trust it.
    if text.startswith('{'):
        return text
    raise ValueError('No JSON found: response is not a JSON object and '
                     'contains no ```json fence')


def _validate_basic(parsed: dict) -> list[str]:
    """Lightweight schema check. Returns list of error strings (empty = valid)."""
    errs: list[str] = []
    if not isinstance(parsed, dict):
        return ['response is not a dict']

    tax = parsed.get('taxonomy')
    if not isinstance(tax, list) or len(tax) != 20:
        errs.append(f'taxonomy must have exactly 20 items, '
                    f'got {len(tax) if isinstance(tax, list) else type(tax).__name__}')

    dds = parsed.get('deep_dives')
    if not isinstance(dds, list) or len(dds) != 3:
        errs.append(f'deep_dives must have exactly 3 items, '
                    f'got {len(dds) if isinstance(dds, list) else type(dds).__name__}')
    elif True:
        required = ['taxonomy_id', 'title', 'claim', 'null_hypothesis',
                    'falsification_test', 'degrees_of_freedom',
                    'entry', 'exit', 'hold_horizon', 'sizing',
                    'honest_assessment', 'code_stub']
        for i, dd in enumerate(dds, 1):
            if not isinstance(dd, dict):
                errs.append(f'deep_dive_{i}: not a dict')
                continue
            for field in required:
                if field not in dd:
                    errs.append(f'deep_dive_{i}: missing field {field!r}')

    fp = parsed.get('final_pick')
    if not isinstance(fp, dict) or 'choice' not in fp:
        errs.append('final_pick missing or malformed (no "choice" key)')
    return errs


def _find_response_file(round_dir: Path) -> Optional[Path]:
    """Look for any of the standard manual-response filenames in the round dir."""
    for name in ('manual_response.json', 'manual_response.md',
                 'manual_response.txt'):
        p = round_dir / name
        if p.exists():
            return p
    return None


def _round_number(round_name: str) -> int:
    """Extract NN from '<date>_round_NN'."""
    m = re.search(r'_round_(\d+)$', round_name)
    return int(m.group(1)) if m else 1


def _date_str(round_name: str) -> str:
    m = re.match(r'(\d{4}-\d{2}-\d{2})_', round_name)
    return m.group(1) if m else datetime.now(timezone.utc).date().isoformat()


def manual_ingest(round_name: str, response_file: Optional[Path] = None,
                  dry_run: bool = False) -> dict:
    """Ingest a manual claude.ai response. Returns a stats dict."""
    round_dir = paths.ROUNDS_DIR / round_name
    if not round_dir.exists():
        raise FileNotFoundError(f'round folder not found: {round_dir}')

    if response_file is None:
        response_file = _find_response_file(round_dir)
    if response_file is None:
        raise FileNotFoundError(
            f'no manual_response.{{json,md,txt}} found in {round_dir}. '
            f'Save your claude.ai response there and rerun.'
        )

    text = response_file.read_text(encoding='utf-8')
    log.info('Reading response from %s (%d bytes)', response_file, len(text))

    raw_json = _extract_json(text)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        # Help user diagnose where in the JSON it failed.
        raise ValueError(
            f'JSON parse failed: {e}. Verify the response is wrapped in '
            f'a ```json fence and the JSON is valid.'
        )

    errs = _validate_basic(parsed)
    if errs:
        raise ValueError(
            'Validation errors:\n  ' + '\n  '.join(errs) +
            '\n(Fix the response or re-run claude.ai with a corrected prompt.)'
        )

    # Persist the validated parsed output where the API path would have written it.
    (round_dir / 'parsed.json').write_text(
        json.dumps(parsed, indent=2, default=str), encoding='utf-8',
    )

    round_n = _round_number(round_name)
    date_str = _date_str(round_name)

    code_block, metas = codegen.render_round(
        parsed['deep_dives'], round_n=round_n, date_str=date_str,
    )
    (round_dir / 'generated.py').write_text(code_block, encoding='utf-8')

    if dry_run:
        log.info('--dry-run: skipping append to %s', paths.HYPOTHESES_PY)
        appended = False
    else:
        codegen.append_to_hypotheses_file(
            paths.HYPOTHESES_PY, code_block, metas,
        )
        appended = True

    stats = {
        'round_n': round_n,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': 'claude.ai-manual',
        'cost_usd': 0.0,
        'input_tokens': None,
        'output_tokens': None,
        'cache_read_tokens': None,
        'cache_write_tokens': None,
        'accepted_strategies': [m['title'] for m in metas],
        'final_pick': parsed.get('final_pick'),
        'status': 'ok_manual' if appended else 'dry_run_manual',
        'response_source': str(response_file.name),
    }
    (round_dir / 'stats.json').write_text(json.dumps(stats, indent=2),
                                          encoding='utf-8')
    log.info('manual_ingest OK: %d strategies accepted, appended=%s',
             len(metas), appended)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='agent.hypothesis_gen.manual_ingest')
    parser.add_argument('--round', required=True,
                        help='Round folder name, e.g. 2026-05-21_round_01')
    parser.add_argument('--response-file', type=str, default=None,
                        help='Path to response file (default: auto-detect '
                             'manual_response.* in round dir)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Codegen + validate but do not append to hypotheses file')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s :: %(message)s',
        stream=sys.stdout, force=True,
    )

    try:
        stats = manual_ingest(
            round_name=args.round,
            response_file=Path(args.response_file) if args.response_file else None,
            dry_run=args.dry_run,
        )
    except Exception as e:
        log.error('manual_ingest failed: %s', e)
        return 1

    log.info('Summary:')
    for k, v in stats.items():
        log.info('  %s: %s', k, v)
    return 0


if __name__ == '__main__':
    sys.exit(main())
