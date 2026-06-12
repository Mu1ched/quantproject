"""One-round driver: build context → API call → validate → codegen → append.

Failures in any step return a RoundResult with `status != 'ok'` and skip
the append. The driver continues to the next round.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import (
    codegen, config, constraints_doc, features_doc,
    paths, prompt, rejection_digest, tried_list,
)


log = logging.getLogger(__name__)


@dataclass
class RoundResult:
    round_n:   int
    round_dir: Path
    status:    str                  # 'ok' | 'api_error' | 'parse_error' | 'codegen_error' | 'dry_run'
    cost_usd:  float = 0.0
    parsed:    Optional[dict] = None
    metas:     list[dict] = field(default_factory=list)
    error:     Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _compute_cost(usage: Any, model: str) -> float:
    table = config.COST_TABLE.get(model)
    if not table:
        return 0.0
    in_tok    = getattr(usage, 'input_tokens', 0) or 0
    out_tok   = getattr(usage, 'output_tokens', 0) or 0
    cache_rd  = getattr(usage, 'cache_read_input_tokens', 0) or 0
    cache_wr  = getattr(usage, 'cache_creation_input_tokens', 0) or 0
    # Anthropic's `input_tokens` excludes cache reads/writes; price each separately.
    return (
        in_tok    * table['input']       / 1_000_000.0 +
        out_tok   * table['output']      / 1_000_000.0 +
        cache_rd  * table['cache_read']  / 1_000_000.0 +
        cache_wr  * table['cache_write'] / 1_000_000.0
    )


def _api_call_with_retry(client: Any, request: dict) -> Any:
    """Retry on rate limit / connection / 5xx. Re-raise on auth / 4xx."""
    import anthropic
    last_err: Optional[Exception] = None
    for attempt in range(1, config.RETRY_MAX_ATTEMPTS + 1):
        try:
            return client.messages.create(**request)
        except anthropic.RateLimitError as e:
            last_err = e
            wait = config.RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            log.warning('rate limit (attempt %d/%d) — sleeping %ds',
                        attempt, config.RETRY_MAX_ATTEMPTS, wait)
            time.sleep(wait)
        except anthropic.APIConnectionError as e:
            last_err = e
            wait = config.RETRY_BASE_DELAY_S
            log.warning('connection error (attempt %d/%d) — sleeping %ds',
                        attempt, config.RETRY_MAX_ATTEMPTS, wait)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            last_err = e
            if e.status_code and e.status_code >= 500:
                wait = config.RETRY_BASE_DELAY_S
                log.warning('5xx error (attempt %d/%d) — sleeping %ds',
                            attempt, config.RETRY_MAX_ATTEMPTS, wait)
                time.sleep(wait)
            else:
                raise
    assert last_err is not None
    raise last_err


def _extract_tool_input(response: Any, tool_name: str) -> dict:
    for block in getattr(response, 'content', []):
        if getattr(block, 'type', None) == 'tool_use' and getattr(block, 'name', None) == tool_name:
            inp = getattr(block, 'input', None) or {}
            if isinstance(inp, dict):
                return inp
    raise ValueError(f'response did not contain a tool_use block named {tool_name!r}')


def run_round(round_n: int, model: str = config.MODEL,
              dry_run: bool = False) -> RoundResult:
    """Execute one round end-to-end."""
    date_str  = date.today().isoformat()
    round_dir = paths.ROUNDS_DIR / f'{date_str}_round_{round_n:02d}'
    round_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build context (markdown strings).
    log.info('Round %d: building context …', round_n)
    tried     = tried_list.generate(round_dir / 'tried.md')
    rejects   = rejection_digest.generate(round_dir / 'rejections.md')
    features  = features_doc.generate(round_dir / 'features.md')
    constr    = constraints_doc.generate(round_dir / 'constraints.md')

    (round_dir / 'context.json').write_text(json.dumps({
        'lens': {'tried': len(tried), 'rejections': len(rejects),
                  'features': len(features), 'constraints': len(constr)},
        'timestamp': _now_iso(),
        'round_n': round_n,
    }, indent=2), encoding='utf-8')

    # Always write the paste-ready prompt for claude.ai manual mode.
    (round_dir / 'prompt.md').write_text(
        prompt.build_manual_prompt(market_name=paths.MARKET_NAME),
        encoding='utf-8',
    )

    # 2. Build API request.
    user_blocks = [
        {'type': 'text', 'text': features,
         'cache_control': {'type': 'ephemeral'}},
        {'type': 'text', 'text': constr,
         'cache_control': {'type': 'ephemeral'}},
        {'type': 'text', 'text': tried},
        {'type': 'text', 'text': rejects},
        {'type': 'text', 'text':
            f'Round {round_n}. Produce the round output as a submit_round '
            f'tool call. Be a harsh critic of your taxonomy. Score most '
            f'ideas 6-9/15.'},
    ]
    request = {
        'model': model,
        'max_tokens': config.MAX_TOKENS_OUTPUT,
        'system': [{
            'type': 'text',
            'text': prompt.SYSTEM_PROMPT.format(market_name=paths.MARKET_NAME),
            'cache_control': {'type': 'ephemeral'},
        }],
        'tools': [prompt.SUBMIT_ROUND_TOOL],
        'tool_choice': {'type': 'tool', 'name': 'submit_round'},
        'messages': [{'role': 'user', 'content': user_blocks}],
    }
    (round_dir / 'request.json').write_text(
        json.dumps(request, indent=2, default=str), encoding='utf-8',
    )

    if dry_run:
        log.info('Round %d: --dry-run: skipping API call.', round_n)
        return RoundResult(
            round_n=round_n, round_dir=round_dir,
            status='dry_run', cost_usd=0.0,
        )

    # 3. API call.
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        return RoundResult(
            round_n=round_n, round_dir=round_dir,
            status='api_error', error=f'anthropic SDK init failed: {e}',
        )

    log.info('Round %d: calling API (model=%s) …', round_n, model)
    try:
        response = _api_call_with_retry(client, request)
    except Exception as e:
        log.exception('Round %d: API call failed', round_n)
        return RoundResult(
            round_n=round_n, round_dir=round_dir,
            status='api_error', error=str(e),
        )

    try:
        (round_dir / 'response.json').write_text(
            response.model_dump_json(indent=2), encoding='utf-8',
        )
    except Exception:
        (round_dir / 'response.json').write_text(
            json.dumps(str(response), indent=2), encoding='utf-8',
        )

    cost_usd = _compute_cost(getattr(response, 'usage', None), model)

    # 4. Extract + persist tool input.
    try:
        parsed = _extract_tool_input(response, tool_name='submit_round')
    except Exception as e:
        return RoundResult(
            round_n=round_n, round_dir=round_dir,
            status='parse_error', cost_usd=cost_usd, error=str(e),
        )
    (round_dir / 'parsed.json').write_text(
        json.dumps(parsed, indent=2, default=str), encoding='utf-8',
    )

    # 5. Codegen.
    try:
        code_block, metas = codegen.render_round(
            parsed.get('deep_dives', []), round_n=round_n, date_str=date_str,
        )
    except Exception as e:
        log.exception('Round %d: codegen failed', round_n)
        return RoundResult(
            round_n=round_n, round_dir=round_dir,
            status='codegen_error', cost_usd=cost_usd, parsed=parsed,
            error=str(e),
        )
    (round_dir / 'generated.py').write_text(code_block, encoding='utf-8')

    # 6. Append to hypotheses file.
    try:
        codegen.append_to_hypotheses_file(
            paths.HYPOTHESES_PY, code_block, metas,
        )
    except Exception as e:
        log.exception('Round %d: append failed', round_n)
        return RoundResult(
            round_n=round_n, round_dir=round_dir,
            status='codegen_error', cost_usd=cost_usd, parsed=parsed,
            error=f'append failed: {e}',
        )

    # 7. Persist stats.
    usage = getattr(response, 'usage', None)
    stats = {
        'round_n': round_n,
        'timestamp': _now_iso(),
        'model': model,
        'cost_usd': cost_usd,
        'input_tokens':       getattr(usage, 'input_tokens', None),
        'output_tokens':      getattr(usage, 'output_tokens', None),
        'cache_read_tokens':  getattr(usage, 'cache_read_input_tokens', None),
        'cache_write_tokens': getattr(usage, 'cache_creation_input_tokens', None),
        'accepted_strategies': [m['title'] for m in metas],
        'final_pick': parsed.get('final_pick'),
        'status': 'ok',
    }
    (round_dir / 'stats.json').write_text(json.dumps(stats, indent=2),
                                          encoding='utf-8')

    log.info('Round %d: OK. cost=$%.3f, accepted=%d',
             round_n, cost_usd, len(metas))
    return RoundResult(
        round_n=round_n, round_dir=round_dir,
        status='ok', cost_usd=cost_usd, parsed=parsed, metas=metas,
    )
