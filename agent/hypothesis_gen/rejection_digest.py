"""Rejection digest — aggregates failure modes across the existing
hypotheses table by calling `agent.scorer.rejection_reason()` on each row.

This is the empirical "what kills strategies in our gauntlet" signal the
LLM consumes. Critical that bucketing matches the LIVE gauntlet exactly —
hence the direct import of scorer.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

from . import paths, config


# Bucket the freeform rejection_reason strings into category keys.
_BUCKETS: list[tuple[str, re.Pattern]] = [
    ('TOO_FEW_TRADES',     re.compile(r'too few trades', re.I)),
    ('BH_NOT_SIG',         re.compile(r'BH correction', re.I)),
    ('NOT_REGIME_STABLE',  re.compile(r'regime-stable', re.I)),
    ('LOW_SHARPE',         re.compile(r'test Sharpe', re.I)),
    ('LOW_DSR',            re.compile(r'DSR too low', re.I)),
    ('TRAIN_TEST_DECAY',   re.compile(r'train→test decay|decay', re.I)),
    ('HIGH_DRAWDOWN',      re.compile(r'drawdown too large', re.I)),
    ('CI_LOW_NEG',         re.compile(r'CI lower bound', re.I)),
    ('PBO_HIGH',           re.compile(r'PBO too high', re.I)),
    ('PSR_LOW',            re.compile(r'PSR too low', re.I)),
    ('WF_LOW',             re.compile(r'WF min Sharpe', re.I)),
    ('MC_PASS_LOW',        re.compile(r'MC eval pass rate', re.I)),
    ('MC_BLOWN_HIGH',      re.compile(r'MC blown pct', re.I)),
    ('OTHER',              re.compile(r'.*')),
]


def _bucket(reason: str) -> str:
    for name, pat in _BUCKETS:
        if pat.search(reason or ''):
            return name
    return 'OTHER'


def _family_from_sweep_id(sweep_id: str) -> str:
    return re.sub(r'_\d{8}_\d{6}$', '', sweep_id or '')


def generate(out_path: Optional[Path] = None) -> str:
    """Build rejections.md."""
    db = paths.EDGE_DB
    if not db.exists():
        body = '# Rejections\n\n(no edge_results.db yet — first round.)\n'
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding='utf-8')
        return body

    # Lazy import to avoid hard dependency if scorer somehow breaks.
    try:
        from agent.scorer import rejection_reason
    except Exception as e:
        body = f'# Rejections\n\n(scorer import failed: {e})\n'
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding='utf-8')
        return body

    try:
        import pandas as pd
        con = sqlite3.connect(str(db))
        # Discover which columns this schema actually has (older DBs lack
        # mc_*, wf_*, psr, pbo_score).
        existing_cols = {
            row[1] for row in con.execute(
                "PRAGMA table_info(hypotheses)"
            ).fetchall()
        }
        wanted = [
            'hypothesis_id', 'sweep_id', 'params_json',
            'train_n', 'train_sharpe',
            'test_n', 'test_sharpe', 'test_pnl', 'test_wr', 'test_max_dd',
            'dsr', 'regime_stable', 'bh_sig', 'p_adj',
            'sharpe_ci_low', 'sharpe_ci_high',
            'psr', 'pbo_score',
            'wf_sharpe_min',
            'mc_eval_pass_pct', 'mc_blown_pct',
            'verdict',
        ]
        cols = [c for c in wanted if c in existing_cols]
        if not cols:
            con.close()
            raise sqlite3.Error('hypotheses table has none of the expected columns')
        sql = f"SELECT {', '.join(cols)} FROM hypotheses WHERE test_sharpe IS NOT NULL"
        df = pd.read_sql_query(sql, con)
        con.close()
    except Exception as e:
        body = f'# Rejections\n\n(DB query failed: {e})\n'
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding='utf-8')
        return body

    if df.empty:
        body = '# Rejections\n\n(hypotheses table is empty.)\n'
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding='utf-8')
        return body

    # Bucket each row (skip VIABLE).
    df['family'] = df['sweep_id'].map(_family_from_sweep_id)
    rejected = df[df['verdict'].fillna('') != 'VIABLE'].copy()
    if rejected.empty:
        body = '# Rejections\n\n(no rejected hypotheses found.)\n'
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding='utf-8')
        return body

    reasons = []
    for _, row in rejected.iterrows():
        try:
            r = rejection_reason(row.to_dict())
        except Exception:
            r = 'unknown'
        reasons.append(r)
    rejected = rejected.copy()
    rejected['_reason'] = reasons
    rejected['_bucket'] = rejected['_reason'].map(_bucket)

    # Truncate to most recent N rows if huge.
    if len(rejected) > config.REJECTION_DIGEST_MAX_ROWS:
        rejected = rejected.tail(config.REJECTION_DIGEST_MAX_ROWS)

    n_total = len(rejected)
    bucket_counts = rejected['_bucket'].value_counts()

    body: list[str] = []
    body += [
        '# Rejection digest — why prior strategies died',
        '',
        f'**Aggregate over {n_total} rejected hypotheses.** Use this to '
        'engineer AROUND the dominant failure modes. The top 2 bucket are '
        'where your design margin needs to be — propose strategies that '
        'do not fail in those ways.',
        '',
        '## Failure-mode distribution',
        '',
        '| gate / reason | count | %  |',
        '|---|---:|---:|',
    ]
    for bucket, count in bucket_counts.items():
        pct = 100.0 * count / n_total
        body.append(f'| `{bucket}` | {int(count)} | {pct:.1f}% |')

    # Per-family modal failure.
    body += [
        '',
        '## Per-family modal failure mode',
        '',
        '| family | n_rejected | modal gate | median test_sharpe | median test_n |',
        '|---|---:|---|---:|---:|',
    ]
    for family, sub in rejected.groupby('family'):
        if len(sub) < 3:
            continue
        modal = sub['_bucket'].mode().iloc[0]
        med_sh = sub['test_sharpe'].median()
        med_n  = sub['test_n'].median()
        body.append(
            f'| `{family}` | {len(sub)} | `{modal}` | '
            f"{med_sh:.2f} | {int(med_n) if med_n == med_n else '—'} |"
        )

    # Observation block.
    top_bucket = bucket_counts.index[0] if not bucket_counts.empty else '—'
    top_pct = (100.0 * bucket_counts.iloc[0] / n_total) if not bucket_counts.empty else 0.0
    med_sh = rejected['test_sharpe'].median()
    near_miss = rejected['test_sharpe'].between(0.20, 0.50).sum()

    body += [
        '',
        '## Observations',
        '',
        f"- Top kill gate: **`{top_bucket}`** ({top_pct:.1f}% of rejections). "
        "If a proposal can systematically avoid this failure mode it has a "
        "structural edge in surviving the gauntlet.",
        f"- Median rejected `test_sharpe` is **{med_sh:.2f}**. Aim for "
        "robust 0.6+ Sharpe on test, not marginal.",
        f"- **{near_miss}** rejected strategies had test_sharpe in [0.20, "
        "0.50] — near-miss territory. Many die on FDR or DSR rather than "
        "raw return — a cleaner per-trade thesis is often more valuable "
        "than more trades.",
    ]

    out = '\n'.join(body) + '\n'

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding='utf-8')
    return out
