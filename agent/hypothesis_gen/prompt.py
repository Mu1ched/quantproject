"""System prompt + submit_round tool schema.

The tool's input_schema gives Anthropic's API a strict contract on output
structure — the response is parsed as JSON, not regex. This kills entire
categories of parsing bugs.
"""
from __future__ import annotations


SYSTEM_PROMPT = """\
You are a quantitative research collaborator working with a {market_name}
trading system. The user has tested ~1,800 strategy hypotheses against a
rigorous gauntlet (Benjamini-Hochberg FDR, Deflated Sharpe, Monte Carlo
prop-firm simulation, walk-forward, regime stability). Zero passed.

The data is clean. The gauntlet is well-calibrated. The pattern of
0/1,800 is consistent with hypothesis generation that has noise-mined
from a feature menu — proposals grounded in trading-textbook clichés
rather than this market's actual structure.

Your job is to break that pattern. Every proposal must be:

1. NEGATIVE CONSTRAINT — not in or conceptually adjacent to the
   tried-list provided. Anything in that list, or with >50% conceptual
   overlap, is off-limits.

2. FAILURE-MODE AWARE — engineer around the dominant rejection modes
   in the rejection digest. If most strategies die on TOO_FEW_TRADES,
   propose ones that fire ≥3×/week. If LOW_SHARPE, propose ones with
   stronger per-trade expectancy. If MC_PASS_LOW, propose ones with
   positive expectancy under bootstrap-resampled equity paths.

3. MECHANISM-GROUNDED — every idea must cite a specific named
   mechanism: academic paper, microstructure effect, event anchor,
   no-arb residual, behavioural bias. Not "I think momentum might
   work". Acceptable forms:
   - "Lo-MacKinlay variance ratio (1988)"
   - "funding-basis no-arb residual"
   - "pre-CPI 30min positioning drift"
   - "Hawkes self-excitation in liquidation cascades"

4. FORMAL CLAIM — every deep-dive must state a conditional
   probabilistic claim, the null hypothesis, and a single statistical
   test that distinguishes them. The math should match the mechanism,
   not exceed it. Decoration is not rigour.

Your output is a single tool call: submit_round. You produce:
- 20-item taxonomy with self-scoring (novelty / plausibility / testability)
- 3 deep-dive proposals (the top-3 by total score)
- Honest final-pick statement

Prop firm context: 6% profit target, 6% max drawdown, **NO time limit**.
Strategies can hold for hours, days, or weeks. The system tests with
realistic costs (spreads + commission + slippage).

Be a harsh critic of your own taxonomy. Score most ideas 6-9/15. Only
genuinely strong proposals should hit 13-15.

In the code_stub field, use exact `row.<feature>` column names from the
features doc. Use `{{p_<name>}}` placeholders for parameters (note: SINGLE
braces around `p_...`, not double — these are template tokens, not Python
format spec). Each placeholder must correspond to an entry in
degrees_of_freedom with a matching name (e.g. `{{p_funding_z}}` <->
degrees_of_freedom entry named `funding_z`).
"""


def build_manual_prompt(market_name: str) -> str:
    """Build a paste-ready prompt for claude.ai (web) instead of the API.

    The user attaches tried.md / rejections.md / features.md / constraints.md
    to a fresh claude.ai conversation, then pastes this prompt. Claude responds
    with a fenced ``json block that manual_ingest.py parses.
    """
    system = SYSTEM_PROMPT.format(market_name=market_name)
    body = '''
---

> **Override:** Where the instructions above mention "submit_round tool call",
> ignore that — this is the claude.ai web workflow. Output a single fenced
> ```json``` block matching the schema below, with no surrounding prose.

---

## Attached files (must be in this conversation)

- `tried.md`       -- every strategy family already tested (forbidden zone)
- `rejections.md`  -- aggregate failure modes from the gauntlet
- `features.md`    -- columns in the prepared M5 cache (use exact names)
- `constraints.md` -- prop-firm rules + gauntlet thresholds

Read all four before answering.

---

## Output format -- IMPORTANT

Respond with ONE fenced JSON code block matching the schema below. NO prose
before or after the block. The user's pipeline parses this JSON to codegen
strategy stubs; malformed JSON fails the round.

```json
{
  "taxonomy": [
    {"id": "t01", "thesis": "...", "mechanism": "...",
     "novelty": 1, "plausibility": 1, "testability": 1},
    ... exactly 20 items, ids t01..t20 ...
  ],
  "deep_dives": [
    {
      "taxonomy_id": "t01",
      "title": "snake_case_friendly_title",
      "claim": "Conditional on [feature state F], r_{t+h} has [property P] of [magnitude M], because [mechanism].",
      "null_hypothesis": "E[r_{t+h} | F] = E[r_{t+h}] unconditional",
      "falsification_test": {
        "statistic": "two-sample t-test of conditional mean",
        "threshold": "|t| > 2.5",
        "min_n": 30,
        "decision_rule": "reject NULL if t > threshold after BH FDR at alpha=0.05"
      },
      "degrees_of_freedom": [
        {"name": "snake_case_param_name",
         "justification": "why this parameter is needed",
         "sweep_range": [n1, n2, n3]},
        ... 1 to 6 items ...
      ],
      "entry": ["row.column condition", "another condition", ...],
      "exit": {
        "sl": "1.5 * atr below entry",
        "tp": "3.0 * atr above entry",
        "time_based": "exit after 24 hours",
        "trailing": "breakeven move at +1R"
      },
      "hold_horizon": "24 hours",
      "sizing": "1% risk per trade, regime_mult applied",
      "honest_assessment": "which constraint does most work, where it borderline cheats, likely failure mode",
      "code_stub": "pseudocode using exact row.<col> names and {p_<name>} placeholders matching degrees_of_freedom"
    },
    ... exactly 3 items, the top 3 by total taxonomy score ...
  ],
  "final_pick": {
    "choice": "deep_dive_1",
    "reason": "honest one-paragraph rationale; \\"none\\" is acceptable"
  }
}
```

Hard rules on the JSON:
- `taxonomy`: exactly 20 items, ids `t01` through `t20`, all scores integers 1-5
- `deep_dives`: exactly 3 items
- Every `{p_<name>}` placeholder in `code_stub` MUST have a matching
  `degrees_of_freedom` entry with the same `name`
- `entry` conditions must reference real columns from `features.md` as
  `row.<column>` (e.g. `row.funding_z`, `row.atr_rank`)
- `final_pick.choice` is one of `deep_dive_1` / `deep_dive_2` / `deep_dive_3` / `none`
- Wrap your output in a SINGLE ```json ... ``` fence. No surrounding prose.
'''
    return system + body


SUBMIT_ROUND_TOOL = {
    'name': 'submit_round',
    'description': (
        'Submit the round output: 20-item taxonomy + 3 deep-dive proposals '
        '+ final pick. Call this tool exactly once.'
    ),
    'input_schema': {
        'type': 'object',
        'required': ['taxonomy', 'deep_dives', 'final_pick'],
        'properties': {
            'taxonomy': {
                'type': 'array',
                'minItems': 20,
                'maxItems': 20,
                'description': 'Exactly 20 candidate strategy theses with self-scoring.',
                'items': {
                    'type': 'object',
                    'required': ['id', 'thesis', 'mechanism',
                                 'novelty', 'plausibility', 'testability'],
                    'properties': {
                        'id': {
                            'type': 'string',
                            'pattern': '^t[0-9]{2}$',
                            'description': 't01..t20',
                        },
                        'thesis': {
                            'type': 'string',
                            'minLength': 20,
                            'maxLength': 280,
                            'description': 'One-line claim, ≤25 words ideally.',
                        },
                        'mechanism': {
                            'type': 'string',
                            'minLength': 5,
                            'description': 'Citable phrase or short reference.',
                        },
                        'novelty': {
                            'type': 'integer', 'minimum': 1, 'maximum': 5,
                            'description': 'Orthogonality to tried-list. 1=overlap, 5=truly new.',
                        },
                        'plausibility': {
                            'type': 'integer', 'minimum': 1, 'maximum': 5,
                            'description': 'Does the cited mechanism exist? 1=speculative, 5=robust.',
                        },
                        'testability': {
                            'type': 'integer', 'minimum': 1, 'maximum': 5,
                            'description': 'Implementable with available features?',
                        },
                    },
                },
            },
            'deep_dives': {
                'type': 'array',
                'minItems': 3,
                'maxItems': 3,
                'description': 'The 3 highest-scoring taxonomy items, fully spelled out.',
                'items': {
                    'type': 'object',
                    'required': [
                        'taxonomy_id', 'title', 'claim', 'null_hypothesis',
                        'falsification_test', 'degrees_of_freedom',
                        'entry', 'exit', 'hold_horizon', 'sizing',
                        'honest_assessment', 'code_stub',
                    ],
                    'properties': {
                        'taxonomy_id': {
                            'type': 'string',
                            'pattern': '^t[0-9]{2}$',
                            'description': 'Which taxonomy row this expands.',
                        },
                        'title': {
                            'type': 'string',
                            'minLength': 5,
                            'maxLength': 80,
                            'description': 'Short identifier, lowercase snake_case friendly.',
                        },
                        'claim': {
                            'type': 'string',
                            'minLength': 40,
                            'description': (
                                "Conditional on [feature state F], r_{t+h} has "
                                "[property P] of [magnitude M], because [mechanism]."
                            ),
                        },
                        'null_hypothesis': {
                            'type': 'string',
                            'minLength': 20,
                            'description': 'Version of the claim that holds if no edge.',
                        },
                        'falsification_test': {
                            'type': 'object',
                            'required': ['statistic', 'threshold', 'min_n', 'decision_rule'],
                            'properties': {
                                'statistic': {'type': 'string'},
                                'threshold': {'type': 'string'},
                                'min_n': {'type': 'integer', 'minimum': 20},
                                'decision_rule': {'type': 'string'},
                            },
                        },
                        'degrees_of_freedom': {
                            'type': 'array',
                            'minItems': 1,
                            'maxItems': 6,
                            'items': {
                                'type': 'object',
                                'required': ['name', 'justification', 'sweep_range'],
                                'properties': {
                                    'name': {
                                        'type': 'string',
                                        'pattern': '^[a-z][a-z0-9_]*$',
                                        'description': 'Snake-case identifier, matches {p_<name>} in code_stub.',
                                    },
                                    'justification': {'type': 'string'},
                                    'sweep_range': {
                                        'type': 'array',
                                        'items': {'type': 'number'},
                                        'minItems': 1,
                                        'maxItems': 6,
                                        'description': 'Round numbers, wide spacing.',
                                    },
                                },
                            },
                        },
                        'entry': {
                            'type': 'array',
                            'items': {'type': 'string'},
                            'minItems': 1,
                            'description': 'One condition per line, using exact feature column names.',
                        },
                        'exit': {
                            'type': 'object',
                            'required': ['sl', 'tp', 'time_based', 'trailing'],
                            'properties': {
                                'sl': {'type': 'string'},
                                'tp': {'type': 'string'},
                                'time_based': {'type': 'string'},
                                'trailing': {'type': 'string'},
                            },
                        },
                        'hold_horizon': {
                            'type': 'string',
                            'description': '"6 hours" / "3 days" / "until funding_z mean-reverts" etc.',
                        },
                        'sizing': {
                            'type': 'string',
                            'description': 'Risk per trade %, conditional sizing rules.',
                        },
                        'honest_assessment': {
                            'type': 'string',
                            'minLength': 80,
                            'description': 'Which constraint does most work, where might it cheat, likely failure mode.',
                        },
                        'code_stub': {
                            'type': 'string',
                            'minLength': 80,
                            'description': (
                                'Pseudocode (Python-style) of the entry function body. Use exact '
                                'row.<feature> names. Use {p_<name>} placeholders matching '
                                'degrees_of_freedom entries.'
                            ),
                        },
                    },
                },
            },
            'final_pick': {
                'type': 'object',
                'required': ['choice', 'reason'],
                'properties': {
                    'choice': {
                        'type': 'string',
                        'enum': ['deep_dive_1', 'deep_dive_2', 'deep_dive_3', 'none'],
                    },
                    'reason': {'type': 'string', 'minLength': 20},
                },
            },
        },
    },
}
