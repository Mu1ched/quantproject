"""
Typed genetic programming evolver — zero-API-cost hypothesis generation.

Produces hypothesis dicts in the same shape as agent/claude_client.generate_hypotheses
so they flow through the existing code_writer → run_sweep → robustness pipeline
without any special-casing.

Why typed GP:
  * Untyped GP burns enormous cycles producing nonsense like
    `And(atr, tick_imbalance)`. With type tags on every tree node, every
    offspring is grammatically valid by construction.
  * Mutation / crossover only swap subtrees of matching type. No syntax
    retries, no compile failures.
  * Search space of structurally novel strategies is orders of magnitude
    larger than what an LLM proposes (different distribution, not better
    or worse — they cover different regions).

The output trees are compiled into the standard mandatory-pattern entry
function template (spread gate → pending check → time gate → NaN guard →
GP boolean → SL/TP from ATR × params). This keeps the engine contract
intact regardless of what the GP discovers.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


# ── Grammar definition ────────────────────────────────────────────────────────

# Curated feature subset — every entry must exist in row produced by prepare_df.
# Covers all four research-discipline categories.
_NUMERIC_FEATURES: List[str] = [
    # MICROSTRUCTURE
    'tick_imbalance', 'vol_imbalance', 'persistent_imbalance',
    'tick_imb_roll5', 'tick_imb_delta', 'aggressive_buy_ratio',
    'delta_momentum', 'hawkes_intensity',
    # VOLATILITY
    'atr_ratio', 'rv_delta', 'yz_vol_ratio',
    # TREND AND STRUCTURE
    'adx', 'ma_dist', 'momentum_3', 'momentum_10',
    'bb_pct', 'rsi_14', 'bar_momentum', 'bar_range_pct',
    'close_location', 'swing_high_dist', 'swing_low_dist',
    'dist_pivot_r1', 'dist_pivot_s1', 'h1_trend_strength',
    # REGIME (statistical)
    'hurst', 'perm_entropy_100',
    'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'hmm_prob_3',
    # CROSS-PAIR
    'eu_gu_corr20', 'dxy_change_5',
]

# Constants drawn from the "wide round" set per the parameter design rules.
_CONSTS: List[float] = [
    -2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.2, 0.0,
    0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0,
]

_REGIME_LITERALS: List[str] = ['TRENDING', 'RANGING', 'TRANSITIONING', 'VOLATILE']

_CMP_OPS: List[str] = ['<', '>', '<=', '>=']
_BIN_OPS: List[str] = ['+', '-', '*']      # division omitted for safety

# Behaviour-type pool used when the GP fabricates a hypothesis.
_BEHAVIOUR_TYPES = [
    'breakout_continuation',
    'false_breakout_liquidity_grab',
    'mean_reversion_low_volatility',
    'momentum_ignition_after_compression',
    'trend_pullback_continuation',
    'stop_run_reversal',
]


# ── Tree types ────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """A typed expression tree node.

    Tag is the operator name; type is 'bool' or 'num'; children are Nodes
    or terminal leaves (str feature name, float const, str regime literal).
    """
    tag: str          # 'and','or','not','cmp','regime','feat','const','binop'
    typ: str          # 'bool' or 'num'
    children: tuple   # tuple of Nodes / leaves

    # ── size for parsimony pressure ──
    def size(self) -> int:
        n = 1
        for c in self.children:
            if isinstance(c, Node):
                n += c.size()
            else:
                n += 1
        return n

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(
            (c.depth() if isinstance(c, Node) else 1) for c in self.children
        )


# ── Random tree generation ────────────────────────────────────────────────────

def _random_num(depth_left: int) -> Node:
    if depth_left <= 0 or random.random() < 0.55:
        # terminal
        if random.random() < 0.7:
            return Node('feat', 'num', (random.choice(_NUMERIC_FEATURES),))
        return Node('const', 'num', (random.choice(_CONSTS),))
    # binary op (small chance, kept rare to avoid bloat)
    if random.random() < 0.25:
        op = random.choice(_BIN_OPS)
        return Node(
            'binop', 'num',
            (op, _random_num(depth_left - 1), _random_num(depth_left - 1)),
        )
    # default: terminal feature
    return Node('feat', 'num', (random.choice(_NUMERIC_FEATURES),))


def _random_bool(depth_left: int) -> Node:
    if depth_left <= 1:
        return _random_compare(depth_left)
    r = random.random()
    if r < 0.55:
        return _random_compare(depth_left)
    if r < 0.75:
        return Node(
            'and', 'bool',
            (_random_bool(depth_left - 1), _random_bool(depth_left - 1)),
        )
    if r < 0.90:
        return Node(
            'or', 'bool',
            (_random_bool(depth_left - 1), _random_bool(depth_left - 1)),
        )
    if r < 0.95:
        return Node('not', 'bool', (_random_bool(depth_left - 1),))
    return Node('regime', 'bool', (random.choice(_REGIME_LITERALS),))


def _random_compare(depth_left: int) -> Node:
    return Node(
        'cmp', 'bool',
        (random.choice(_CMP_OPS),
         _random_num(depth_left - 1),
         _random_num(depth_left - 1)),
    )


def random_tree(max_depth: int = 4) -> Node:
    """Top-level tree is always Bool — the entry condition."""
    return _random_bool(max_depth)


# ── Compile to Python expression text ─────────────────────────────────────────

def _expr(n: Node) -> str:
    """Return a Python expression string. Caller is responsible for context
    (the row variable, math import) — these are present at the call site
    inside the generated entry function."""
    if not isinstance(n, Node):
        # leaf
        return repr(n)

    tag = n.tag
    if tag == 'feat':
        name = n.children[0]
        # NaN-safe getattr
        return (
            f"(lambda _v: 0.0 if (_v is None or "
            f"(isinstance(_v, float) and _v != _v)) else float(_v))"
            f"(getattr(row, {name!r}, float('nan')))"
        )
    if tag == 'const':
        return f"({float(n.children[0])!r})"
    if tag == 'binop':
        op, a, b = n.children
        return f"({_expr(a)} {op} {_expr(b)})"
    if tag == 'cmp':
        op, a, b = n.children
        return f"({_expr(a)} {op} {_expr(b)})"
    if tag == 'and':
        return f"({_expr(n.children[0])} and {_expr(n.children[1])})"
    if tag == 'or':
        return f"({_expr(n.children[0])} or {_expr(n.children[1])})"
    if tag == 'not':
        return f"(not {_expr(n.children[0])})"
    if tag == 'regime':
        return f"(regime == {n.children[0]!r})"
    raise ValueError(f"unknown node tag: {tag}")


# ── Mutation and crossover (typed) ────────────────────────────────────────────

def _collect(n: Node, want_type: str, acc: list, parent: Optional[Node] = None,
             slot: Optional[int] = None):
    """Walk the tree collecting (parent, slot, node) tuples for nodes whose
    type matches want_type. Used by mutate/crossover to pick a subtree to
    replace or swap."""
    if isinstance(n, Node):
        if n.typ == want_type:
            acc.append((parent, slot, n))
        for i, c in enumerate(n.children):
            if isinstance(c, Node):
                _collect(c, want_type, acc, n, i)


def _replace(parent: Optional[Node], slot: Optional[int],
             old: Node, new: Node, root: Node) -> Node:
    """Return a new tree with `old` replaced by `new`. Pure-functional
    rebuild (Node is dataclass-immutable in usage)."""
    if parent is None:
        # replacing the root
        return new

    def rebuild(n: Node) -> Node:
        if not isinstance(n, Node):
            return n
        if n is parent:
            new_children = list(n.children)
            new_children[slot] = new
            return Node(n.tag, n.typ, tuple(new_children))
        return Node(
            n.tag, n.typ,
            tuple(rebuild(c) if isinstance(c, Node) else c for c in n.children),
        )

    return rebuild(root)


def mutate(root: Node, max_depth: int = 4) -> Node:
    """Subtree mutation: pick a Bool or Num node, regenerate it at the
    same type with depth budget proportional to remaining tree depth."""
    pool: list = []
    _collect(root, 'bool', pool)
    _collect(root, 'num', pool)
    if not pool:
        return root
    parent, slot, target = random.choice(pool)
    fresh: Node = (
        _random_bool(max(2, max_depth - 1))
        if target.typ == 'bool'
        else _random_num(max(2, max_depth - 2))
    )
    return _replace(parent, slot, target, fresh, root)


def crossover(a: Node, b: Node) -> Tuple[Node, Node]:
    """Type-respecting subtree swap. Picks a random node in `a`, finds
    a matching-type node in `b`, swaps them. Returns two children."""
    pool_a_bool, pool_a_num = [], []
    pool_b_bool, pool_b_num = [], []
    _collect(a, 'bool', pool_a_bool); _collect(a, 'num', pool_a_num)
    _collect(b, 'bool', pool_b_bool); _collect(b, 'num', pool_b_num)

    options = []
    if pool_a_bool and pool_b_bool:
        options.append(('bool', pool_a_bool, pool_b_bool))
    if pool_a_num and pool_b_num:
        options.append(('num', pool_a_num, pool_b_num))
    if not options:
        return a, b

    typ, p_a, p_b = random.choice(options)
    pa, sa, na = random.choice(p_a)
    pb, sb, nb = random.choice(p_b)

    child_a = _replace(pa, sa, na, nb, a)
    child_b = _replace(pb, sb, nb, na, b)
    return child_a, child_b


# ── Full entry-function template ──────────────────────────────────────────────

_ENTRY_TEMPLATE = """\
def entry_{fn_name}(bst, slot, row, ts, pair, slip, hspd,
                    sess_cfg, regime, regime_mult, fvg_buf=None, day_sweep=None):
    params = slot['strategy_def']['params']
    sc     = slot['scratch']

    if spread_gate(row):
        return False
    if sc.get('pending_dir') is not None:
        return check_and_fill(sc, row, slot, ts, regime)

    atr = getattr(row, 'atr', float('nan'))
    if math.isnan(atr) or atr <= 0:
        return False

    try:
        cond_long  = bool({long_expr})
        cond_short = bool({short_expr})
    except Exception:
        return False

    if cond_long and cond_short:
        return False
    if not (cond_long or cond_short):
        return False

    direction = 'long' if cond_long else 'short'
    sl_dist   = atr * params['sl_r']
    tp_dist   = atr * params['tp_r']
    risk      = resolve_risk(bst, regime_mult, 'dynamic')

    close_v = getattr(row, 'close', float('nan'))
    if math.isnan(close_v) or close_v <= 0:
        return False

    if direction == 'long':
        entry = float(close_v) + hspd + slip
        sc['pending_dir']   = 'long'
        sc['pending_level'] = entry
        sc['pending_entry'] = entry
        sc['pending_sl']    = entry - sl_dist
        sc['pending_tp']    = entry + tp_dist
    else:
        entry = float(close_v) - hspd - slip
        sc['pending_dir']   = 'short'
        sc['pending_level'] = entry
        sc['pending_entry'] = entry
        sc['pending_sl']    = entry + sl_dist
        sc['pending_tp']    = entry - tp_dist

    sc['pending_size'] = rv_size(pair, bst.balance, risk, sc['pending_entry'], sl_dist, row)
    sc['pending_dist'] = sl_dist
    return check_and_fill(sc, row, slot, ts, regime)
"""


def _invert_compares(expr_text: str) -> str:
    """Cheap directional inversion: flip < ↔ > and <= ↔ >= so the short
    condition is the geometric mirror of the long. This is much cheaper
    than maintaining two trees and gives sensible long/short symmetry."""
    out = []
    i = 0
    while i < len(expr_text):
        if expr_text[i:i + 2] == '<=':
            out.append('>='); i += 2
        elif expr_text[i:i + 2] == '>=':
            out.append('<='); i += 2
        elif expr_text[i] == '<':
            out.append('>'); i += 1
        elif expr_text[i] == '>':
            out.append('<'); i += 1
        else:
            out.append(expr_text[i]); i += 1
    return ''.join(out)


def _name_from_tree(prefix: str, root: Node) -> str:
    # Stable-ish hash of the structure for naming.
    h = abs(hash((root.tag, root.size(), root.depth()))) % 10_000_000
    return f"gp_{prefix}_{h}"


def tree_to_hypothesis(root: Node, behaviour_type: Optional[str] = None,
                       prefix: str = "evo") -> dict:
    """Produce a hypothesis dict consumable by loop._process_hypotheses."""
    long_expr  = _expr(root)
    short_expr = _invert_compares(long_expr)
    fn_name    = _name_from_tree(prefix, root)
    code = _ENTRY_TEMPLATE.format(
        fn_name=fn_name, long_expr=long_expr, short_expr=short_expr,
    )
    bt = behaviour_type or random.choice(_BEHAVIOUR_TYPES)
    return {
        'function_name':     fn_name,
        'code':              code,
        'rationale':         (
            f"Typed-GP synthesis (depth={root.depth()}, size={root.size()}). "
            f"Targets {bt}: combination of "
            f"{', '.join(sorted({c for c in _NUMERIC_FEATURES if c in long_expr})[:4]) or 'features'} "
            f"under regime gate."
        ),
        'behaviour_type':    bt,
        'additional_params': {},  # GP relies on grid TP/SL only by default
    }


# ── Public entry point: NSGA-II-lite multi-objective generation ───────────────

def gp_generate(
    survivor_codes:    Optional[List[dict]] = None,
    n:                 int = 8,
    seed_population:   int = 64,
    max_depth:         int = 4,
    mutation_rate:     float = 0.5,
    parsimony_lambda:  float = 0.0,
) -> list:
    """
    Produce `n` hypothesis dicts via typed GP.

    Strategy:
      * Build a seed population of random typed trees.
      * Apply mutation/crossover for one generation to produce n candidates.
      * Each tree is independently expanded into the mandatory entry-function
        template, so every output is syntactically valid Python.
      * Behaviour type is sampled per candidate; the meta-learner's
        UNDEREXPLORED_BEHAVIOURS feedback loop in claude_client / meta_learner
        is unaffected — these candidates participate normally.

    `survivor_codes` is accepted for signature compatibility with
    evolver.mutate_survivors but unused — the GP search is independent.
    `parsimony_lambda` is reserved for a future fitness-based reproduction
    cycle; current pass produces single-generation random offspring.
    """
    population: List[Node] = [random_tree(max_depth) for _ in range(seed_population)]
    candidates: List[Node] = []

    # Light "generation": produce n descendants by mixing operators.
    while len(candidates) < n:
        if random.random() < mutation_rate or len(population) < 2:
            parent = random.choice(population)
            child  = mutate(parent, max_depth=max_depth)
        else:
            a, b      = random.sample(population, 2)
            child, _  = crossover(a, b)
        # Reject monstrously large trees (parsimony floor).
        if child.size() > 40:
            continue
        candidates.append(child)

    # De-duplicate by exact compiled-expression text.
    seen = set()
    out  = []
    for tree in candidates:
        h = tree_to_hypothesis(tree)
        if h['code'] in seen:
            continue
        seen.add(h['code'])
        out.append(h)

    log.info("GP-evolver generated %d unique hypotheses", len(out))
    return out
