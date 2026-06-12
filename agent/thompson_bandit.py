"""
Thompson-sampling bandit for live capital allocation across strategies.

Maintains a Normal-Inverse-Gamma (NIG) conjugate posterior over each
strategy's true mean daily PnL (proxy for Sharpe). Each allocation
decision draws one sample from each posterior; weights are proportional
to max(sample, 0). Strategies with wide posteriors get probabilistic
exploration; tight posteriors dominate exploitation.

Why this beats fixed/equal weights or fractional Kelly:
  * Provably bounded regret O(√(KT log T)) under sub-Gaussian rewards.
  * Naturally handles strategy decay via exponential forgetting on the
    sufficient statistics (recent observations carry more weight).
  * Survives BOCPD-style regime breaks: when a change-point is detected
    on a strategy's PnL stream, you can collapse its posterior back to
    the weak prior to force re-exploration.

Persistent state:
  * Sufficient stats are stored in agent_results.db so the bandit
    survives process restarts without losing learned posteriors.

Public API:
    record_pnl(strategy_name: str, pnl: float) -> None
    allocate(strategy_names: list[str]) -> dict[str, float]
    reset_strategy(strategy_name: str) -> None     # post-changepoint reset
    get_posterior(strategy_name: str) -> dict       # diagnostics
"""

from __future__ import annotations

import logging
import math
import sqlite3
from typing import Dict, List

import numpy as np

from agent.config import AGENT_DB_PATH

log = logging.getLogger(__name__)


# ── Hyperparameters ───────────────────────────────────────────────────────────

# Weak prior — gets overwritten quickly with real data.
PRIOR_MU      = 0.0       # prior mean daily PnL
PRIOR_KAPPA   = 1.0       # prior strength (number of pseudo-observations)
PRIOR_ALPHA   = 1.5       # NIG alpha (degrees of freedom = 2*alpha)
PRIOR_BETA    = 100.0     # NIG beta (in (PnL units)² )

# Exponential forgetting: each observation carries weight 1.0; older
# observations decay so the posterior tracks current performance.
# Decay = 0.99 means after 100 trades the oldest obs has weight ~0.37.
DECAY_FACTOR  = 0.99

# Hard cap on any single strategy's allocation, even if its posterior
# dominates. Prevents one regime-favourable strategy from owning the book.
MAX_WEIGHT    = 0.40
MIN_WEIGHT    = 0.02      # exploration floor

# Random source used for sampling (seedable for reproducible tests).
_rng = np.random.default_rng(seed=0xBA1D17)


# ── DB schema ─────────────────────────────────────────────────────────────────

def _init_db():
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS bandit_posterior (
            strategy_name TEXT PRIMARY KEY,
            mu            REAL NOT NULL,
            kappa         REAL NOT NULL,
            alpha         REAL NOT NULL,
            beta          REAL NOT NULL,
            n_obs         INTEGER NOT NULL DEFAULT 0,
            updated_at    TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def _load(strategy_name: str) -> dict:
    _init_db()
    con = sqlite3.connect(AGENT_DB_PATH)
    row = con.execute(
        "SELECT mu, kappa, alpha, beta, n_obs FROM bandit_posterior WHERE strategy_name = ?",
        (strategy_name,),
    ).fetchone()
    con.close()
    if row is None:
        return {
            'mu': PRIOR_MU, 'kappa': PRIOR_KAPPA,
            'alpha': PRIOR_ALPHA, 'beta': PRIOR_BETA,
            'n_obs': 0,
        }
    return {'mu': row[0], 'kappa': row[1], 'alpha': row[2], 'beta': row[3], 'n_obs': row[4]}


def _save(strategy_name: str, post: dict):
    from datetime import datetime, timezone
    _init_db()
    con = sqlite3.connect(AGENT_DB_PATH)
    con.execute("""
        INSERT INTO bandit_posterior (strategy_name, mu, kappa, alpha, beta, n_obs, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy_name) DO UPDATE SET
            mu=excluded.mu, kappa=excluded.kappa,
            alpha=excluded.alpha, beta=excluded.beta,
            n_obs=excluded.n_obs, updated_at=excluded.updated_at
    """, (
        strategy_name,
        float(post['mu']), float(post['kappa']),
        float(post['alpha']), float(post['beta']),
        int(post['n_obs']),
        datetime.now(timezone.utc).isoformat(),
    ))
    con.commit()
    con.close()


# ── Posterior update ──────────────────────────────────────────────────────────

def record_pnl(strategy_name: str, pnl: float) -> None:
    """
    Update the strategy's NIG posterior with a new daily PnL observation.

    Conjugate update (Murphy 2007 §3.3):
      μ'   = (κμ + x) / (κ + 1)
      κ'   = κ + 1
      α'   = α + 1/2
      β'   = β + κ(x - μ)² / (2(κ + 1))

    Then we apply an exponential decay so old data fades out:
      κ ← κ · DECAY_FACTOR
      α ← α · DECAY_FACTOR
      β ← β · DECAY_FACTOR
    (μ unchanged — decay only the precision/strength).
    """
    p = _load(strategy_name)
    x = float(pnl)
    new_mu     = (p['kappa'] * p['mu'] + x) / (p['kappa'] + 1.0)
    new_kappa  = p['kappa'] + 1.0
    new_alpha  = p['alpha'] + 0.5
    new_beta   = p['beta']  + (p['kappa'] * (x - p['mu']) ** 2) / (2.0 * (p['kappa'] + 1.0))

    # Exponential forgetting on the precision parameters.
    new_kappa *= DECAY_FACTOR
    new_alpha *= DECAY_FACTOR
    new_beta  *= DECAY_FACTOR

    # Floors so the posterior cannot collapse to a degenerate point mass.
    new_kappa = max(new_kappa, PRIOR_KAPPA * 0.5)
    new_alpha = max(new_alpha, PRIOR_ALPHA * 0.5)
    new_beta  = max(new_beta,  PRIOR_BETA  * 0.5)

    _save(strategy_name, {
        'mu': new_mu, 'kappa': new_kappa,
        'alpha': new_alpha, 'beta': new_beta,
        'n_obs': p['n_obs'] + 1,
    })


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_mean(post: dict) -> float:
    """Single Thompson sample from a NIG posterior:
       σ²  ~ InvGamma(α, β)
       μ   ~ N(μ₀, σ² / κ)
    Return one draw of μ (the latent mean PnL)."""
    alpha = max(post['alpha'], 0.5)
    beta  = max(post['beta'],  1e-9)
    kappa = max(post['kappa'], 1e-3)
    sigma2 = beta / _rng.gamma(shape=alpha, scale=1.0)  # inverse-gamma sample
    mu     = _rng.normal(loc=post['mu'], scale=math.sqrt(sigma2 / kappa))
    return float(mu)


# ── Public allocation API ─────────────────────────────────────────────────────

def allocate(strategy_names: List[str]) -> Dict[str, float]:
    """
    Return capital weights {name: weight, …} summing to 1.0.

    Procedure:
      1. Sample one μ from each posterior.
      2. Cap negative samples to 0 (won't allocate to negative-edge draws,
         but the strategy still gets a chance via its posterior next call).
      3. Apply MIN_WEIGHT floor for exploration, MAX_WEIGHT cap for safety.
      4. Renormalise to sum to 1.0.
    """
    if not strategy_names:
        return {}
    if len(strategy_names) == 1:
        return {strategy_names[0]: 1.0}

    samples = {n: max(_sample_mean(_load(n)), 0.0) for n in strategy_names}
    total = sum(samples.values())

    if total <= 0:
        # All draws negative — fall back to equal weights for exploration.
        eq = 1.0 / len(strategy_names)
        return {n: eq for n in strategy_names}

    raw = {n: v / total for n, v in samples.items()}

    # Apply min/max caps then renormalise.
    capped = {n: min(max(w, MIN_WEIGHT), MAX_WEIGHT) for n, w in raw.items()}
    s = sum(capped.values())
    return {n: w / s for n, w in capped.items()}


def reset_strategy(strategy_name: str) -> None:
    """Collapse the posterior back to the weak prior. Use this when
    BOCPD detects a change-point on this strategy's PnL — the bandit
    will re-explore at full uncertainty rather than carry stale beliefs."""
    _save(strategy_name, {
        'mu': PRIOR_MU, 'kappa': PRIOR_KAPPA,
        'alpha': PRIOR_ALPHA, 'beta': PRIOR_BETA,
        'n_obs': 0,
    })
    log.info("Bandit: posterior reset for '%s' (BOCPD trigger or manual)", strategy_name)


def get_posterior(strategy_name: str) -> dict:
    """Diagnostics: posterior summary plus 95% credible interval on μ."""
    p = _load(strategy_name)
    # Marginal of μ under NIG is Student-t with df = 2α, scale = √(β/(αk)).
    df    = 2 * p['alpha']
    scale = math.sqrt(max(p['beta'], 0.0) / max(p['alpha'] * p['kappa'], 1e-9))
    # Use Welch-Satterthwaite-style 95% CI on the t (df > 2): t-crit ≈ 1.96 + 2.4/df
    t_crit = 1.96 + 2.4 / max(df, 1.0)
    return {
        'mu':       p['mu'],
        'kappa':    p['kappa'],
        'alpha':    p['alpha'],
        'beta':     p['beta'],
        'n_obs':    p['n_obs'],
        'mu_low':   p['mu'] - t_crit * scale,
        'mu_high':  p['mu'] + t_crit * scale,
        'mu_std':   scale,
    }
