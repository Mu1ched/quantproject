"""Hypothesis-generation pipeline (automated, API-driven).

Independent submodule. Does NOT import agent.main or agent.runtime — only
read-only access to agent.scorer (for gate definitions) and agent.config
(for thresholds in the constraints doc).

Run a round: `python -m agent.hypothesis_gen.driver --max-rounds 1`
"""
