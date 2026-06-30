"""Agent-side evaluation harnesses owned by the agent work-stream.

This package holds evals that live WITH the agent (``ml/agent/``) rather than in
the shared :mod:`ml.eval` benchmark surface, so the agent team can iterate on
copilot-specific measurements without touching the public benchmark harness.

Modules:

- :mod:`ml.agent.eval.hedge_ab_eval` -- the out-of-vocabulary hedge A/B (US-081
  AC8): does the reasoner's analysis of a parcel whose true crop is OUTSIDE the
  resolved vocabulary improve WITH ``retrieve_context`` (RAG + phenology
  grounding) versus WITHOUT it?
"""

from __future__ import annotations

__all__: list[str] = []
