"""Rubric ensembles for AgroSatCopilot (US-040, EPIC 6).

Hosts the four mandatory ensembles (Voting / Bagging / Stacking / Blending),
each consuming the US-031 out-of-fold (OOF) probabilities and reporting on the
held-out fold-5 ONLY. Phase 1 ships the shared abstract base
:class:`ml.ensemble.base.EnsembleModel`; the four concrete ensembles are added in
phase 3 and re-exported here.
"""

from __future__ import annotations

from ml.ensemble.bagging import BaggingEnsemble
from ml.ensemble.base import (
    DEFAULT_OOF_DIR,
    ENSEMBLE_EXPERIMENT,
    EnsembleModel,
    Space,
)
from ml.ensemble.blending import DEFAULT_BLENDING_MEMBERS, BlendingEnsemble
from ml.ensemble.stacking import DEFAULT_BASE_MEMBERS, StackingEnsemble
from ml.ensemble.voting import DEFAULT_VOTING_MEMBERS, VotingEnsemble

__all__ = [
    "DEFAULT_BASE_MEMBERS",
    "DEFAULT_BLENDING_MEMBERS",
    "DEFAULT_OOF_DIR",
    "DEFAULT_VOTING_MEMBERS",
    "ENSEMBLE_EXPERIMENT",
    "BaggingEnsemble",
    "BlendingEnsemble",
    "EnsembleModel",
    "Space",
    "StackingEnsemble",
    "VotingEnsemble",
]
