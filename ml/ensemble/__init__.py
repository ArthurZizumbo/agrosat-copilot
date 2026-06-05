"""ml/ensemble/__init__.py
=========================
Ensemble strategies for dense semantic segmentation on PASTIS-R.

Four strategies covering both homogeneous and heterogeneous ensembles
as required by the Avance 5 rubric:

Homogeneous (same model family)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. :class:`~ml.ensemble.voting.SoftVotingEnsemble`
   — Average softmax probabilities across temporal models
     (TSViT-pheno + TSViT-base + U-TAE).

2. :class:`~ml.ensemble.voting.HardVotingEnsemble`
   — Majority vote on argmax predictions.

Heterogeneous (different model families)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
3. :class:`~ml.ensemble.blending.LogitBlender`
   — Optuna-optimised weighted average of raw logits across
     temporal + spatial + foundation models
     (TSViT + AnySat + DeepLabv3+).

4. :class:`~ml.ensemble.stacking.StackingEnsemble`
   — Meta-learner (Conv 1×1 head or LogisticRegression) trained
     on concatenated base-model probability maps.

Quick-start::

    from ml.ensemble import SoftVotingEnsemble, LogitBlender, StackingEnsemble
"""

from ml.ensemble.voting import (
    SoftVotingEnsemble,
    HardVotingEnsemble,
    soft_vote_from_logits,
    hard_vote_from_logits,
)
from ml.ensemble.blending import (
    LogitBlender,
    blend_logits,
    optimise_blend_weights,
)
from ml.ensemble.stacking import StackingEnsemble, ConvMetaLearner

__all__ = [
    # Homogeneous
    "SoftVotingEnsemble",
    "HardVotingEnsemble",
    "soft_vote_from_logits",
    "hard_vote_from_logits",
    # Heterogeneous — blending
    "LogitBlender",
    "blend_logits",
    "optimise_blend_weights",
    # Heterogeneous — stacking
    "StackingEnsemble",
    "ConvMetaLearner",
]
