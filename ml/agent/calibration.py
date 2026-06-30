"""Honest confidence calibration for the copilot's restricted classifier (AC11).

The ``classify_new_parcel`` tool restricts its posterior to the resolved classes
of the active label-space and RENORMALIZES over them (see
:func:`ml.eval.class_remap.restrict_posterior`). That renormalization is
necessary -- the model genuinely cannot resolve the dropped classes -- but it has
a SHARP honesty cost: the restricted top class can look very confident
(``confidence`` near 1.0) even when most of the RAW (unrestricted) probability
mass actually landed OUTSIDE the resolved vocabulary. The renormalized headline
then over-states certainty.

This module exposes that gap as a PURE function over a posterior (no network, no
DB, no model load), so the reasoner and the UI can show an HONEST confidence
indicator instead of the inflated renormalized number alone:

- ``raw_confidence``  -- the model's RAW top-class probability over the FULL
  18-class space (what the model really thinks, including out-of-vocabulary mass).
- ``restricted_confidence`` -- the renormalized top-class probability over the
  resolved classes (what ``ClassificationResult.confidence`` reports today).
- ``resolved_mass`` -- the share of RAW mass that landed on the resolved classes
  (``1 - dropped_mass``); the renormalization divides by exactly this, so a small
  ``resolved_mass`` is precisely the regime where the restricted number is
  inflated.
- ``unresolved_candidate`` -- the out-of-vocabulary crop the RAW argmax leaned
  toward (``None`` when the raw top is in vocabulary), mirroring
  :attr:`ml.agent.schemas.ClassificationResult.unresolved_candidate`.

The central honesty signal is :meth:`ConfidenceReport.is_inflated`: ``True`` when
the RAW argmax is out-of-vocabulary OR the resolved mass is below a (documented,
non-probabilistic) floor -- i.e. when the restricted headline should be HEDGED,
not reported as confident. This is a heuristic flag, NOT a calibrated probability
(no Platt/temperature fit), and it is documented as such so it is never
over-sold.

This module is deliberately decoupled from :mod:`ml.agent.tools.classify` (which
a sibling US owns and which this US must not edit): it reads the SAME
:class:`~ml.eval.class_remap.LabelSpace` registry and operates either on a raw
``(18,)`` posterior or on an existing
:class:`~ml.agent.schemas.ClassificationResult` (its
``class_probabilities`` + ``unresolved_candidate`` fields), so it can annotate a
result the tool already produced without re-running any inference.

Project conventions: identifiers and docstrings in English (Google style);
``structlog`` (never ``print``); full type hints; no emojis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ml.eval.class_remap import LabelSpace, get_label_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.agent.schemas import ClassificationResult

logger = structlog.get_logger(__name__)

__all__ = [
    "ConfidenceReport",
    "calibrate_from_posterior",
    "calibrate_from_result",
]

#: Number of contiguous agronomic classes in the semantic18 space (mirrors
#: :data:`ml.eval.class_remap.HARNESS_NUM_CLASSES`; kept local so this pure module
#: needs no torch-importing dependency at module load).
_SEMANTIC18_SIZE: int = 18

#: Numerical floor for "no probability mass" (mirrors ``restrict_posterior``).
_MASS_EPS: float = 1e-12

#: Default resolved-mass floor below which the restricted headline is flagged as
#: INFLATED. NON-PROBABILISTIC, documented heuristic: when less than half of the
#: RAW mass survives the restriction, the renormalized top class has multiplied a
#: minority signal by more than 2x and should be read as a hedge, not a confident
#: call. Tunable per call; never presented as a calibrated threshold.
_DEFAULT_RESOLVED_MASS_FLOOR: float = 0.5


@dataclass(frozen=True)
class ConfidenceReport:
    """Raw-vs-restricted confidence breakdown for one classification (AC11).

    Exposes both confidence numbers plus the mass-loss the restriction caused, so
    a UI / reasoner can show an HONEST indicator instead of the (possibly
    inflated) renormalized headline alone. Every field is derived from a single
    posterior by an audited pure function; nothing here is a calibrated
    probability.

    Attributes:
        crop_class: The restricted top-class crop name (the headline the tool
            reports), or ``""`` when the resolved classes carry no mass.
        raw_confidence: The RAW top-class probability over the FULL 18-class space
            (what the model believes before restriction, out-of-vocabulary mass
            included).
        restricted_confidence: The renormalized top-class probability over the
            resolved classes (equals ``ClassificationResult.confidence``).
        resolved_mass: Share of RAW mass on the resolved classes in ``[0, 1]``
            (``1 - dropped_mass``); the restriction divides by exactly this.
        dropped_mass: Share of RAW mass on the out-of-vocabulary classes in
            ``[0, 1]`` (``1 - resolved_mass``).
        raw_top_class: The RAW argmax crop name over the full space (may be an
            out-of-vocabulary crop the restricted headline hides).
        unresolved_candidate: The out-of-vocabulary crop the RAW argmax leaned
            toward, or ``None`` when the raw top is in vocabulary (mirrors
            :attr:`ml.agent.schemas.ClassificationResult.unresolved_candidate`).
        resolved_mass_floor: The floor used by :attr:`is_inflated` (echoed for
            transparency).
    """

    crop_class: str
    raw_confidence: float
    restricted_confidence: float
    resolved_mass: float
    dropped_mass: float
    raw_top_class: str
    unresolved_candidate: str | None = None
    resolved_mass_floor: float = _DEFAULT_RESOLVED_MASS_FLOOR

    @property
    def is_inflated(self) -> bool:
        """Whether the restricted confidence over-states certainty (hedge cue).

        ``True`` when EITHER the RAW argmax is out-of-vocabulary (the headline is a
        renormalization artifact of a minority in-vocabulary signal) OR the
        resolved mass is below :attr:`resolved_mass_floor` (the restriction divided
        by a minority of the RAW mass). Both are the regimes where the UI should
        show a hedge instead of the confident-looking renormalized number.

        This is a HEURISTIC, not a calibrated probability (no temperature/Platt
        fit); it is the honest "treat with caution" flag US-081 AC11 asks for.

        Returns:
            ``True`` when the restricted headline should be hedged.
        """
        if self.unresolved_candidate is not None:
            return True
        return self.resolved_mass < self.resolved_mass_floor

    @property
    def confidence_gap(self) -> float:
        """Restricted minus raw confidence (the inflation magnitude).

        A large positive gap quantifies how much the renormalization lifted the
        headline above what the model raw-believed. Reported so the magnitude of
        the inflation is visible, not just its boolean flag.

        Returns:
            ``restricted_confidence - raw_confidence`` (>= 0 in practice, since
            restriction can only raise the top class's share).
        """
        return self.restricted_confidence - self.raw_confidence


def _resolve_space(label_space: LabelSpace | str | None) -> LabelSpace:
    """Resolve a label-space argument to a concrete :class:`LabelSpace`.

    Args:
        label_space: A :class:`LabelSpace`, a registered name, or ``None`` (the
            configured :data:`~ml.eval.class_remap.DEFAULT_LABEL_SPACE`).

    Returns:
        The resolved :class:`LabelSpace`.
    """
    if isinstance(label_space, LabelSpace):
        return label_space
    return get_label_space(label_space)


def calibrate_from_posterior(
    proba: np.ndarray,
    *,
    label_space: LabelSpace | str | None = None,
    resolved_mass_floor: float = _DEFAULT_RESOLVED_MASS_FLOOR,
) -> ConfidenceReport:
    """Build a :class:`ConfidenceReport` from a RAW ``(18,)`` posterior.

    Computes the RAW top-class confidence over the full space, the resolved /
    dropped mass split induced by ``label_space``, and the RENORMALIZED restricted
    confidence (the same number ``restrict_posterior`` + the tool report). Pure:
    no network, no DB, no model load.

    Args:
        proba: A ``(18,)`` post-softmax distribution over the contiguous
            semantic18 space (the model's RAW, unrestricted posterior).
        label_space: The active label-space (a :class:`LabelSpace`, a registered
            name, or ``None`` for the configured default). Its ``kept_class_ids``
            define the resolved vocabulary.
        resolved_mass_floor: Floor below which :attr:`ConfidenceReport.is_inflated`
            fires (documented heuristic, not a calibrated threshold).

    Returns:
        The :class:`ConfidenceReport` describing the raw-vs-restricted confidence.

    Raises:
        ValueError: if ``proba`` is not a 1-D vector of length 18.
    """
    arr = np.asarray(proba, dtype=np.float64).ravel()
    if arr.size != _SEMANTIC18_SIZE:
        raise ValueError(
            f"calibrate_from_posterior expects a ({_SEMANTIC18_SIZE},) semantic18 "
            f"posterior; received size {arr.size}."
        )
    space = _resolve_space(label_space)

    # Names for the full space come from the label-space class names where known,
    # falling back to the global semantic18 table for the dropped ids.
    full_names = _full_class_names(space)

    total = float(arr.sum())
    raw_norm = arr / total if total > _MASS_EPS else arr
    raw_top_idx = int(np.argmax(arr))
    raw_top_class = full_names.get(raw_top_idx, str(raw_top_idx))
    raw_confidence = float(raw_norm[raw_top_idx])

    kept = list(space.kept_class_ids)
    resolved_mass = float(raw_norm[kept].sum()) if kept else 0.0
    resolved_mass = max(0.0, min(1.0, resolved_mass))
    dropped_mass = max(0.0, 1.0 - resolved_mass)

    # Restricted (renormalized) headline over the resolved classes only.
    if resolved_mass > _MASS_EPS:
        restricted_top_idx = max(kept, key=lambda cid: raw_norm[cid])
        restricted_confidence = float(raw_norm[restricted_top_idx]) / resolved_mass
        crop_class = space.class_names.get(
            restricted_top_idx, full_names.get(restricted_top_idx, str(restricted_top_idx))
        )
    else:
        # No resolved-class mass: honest "none of the resolved classes apply".
        restricted_confidence = 0.0
        crop_class = ""

    # The raw argmax is out-of-vocabulary iff it is among the dropped ids.
    unresolved_candidate = space.dropped_class_names.get(raw_top_idx)

    report = ConfidenceReport(
        crop_class=crop_class,
        raw_confidence=raw_confidence,
        restricted_confidence=restricted_confidence,
        resolved_mass=resolved_mass,
        dropped_mass=dropped_mass,
        raw_top_class=raw_top_class,
        unresolved_candidate=unresolved_candidate,
        resolved_mass_floor=resolved_mass_floor,
    )
    logger.info(
        "confidence_calibrated",
        label_space=space.name,
        crop_class=report.crop_class,
        raw_confidence=round(report.raw_confidence, 4),
        restricted_confidence=round(report.restricted_confidence, 4),
        resolved_mass=round(report.resolved_mass, 4),
        is_inflated=report.is_inflated,
    )
    return report


def calibrate_from_result(
    result: ClassificationResult,
    *,
    label_space: LabelSpace | str | None = None,
    resolved_mass_floor: float = _DEFAULT_RESOLVED_MASS_FLOOR,
) -> ConfidenceReport:
    """Build a :class:`ConfidenceReport` from an existing ``ClassificationResult``.

    Reconstructs the RAW resolved/dropped mass split from the result's RESTRICTED
    ``class_probabilities`` (which sum to ~1 over the resolved classes) plus the
    ``unresolved_candidate`` hint, WITHOUT re-running inference and WITHOUT editing
    :mod:`ml.agent.tools.classify`. This is the on-result path for the SSE / UI: a
    sibling produced the result; this annotates it with the honest confidence.

    The restricted ``class_probabilities`` already lost the dropped mass (they were
    renormalized), so the exact RAW split is not recoverable from them alone. This
    path therefore reports what IS knowable honestly:

    - ``restricted_confidence`` is the result's reported ``confidence`` (or the max
      of its ``class_probabilities``).
    - ``unresolved_candidate`` is taken verbatim from the result.
    - ``is_inflated`` keys off ``unresolved_candidate`` (the in-result honest cue),
      since the RAW resolved mass is not reconstructable from a restricted dump.
      ``resolved_mass`` is reported as NaN-free best effort: ``1.0`` when the raw
      top is in vocabulary, and left as the (unknown) restricted sum otherwise,
      with ``raw_confidence`` set equal to ``restricted_confidence`` (no
      fabrication of a smaller raw number we cannot derive).

    Prefer :func:`calibrate_from_posterior` whenever the RAW ``(18,)`` posterior is
    available (it derives the true mass split); this function is the lossy fallback
    for when only the restricted result is on hand.

    Args:
        result: The :class:`~ml.agent.schemas.ClassificationResult` to annotate.
        label_space: The active label-space (resolved for naming/floor only).
        resolved_mass_floor: Floor echoed onto the report (unused on this lossy
            path beyond the ``unresolved_candidate`` gate).

    Returns:
        The :class:`ConfidenceReport` derived from the restricted result.
    """
    space = _resolve_space(label_space)
    probs = dict(result.class_probabilities)
    restricted_confidence = float(result.confidence)
    if not restricted_confidence and probs:
        restricted_confidence = float(max(probs.values()))

    unresolved = result.unresolved_candidate
    # On the lossy path the RAW mass split is not recoverable from a renormalized
    # dump; we do NOT invent one. ``raw_confidence`` is reported equal to the
    # restricted value (the only honest scalar we have), and the inflation flag is
    # driven by the in-result ``unresolved_candidate`` cue.
    in_vocab = unresolved is None
    resolved_mass = 1.0 if in_vocab else float("nan")
    report = ConfidenceReport(
        crop_class=result.crop_class,
        raw_confidence=restricted_confidence,
        restricted_confidence=restricted_confidence,
        resolved_mass=resolved_mass,
        dropped_mass=0.0 if in_vocab else float("nan"),
        raw_top_class=unresolved or result.crop_class,
        unresolved_candidate=unresolved,
        resolved_mass_floor=resolved_mass_floor,
    )
    logger.info(
        "confidence_calibrated_from_result",
        label_space=space.name,
        crop_class=report.crop_class,
        restricted_confidence=round(report.restricted_confidence, 4),
        unresolved_candidate=report.unresolved_candidate,
        is_inflated=report.is_inflated,
    )
    return report


def _full_class_names(space: LabelSpace) -> dict[int, str]:
    """Return a ``{semantic18_id: name}`` map covering all 18 ids for a space.

    Merges the label-space's kept-class names with the dropped-class names so the
    raw argmax (which may be an out-of-vocabulary id) is always nameable.

    Args:
        space: The active label-space.

    Returns:
        A mapping covering every id the space references (kept + dropped).
    """
    names: dict[int, str] = dict(space.class_names)
    names.update(space.dropped_class_names)
    return names
