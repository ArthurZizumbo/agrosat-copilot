"""Perceiver layer of the "Be My Eyes" pattern (Huang et al., 2025).

The perceiver is the eyes of the agent: it looks at a parcel (or an AOI polygon)
through the trained crop models and emits a *structured TEXT* observation that a
frozen reasoner LLM can read. It deliberately does NOT run the LLM and never
exposes tensors/logits to the reasoner -- only natural-language fields and a
``to_prompt_block`` rendering of them. This keeps the perceiver/reasoner contract
text-only (AC-2 of US-046).

What the perceiver wraps
------------------------
Conceptually the perceiver is the wrapper over the phenology-aware crop models
(TSViT-pheno / FarSLIP-pheno / AlphaEarth+XGBoost). In practice the only *direct,
inline, per-parcel* inference available without a raster is the tabular
``xgb-alphaearth`` member (the same model behind ``classify_new_parcel``) plus the
real phenology descriptor behind ``explain_prediction``. The dense temporal models
(TSViT-pheno, FarSLIP-pheno) require a Sentinel-2 raster cube and must run out of
band (OOF materialisation / Pub/Sub worker), so they are NOT invoked here -- doing
so inline would violate the ``ml/agent/AGENTS.md`` rule against heavy GPU inference
inside a tool. The perceiver therefore composes:

* the posterior over the 18 PASTIS semantic classes from the cached
  XGBoost-AlphaEarth classifier (reused from :mod:`ml.agent.tools.classify`), and
* the phenology / vigor / natural-language description from the real Wen et al.
  (2025) descriptor (reused from :mod:`ml.agent.tools.explain`).

Both data paths are session-scoped (multi-tenant): the embedding and the
phenology features are read only for parcels visible to ``ctx.session_id``.
"""

from __future__ import annotations

import time

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field

from ml.agent.context import ToolContext
from ml.agent.schemas import (
    ClassificationResult,
    ClassifyParcelInput,
    ExplainPredictionInput,
    Explanation,
    GeoJSONGeometry,
)
from ml.agent.tools import classify, explain

logger = structlog.get_logger(__name__)

__all__ = ["PerceiverLayer", "PerceiverObservation"]

#: Fallback crop label when no stored prediction nor classifier output is found.
_UNKNOWN_CROP: str = "unknown"

#: Default campaign year of the AlphaEarth annual embedding used for the
#: per-parcel classifier posterior (matches ``ClassifyParcelInput`` default).
_DEFAULT_YEAR: int = 2019

#: Sentinel parcel id for AOI-level observations (the AOI is not a stored parcel,
#: so it has no integer primary key). ``-1`` flags a synthetic/AOI observation.
_AOI_PARCEL_ID: int = -1


class PerceiverObservation(BaseModel):
    """Structured TEXT observation the reasoner consumes (never logits/tensors).

    This is the perceiver's only output type. Every field is plain text or a
    JSON-serialisable scalar/mapping so the reasoner reads language, not model
    internals. :meth:`to_prompt_block` renders these fields into the grounding
    block injected into the reasoner prompt.

    Attributes:
        parcel_id: Stored parcel primary key, or ``-1`` for an AOI-level
            observation that does not map to a persisted parcel.
        crop_class: Argmax crop class (human-readable PASTIS semantic18 name).
        confidence: Probability of ``crop_class`` in ``[0, 1]``.
        phenology_text: Structured phenology landmark text (SOG / peak /
            senescence / AUC), grounded in the measured scalar metrics.
        vigor: Qualitative vigor label (``"high"`` / ``"moderate"`` / ``"low"`` /
            ``"unknown"``) derived from the peak NDVI.
        class_probabilities: Full posterior over the crop classes
            (``{class_name: probability}``), summing to ~1.
        description: Natural-language summary suitable for the final answer
            (the Wen et al. 2025 phenology descriptor output).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    parcel_id: int
    crop_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    phenology_text: str
    vigor: str
    class_probabilities: dict[str, float]
    description: str

    def to_prompt_block(self) -> str:
        """Render the observation as a grounding text block for the reasoner.

        Produces a compact, human-readable Spanish block (crop, phenology, vigor,
        confidence, top alternatives) that the reasoner reads as context. It is
        pure text -- no tensors, no logits -- honouring the Be My Eyes contract.

        Returns:
            A multi-line Spanish text block describing the observation.
        """
        confidence_pct = f"{self.confidence * 100:.1f}%"
        top_alternatives = sorted(
            self.class_probabilities.items(), key=lambda kv: kv[1], reverse=True
        )[:3]
        alternatives_line = ", ".join(
            f"{name} ({prob * 100:.1f}%)" for name, prob in top_alternatives
        )
        lines = [
            "Observacion del perceiver (TEXTO, sin logits):",
            f"- Cultivo estimado: {self.crop_class} (confianza {confidence_pct}).",
            f"- {self.phenology_text}",
            f"- Vigor del cultivo: {self.vigor}.",
            f"- Clases mas probables: {alternatives_line}.",
            f"- Descripcion: {self.description}",
        ]
        return "\n".join(lines)


class PerceiverLayer:
    """Compose the crop classifier and phenology descriptor into TEXT observations.

    The layer is the inline, raster-free perceiver: it reuses the cached
    XGBoost-AlphaEarth classifier and the real phenology descriptor (never
    re-implementing their logic) to build a :class:`PerceiverObservation` for a
    stored parcel (:meth:`observe`) or a freshly drawn AOI polygon
    (:meth:`observe_aoi`). It does not call any LLM and emits no tensors.
    """

    def __init__(self, ctx: ToolContext) -> None:
        """Initialise the perceiver with the shared tool execution context.

        Args:
            ctx: Tool execution context (asyncpg pool, settings, session id).
        """
        self._ctx = ctx

    async def observe(self, parcel_id: int) -> PerceiverObservation:
        """Observe a stored parcel and emit a structured TEXT observation.

        Reuses :func:`ml.agent.tools.explain.run` for the crop class, confidence,
        phenology text, vigor and natural-language description (all grounded in the
        parcel's stored prediction and real phenology features), and enriches it
        with the full class posterior from the cached XGBoost-AlphaEarth classifier
        (:func:`ml.agent.tools.classify._load_classifier` over the parcel's
        AlphaEarth embedding). Both reads are session-scoped.

        Args:
            parcel_id: Primary key of the parcel to observe (must belong to the
                current session).

        Returns:
            A :class:`PerceiverObservation` with text fields and the class
            posterior. No tensors/logits are exposed to the caller.
        """
        start = time.perf_counter()
        logger.info(
            "perceiver_observe_started",
            session_id=str(self._ctx.session_id),
            parcel_id=parcel_id,
        )

        explanation = await explain.run(
            ExplainPredictionInput(
                session_id=self._ctx.session_id, parcel_id=parcel_id
            ),
            self._ctx,
        )
        class_probabilities = await self._class_posterior(
            crop_class=explanation.crop_class, year=_DEFAULT_YEAR
        )

        observation = self._observation_from_explanation(
            parcel_id=parcel_id,
            explanation=explanation,
            class_probabilities=class_probabilities,
        )
        logger.info(
            "perceiver_observe_finished",
            session_id=str(self._ctx.session_id),
            parcel_id=parcel_id,
            crop_class=observation.crop_class,
            vigor=observation.vigor,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )
        return observation

    async def observe_aoi(
        self, aoi: GeoJSONGeometry, year: int
    ) -> PerceiverObservation:
        """Observe a freshly drawn AOI polygon and emit a TEXT observation.

        An AOI is not a persisted parcel, so there is no stored phenology row to
        explain. The perceiver classifies the AOI with the XGBoost-AlphaEarth
        model (reusing :func:`ml.agent.tools.classify.run`, which resolves the
        session parcel's AlphaEarth embedding for ``year``) and derives the vigor
        and phenology text from that classifier output. When the AOI has no
        persisted embedding yet, ``classify.run`` returns the controlled
        ``needs_gee_sampling`` result, which is surfaced verbatim (no hallucinated
        crop).

        Args:
            aoi: Polygon geometry of the area to observe.
            year: Campaign year of the AlphaEarth annual embedding.

        Returns:
            A :class:`PerceiverObservation` for the AOI, with ``parcel_id == -1``.
        """
        start = time.perf_counter()
        logger.info(
            "perceiver_observe_started",
            session_id=str(self._ctx.session_id),
            parcel_id=_AOI_PARCEL_ID,
            geometry_type=aoi.type,
            year=year,
        )

        result = await classify.run(
            ClassifyParcelInput(
                session_id=self._ctx.session_id, aoi=aoi, year=year
            ),
            self._ctx,
        )
        observation = self._observation_from_classification(result)
        logger.info(
            "perceiver_observe_finished",
            session_id=str(self._ctx.session_id),
            parcel_id=_AOI_PARCEL_ID,
            crop_class=observation.crop_class,
            vigor=observation.vigor,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )
        return observation

    async def _class_posterior(
        self, *, crop_class: str, year: int
    ) -> dict[str, float]:
        """Compute the full class posterior for the session's parcel embedding.

        Reuses the cached XGBoost-AlphaEarth classifier and the session-scoped
        embedding fetch from :mod:`ml.agent.tools.classify`. When the session has
        no persisted embedding for ``year``, falls back to a degenerate posterior
        that places all mass on the (stored) ``crop_class`` so the observation
        still carries a coherent distribution without inventing alternatives.

        Args:
            crop_class: Crop class of the stored prediction, used for the fallback
                degenerate posterior.
            year: Campaign year of the AlphaEarth annual embedding.

        Returns:
            A ``{class_name: probability}`` posterior summing to ~1.
        """
        embedding = await classify._fetch_parcel_embedding(self._ctx, year)
        if embedding is None:
            logger.info(
                "perceiver_posterior_fallback",
                session_id=str(self._ctx.session_id),
                reason="no persisted AlphaEarth embedding for the session/year",
            )
            return {crop_class: 1.0}

        classifier = classify._load_classifier()
        proba = classifier.predict_proba_18(embedding)
        return {
            classifier.class_names.get(idx, str(idx)): float(proba[idx])
            for idx in range(proba.shape[0])
        }

    @staticmethod
    def _observation_from_explanation(
        *,
        parcel_id: int,
        explanation: Explanation,
        class_probabilities: dict[str, float],
    ) -> PerceiverObservation:
        """Assemble a :class:`PerceiverObservation` from an explanation + posterior.

        Args:
            parcel_id: Parcel the observation refers to.
            explanation: The ``explain_prediction`` result (text fields).
            class_probabilities: Full class posterior from the classifier.

        Returns:
            The composed :class:`PerceiverObservation`.
        """
        return PerceiverObservation(
            parcel_id=parcel_id,
            crop_class=explanation.crop_class or _UNKNOWN_CROP,
            confidence=float(np.clip(explanation.confidence, 0.0, 1.0)),
            phenology_text=explanation.phenology_text,
            vigor=explanation.vigor,
            class_probabilities=class_probabilities,
            description=explanation.description,
        )

    @staticmethod
    def _observation_from_classification(
        result: ClassificationResult,
    ) -> PerceiverObservation:
        """Assemble an AOI-level observation from a classification result.

        The AOI has no stored phenology row, so the phenology text is derived from
        the classifier output: the qualitative vigor is mapped from the top-class
        probability (a confidence proxy) via the shared
        :func:`ml.agent.tools.explain._vigor_from_peak` thresholds, and the
        description states the estimated crop and confidence in plain language.

        Args:
            result: The ``classify_new_parcel`` result for the AOI.

        Returns:
            An AOI-level :class:`PerceiverObservation` (``parcel_id == -1``).
        """
        vigor = explain._vigor_from_peak(result.confidence)
        confidence_pct = f"{result.confidence * 100:.1f}%"
        phenology_text = (
            "Fenologia: sin metricas fenologicas almacenadas para el AOI; "
            "observacion derivada del clasificador AlphaEarth+XGBoost."
        )
        description = (
            f"Cultivo estimado para el AOI: {result.crop_class} "
            f"(confianza {confidence_pct}), inferido del embedding AlphaEarth "
            "anual sin descriptor fenologico denso (TSViT/FarSLIP requieren "
            "raster y corren fuera de linea)."
        )
        return PerceiverObservation(
            parcel_id=_AOI_PARCEL_ID,
            crop_class=result.crop_class or _UNKNOWN_CROP,
            confidence=float(np.clip(result.confidence, 0.0, 1.0)),
            phenology_text=phenology_text,
            vigor=vigor,
            class_probabilities=dict(result.class_probabilities),
            description=description,
        )
