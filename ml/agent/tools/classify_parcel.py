"""FunctionTool: classify the crop of parcels in an AOI (XGBoost + AlphaEarth).

The vision agent calls this tool to label each parcel of an AOI with a crop
class. The real path runs the trained tabular baseline (XGBoost over the 64-dim
AlphaEarth Satellite Embedding) loaded from the MLflow Model Registry. When the
model or the registry is unreachable (CI, laptop without MLflow), it falls back
HONESTLY to the ``crop_class`` already stored in :class:`ParcelRecord` and marks
the citation ``source="stored:crop_class"`` so the provenance never lies.

NON-NEGOTIABLE: every figure carries a :class:`Citation` tied to the tool call.
Crop ids are mapped to readable names with ``PASTIS_R_CLASSES``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from ml.agent.events import Citation, Finding
from ml.ingest.pastis_loader import PASTIS_R_CLASSES

if TYPE_CHECKING:  # avoid importing heavy deps / ports at runtime if unused
    from ml.agent.ports import ParcelReader

logger = structlog.get_logger(__name__)

# Cache of the loaded model so repeated tool calls within a process do not hit
# the registry every time. ``None`` distinct from "not loaded yet": we use a
# sentinel so a known-unavailable registry is not retried on every call.
_MODEL_CACHE: dict[str, object | None] = {}
_REGISTRY_TRIED: set[str] = set()


# ---------------------------------------------------------------------------
# Schemas.
# ---------------------------------------------------------------------------


class ClassifyParcelInput(BaseModel):
    """Input contract for :func:`classify_parcel`."""

    session_id: str = Field(description="Tenant scope (multi-tenant NON-NEGOTIABLE).")
    aoi_id: int | None = Field(
        default=None, description="AOI whose parcels to classify; None = all session parcels."
    )
    year: int | None = Field(default=None, description="Feature/label year filter.")


class ClassifyParcelOutput(BaseModel):
    """Output contract: structured findings plus a human-readable summary."""

    findings: list[Finding] = Field(default_factory=list)
    summary: str
    used_model: bool = Field(
        description="True if the XGBoost model ran; False if the stored fallback was used."
    )


# ---------------------------------------------------------------------------
# Model loading (MLflow Model Registry).
# ---------------------------------------------------------------------------


def _load_classifier(model_uri: str) -> object | None:
    """Load the trained classifier bundle from MLflow, or ``None`` if absent.

    The artifact is expected to be a ``BaselineResult`` (pickled) exposing
    ``model``, ``label_encoder``/``label_classes`` and ``feature_cols``. We load
    it via ``mlflow.sklearn``/``mlflow.pyfunc`` defensively: any failure (no
    registry, no run, no MLflow installed) returns ``None`` so the caller can
    fall back. The result is cached per URI.
    """
    if model_uri in _MODEL_CACHE:
        return _MODEL_CACHE[model_uri]
    if model_uri in _REGISTRY_TRIED:
        return None
    _REGISTRY_TRIED.add(model_uri)
    try:
        import mlflow  # type: ignore[import-not-found]

        bundle: object = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001 - registry/MLflow optional in dev
        logger.warning("crop_classifier_load_failed", uri=model_uri, error=str(exc))
        _MODEL_CACHE[model_uri] = None
        return None
    logger.info("crop_classifier_loaded", uri=model_uri)
    _MODEL_CACHE[model_uri] = bundle
    return bundle


def _crop_name(class_id: int) -> str:
    """Map a PASTIS-R class id to its readable crop name."""
    return PASTIS_R_CLASSES.get(int(class_id), f"class_{class_id}")


def _predict_one(bundle: object, embedding: list[float]) -> tuple[str, float] | None:
    """Run the bundled classifier on one AlphaEarth embedding vector.

    Returns ``(crop_name, confidence)`` or ``None`` if the embedding shape does
    not match the model's expected feature count (defensive: never crash a chat
    turn over a malformed feature row).
    """
    import numpy as np

    model = getattr(bundle, "model", None)
    if model is None:
        return None
    label_classes = getattr(bundle, "label_classes", None)
    feature_cols = getattr(bundle, "feature_cols", None)
    x = np.asarray([embedding], dtype=np.float64)
    if feature_cols is not None and x.shape[1] != len(feature_cols):
        logger.warning(
            "crop_classifier_feature_mismatch",
            got=x.shape[1],
            expected=len(feature_cols),
        )
        return None
    proba_fn = getattr(model, "predict_proba", None)
    if proba_fn is not None:
        proba = np.asarray(proba_fn(x))[0]
        best = int(np.argmax(proba))
        confidence = float(proba[best])
        classes = getattr(model, "classes_", None)
        encoded = int(classes[best]) if classes is not None else best
    else:
        encoded = int(np.asarray(model.predict(x))[0])
        confidence = float("nan")
    # Map the (encoded) label back to the original PASTIS-R class id.
    if label_classes is not None and 0 <= encoded < len(label_classes):
        class_id = int(label_classes[encoded])
    else:
        class_id = encoded
    return _crop_name(class_id), confidence


# ---------------------------------------------------------------------------
# Tool entry-point.
# ---------------------------------------------------------------------------


async def classify_parcel(
    payload: ClassifyParcelInput,
    *,
    parcels: ParcelReader,
    model_uri: str | None = None,
) -> ClassifyParcelOutput:
    """Classify the crop of each parcel in the AOI.

    Args:
        payload: Validated input (session, AOI, year).
        parcels: Read-only parcel/feature port (injected from ``AgentDeps``).
        model_uri: Override the MLflow registry URI (defaults to settings).

    Returns:
        A :class:`ClassifyParcelOutput` with one :class:`Finding` per parcel.

    Note:
        ``citation.tool_call_id`` is left empty here; the caller (vision agent)
        stamps the real ``call_id`` so the wire-level link is authoritative.
    """
    from ml.agent.settings import get_settings

    uri = model_uri or get_settings().crop_classifier_model_uri
    bundle = _load_classifier(uri)

    records = await parcels.list_parcels_in_aoi(
        session_id=payload.session_id, aoi_id=payload.aoi_id, year=payload.year
    )

    findings: list[Finding] = []
    used_model = False
    for rec in records:
        crop_name: str | None = rec.crop_class
        confidence: float | None = rec.confidence
        source = "stored:crop_class"

        if bundle is not None:
            feature = await parcels.get_features(
                session_id=payload.session_id, parcel_id=rec.id, year=rec.year
            )
            embedding = feature.alphaearth_embedding if feature else None
            if embedding:
                prediction = _predict_one(bundle, embedding)
                if prediction is not None:
                    crop_name, confidence = prediction
                    source = "XGBoost+AlphaEarth"
                    used_model = True

        findings.append(
            Finding(
                parcel_id=rec.id,
                crop_class=crop_name,
                confidence=confidence,
                area_ha=rec.area_ha,
                geometry=rec.geometry,
                citation=Citation(
                    tool_call_id="",
                    source=source,
                    parcel_id=rec.id,
                    aoi_id=rec.aoi_id,
                ),
            )
        )

    logger.info(
        "classify_parcel_done",
        session_id=payload.session_id,
        aoi_id=payload.aoi_id,
        n_parcels=len(findings),
        used_model=used_model,
        source=("XGBoost+AlphaEarth" if used_model else "stored:crop_class"),
    )
    summary = (
        f"{len(findings)} parcelas clasificadas "
        f"({'modelo XGBoost+AlphaEarth' if used_model else 'crop_class almacenado'})."
    )
    return ClassifyParcelOutput(findings=findings, summary=summary, used_model=used_model)


__all__ = [
    "ClassifyParcelInput",
    "ClassifyParcelOutput",
    "classify_parcel",
]
