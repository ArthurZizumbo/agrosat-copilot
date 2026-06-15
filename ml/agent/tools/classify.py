"""``classify_new_parcel`` tool: per-parcel crop classification (synchronous).

For a NEW parcel polygon the only direct per-parcel inference available is the
tabular ``xgb-alphaearth`` member of the EPIC 6 ensemble: an XGBoost trained over
the 64-dim AlphaEarth annual embedding. This tool:

1. Resolves the AlphaEarth embedding of the parcel. If the polygon already maps to
   a persisted parcel of the session (``features_parcels.alphaearth_embedding``),
   that embedding is used; the query is session-scoped (multi-tenant). If no
   embedding is found (a fresh AOI with no persisted parcel/embedding), the tool
   does NOT hallucinate a class: it returns a controlled low-confidence result
   flagged ``needs_gee_sampling`` (the AlphaEarth GEE sampler that would produce
   the embedding is out of scope for this US).
2. Loads the XGBoost-AlphaEarth classifier (CPU, cached with ``functools.lru_cache``)
   trained leak-free on folds 1-4 of ``features_fused_pastis.parquet`` -- the exact
   same recipe the EPIC 6 stacking ensemble uses to materialize the
   ``xgb-alphaearth`` base member (``scripts/run_us040_ensembles.py``).
3. Runs ``predict_proba`` on the embedding and returns a
   :class:`~ml.agent.schemas.ClassificationResult` with the argmax crop class, its
   confidence and the full posterior over the 18 PASTIS semantic classes.

The classifier load is CPU-light (a single XGBoost fit over ~13k tabular rows) and
cached process-wide, so repeated calls reuse the fitted estimator (no GPU, per the
``ml/agent/AGENTS.md`` rule that heavy GPU inference must go behind Pub/Sub).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import structlog

from ml.agent.context import ToolContext
from ml.agent.schemas import ClassificationResult, ClassifyParcelInput

logger = structlog.get_logger(__name__)

__all__ = ["run"]

#: Repo root resolved from this file (``ml/agent/tools/classify.py`` -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Fused tabular features parquet holding the AlphaEarth ``dim_*`` columns,
#: the PASTIS ``class_id`` and the spatial ``fold`` (1..5).
_FEATURES_PATH = _REPO_ROOT / "data" / "features" / "features_fused_pastis.parquet"

#: Column prefix of the 64-dim AlphaEarth embedding in the features parquet.
_ALPHAEARTH_PREFIX: str = "dim_"

#: Number of AlphaEarth embedding dimensions (annual Satellite Embedding V1).
_EMBED_DIM: int = 64

#: Number of contiguous agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18

#: Spatial fold held out by the harness; the classifier trains on folds 1-4 so
#: its fold-5 predictions stay leak-free (R-LEAK), matching the EPIC 6 stacking
#: base member materialization.
_HELD_OUT_FOLD: int = 5

#: Sentinel crop class emitted when the parcel has no AlphaEarth embedding (a new
#: AOI that still needs GEE sampling). It is NOT a real PASTIS class: it signals
#: the agent loop that an out-of-band embedding step is required.
_NEEDS_GEE_SAMPLING: str = "needs_gee_sampling"


class _XgbAlphaEarthClassifier:
    """Fitted XGBoost-AlphaEarth classifier with a fixed 18-class probability head.

    Wraps the sklearn estimator plus the mapping from its local class columns to
    the global ``[0, 18)`` semantic18 space, so :meth:`predict_proba_18` always
    returns a ``(n, 18)`` post-softmax row regardless of which classes the
    training folds happened to cover.

    Attributes:
        estimator: The fitted ``SpatialXGBClassifier`` (sklearn-compatible).
        global_classes: Global semantic18 class ids of the estimator's local
            ``predict_proba`` columns, in order.
        class_names: Mapping ``{global_class_id: human-readable crop name}``.
    """

    def __init__(
        self,
        estimator: object,
        global_classes: np.ndarray,
        class_names: dict[int, str],
    ) -> None:
        self.estimator = estimator
        self.global_classes = global_classes
        self.class_names = class_names

    def predict_proba_18(self, embedding: np.ndarray) -> np.ndarray:
        """Predict the full 18-class posterior for one AlphaEarth embedding.

        Args:
            embedding: A single ``(64,)`` AlphaEarth embedding vector.

        Returns:
            A ``(18,)`` ``float64`` post-softmax distribution summing to 1.
        """
        x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        x = np.where(np.isfinite(x), x, 0.0)
        proba_local = np.asarray(self.estimator.predict_proba(x), dtype=np.float64)[0]
        full = np.zeros(_NUM_CLASSES, dtype=np.float64)
        for col, gid in enumerate(self.global_classes):
            gid_int = int(gid)
            if 0 <= gid_int < _NUM_CLASSES:
                full[gid_int] = proba_local[col]
        total = full.sum()
        return full / total if total > 1e-12 else full


@functools.lru_cache(maxsize=1)
def _load_classifier() -> _XgbAlphaEarthClassifier:
    """Load (and cache) the XGBoost-AlphaEarth classifier.

    Fits an XGBoost over the AlphaEarth ``dim_*`` columns of folds 1-4 of
    ``features_fused_pastis.parquet`` and maps its labels to the contiguous
    semantic18 space, exactly as the EPIC 6 stacking ensemble materializes its
    ``xgb-alphaearth`` base member. The result is cached process-wide
    (``maxsize=1``) so the CPU-light fit happens once.

    Returns:
        A ready :class:`_XgbAlphaEarthClassifier`.

    Raises:
        FileNotFoundError: if the fused features parquet is absent (run
            ``dvc pull data/features``).
        ValueError: if the parquet lacks the required columns or fold-5 leaves no
            training rows.
    """
    import polars as pl
    from sklearn.preprocessing import LabelEncoder

    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
    from ml.data.pastis_seg_dataset import _build_semantic18_lut
    from ml.train.baseline import build_estimator

    if not _FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"AlphaEarth fused features parquet not found: {_FEATURES_PATH}. "
            "Run `dvc pull data/features` to fetch it."
        )

    df = pl.read_parquet(_FEATURES_PATH)
    for col in ("class_id", "fold"):
        if col not in df.columns:
            raise ValueError(f"features parquet is missing the `{col}` column.")
    feature_cols = [c for c in df.columns if c.startswith(_ALPHAEARTH_PREFIX)]
    if not feature_cols:
        raise ValueError(
            f"no AlphaEarth feature column with prefix {_ALPHAEARTH_PREFIX!r} in "
            f"{_FEATURES_PATH}."
        )

    # Train on folds 1-4 only (leak-free: the classifier never sees fold-5).
    train = df.filter(pl.col("fold") != _HELD_OUT_FOLD).filter(
        pl.col("class_id").is_not_null()
    )
    if train.height == 0:
        raise ValueError("no training rows on folds 1-4 in the features parquet.")

    x_train = train.select(feature_cols).to_numpy().astype(np.float64)
    x_train = np.where(np.isfinite(x_train), x_train, 0.0)

    # Map the raw PASTIS class_id (1..18) to the contiguous semantic18 space
    # [0..17] used by every ensemble member; drop Background/Void parcels (255).
    label_lut = _build_semantic18_lut(255)
    pastis_train = np.clip(
        train.get_column("class_id").to_numpy().astype(np.int64), 0, 19
    )
    y_raw = label_lut[pastis_train]
    keep = y_raw != 255
    x_train = x_train[keep]
    y_raw = y_raw[keep]
    if x_train.shape[0] == 0:
        raise ValueError("no semantic18-labelled parcels left after dropping Background/Void.")

    encoder = LabelEncoder().fit(y_raw)
    y_train = encoder.transform(y_raw).astype(np.int64)

    estimator = build_estimator(
        "xgb",
        {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "objective": "multi:softprob",
            "random_state": 42,
        },
    )
    estimator.fit(x_train, y_train)

    global_classes = encoder.classes_.astype(np.int64)
    logger.info(
        "classify_classifier_loaded",
        n_train=int(x_train.shape[0]),
        n_features=len(feature_cols),
        n_classes=int(global_classes.size),
    )
    return _XgbAlphaEarthClassifier(
        estimator=estimator,
        global_classes=global_classes,
        class_names=dict(SEMANTIC18_CLASS_NAMES),
    )


async def _fetch_parcel_embedding(ctx: ToolContext, year: int) -> np.ndarray | None:
    """Fetch the AlphaEarth embedding of the session's most recent parcel.

    The polygon-to-parcel resolution for a brand-new AOI is owned by the GEE
    sampler (out of scope here), so this tool reads the embedding from
    ``features_parcels`` for the parcels of the current session. The query is
    session-scoped: it joins ``features_parcels`` to ``parcels`` and filters by
    ``parcels.session_id`` (multi-tenant defence in depth) plus the requested
    ``year``, returning the latest non-null embedding.

    Args:
        ctx: Tool execution context (pool, session id).
        year: Campaign year of the annual embedding.

    Returns:
        A ``(64,)`` ``float64`` embedding, or ``None`` if the session has no
        parcel with a persisted embedding for that year.
    """
    from ml.agent.db import session_scoped_conn

    query = """
        SELECT fp.alphaearth_embedding
        FROM features_parcels fp
        JOIN parcels p ON p.id = fp.parcel_id
        WHERE p.session_id = $1
          AND fp.year = $2
          AND fp.alphaearth_embedding IS NOT NULL
        ORDER BY fp.updated_at DESC
        LIMIT 1
    """
    async with session_scoped_conn(ctx.session_id) as conn:
        row = await conn.fetchrow(query, ctx.session_id, year)

    if row is None or row["alphaearth_embedding"] is None:
        return None

    raw = row["alphaearth_embedding"]
    # pgvector returns the embedding as a string like "[0.1,0.2,...]" over
    # asyncpg unless a codec is registered; parse both that and native sequences.
    if isinstance(raw, str):
        values = [float(v) for v in raw.strip().strip("[]").split(",") if v.strip()]
    else:
        values = [float(v) for v in raw]
    embedding = np.asarray(values, dtype=np.float64)
    if embedding.size != _EMBED_DIM:
        logger.warning(
            "classify_embedding_unexpected_dim",
            expected=_EMBED_DIM,
            got=int(embedding.size),
        )
        return None
    return embedding


def _needs_gee_result() -> ClassificationResult:
    """Build the controlled result for a parcel without an AlphaEarth embedding.

    Returns a uniform low-confidence posterior tagged with the
    ``needs_gee_sampling`` sentinel class so the agent loop can route the request
    to the GEE sampler instead of trusting a hallucinated crop label.

    Returns:
        A :class:`ClassificationResult` with ``crop_class="needs_gee_sampling"``,
        ``confidence`` equal to a uniform prior, and a flat 18-class posterior.
    """
    uniform = 1.0 / _NUM_CLASSES
    return ClassificationResult(
        crop_class=_NEEDS_GEE_SAMPLING,
        confidence=uniform,
        class_probabilities={_NEEDS_GEE_SAMPLING: 1.0},
    )


async def run(inp: ClassifyParcelInput, ctx: ToolContext) -> ClassificationResult:
    """Classify the crop of a new parcel polygon with the XGBoost-AlphaEarth model.

    Args:
        inp: Validated arguments (session id, AOI polygon, campaign year).
        ctx: Tool execution context (asyncpg pool, settings, session id).

    Returns:
        A :class:`ClassificationResult` with the argmax crop class, its confidence
        and the full 18-class posterior. When no AlphaEarth embedding is available
        for the session's parcel (a fresh AOI), a controlled
        ``needs_gee_sampling`` result is returned instead of a guessed class.
    """
    logger.info(
        "classify_new_parcel_started",
        session_id=str(inp.session_id),
        year=inp.year,
        geometry_type=inp.aoi.type,
    )

    embedding = await _fetch_parcel_embedding(ctx, inp.year)
    if embedding is None:
        logger.info(
            "classify_new_parcel_needs_gee",
            session_id=str(inp.session_id),
            year=inp.year,
            reason="no persisted AlphaEarth embedding for the session/year",
        )
        return _needs_gee_result()

    classifier = _load_classifier()
    proba = classifier.predict_proba_18(embedding)
    top_idx = int(np.argmax(proba))
    crop_class = classifier.class_names.get(top_idx, str(top_idx))
    class_probabilities = {
        classifier.class_names.get(idx, str(idx)): float(proba[idx])
        for idx in range(_NUM_CLASSES)
    }

    result = ClassificationResult(
        crop_class=crop_class,
        confidence=float(proba[top_idx]),
        class_probabilities=class_probabilities,
    )
    logger.info(
        "classify_new_parcel_finished",
        session_id=str(inp.session_id),
        crop_class=result.crop_class,
        confidence=round(result.confidence, 4),
    )
    return result
