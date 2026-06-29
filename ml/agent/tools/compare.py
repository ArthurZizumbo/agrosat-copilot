"""``compare_models`` tool: cross-model prediction comparison (deferred).

Compares the per-parcel crop predictions of several EPIC 5/6 models for a single
parcel, reading the out-of-fold (OOF) parquet artifacts already dumped by US-031
(``ml/eval/oof/oof_parcel_<model>_fold5.parquet``). For each requested model it:

1. Loads its parcel OOF parquet (post-softmax ``prob_000..prob_017`` per parcel).
2. Selects the row whose ``canonical_parcel_id`` matches the requested parcel and
   takes its argmax class as that model's prediction.
3. A model whose OOF parquet is missing -- or that has no row for the parcel -- is
   omitted from the comparison with a structured warning (never fabricated).

``agreement`` is the fraction of compared models that predict the MAJORITY class
(the modal prediction), in ``[0, 1]``. With two models it is ``1.0`` when they
agree and ``0.5`` when they disagree.

Deferred tool: it is registered ``deferred=True`` so it does not block the demo's
synchronous loop. If a deferred executor (``ctx.defer``) is wired by the agent
loop, the comparison is enqueued and its handle returned via the controlled
:class:`~ml.agent.schemas.ModelComparison`; otherwise the comparison is computed
inline from the local OOF parquets (they are CPU-light parquet reads, no GPU).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from ml.agent.context import ToolContext
from ml.agent.schemas import CompareModelsInput, ModelComparison
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

logger = structlog.get_logger(__name__)

__all__ = ["run"]

#: Repo root resolved from this file (``ml/agent/tools/compare.py`` -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Directory holding the US-031 per-parcel OOF parquet artifacts.
_OOF_DIR = _REPO_ROOT / "ml" / "eval" / "oof"

#: Held-out fold of the dumped OOF parquets (anti-leakage: fold-5 only).
_HELD_OUT_FOLD: int = 5

#: Canonical key column shared by every parcel OOF frame.
_KEY: str = "canonical_parcel_id"

#: Background job name used when a deferred executor is wired.
_DEFER_JOB: str = "compare_models"


def _empty_comparison(parcel_id: int) -> ModelComparison:
    """Return the controlled empty comparison for a parcel.

    Args:
        parcel_id: Parcel the (empty) comparison refers to.

    Returns:
        A :class:`ModelComparison` with no predictions and zero agreement, used
        both when the parcel is not visible to the session and when no requested
        model had a usable OOF row for it.
    """
    return ModelComparison(parcel_id=parcel_id, predictions={}, agreement=0.0)


async def _parcel_belongs_to_session(ctx: ToolContext, parcel_id: int) -> bool:
    """Check the parcel belongs to the current session (multi-tenant gate).

    The OOF parquets are global artifacts keyed only by ``parcel_id``; reading
    them for an arbitrary id would leak another tenant's predictions. Before any
    OOF read this resolves the id against ``parcels`` inside a session-scoped
    connection, mirroring ``explain._fetch_parcel`` (``WHERE id=$1 AND
    session_id=$2``). The query is parametrised ($1/$2), never f-string built.

    Args:
        ctx: Tool execution context (pool, session id).
        parcel_id: Parcel whose session ownership is verified.

    Returns:
        ``True`` if a ``parcels`` row with that id exists for the session,
        ``False`` otherwise.
    """
    from ml.agent.db import session_scoped_conn

    query = "SELECT 1 FROM parcels WHERE id = $1 AND session_id = $2 LIMIT 1"
    async with session_scoped_conn(ctx.session_id) as conn:
        row = await conn.fetchrow(query, parcel_id, ctx.session_id)
    return row is not None


def _oof_parquet_path(model: str) -> Path:
    """Return the per-parcel OOF parquet path of a model member.

    Args:
        model: Model member name (e.g. ``"tsvit-pheno"``, ``"utae"``,
            ``"xgb-alphaearth"``).

    Returns:
        The path ``ml/eval/oof/oof_parcel_<model>_fold5.parquet``.
    """
    return _OOF_DIR / f"oof_parcel_{model}_fold{_HELD_OUT_FOLD}.parquet"


def _predict_for_parcel(model: str, canonical_id: str) -> int | None:
    """Read a model's OOF parquet and return its argmax class for one parcel.

    Args:
        model: Model member name.
        canonical_id: Canonical parcel id (``"{patch}_{local}"`` or the Utf8 form
            of a numeric id) to look up in the parquet.

    Returns:
        The argmax class index (``0..17``) the model predicted for the parcel, or
        ``None`` if the OOF parquet is missing or has no row for the parcel.
    """
    path = _oof_parquet_path(model)
    if not path.exists():
        logger.warning(
            "compare_models_oof_missing",
            model=model,
            path=str(path),
            note="OOF parquet absent; model omitted from the comparison.",
        )
        return None

    frame = canonical_parcel_id(pl.read_parquet(path), col=_KEY)
    row = frame.filter(pl.col(_KEY) == canonical_id)
    if row.height == 0:
        logger.warning(
            "compare_models_parcel_absent",
            model=model,
            canonical_parcel_id=canonical_id,
            note="parcel not present in this model's OOF; model omitted.",
        )
        return None

    probs = row.select(PROB_COLUMNS).to_numpy().astype(np.float64)[0]
    return int(np.argmax(probs))


def _compute_comparison(
    inp: CompareModelsInput, canonical_id: str | None = None
) -> ModelComparison:
    """Compute the cross-model comparison from the local OOF parquets.

    Args:
        inp: Validated arguments (parcel id and the model names to compare).
        canonical_id: The parcel's stored canonical PASTIS-R OOF key
            (``parcels.canonical_parcel_id``, US-079) when present; ``None`` falls
            back to the Utf8 cast of the integer parcel id (legacy behaviour).

    Returns:
        A :class:`ModelComparison` whose ``predictions`` maps each model with a
        usable OOF row to its predicted crop class label, and whose ``agreement``
        is the fraction of those models predicting the majority class.
    """
    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

    # Resolve the OOF key: prefer the stored canonical PASTIS-R id, falling back to
    # the Utf8 cast of the integer parcel id (which only matches numeric OOF keys).
    if canonical_id is None:
        canonical_id = canonical_parcel_id(pl.DataFrame({_KEY: [inp.parcel_id]}), col=_KEY)[_KEY][0]

    predictions: dict[str, str] = {}
    class_ids: list[int] = []
    for model in inp.models:
        class_idx = _predict_for_parcel(model, canonical_id)
        if class_idx is None:
            continue
        predictions[model] = SEMANTIC18_CLASS_NAMES.get(class_idx, str(class_idx))
        class_ids.append(class_idx)

    if not class_ids:
        logger.warning(
            "compare_models_no_usable_models",
            parcel_id=inp.parcel_id,
            requested=inp.models,
            note="no requested model had a usable OOF row for the parcel.",
        )
        return _empty_comparison(inp.parcel_id)

    majority_count = Counter(class_ids).most_common(1)[0][1]
    agreement = majority_count / len(class_ids)

    logger.info(
        "compare_models_computed",
        parcel_id=inp.parcel_id,
        n_compared=len(class_ids),
        agreement=round(agreement, 4),
    )
    return ModelComparison(
        parcel_id=inp.parcel_id,
        predictions=predictions,
        agreement=float(agreement),
    )


async def run(inp: CompareModelsInput, ctx: ToolContext) -> ModelComparison:
    """Compare several models' crop predictions for one parcel (deferred tool).

    When a deferred executor is wired (``ctx.defer`` is not ``None``), the
    comparison is enqueued as a background job and its handle is logged; the
    comparison itself is still computed inline from the local OOF parquets (CPU
    only) so the agent always receives a typed result. When no executor is wired,
    the inline computation is the sole path -- the tool never crashes for a
    missing deferred backend (controlled degradation per ``ml/agent/AGENTS.md``).

    Args:
        inp: Validated arguments (parcel id and the >=2 unique model names).
        ctx: Tool execution context (carries the optional ``defer`` hook).

    Returns:
        A :class:`ModelComparison` with the per-model predicted crop classes and
        the majority-agreement fraction. Models without a usable OOF row are
        omitted (logged), so ``predictions`` may have fewer keys than requested.
        If the parcel does not belong to the current session, the controlled
        empty comparison (``predictions={}``, ``agreement=0.0``) is returned.
    """
    logger.info(
        "compare_models_started",
        parcel_id=inp.parcel_id,
        models=inp.models,
        deferred_executor=ctx.defer is not None,
    )

    # Multi-tenant gate (NON-NEGOTIABLE): the OOF parquets are global and keyed
    # only by parcel id, so a parcel must be resolved against the session before
    # any OOF read or deferred enqueue. A parcel not visible to the session
    # yields the controlled empty comparison (never another tenant's prediction).
    if not await _parcel_belongs_to_session(ctx, inp.parcel_id):
        logger.warning(
            "compare_models_parcel_not_in_session",
            session_id=str(ctx.session_id),
            parcel_id=inp.parcel_id,
            note="parcel not visible to the session; returning empty comparison.",
        )
        return _empty_comparison(inp.parcel_id)

    # Resolve the parcel's canonical PASTIS-R OOF key (US-079) so the OOF lookup
    # hits the real fold-5 row; ``None`` falls back to the legacy numeric cast.
    from ml.agent.tools.classify import fetch_canonical_parcel_id

    canonical_id = await fetch_canonical_parcel_id(ctx, inp.parcel_id)

    if ctx.defer is not None:
        handle = await ctx.defer(
            _DEFER_JOB,
            {"parcel_id": inp.parcel_id, "models": list(inp.models)},
        )
        logger.info(
            "compare_models_deferred_enqueued",
            parcel_id=inp.parcel_id,
            handle=str(handle),
        )

    return _compute_comparison(inp, canonical_id)
