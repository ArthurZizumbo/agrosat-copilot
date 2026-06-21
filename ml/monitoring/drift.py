"""Evidently-based drift detection pipeline (US-060, plan v8 §US-060).

Pure, framework-free functions that compare a *reference* distribution (the
training baseline) against a *current* batch and quantify distribution drift.
The Dagster asset ``drift_check`` orchestrates these functions weekly; tests in
``tests/ml/monitoring/test_drift.py`` exercise them on real repo parquets.

Statistical tests per feature family (Evidently 0.7.x modern API):

- Numerical Sentinel-2 bands / spectral indices -> **Kolmogorov-Smirnov (KS)**
  two-sample test (``num_method="ks"`` in :class:`DataDriftPreset`).
- AlphaEarth / FarSLIP embeddings -> **Maximum Mean Discrepancy (MMD)** via the
  :class:`~evidently.metrics.EmbeddingsDrift` metric (Evidently's native MMD
  detector over the grouped embedding columns). No proxy is used: MMD is the
  real method exposed by Evidently 0.7.21.
- Predicted classes -> **Chi-squared** categorical drift test
  (``cat_method="chisquare"``), evaluated on the US-030 18-class contiguous
  space or the US-074 HCAT macro space when the current set is multi-region.

The global ``drift_score`` is the share of monitored columns flagged as drifted
by Evidently's ``DriftedColumnsCount`` (range ``[0, 1]``). The alert threshold
is :data:`DRIFT_SCORE_THRESHOLD` (``0.3``).

AlphaEarth attribution: ``SATELLITE_EMBEDDING/V1/ANNUAL`` (data v1.1, 64-dim,
global incl. Mexico, CC-BY-4.0). NOT "v2.1".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids hard import at module load
    import polars as pl

_log = structlog.get_logger(__name__)

#: Alert threshold for the global drift score (plan v8 §US-060: "alerta si
#: drift score > 0.3"). Comparison is strict (``>``).
DRIFT_SCORE_THRESHOLD: float = 0.3

#: KS / Chi-squared per-column p-value threshold Evidently uses to flag a single
#: column as drifted (Evidently default for these stat tests).
_COLUMN_PVALUE_THRESHOLD: float = 0.05

#: Decision threshold for the MMD embedding drift score returned by Evidently's
#: ``EmbeddingsDrift`` metric. That score is a separability measure in ``[0, 1]``
#: where ``~0.5`` means the reference and current embedding clouds are
#: indistinguishable (no drift) and higher values mean the distributions pull
#: apart. Empirically (US-060 recon on the FarSLIP parquet) identical clouds
#: score ``~0.48`` and a single-class shift scores ``~0.79``; ``0.55`` is a
#: conservative ceiling above the no-drift baseline. Documented in
#: ``docs/blockers/epic10-notas.md``.
_EMBEDDING_MMD_DRIFT_THRESHOLD: float = 0.55

#: Logical name of the embedding group registered in the Evidently
#: ``DataDefinition`` (used both for MMD config and report wiring).
_EMBEDDING_GROUP: str = "embeddings"


@dataclass(frozen=True)
class DriftSummary:
    """Structured result of a drift comparison.

    Attributes:
        drift_score: Share of monitored columns flagged as drifted in
            ``[0, 1]`` (Evidently ``DriftedColumnsCount.share``). This is the
            scalar compared against :data:`DRIFT_SCORE_THRESHOLD`.
        n_columns: Total number of monitored columns (numerical + categorical).
        n_columns_drifted: Number of columns flagged as drifted.
        column_drift: Mapping ``column -> drift_detected`` per feature.
        column_pvalues: Mapping ``column -> p-value`` for the per-column stat
            test (KS for bands, Chi-squared for classes). ``None`` for the
            embedding group (MMD does not return a p-value).
        embedding_drift: ``True``/``False``/``None`` MMD verdict for the grouped
            embedding columns, or ``None`` when no embeddings were monitored.
        embedding_mmd_score: Raw MMD separability score in ``[0, 1]`` from
            Evidently's ``EmbeddingsDrift`` (``~0.5`` = no drift), or ``None``.
        n_embedding_dims: Number of embedding dimensions monitored (e.g. 64 for
            AlphaEarth, 512 for FarSLIP), ``0`` when none.
    """

    drift_score: float
    n_columns: int
    n_columns_drifted: int
    column_drift: dict[str, bool] = field(default_factory=dict)
    column_pvalues: dict[str, float | None] = field(default_factory=dict)
    embedding_drift: bool | None = None
    embedding_mmd_score: float | None = None
    n_embedding_dims: int = 0


def embedding_columns(columns: list[str], prefixes: tuple[str, ...] = ("emb_", "ae_")) -> list[str]:
    """Return the embedding columns among ``columns`` by name prefix.

    Args:
        columns: Candidate column names (e.g. a parquet schema).
        prefixes: Case-insensitive prefixes identifying embedding dimensions.
            Defaults cover AlphaEarth (``ae_``) and FarSLIP/consolidated
            (``emb_``) conventions used across the repo.

    Returns:
        Embedding column names in their original order.
    """
    lowered = tuple(p.lower() for p in prefixes)
    return [c for c in columns if c.lower().startswith(lowered)]


def _to_pandas(frame: pl.DataFrame, columns: list[str]) -> Any:
    """Convert the selected columns of a Polars frame to pandas.

    Conversion to pandas happens only at the Evidently boundary; the rest of the
    pipeline stays Polars (``ml/`` rule). Embedding columns are kept as float to
    satisfy Evidently's numeric expectations.

    Args:
        frame: Source Polars DataFrame.
        columns: Columns to keep (drops any that are absent defensively).

    Returns:
        A ``pandas.DataFrame`` with exactly the present requested columns.
    """
    present = [c for c in columns if c in frame.columns]
    return frame.select(present).to_pandas()


def build_drift_report(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    band_columns: list[str] | None = None,
    embedding_cols: list[str] | None = None,
    class_column: str | None = None,
) -> tuple[str, DriftSummary]:
    """Compute drift between ``reference`` and ``current`` and render HTML.

    Builds a single Evidently ``Report`` combining KS (numerical bands),
    Chi-squared (predicted classes) and MMD (embedding group), runs it on the
    two datasets and extracts a :class:`DriftSummary` plus the HTML report body.

    Args:
        reference: Baseline distribution (e.g. training set / base year).
        current: Batch under surveillance (e.g. new region / current quarter).
        band_columns: Numerical Sentinel-2 band / spectral index columns to test
            with KS. If ``None``, every numerical column not in
            ``embedding_cols`` and not equal to ``class_column`` is used.
        embedding_cols: Embedding dimension columns (AlphaEarth 64-dim / FarSLIP
            512-dim) tested as a group with MMD. If ``None``, columns are
            detected by prefix via :func:`embedding_columns`.
        class_column: Categorical predicted-class column tested with Chi-squared.
            ``None`` to skip categorical drift.

    Returns:
        Tuple ``(html, summary)`` where ``html`` is the standalone Evidently
        report and ``summary`` is the :class:`DriftSummary`.

    Raises:
        ValueError: If no monitored column can be derived from the inputs
            (nothing numerical, categorical nor embedding to compare).
    """
    # Deferred imports: Evidently / pandas / numpy are heavy and must not load at
    # module import time (keeps ``dagster definitions validate`` and ml imports
    # cheap; mirrors the lazy-import convention of the Dagster assets).
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import EmbeddingsDrift
    from evidently.presets import DataDriftPreset

    ref_cols = list(reference.columns)
    cur_cols = list(current.columns)
    common = set(ref_cols) & set(cur_cols)

    # Resolve explicit band columns first so auto-detected embeddings can exclude
    # them (a column is either a KS band or part of the MMD embedding group,
    # never both — otherwise Evidently sees a duplicate projection).
    explicit_bands = band_columns is not None
    if explicit_bands:
        band_columns = [c for c in (band_columns or []) if c in common]

    if embedding_cols is None:
        embedding_cols = embedding_columns(ref_cols)
    embedding_cols = [
        c
        for c in embedding_cols
        if c in common and c != class_column and c not in (band_columns or [])
    ]

    if not explicit_bands:
        numeric_dtypes = {"Int", "Float", "UInt"}
        band_columns = [
            c
            for c in ref_cols
            if c in common
            and c not in embedding_cols
            and c != class_column
            and any(str(reference.schema[c]).startswith(p) for p in numeric_dtypes)
        ]
    band_columns = band_columns or []

    if class_column is not None and class_column not in (set(ref_cols) & set(cur_cols)):
        _log.warning("drift.class_column_absent", class_column=class_column)
        class_column = None

    if not band_columns and not embedding_cols and class_column is None:
        raise ValueError(
            "No monitorable columns: provide band_columns, embedding_cols or class_column"
        )

    _log.info(
        "drift.build_report.start",
        n_bands=len(band_columns),
        n_embedding_dims=len(embedding_cols),
        has_class=class_column is not None,
        n_reference=reference.height,
        n_current=current.height,
    )

    keep = [*band_columns, *embedding_cols]
    if class_column is not None:
        keep.append(class_column)
    ref_pd = _to_pandas(reference, keep)
    cur_pd = _to_pandas(current, keep)

    # Predicted classes are categorical regardless of their physical dtype
    # (class_id is Int in the FarSLIP parquet) -> cast to str so Evidently runs
    # the Chi-squared test, not a numeric KS test.
    if class_column is not None:
        ref_pd[class_column] = ref_pd[class_column].astype(str)
        cur_pd[class_column] = cur_pd[class_column].astype(str)

    embeddings_map = {_EMBEDDING_GROUP: embedding_cols} if embedding_cols else None
    data_definition = DataDefinition(
        numerical_columns=band_columns or None,
        categorical_columns=[class_column] if class_column else None,
        embeddings=embeddings_map,
    )
    ref_ds = Dataset.from_pandas(ref_pd, data_definition=data_definition)
    cur_ds = Dataset.from_pandas(cur_pd, data_definition=data_definition)

    metrics: list[Any] = [
        DataDriftPreset(
            columns=(band_columns + ([class_column] if class_column else [])) or None,
            num_method="ks",
            cat_method="chisquare",
        )
    ]
    if embedding_cols:
        # Evidently's EmbeddingsDrift defaults to the MMD detector over the
        # grouped embedding columns (plan v8 AC-1: AlphaEarth embeddings -> MMD).
        metrics.append(EmbeddingsDrift(embeddings_name=_EMBEDDING_GROUP))

    snapshot = Report(metrics=metrics).run(reference_data=ref_ds, current_data=cur_ds)
    summary = _summarize(snapshot.dict(), n_embedding_dims=len(embedding_cols))
    html = snapshot.get_html_str(as_iframe=False)

    _log.info(
        "drift.build_report.done",
        drift_score=summary.drift_score,
        n_columns=summary.n_columns,
        n_columns_drifted=summary.n_columns_drifted,
        embedding_drift=summary.embedding_drift,
    )
    return html, summary


def _summarize(report_dict: dict[str, Any], *, n_embedding_dims: int) -> DriftSummary:
    """Reduce Evidently's report dict to a :class:`DriftSummary`.

    Args:
        report_dict: ``Snapshot.dict()`` output of the drift report.
        n_embedding_dims: Number of embedding dimensions monitored.

    Returns:
        A populated :class:`DriftSummary`.
    """
    drift_score = 0.0
    n_columns = 0
    n_columns_drifted = 0
    column_drift: dict[str, bool] = {}
    column_pvalues: dict[str, float | None] = {}
    embedding_drift: bool | None = None
    embedding_mmd_score: float | None = None

    for metric in report_dict.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        config = metric.get("config", {})
        value = metric.get("value")

        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drift_score = float(value.get("share", 0.0))
                n_columns_drifted = int(value.get("count", 0))
            continue

        if name.startswith("ValueDrift"):
            column = str(config.get("column", name))
            n_columns += 1
            pvalue = float(value) if isinstance(value, (int, float)) else None
            column_pvalues[column] = pvalue
            drifted = pvalue is not None and pvalue < _COLUMN_PVALUE_THRESHOLD
            column_drift[column] = drifted
            continue

        if name.startswith("EmbeddingsDrift"):
            # Evidently returns a continuous MMD separability score in [0, 1]
            # (~0.5 == indistinguishable clouds == no drift). We keep the raw
            # score and derive a verdict against a documented threshold instead
            # of trusting truthiness of a float.
            if isinstance(value, (int, float)):
                embedding_mmd_score = float(value)
                embedding_drift = embedding_mmd_score > _EMBEDDING_MMD_DRIFT_THRESHOLD
            continue

    # Recompute the column count from per-column ValueDrift entries when the
    # DriftedColumnsCount did not arrive (defensive); otherwise keep its share.
    if n_columns == 0 and column_drift:
        n_columns = len(column_drift)

    return DriftSummary(
        drift_score=drift_score,
        n_columns=n_columns,
        n_columns_drifted=n_columns_drifted,
        column_drift=column_drift,
        column_pvalues=column_pvalues,
        embedding_drift=embedding_drift,
        embedding_mmd_score=embedding_mmd_score,
        n_embedding_dims=n_embedding_dims,
    )


def extract_drift_score(summary: DriftSummary) -> float:
    """Return the scalar global drift score from a :class:`DriftSummary`.

    Args:
        summary: Result of :func:`build_drift_report`.

    Returns:
        The share of drifted columns in ``[0, 1]``.
    """
    return summary.drift_score


def exceeds_threshold(score: float, threshold: float = DRIFT_SCORE_THRESHOLD) -> bool:
    """Whether a drift score crosses the alert threshold.

    Args:
        score: Global drift score in ``[0, 1]``.
        threshold: Alert threshold (default :data:`DRIFT_SCORE_THRESHOLD`).

    Returns:
        ``True`` if ``score > threshold`` (strict, per plan v8 "score > 0.3").
    """
    return score > threshold
