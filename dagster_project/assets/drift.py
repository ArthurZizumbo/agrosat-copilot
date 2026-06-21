"""Dagster asset ``drift_check`` — weekly data/prediction drift monitor (US-060).

Orchestrates the pure pipeline in ``ml.monitoring.drift`` over the project's
real ingestion parquets and publishes a weekly Evidently HTML report. The asset:

1. Reads a reference parquet (training baseline) and a current parquet (batch to
   surveil). With no date-partitioned ingestion yet, the default current set is
   a distinct slice of the same real corpus (documented Plan B in
   ``docs/blockers/epic10-notas.md``); the asset is already parametrised to take
   a true current quarter once such ingestion exists.
2. Computes KS (Sentinel-2 bands / spectral indices), MMD (AlphaEarth / FarSLIP
   embeddings) and Chi-squared (predicted classes, US-030 18-class space) drift.
3. Writes ``report_{week}.html`` to ``data/monitoring/drift/`` and, if GCS
   credentials are present, uploads it to ``gs://agrosat-reports/drift/{week}/``.
4. Emits ``MaterializeResult`` with ``drift_score``, ``data_version``,
   ``code_version`` and ``status``.
5. Fires the ``drift_notifier`` resource when ``drift_score`` exceeds
   :data:`ml.monitoring.drift.DRIFT_SCORE_THRESHOLD` (0.3).

Graceful skip (CI / dev without secrets): if the upstream parquet is missing the
asset returns ``status="skipped_no_upstream"`` with ``rows=0`` instead of
raising; a GCS auth failure degrades to a local-only report (not a failure).

AlphaEarth attribution: ``SATELLITE_EMBEDDING/V1/ANNUAL`` (data v1.1, 64-dim,
CC-BY-4.0). NOT "v2.1".
"""

import datetime as _dt
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from ml.utils.git_meta import git_sha

#: Reference (training-baseline) parquet — FarSLIP consolidated embeddings with
#: PASTIS ``class_id`` in the US-030 18-class contiguous space.
DEFAULT_REFERENCE_PARQUET = Path("data/farslip/embeddings_pastis.parquet")

#: Current (batch-to-surveil) parquet. Defaults to the same corpus until a
#: date-partitioned ingestion exists (Plan B, US-060 §5 R2).
DEFAULT_CURRENT_PARQUET = DEFAULT_REFERENCE_PARQUET

#: Local output directory for the HTML report (mirrored to GCS if ADC present).
DRIFT_REPORT_DIR = Path("data/monitoring/drift")

#: GCS destination per plan v8 §US-060 (NOT ``agrosat-artifacts``).
DRIFT_REPORT_BUCKET = "agrosat-reports"
DRIFT_REPORT_PREFIX = "drift"

#: DVC tag for the reference parquet lineage (FarSLIP consolidated, US-022b-B).
DATA_VERSION_DVC_PATH = "data/farslip/embeddings_pastis.parquet"

#: Categorical predicted-class column in the FarSLIP parquet.
CLASS_COLUMN = "class_id"

#: Rows sampled per side to keep the weekly run light (full corpus is ~82k).
MAX_ROWS_PER_SIDE = 5000


def _iso_week(now: _dt.date | None = None) -> str:
    """Return the current ISO week key ``YYYY-Www`` (e.g. ``2026-W25``)."""
    day = now or _dt.date.today()
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _upload_to_gcs(local_path: Path, blob_uri: str) -> str | None:
    """Upload ``local_path`` to ``blob_uri`` if GCS credentials are available.

    Args:
        local_path: Local HTML report path.
        blob_uri: Target ``gs://bucket/key`` URI.

    Returns:
        The ``gs://`` URI on success, or ``None`` if upload is skipped because
        credentials/network are unavailable (graceful, never raises).
    """
    from ml.utils.gcs_errors import is_gcs_auth_error

    try:
        from google.cloud import storage  # type: ignore[import-not-found]

        _, _, rest = blob_uri.partition("gs://")
        bucket_name, _, key = rest.partition("/")
        client = storage.Client()
        client.bucket(bucket_name).blob(key).upload_from_filename(str(local_path))
        return blob_uri
    except ImportError:
        return None
    except Exception as exc:
        if is_gcs_auth_error(exc):
            return None
        raise


@asset(
    deps=[
        AssetKey("farslip_embeddings_consolidated"),
        AssetKey("parcel_features_fused"),
    ],
    group_name="monitoring",
    compute_kind="evidently",
    required_resource_keys={"mlflow", "drift_notifier"},
    description=(
        "Drift detection semanal (Evidently): KS bandas Sentinel-2, MMD "
        "embeddings AlphaEarth/FarSLIP, Chi-cuadrado clases (espacio 18-clase "
        "US-030). Publica HTML en gs://agrosat-reports/drift/ y alerta si "
        "drift_score > 0.3. US-060."
    ),
)
def drift_check(context: AssetExecutionContext) -> MaterializeResult:
    """Materialize the weekly drift report over the real ingestion parquets.

    Args:
        context: Dagster context (``context.log``, ``context.resources.mlflow``,
            ``context.resources.drift_notifier``).

    Returns:
        ``MaterializeResult`` with ``drift_score``, ``n_columns_drifted``,
        ``embedding_drift``, ``report_url``, ``data_version``, ``code_version``,
        ``status``. Graceful skip (``status="skipped_no_upstream"``, ``rows=0``)
        when the reference parquet is absent.
    """
    import polars as pl

    from ml.monitoring.drift import (
        DRIFT_SCORE_THRESHOLD,
        build_drift_report,
        embedding_columns,
        exceeds_threshold,
    )
    from ml.utils.git_meta import dvc_data_version

    code_version = git_sha(short=True)
    week = _iso_week()

    reference_path = DEFAULT_REFERENCE_PARQUET
    current_path = DEFAULT_CURRENT_PARQUET
    if not reference_path.exists():
        context.log.warning(
            "drift_check.skip reference parquet ausente path=%s", str(reference_path)
        )
        return MaterializeResult(
            metadata={
                "status": MetadataValue.text("skipped_no_upstream"),
                "rows": MetadataValue.int(0),
                "week": MetadataValue.text(week),
                "code_version": MetadataValue.text(code_version),
            }
        )

    reference = pl.read_parquet(reference_path)
    current = pl.read_parquet(current_path)

    # Plan B (US-060 §5 R2): with no date-partitioned ingestion, build a real
    # contrast from the same corpus — reference = a base-class slice, current =
    # the complementary slice. Both are real rows; the detected drift is the
    # genuine class/embedding shift between PASTIS subpopulations, never
    # synthetic. Once a dated current quarter exists, point current_path at it.
    if reference_path == current_path and CLASS_COLUMN in reference.columns:
        majority_class = (
            reference.group_by(CLASS_COLUMN)
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .row(0)[0]
        )
        reference = reference.filter(pl.col(CLASS_COLUMN) != majority_class)
        current = current.filter(pl.col(CLASS_COLUMN) == majority_class)
        context.log.info(
            "drift_check.planB majority_class=%s ref_rows=%d cur_rows=%d",
            str(majority_class),
            reference.height,
            current.height,
        )

    reference = reference.head(MAX_ROWS_PER_SIDE)
    current = current.head(MAX_ROWS_PER_SIDE)

    if reference.height == 0 or current.height == 0:
        context.log.warning("drift_check.skip reference/current vacio tras slice")
        return MaterializeResult(
            metadata={
                "status": MetadataValue.text("skipped_no_upstream"),
                "rows": MetadataValue.int(0),
                "week": MetadataValue.text(week),
                "code_version": MetadataValue.text(code_version),
            }
        )

    emb_cols = embedding_columns(list(reference.columns))
    # Use the first 64 embedding dims to match the AlphaEarth 64-dim contract
    # (the FarSLIP parquet is 512-dim; the AlphaEarth block is 64-dim) and keep
    # the MMD computation bounded.
    emb_cols = emb_cols[:64]

    html, summary = build_drift_report(
        reference,
        current,
        embedding_cols=emb_cols,
        class_column=CLASS_COLUMN if CLASS_COLUMN in reference.columns else None,
    )

    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    local_report = DRIFT_REPORT_DIR / f"report_{week}.html"
    local_report.write_text(html, encoding="utf-8")

    blob_uri = f"gs://{DRIFT_REPORT_BUCKET}/{DRIFT_REPORT_PREFIX}/{week}/report.html"
    uploaded = _upload_to_gcs(local_report, blob_uri)
    report_url = uploaded or str(local_report.resolve())

    drift_score = summary.drift_score
    alert = exceeds_threshold(drift_score, DRIFT_SCORE_THRESHOLD)
    context.log.info(
        "drift_check.done week=%s drift_score=%.4f drifted=%d/%d emb_drift=%s alert=%s",
        week,
        drift_score,
        summary.n_columns_drifted,
        summary.n_columns,
        str(summary.embedding_drift),
        str(alert),
    )

    if alert:
        subject = (
            f"[AgroSatCopilot] Drift {week}: score={drift_score:.3f} > {DRIFT_SCORE_THRESHOLD}"
        )
        body = (
            f"Semana {week}\n"
            f"drift_score={drift_score:.4f} (umbral {DRIFT_SCORE_THRESHOLD})\n"
            f"columnas con drift={summary.n_columns_drifted}/{summary.n_columns}\n"
            f"embedding_drift={summary.embedding_drift} "
            f"(mmd_score={summary.embedding_mmd_score})\n"
            f"reporte={report_url}\n"
        )
        context.resources.drift_notifier.send(subject, body)

    data_version = dvc_data_version(DATA_VERSION_DVC_PATH)

    # Best-effort MLflow lineage tags (silent if the server is offline).
    try:
        context.resources.mlflow.log_metric("drift_score", float(drift_score))
        context.resources.mlflow.set_tags(
            {"data_version": data_version, "code_version": code_version, "week": week}
        )
    except Exception as exc:  # noqa: BLE001 - MLflow logging is best-effort
        context.log.debug("drift_check.mlflow_skip error=%s", str(exc))

    return MaterializeResult(
        metadata={
            "status": MetadataValue.text("ok"),
            "week": MetadataValue.text(week),
            "rows": MetadataValue.int(reference.height + current.height),
            "drift_score": MetadataValue.float(float(drift_score)),
            "n_columns": MetadataValue.int(summary.n_columns),
            "n_columns_drifted": MetadataValue.int(summary.n_columns_drifted),
            "n_embedding_dims": MetadataValue.int(summary.n_embedding_dims),
            "embedding_drift": MetadataValue.bool(bool(summary.embedding_drift)),
            "alert_triggered": MetadataValue.bool(alert),
            "report_url": MetadataValue.text(report_url),
            "report_uploaded_gcs": MetadataValue.bool(uploaded is not None),
            "data_version": MetadataValue.text(data_version),
            "code_version": MetadataValue.text(code_version),
        }
    )
