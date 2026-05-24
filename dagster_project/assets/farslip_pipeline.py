"""Assets Dagster — US-022b-B re-materializacion del pipeline FarSLIP.

Saldo la deuda de US-017 Fase 4 declarando explicitamente en Dagster los tres
artefactos del paper FarSLIP (arXiv:2511.14901) con lineage end-to-end:

::

    sentinel2_crops_256  ──┐
                            ├─► farslip_pairs_italy ──┐
    cap_vocabulary ─────────┘                          │
                                                       ├─► farslip_embeddings_italy
    farslip_clip_italy_v1 (model, MLflow Registry) ────┘                │
                                                                         │
                                                                         ▼
                                                       farslip_embeddings_consolidated
                                                       (data/farslip/embeddings_italy.parquet)

Mapeo a criterios de aceptacion (docs/us-planning/us-022b.md §3.2):

- **B-4**: ``farslip_embeddings_consolidated`` produce
  ``data/farslip/embeddings_italy.parquet`` consumido por
  ``ml.features.fusion._DEFAULT_FARSLIP_PATH``.
- **B-5**: tags ``farslip-pairs-italy-v1`` (en ``farslip_pairs_italy``),
  ``farslip-embeddings-italy-v1`` (en ``farslip_embeddings_consolidated``),
  ``farslip-student-italy-v1`` (en ``farslip_clip_italy_v1`` external asset).
- **Lineage declarativo**: ``farslip_pairs_italy`` -> ``farslip_clip_italy_v1``
  (modelo) -> ``farslip_embeddings_italy`` (extraccion) ->
  ``farslip_embeddings_consolidated`` (parquet final).
- **MLflow metrics**: vias resource ``mlflow`` (``dagster-mlflow``), tags
  ``data_version`` + ``code_version`` (regla ML/CLAUDE.md NON-NEGOTIABLE).

El asset que ejecuta training real (``farslip_clip_italy_v1``) NO se
materializa desde Dagster — el training se lanza via ``make train-l4`` sobre
GCP L4 spot (regla ml/CLAUDE.md). Dagster lo modela como ``AssetSpec`` external
con ``auto_materialize_policy=None`` para que aparezca en el lineage UI sin
intentar ejecutarse.

Importante (scope US-022b-B Dagster):

- Este archivo SOLO declara los assets nuevos y el wiring del lineage. La
  consolidacion lee los parquets ya escritos por ``farslip_embeddings_italy``
  (asset existente US-017). NO recodifica el pipeline FarSLIP — solo lo
  orquesta y publica los artefactos a las paths canonicas (B-4).
- La materializacion real (con datos en GCS y pesos student) ocurre fuera de
  esta US (Isaac + Arthur, Fase 4 022b-B). Aqui se entrega la declaracion para
  que ``dagster definitions validate`` pase y el lineage UI muestre el flujo
  completo.
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetSpec,
    MaterializeResult,
    MetadataValue,
    asset,
)

from dagster_project.assets.farslip import (
    DATA_FARSLIP_EMBEDDINGS_DIR,
    EMBEDDING_DIM,
    farslip_embeddings_italy,
)
from dagster_project.assets.farslip import (
    DATA_VERSION_TAG as EMBEDDINGS_DATA_VERSION_TAG,
)
from dagster_project.assets.sentinel2_crops import (
    DATA_FARSLIP_PAIRS_DIR,
    ITALY_REGIONS,
    sentinel2_crops_256,
)
from dagster_project.assets.sentinel2_crops import (
    DATA_VERSION_TAG as PAIRS_DATA_VERSION_TAG,
)
from ml.utils.git_meta import git_sha

#: Ruta canonica del parquet consolidado consumido por ``fusion.py``.
#: Sincronizada con ``ml.features.fusion._DEFAULT_FARSLIP_PATH``.
DATA_FARSLIP_CONSOLIDATED_PATH = Path("data/farslip/embeddings_italy.parquet")

#: Tags DVC + MLflow Registry definidos en US-022b §3.2 B-5.
PAIRS_TAG = PAIRS_DATA_VERSION_TAG  # farslip-pairs-italy-v1
EMBEDDINGS_TAG = EMBEDDINGS_DATA_VERSION_TAG  # farslip-embeddings-italy-v1
STUDENT_TAG = "farslip-student-italy-v1"  # promovido a MLflow @Production

#: URI MLflow Registry del modelo destilado (B-5).
FARSLIP_REGISTRY_URI = "models:/farslip-clip-italy-v1/Production"

#: AssetKey del modelo destilado — referenciado por
#: ``farslip_embeddings_italy`` como dep externa (lineage explicito).
FARSLIP_MODEL_ASSET_KEY = AssetKey("farslip_clip_italy_v1")


# -----------------------------------------------------------------------------
# AssetSpec externos (no materializables desde Dagster) — lineage declarativo.
# -----------------------------------------------------------------------------

#: Alias semantico de ``sentinel2_crops_256`` para alinear con el contrato del
#: plan US-022b §4.1 ("farslip_pairs_italy"). Es el mismo artefacto fisico
#: (``data/farslip_pairs/{roi}/manifest.parquet`` + crops); declararlo aqui
#: como ``AssetSpec`` external mantiene el lineage del paper visible en la UI.
farslip_pairs_italy_spec = AssetSpec(
    key=AssetKey("farslip_pairs_italy"),
    description=(
        "Alias semantico del dataset FarSLIP de pares (imagen 256x256 + texto "
        "agronomico) por ROI italiana. Materializado por sentinel2_crops_256; "
        "este AssetSpec sostiene el contrato de nombre del paper Wen et al. "
        "Tag DVC: farslip-pairs-italy-v1."
    ),
    deps=[sentinel2_crops_256],
    kinds={"polars", "geotiff"},
    group_name="farslip",
    metadata={
        "data_version": MetadataValue.text(PAIRS_TAG),
        "expected_path": MetadataValue.path(str(DATA_FARSLIP_PAIRS_DIR.resolve())),
        "alias_of": MetadataValue.text("sentinel2_crops_256"),
        "us": MetadataValue.text("US-022b-B"),
    },
)

#: Modelo destilado FarSLIP CLIP ViT-B/16 4-bandas. Vive en MLflow Registry,
#: NO se materializa desde Dagster — el training real lo lanza ``make train-l4``
#: sobre GCP L4 spot. Se declara como external AssetSpec con dep upstream a
#: ``farslip_pairs_italy`` para que el lineage UI muestre el flujo
#: ``pairs -> model -> embeddings``.
farslip_clip_italy_v1_spec = AssetSpec(
    key=FARSLIP_MODEL_ASSET_KEY,
    description=(
        "Modelo FarSLIP CLIP ViT-B/16 destilado a 4 bandas Sentinel-2 "
        "(arXiv:2511.14901). Entrenado en GCP L4 spot (US-022b-A), registrado "
        "en MLflow Registry como farslip-clip-italy-v1@Production. Tag DVC: "
        "farslip-student-italy-v1. Lineage upstream: farslip_pairs_italy."
    ),
    deps=[AssetKey("farslip_pairs_italy")],
    kinds={"mlflow", "pytorch"},
    group_name="farslip",
    metadata={
        "data_version": MetadataValue.text(STUDENT_TAG),
        "registry_uri": MetadataValue.text(FARSLIP_REGISTRY_URI),
        "experiment": MetadataValue.text("farslip-clip-italy"),
        "run_name": MetadataValue.text("farslip-clip-italy-v1"),
        "training_window": MetadataValue.text("GCP L4 spot ~6h ~$1.7 USD"),
        "us": MetadataValue.text("US-022b-B"),
    },
)


# -----------------------------------------------------------------------------
# Asset materializable: consolida los embeddings por (roi, year) en un unico
# parquet consumido por ml/features/fusion.py.
# -----------------------------------------------------------------------------


def _resolve_consolidated_path() -> Path:
    """Resuelve la ruta consolidada relativa al cwd.

    Returns:
        ``Path`` absoluto de ``data/farslip/embeddings_italy.parquet`` listo
        para ``parent.mkdir(parents=True, exist_ok=True)``.
    """
    return DATA_FARSLIP_CONSOLIDATED_PATH


def _iter_partition_parquets(
    embeddings_root: Path,
) -> list[tuple[str, int, Path]]:
    """Itera los parquets escritos por ``farslip_embeddings_italy``.

    Args:
        embeddings_root: raiz ``data/farslip_embeddings/`` con layout
            ``{roi}/{year}/embeddings.parquet`` (output del asset upstream
            particionado por ROI).

    Returns:
        Lista de tuplas ``(roi, year, path)`` ordenada por roi luego year.
        Vacia si ``embeddings_root`` no existe.
    """
    if not embeddings_root.exists():
        return []
    found: list[tuple[str, int, Path]] = []
    for roi_dir in sorted(embeddings_root.iterdir()):
        if not roi_dir.is_dir():
            continue
        for year_dir in sorted(roi_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            parquet_path = year_dir / "embeddings.parquet"
            if not parquet_path.exists():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            found.append((roi_dir.name, year, parquet_path))
    return found


@asset(
    deps=[farslip_embeddings_italy, FARSLIP_MODEL_ASSET_KEY],
    group_name="farslip",
    compute_kind="polars",
    required_resource_keys={"mlflow"},
    description=(
        "Consolida los embeddings FarSLIP 512-dim de las 3 ROIs italianas en "
        "un unico parquet ``data/farslip/embeddings_italy.parquet`` (B-4 del "
        "plan US-022b). Anade columna ``region`` y persiste con schema "
        "compatible con ``ml/features/fusion.py``. Registra metrics + tags en "
        "MLflow (data_version, code_version, n_embeddings, embedding_dim). "
        "Tag DVC: farslip-embeddings-italy-v1."
    ),
)
def farslip_embeddings_consolidated(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Consolida embeddings por (roi, year) en ``data/farslip/embeddings_italy.parquet``.

    Lee los parquets escritos por las particiones de
    ``farslip_embeddings_italy`` (``data/farslip_embeddings/{roi}/{year}/``),
    los concatena con Polars (NO pandas — regla ML CLAUDE.md), anade columna
    ``region`` y persiste a la ruta canonica consumida por ``fusion.py``.

    Args:
        context: contexto Dagster. ``context.resources.mlflow`` provee el
            cliente MLflow para registrar metrics + tags (B-5).

    Returns:
        ``MaterializeResult`` con metadata ``rows``, ``embedding_dim``,
        ``rois``, ``output_path``, ``data_version`` (DVC tag),
        ``code_version`` (git SHA short).
        Si no hay parquets upstream: ``status="skipped_no_upstream"`` y
        ``rows=0`` (no error — el extractor FarSLIP puede haber skipped por
        GCS auth en CI).

    Notes:
        Schema del parquet final:
        ``{parcel_id: int64, region: str, embedding: list[float32]}``.
        Mapea ``crop_id`` -> ``parcel_id`` (cast int via Path.stem). El cast
        es defensivo: si el ``crop_id`` no es numerico se usa hash truncado.
    """
    import polars as pl

    output_path = _resolve_consolidated_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    code_version = git_sha(short=True)
    embeddings_root = DATA_FARSLIP_EMBEDDINGS_DIR

    context.log.info(
        "farslip_embeddings_consolidated.start root=%s output=%s",
        embeddings_root,
        output_path,
    )

    partitions = _iter_partition_parquets(embeddings_root)
    if not partitions:
        context.log.warning(
            "farslip_embeddings_consolidated: no se encontraron parquets en "
            "%s. Materializa primero farslip_embeddings_italy para las 3 "
            "ROIs (pianura_padana, toscana, puglia).",
            embeddings_root,
        )
        return MaterializeResult(
            metadata={
                "status": MetadataValue.text("skipped_no_upstream"),
                "rows": MetadataValue.int(0),
                "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
                "rois": MetadataValue.text(""),
                "data_version": MetadataValue.text(EMBEDDINGS_TAG),
                "code_version": MetadataValue.text(code_version),
                "output_path": MetadataValue.path(str(output_path.resolve())),
            }
        )

    frames: list[pl.DataFrame] = []
    rois_seen: set[str] = set()
    years_seen: set[int] = set()
    for roi, year, parquet_path in partitions:
        df = pl.read_parquet(parquet_path)
        # Anade columnas region/year (compat con fusion.py LEFT JOIN por parcel_id).
        df = df.with_columns(
            pl.lit(roi).alias("region"),
            pl.lit(year).cast(pl.Int32).alias("year"),
        )
        frames.append(df)
        rois_seen.add(roi)
        years_seen.add(year)
        context.log.info(
            "farslip_embeddings_consolidated.append roi=%s year=%d rows=%d",
            roi,
            year,
            df.height,
        )

    consolidated = pl.concat(frames, how="diagonal_relaxed")
    n_rows = consolidated.height

    # Garantiza el contrato con fusion.py: columna ``parcel_id`` numerica.
    # ``crop_id`` del upstream tiene formato libre (Path.stem); intentamos cast
    # int sin perder filas — fallback a hash si no es numerico.
    if "crop_id" in consolidated.columns and "parcel_id" not in consolidated.columns:
        consolidated = consolidated.with_columns(
            pl.col("crop_id").cast(pl.Int64, strict=False).alias("parcel_id"),
        )

    consolidated.write_parquet(output_path, compression="zstd")

    context.log.info(
        "farslip_embeddings_consolidated.complete rows=%d rois=%s years=%s output=%s",
        n_rows,
        sorted(rois_seen),
        sorted(years_seen),
        output_path,
    )

    # B-5: metrics + tags MLflow via resource dagster-mlflow.
    # El resource gestiona el run; aqui solo emitimos params/metrics.
    mlflow_client = context.resources.mlflow
    try:
        mlflow_client.log_metric("n_embeddings", float(n_rows))
        mlflow_client.log_metric("embedding_dim", float(EMBEDDING_DIM))
        mlflow_client.log_metric("n_rois", float(len(rois_seen)))
        mlflow_client.log_param("data_version", EMBEDDINGS_TAG)
        mlflow_client.log_param("code_version", code_version)
        mlflow_client.log_param("model_version", STUDENT_TAG)
        mlflow_client.log_param("pairs_version", PAIRS_TAG)
        mlflow_client.set_tag("us", "US-022b-B")
        mlflow_client.set_tag("pipeline", "farslip")
    except Exception as exc:  # noqa: BLE001 — MLflow offline no debe romper la materializacion
        context.log.warning(
            "farslip_embeddings_consolidated: MLflow logging failed (offline?) %s: %s",
            type(exc).__name__,
            exc,
        )

    return MaterializeResult(
        metadata={
            "rows": MetadataValue.int(n_rows),
            "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
            "rois": MetadataValue.text(",".join(sorted(rois_seen))),
            "years": MetadataValue.text(",".join(str(y) for y in sorted(years_seen))),
            "n_partitions_in": MetadataValue.int(len(partitions)),
            "output_path": MetadataValue.path(str(output_path.resolve())),
            "data_version": MetadataValue.text(EMBEDDINGS_TAG),
            "model_version": MetadataValue.text(STUDENT_TAG),
            "pairs_version": MetadataValue.text(PAIRS_TAG),
            "code_version": MetadataValue.text(code_version),
            "mlflow_run_name": MetadataValue.text("farslip-clip-italy-v1"),
            "consumed_by": MetadataValue.text(
                "ml.features.fusion.build_fused_features(include_farslip=True)"
            ),
        }
    )


__all__ = [
    "DATA_FARSLIP_CONSOLIDATED_PATH",
    "EMBEDDINGS_TAG",
    "FARSLIP_MODEL_ASSET_KEY",
    "FARSLIP_REGISTRY_URI",
    "ITALY_REGIONS",
    "PAIRS_TAG",
    "STUDENT_TAG",
    "farslip_clip_italy_v1_spec",
    "farslip_embeddings_consolidated",
    "farslip_pairs_italy_spec",
]
