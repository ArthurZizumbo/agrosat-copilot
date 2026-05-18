"""Assets Dagster para US-017 — Bulk extraction de embeddings FarSLIP.

Materializa el asset downstream ``farslip_embeddings_italy`` particionado
por ROI italiana. Para cada partition:

1. Lee el manifest ``data/farslip_pairs/{roi}/manifest.parquet`` (output
   del asset upstream ``sentinel2_crops_256``) con Polars.
2. Instancia el ``FarSLIPExtractor`` cargando pesos student desde
   ``gs://agrosat-models/farslip/farslip-clip-italy-v1/`` (cache local en
   ``~/.cache/agrosat/farslip/``).
3. Itera los crops en batches (default 32) y extrae embeddings 512-dim.
4. Persiste a ``data/farslip_embeddings/{roi}/{year}/embeddings.parquet``
   con schema ``{crop_id: str, embedding: list[float32], crop_doy: int,
   cap_class: str}``.

Lineage declarada vía ``deps=[sentinel2_crops_256]`` para que Dagster
materialice automáticamente el upstream si no está fresco.

Producción (NO disponible en CI ni dev local sin GCS creds):
    Se inyectaría un ``GCSResource`` desde ``dagster_project/resources/``
    para autenticar la descarga de pesos del Model Registry MLflow y
    para persistir embeddings a ``gs://agrosat-features/farslip/`` vía
    DVC remote. En US-017 el extractor gestiona GCS internamente
    (cache local) — la inyección formal del resource queda para US-025.

Smoke / dev local:
    Si ``FarSLIPExtractor`` falla porque GCS no es accesible (creds
    ausentes, offline) la materialización devuelve un ``MaterializeResult``
    con ``status="skipped_no_gcs"`` y warning — NO falla. Esto permite que
    `make check` y la CI sin secrets GCS pasen el smoke de Dagster.

Integración MLflow (documentada, no implementada en US-017):
    El extractor lee el run ``farslip-clip-italy-v1`` del Model Registry
    y aplica los tags ``data_version=farslip-pairs-italy-v1`` +
    ``code_version=<git_sha>`` a los embeddings persistidos. Se podría
    materializar también un MLflow run por asset para tracking explícito
    del bulk extraction; en US-017 se difiere a US-025 cuando el cabezal
    SegFormer consuma estos embeddings.
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from dagster_project.assets.sentinel2_crops import (
    DATA_FARSLIP_PAIRS_DIR,
    ITALY_REGIONS,
    sentinel2_crops_256,
)
from ml.utils.gcs_errors import is_gcs_auth_error
from ml.utils.git_meta import git_sha

#: Rutas relativas al cwd. Persistencia particionada por ROI y año (anio
#: derivado del manifest, no hardcoded — Q4 fix).
DATA_FARSLIP_EMBEDDINGS_DIR = Path("data/farslip_embeddings")

#: URI del student FarSLIP en GCS (MLflow artifact + DVC tag).
#: En CI sin creds GCS el extractor cae a cache local o lanza
#: DefaultCredentialsError; en ese caso el asset reporta skipped.
DEFAULT_WEIGHTS_URI = "gs://agrosat-models/farslip/farslip-clip-italy-v1/"

#: Tag de version del banco de embeddings (DVC).
DATA_VERSION_TAG = "farslip-embeddings-italy-v1"

#: Dimension del embedding student (CLIP ViT-B/16 projection head).
EMBEDDING_DIM = 512

#: Tamano de batch para bulk extraction (cabe holgado en L4 24 GB).
EXTRACTION_BATCH_SIZE = 32

#: Anio fallback si el manifest no expone ``crop_year``. Solo se usa cuando
#: el manifest carece de la columna; los anios reales del manifest priman.
FALLBACK_YEAR = 2024


def _skipped_result(context: AssetExecutionContext, reason: str, roi: str) -> MaterializeResult:
    """Construye un MaterializeResult de skip uniforme para fallos esperados.

    Args:
        context: contexto Dagster para emitir warning.
        reason: razon del skip (se incluye en metadata + log).
        roi: partition key de la ROI activa.

    Returns:
        MaterializeResult con ``status="skipped_no_gcs"`` y metadata util.
    """
    context.log.warning(
        "farslip_embeddings_italy roi=%s SKIPPED: %s",
        roi,
        reason,
    )
    return MaterializeResult(
        metadata={
            "roi": MetadataValue.text(roi),
            "status": MetadataValue.text("skipped_no_gcs"),
            "reason": MetadataValue.text(reason),
            "n_embeddings": MetadataValue.int(0),
            "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
            "data_version": MetadataValue.text(DATA_VERSION_TAG),
            "code_version": MetadataValue.text(git_sha(short=True)),
        }
    )


@asset(
    deps=[sentinel2_crops_256],
    partitions_def=ITALY_REGIONS,
    group_name="farslip",
    compute_kind="python",
    description=(
        "Bulk extraction de embeddings FarSLIP 512-dim sobre los crops "
        "Sentinel-2 256x256 de la ROI activa. Persiste a "
        "data/farslip_embeddings/{roi}/{year}/embeddings.parquet."
    ),
)
def farslip_embeddings_italy(context: AssetExecutionContext) -> MaterializeResult:
    """Materializa embeddings FarSLIP 512-dim por ROI italiana.

    Args:
        context: contexto Dagster. ``context.partition_key`` indica la ROI.

    Returns:
        ``MaterializeResult`` con metadata ``n_embeddings``,
        ``embedding_dim=512``, ``output_path``, ``roi``, ``year``,
        ``data_version`` (DVC tag), ``code_version`` (git SHA short).
        Si GCS no es accesible: metadata ``status="skipped_no_gcs"``.

    Raises:
        FileNotFoundError: si el manifest upstream no existe (mensaje
            indica al usuario materializar primero ``sentinel2_crops_256``).
    """
    import polars as pl

    roi = context.partition_key
    manifest_path = DATA_FARSLIP_PAIRS_DIR / roi / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest upstream ausente: {manifest_path}. Materializa primero "
            "sentinel2_crops_256 con partition_key={roi}."
        )

    context.log.info(
        "farslip_embeddings_italy.start roi=%s manifest=%s",
        roi,
        manifest_path,
    )

    # Polars (NON-NEGOTIABLE: no pandas). LazyFrame para volúmenes grandes.
    manifest = pl.read_parquet(manifest_path)
    n_crops = manifest.height

    if n_crops == 0:
        context.log.warning(
            "farslip_embeddings_italy roi=%s: manifest vacío, nada que extraer",
            roi,
        )
        return _skipped_result(context, "manifest empty", roi)

    # Carga del extractor — cae limpiamente si GCS no es accesible.
    try:
        from ml.extractors.farslip_extractor import (  # type: ignore[import-not-found]
            FarSLIPExtractor,
        )

        extractor = FarSLIPExtractor(weights_uri=DEFAULT_WEIGHTS_URI)
    except ImportError as exc:
        return _skipped_result(
            context,
            f"FarSLIPExtractor no instalable todavia ({exc})",
            roi,
        )
    except Exception as exc:
        if is_gcs_auth_error(exc):
            return _skipped_result(
                context, f"GCS auth failed: {type(exc).__name__}", roi
            )
        # AttributeError, KeyError, ValueError reales burbujean — son bugs.
        raise

    # Year derivado del manifest (Q4 fix): si la columna existe, agrupamos
    # por crop_year y escribimos una particion por (roi, year). Si no
    # existe, caemos a FALLBACK_YEAR documentado.
    has_year_col = "crop_year" in manifest.columns
    if has_year_col:
        years_present = sorted(set(int(y) for y in manifest["crop_year"].to_list()))
    else:
        years_present = [FALLBACK_YEAR]
        context.log.warning(
            "manifest sin crop_year; usando FALLBACK_YEAR=%d", FALLBACK_YEAR
        )

    # Bulk extraction en batches. Q9 fix: acumulamos tensores de embeddings
    # en una sola estructura columnar (np.ndarray + list[str/int]) y al final
    # serializamos con pl.DataFrame({...}) en bulk — sin list[dict] intermedio
    # ni Python overhead por fila. Para 30k pares x 512 floats esto baja de
    # ~60 MB de Python dicts a ~60 MB de arrays nativos (alocados 1 sola vez).
    import numpy as np

    output_paths_by_year: dict[int, Path] = {}

    crop_paths = manifest["crop_path"].to_list()
    crop_doys = (
        manifest["crop_doy"].to_list() if "crop_doy" in manifest.columns else [0] * n_crops
    )
    cap_classes = (
        manifest["cap_class"].to_list()
        if "cap_class" in manifest.columns
        else [""] * n_crops
    )
    crop_years = (
        [int(y) for y in manifest["crop_year"].to_list()]
        if has_year_col
        else [FALLBACK_YEAR] * n_crops
    )

    # Buffers por anio: indices del manifest cuyo crop_year coincide.
    indices_by_year: dict[int, list[int]] = {y: [] for y in years_present}
    embeddings_buffer: list[np.ndarray] = []
    valid_indices: list[int] = []  # indices del manifest que produjeron embedding

    for start in range(0, n_crops, EXTRACTION_BATCH_SIZE):
        end = min(start + EXTRACTION_BATCH_SIZE, n_crops)
        batch_paths = crop_paths[start:end]

        try:
            batch_tensor = extractor.load_crops_batch(batch_paths)
            embeddings = extractor.extract_embeddings(batch_tensor)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            context.log.error(
                "farslip_embeddings_italy roi=%s batch=%d-%d failed: %s",
                roi,
                start,
                end,
                exc,
            )
            continue

        # Single tensor -> single numpy array (1 copia CPU, no list[float]).
        batch_np = embeddings.detach().cpu().numpy().astype(np.float32)
        embeddings_buffer.append(batch_np)
        for offset in range(end - start):
            global_idx = start + offset
            valid_indices.append(global_idx)
            indices_by_year.setdefault(crop_years[global_idx], []).append(
                len(valid_indices) - 1
            )

    if not embeddings_buffer:
        return _skipped_result(context, "todos los batches fallaron", roi)

    all_embeddings = np.concatenate(embeddings_buffer, axis=0)  # (N, 512)
    n_embeddings = all_embeddings.shape[0]

    for year, local_idxs in indices_by_year.items():
        if not local_idxs:
            continue
        output_dir = DATA_FARSLIP_EMBEDDINGS_DIR / roi / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "embeddings.parquet"

        # Indices globales del manifest para este anio.
        manifest_idxs = [valid_indices[li] for li in local_idxs]
        year_df = pl.DataFrame(
            {
                "crop_id": [
                    Path(str(crop_paths[mi])).stem for mi in manifest_idxs
                ],
                "embedding": [
                    all_embeddings[li].tolist() for li in local_idxs
                ],
                "crop_doy": [int(crop_doys[mi]) for mi in manifest_idxs],
                "cap_class": [str(cap_classes[mi]) for mi in manifest_idxs],
            }
        )
        year_df.write_parquet(output_path, compression="zstd")
        output_paths_by_year[year] = output_path

    code_version = git_sha(short=True)
    context.log.info(
        "farslip_embeddings_italy.complete roi=%s n_embeddings=%d years=%s",
        roi,
        n_embeddings,
        sorted(output_paths_by_year.keys()),
    )

    from dagster import MetadataValue as _MV

    metadata: dict[str, _MV] = {
        "roi": MetadataValue.text(roi),
        "years": MetadataValue.text(
            ",".join(str(y) for y in sorted(output_paths_by_year.keys())) or "none"
        ),
        "n_embeddings": MetadataValue.int(n_embeddings),
        "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
        "batch_size": MetadataValue.int(EXTRACTION_BATCH_SIZE),
        "data_version": MetadataValue.text(DATA_VERSION_TAG),
        "code_version": MetadataValue.text(code_version),
    }
    if output_paths_by_year:
        # Path "principal" = primer year (mantiene compat con tests).
        first_year = sorted(output_paths_by_year.keys())[0]
        metadata["output_path"] = MetadataValue.path(
            str(output_paths_by_year[first_year].resolve())
        )
    return MaterializeResult(metadata=metadata)
