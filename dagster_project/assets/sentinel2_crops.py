"""Assets Dagster para US-017 — Crops Sentinel-2 256x256 px para FarSLIP.

Materializa el asset upstream ``sentinel2_crops_256`` particionado por ROI
italiana (Pianura Padana, Toscana, Puglia). Para cada partition:

1. Lee parcelas con ``crop_class`` etiquetado desde el subset GSAA Italia.
2. Genera crops 256x256 px (4 bandas B02/B03/B04/B08) centrados en el
   centroide de cada parcela.
3. Aplica QA mask Cloud Score+ (umbral 0.2) descartando crops nublados.
4. Escribe COGs en ``data/farslip_pairs/{roi}/crops/*.tif`` y appendea el
   manifest ``data/farslip_pairs/{roi}/manifest.parquet`` con metadata por
   par (path, DOY, year, clase CAP, texto agronómico it/es/en, lat/lon).

La lógica de negocio vive en ``ml.farslip.dataset.build_farslip_pairs`` —
este asset es un wrapper de orquestación que:

- Pasa la ROI activa como única partition al builder (one-ROI-per-run).
- Captura métricas para metadata Dagster (n_pairs, rango DOY, clases).
- Propaga tags ``data_version`` y ``code_version`` (git SHA short) para
  lineage MLflow aguas abajo (asset ``farslip_embeddings_italy``).

Producción (NO disponible en CI ni dev local sin GEE creds):
    Se inyectaría un ``GoogleEarthEngineResource`` y un ``GCSResource``
    desde ``dagster_project/resources/`` para autenticar GEE/CDSE y para
    persistir los COGs directamente a ``gs://agrosat-data/farslip-pairs/``
    vía DVC remote. El builder soporta cualquiera de los dos backends.

Smoke / dev local:
    Si ``build_farslip_pairs`` devuelve ``n_pairs < 1000`` se emite
    warning pero NO se falla la materialización — el fixture sintético
    de tests vive en ``data/test_fixtures/farslip_synthetic/`` con 10
    pares y debe poder ejecutar el asset sin romper Dagster.

Integración MLflow (documentada, no implementada en US-017):
    Se podría wrappear este asset con el decorator ``@mlflow_resource``
    de ``dagster-mlflow`` para crear automáticamente un run por partición
    y registrar ``data_version`` como tag. En US-017 los tags se emiten
    como metadata Dagster; la promoción a MLflow ocurre en el trainer
    (``ml/farslip/train.py``) que consume estos crops.
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    StaticPartitionsDefinition,
    asset,
)

from ml.utils.git_meta import git_sha

#: Particiones estáticas — una ROI italiana por partition key.
#: Mantener sincronizado con ``ml/farslip/cap_vocabulary.yaml`` y con la
#: tabla GSAA Italia subset (R1 del riesgo: bajar a 2 ROIs si AC-3 falla).
ITALY_REGIONS = StaticPartitionsDefinition(["pianura_padana", "toscana", "puglia"])

#: Rutas relativas al cwd (consistente con assets US-016 features.py).
DATA_FARSLIP_PAIRS_DIR = Path("data/farslip_pairs")
DEFAULT_VOCABULARY_PATH = Path("ml/farslip/cap_vocabulary.yaml")

#: Tag de versión del dataset FarSLIP (DVC) — se promueve a tag DVC
#: ``farslip-pairs-italy-v1`` al cierre de US-017.
DATA_VERSION_TAG = "farslip-pairs-italy-v1"

#: Umbral mínimo de pares por ROI por debajo del cual se emite warning
#: (smoke local con fixture sintético puede devolver muy pocos pares).
MIN_PAIRS_WARNING_THRESHOLD = 1000

#: Parámetros del builder (paper FarSLIP §3.1 + AC-3 del planning).
N_PER_ROI = 10000
CROP_SIZE_PX = 256
QA_CLOUD_THRESHOLD = 0.2
SEED = 42


@asset(
    partitions_def=ITALY_REGIONS,
    group_name="farslip",
    compute_kind="python",
    description=(
        "Crops Sentinel-2 256x256 px (4 bandas B02/B03/B04/B08) + manifest "
        "Parquet con texto agronómico it/es/en, particionado por ROI italiana. "
        "Upstream del asset farslip_embeddings_italy."
    ),
)
def sentinel2_crops_256(context: AssetExecutionContext) -> MaterializeResult:
    """Materializa crops Sentinel-2 256x256 px por ROI italiana (one per run).

    Args:
        context: contexto Dagster. ``context.partition_key`` indica la ROI
            activa (``pianura_padana`` | ``toscana`` | ``puglia``).
            ``context.log`` emite a la UI de Dagster.

    Returns:
        ``MaterializeResult`` con metadata: ``n_pairs``, ``min_doy``,
        ``max_doy``, ``n_classes``, ``output_path``, ``roi``,
        ``data_version`` (DVC tag), ``code_version`` (git SHA short).

    Raises:
        ImportError: si ``ml.farslip.dataset`` no está instalado todavía
            (el módulo lo crea el subagente ml-engineer en paralelo;
            cuando esté disponible este asset funcionará sin cambios).
    """
    # Import diferido para que la introspección de assets no requiera el
    # módulo ml.farslip — útil en CI antes de que ml-engineer aterrice.
    from ml.farslip.dataset import build_farslip_pairs  # type: ignore[import-not-found]

    roi = context.partition_key
    output_root = DATA_FARSLIP_PAIRS_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    code_version = git_sha(short=True)
    context.log.info(
        "sentinel2_crops_256.start roi=%s n_per_roi=%d crop_size=%d qa=%.2f",
        roi,
        N_PER_ROI,
        CROP_SIZE_PX,
        QA_CLOUD_THRESHOLD,
    )

    manifest = build_farslip_pairs(
        rois=(roi,),
        n_per_roi=N_PER_ROI,
        crop_size_px=CROP_SIZE_PX,
        qa_cloud_threshold=QA_CLOUD_THRESHOLD,
        output_root=output_root,
        vocabulary_path=DEFAULT_VOCABULARY_PATH,
        seed=SEED,
    )

    n_pairs = manifest.height if manifest is not None else 0
    if n_pairs < MIN_PAIRS_WARNING_THRESHOLD:
        context.log.warning(
            "sentinel2_crops_256 produjo solo %d pares para roi=%s (esperado>=%d). "
            "Smoke con fixture sintético es OK; en producción revisar gate AC-3.",
            n_pairs,
            roi,
            MIN_PAIRS_WARNING_THRESHOLD,
        )

    # Métricas derivadas (defensivas: el manifest puede venir vacío en smoke).
    if n_pairs > 0 and "crop_doy" in manifest.columns:
        # Polars Series.min/max() devuelve PythonLiteral (Union amplio).
        # Cast intermedio a str para satisfacer mypy SupportsInt.
        min_doy = int(str(manifest["crop_doy"].min()))
        max_doy = int(str(manifest["crop_doy"].max()))
    else:
        min_doy = 0
        max_doy = 0

    if n_pairs > 0 and "cap_class" in manifest.columns:
        n_classes = int(manifest["cap_class"].n_unique())
    else:
        n_classes = 0

    roi_output_path = (output_root / roi).resolve()
    context.log.info(
        "sentinel2_crops_256.complete roi=%s n_pairs=%d n_classes=%d",
        roi,
        n_pairs,
        n_classes,
    )

    return MaterializeResult(
        metadata={
            "roi": MetadataValue.text(roi),
            "n_pairs": MetadataValue.int(n_pairs),
            "min_doy": MetadataValue.int(min_doy),
            "max_doy": MetadataValue.int(max_doy),
            "n_classes": MetadataValue.int(n_classes),
            "crop_size_px": MetadataValue.int(CROP_SIZE_PX),
            "qa_cloud_threshold": MetadataValue.float(QA_CLOUD_THRESHOLD),
            "output_path": MetadataValue.path(str(roi_output_path)),
            "data_version": MetadataValue.text(DATA_VERSION_TAG),
            "code_version": MetadataValue.text(code_version),
        }
    )
