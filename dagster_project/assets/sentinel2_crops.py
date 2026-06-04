"""Dagster assets for US-017 — Sentinel-2 256x256 px crops for FarSLIP.

Materializes the upstream asset ``sentinel2_crops_256`` partitioned by Italian
ROI (Pianura Padana, Toscana, Puglia). For each partition:

1. Reads parcels with labeled ``crop_class`` from the GSAA Italy subset.
2. Generates 256x256 px crops (4 bands B02/B03/B04/B08) centered on the
   centroid of each parcel.
3. Applies a Cloud Score+ QA mask (threshold 0.2) discarding cloudy crops.
4. Writes COGs to ``data/farslip_pairs/{roi}/crops/*.tif`` and appends the
   manifest ``data/farslip_pairs/{roi}/manifest.parquet`` with per-pair
   metadata (path, DOY, year, CAP class, agronomic text it/es/en, lat/lon).

The business logic lives in ``ml.farslip.dataset.build_farslip_pairs`` —
this asset is an orchestration wrapper that:

- Passes the active ROI as the single partition to the builder (one-ROI-per-run).
- Captures metrics for Dagster metadata (n_pairs, DOY range, classes).
- Propagates ``data_version`` and ``code_version`` (git SHA short) tags for
  downstream MLflow lineage (asset ``farslip_embeddings_italy``).

Production (NOT available in CI nor local dev without GEE creds):
    A ``GoogleEarthEngineResource`` and a ``GCSResource`` would be injected
    from ``dagster_project/resources/`` to authenticate GEE/CDSE and to
    persist the COGs directly to ``gs://agrosat-data/farslip-pairs/`` via DVC
    remote. The builder supports either backend.

Smoke / local dev:
    If ``build_farslip_pairs`` returns ``n_pairs < 1000`` a warning is emitted
    but the materialization does NOT fail — the synthetic test fixture lives in
    ``data/test_fixtures/farslip_synthetic/`` with 10 pairs and must be able to
    run the asset without breaking Dagster.

MLflow integration (documented, not implemented in US-017):
    This asset could be wrapped with the ``@mlflow_resource`` decorator of
    ``dagster-mlflow`` to automatically create a run per partition and register
    ``data_version`` as a tag. In US-017 the tags are emitted as Dagster
    metadata; the promotion to MLflow happens in the trainer
    (``ml/farslip/train.py``) that consumes these crops.
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

#: Static partitions — one Italian ROI per partition key.
#: Keep in sync with ``ml/farslip/cap_vocabulary.yaml`` and with the GSAA Italy
#: subset table (risk R1: drop to 2 ROIs if AC-3 fails).
ITALY_REGIONS = StaticPartitionsDefinition(["pianura_padana", "toscana", "puglia"])

#: Paths relative to the cwd (consistent with US-016 assets features.py).
DATA_FARSLIP_PAIRS_DIR = Path("data/farslip_pairs")
DEFAULT_VOCABULARY_PATH = Path("ml/farslip/cap_vocabulary.yaml")

#: Version tag of the FarSLIP dataset (DVC) — promoted to the DVC tag
#: ``farslip-pairs-italy-v1`` at the closing of US-017.
DATA_VERSION_TAG = "farslip-pairs-italy-v1"

#: Minimum threshold of pairs per ROI below which a warning is emitted
#: (local smoke with synthetic fixture may return very few pairs).
MIN_PAIRS_WARNING_THRESHOLD = 1000

#: Builder parameters (FarSLIP paper §3.1 + AC-3 of the planning).
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
    """Materialize Sentinel-2 256x256 px crops per Italian ROI (one per run).

    Args:
        context: Dagster context. ``context.partition_key`` indicates the
            active ROI (``pianura_padana`` | ``toscana`` | ``puglia``).
            ``context.log`` emits to the Dagster UI.

    Returns:
        ``MaterializeResult`` with metadata: ``n_pairs``, ``min_doy``,
        ``max_doy``, ``n_classes``, ``output_path``, ``roi``,
        ``data_version`` (DVC tag), ``code_version`` (git SHA short).

    Raises:
        ImportError: if ``ml.farslip.dataset`` is not installed yet (the
            module is created by the ml-engineer subagent in parallel; when it
            is available this asset will work without changes).
    """
    # Deferred import so that asset introspection does not require the
    # ml.farslip module — useful in CI before ml-engineer lands.
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

    # Derived metrics (defensive: the manifest may be empty in smoke).
    if n_pairs > 0 and "crop_doy" in manifest.columns:
        # Polars Series.min/max() returns a PythonLiteral (broad Union).
        # Intermediate cast to str to satisfy mypy SupportsInt.
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
