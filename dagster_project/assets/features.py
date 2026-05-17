"""Assets Dagster para US-016 — Fusión multisensor a nivel parcela.

Materializa tres artefactos versionables del pipeline de feature engineering:

1. ``parcel_features_fused`` → ``data/features/features_fused_v1.parquet``
   matriz tabular (N, 2 + 189) con bloques AlphaEarth (64), índices×stats (85),
   Sentinel-1 (10), SRTM (3), ERA5 mensual (24) y geometría (3). Bloque
   opcional FarSLIP (512) si ``data/farslip/embeddings_italy.parquet`` existe.

2. ``parcel_splits_spatial_kfold`` → ``data/splits/spatial_kfold_v1/fold_{0..4}/``
   K=5 folds espaciales no contiguos generados con tessellation H3 res 5 +
   KMeans sobre centroides hex + buffer de exclusión de 1 km. Sin leakage
   espacial entre folds.

3. ``parcel_features_scaler`` → ``artifacts/scaler_v1.pkl``
   ``StandardScaler`` (joblib) ajustado **solo** sobre el split train del
   ``fold_0`` para evitar fuga de val/test a la normalización global.

Los tres assets delegan la lógica de negocio en ``ml.features.fusion``,
``ml.features.spatial_split`` y ``ml.features.scaler`` (DRY: el script CLI
``scripts/build_parcel_features.py`` consume las mismas funciones).

Lineage declarada vía ``deps=[...]``:

::

    alphaearth_embeddings_italy ─┐
    features_parcels ────────────┼─→ parcel_features_fused ─┐
    parcels ─────────────────────┘                            │
                                                              ├─→ parcel_features_scaler
    parcels ─→ parcel_splits_spatial_kfold ───────────────────┘

Las dependencias upstream ``alphaearth_embeddings_italy``, ``features_parcels``
y ``parcels`` son ``SourceAsset`` lógicas — los nombres existen como contrato
hacia US-006/US-015; cuando esos assets concretos se registren en
``definitions.py`` la lineage se enlazará automáticamente sin tocar este
módulo.
"""

import hashlib
from pathlib import Path

import structlog
from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

log = structlog.get_logger(__name__)

# Convenciones de paths (relativos a la raíz del repo). Los assets resuelven
# las rutas desde ``Path.cwd()`` o desde ``DAGSTER_HOME`` si está configurado.
DATA_FEATURES_DIR = Path("data/features")
DATA_SPLITS_DIR = Path("data/splits/spatial_kfold_v1")
ARTIFACTS_DIR = Path("artifacts")
DEFAULT_FUSED_PATH = DATA_FEATURES_DIR / "features_fused_v1.parquet"
DEFAULT_SCALER_PATH = ARTIFACTS_DIR / "scaler_v1.pkl"
DEFAULT_PARCELS_FIXTURE = Path("data/test_fixtures/parcels_demo_3regions.parquet")
DEFAULT_FARSLIP_PATH = Path("data/farslip/embeddings_italy.parquet")

# Bloques activos para la materialización por defecto (sin FarSLIP).
DEFAULT_BLOCKS: tuple[str, ...] = (
    "alphaearth",
    "indices_stats",
    "sentinel1",
    "srtm",
    "era5_monthly",
    "geometry",
)
DEFAULT_YEAR = 2024
DATA_VERSION_TAG = "fused-features-italy-v1"
CODE_VERSION_LABEL = "us-016"


def _hash_file_md5(path: Path) -> str:
    """Calcula el hash MD5 de un fichero existente para metadata de lineage."""
    hasher = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_parcels_geodataframe(parcels_path: Path):
    """Carga el GeoDataFrame de parcelas desde un parquet o fixture demo.

    Args:
        parcels_path: ruta al parquet con columnas ``parcel_id``, ``year``,
            ``geom`` (WKT EPSG:4326). En producción será un asset upstream
            ``parcels`` materializado desde Postgres; en local/CI se usa el
            fixture demo de 9 parcelas.

    Returns:
        GeoDataFrame en CRS EPSG:4326 listo para ``build_fused_features``.

    Raises:
        FileNotFoundError: si el parquet no existe en disco.
    """
    import geopandas as gpd
    import polars as pl
    from shapely import wkt

    if not parcels_path.exists():
        raise FileNotFoundError(
            f"parcels fixture not found at {parcels_path}; ejecuta el script "
            "de generación o monta el asset upstream `parcels`."
        )
    frame = pl.read_parquet(parcels_path).to_pandas()
    geoms = [wkt.loads(g) for g in frame["geom"].tolist()]
    gdf = gpd.GeoDataFrame(frame.drop(columns=["geom"]), geometry=geoms, crs="EPSG:4326")
    return gdf


@asset(
    deps=[
        AssetKey("alphaearth_embeddings_italy"),
        AssetKey("features_parcels"),
        AssetKey("parcels"),
    ],
    compute_kind="polars",
    group_name="feature_engineering",
    description=(
        "Vector tabular fusionado por (parcel_id, year) con 6 bloques "
        "heterogéneos alineados (AE 64 + idx×stats 85 + S1 10 + SRTM 3 + "
        "ERA5 24 + geom 3 = 189 cols). Bloque opcional FarSLIP (+512)."
    ),
)
def parcel_features_fused(context: AssetExecutionContext) -> MaterializeResult:
    """Materializa ``data/features/features_fused_v1.parquet``.

    Args:
        context: contexto de ejecución Dagster. ``context.log`` emite a la UI
            mientras ``structlog`` registra estructurado para CI/observabilidad.

    Returns:
        ``MaterializeResult`` con metadata ``rows``, ``cols``, ``md5``,
        ``year``, ``regions``, ``blocks``, ``data_version``, ``code_version``.
    """
    from ml.features.fusion import build_fused_features  # type: ignore[import-not-found]

    parcels_path = DEFAULT_PARCELS_FIXTURE
    out_path = DEFAULT_FUSED_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "parcel_features_fused.start",
        parcels_path=str(parcels_path),
        out_path=str(out_path),
        year=DEFAULT_YEAR,
    )
    context.log.info(f"Cargando parcelas desde {parcels_path}")
    parcels = _load_parcels_geodataframe(parcels_path)

    farslip_present = DEFAULT_FARSLIP_PATH.exists()
    include_farslip = farslip_present
    if not farslip_present:
        log.warning(
            "parcel_features_fused.farslip_block_skipped",
            farslip_path=str(DEFAULT_FARSLIP_PATH),
        )
        context.log.warning(
            "FarSLIP block omitido: no existe data/farslip/embeddings_italy.parquet"
        )

    df = build_fused_features(
        parcels,
        year=DEFAULT_YEAR,
        blocks=DEFAULT_BLOCKS,
        include_farslip=include_farslip,
        farslip_path=str(DEFAULT_FARSLIP_PATH) if include_farslip else None,
        lazy=True,
    )
    df.write_parquet(out_path, compression="zstd")

    md5 = _hash_file_md5(out_path)
    regions: list[str] = (
        sorted({str(r) for r in parcels["region"].tolist()})
        if "region" in parcels.columns
        else []
    )
    log.info(
        "parcel_features_fused.complete",
        rows=df.height,
        cols=df.width,
        md5=md5,
        year=DEFAULT_YEAR,
        regions=regions,
        farslip=include_farslip,
    )

    return MaterializeResult(
        metadata={
            "rows": MetadataValue.int(df.height),
            "cols": MetadataValue.int(df.width),
            "md5": MetadataValue.text(md5),
            "year": MetadataValue.int(DEFAULT_YEAR),
            "regions": MetadataValue.text(", ".join(regions) or "n/a"),
            "blocks": MetadataValue.text(", ".join(DEFAULT_BLOCKS)),
            "farslip": MetadataValue.bool(include_farslip),
            "output_path": MetadataValue.path(str(out_path.resolve())),
            "data_version": MetadataValue.text(DATA_VERSION_TAG),
            "code_version": MetadataValue.text(CODE_VERSION_LABEL),
            "preview": MetadataValue.md(
                f"```\nshape: ({df.height}, {df.width})\nfirst_cols: "
                f"{df.columns[:6]}\n```"
            ),
        }
    )


@asset(
    deps=[AssetKey("parcels")],
    compute_kind="geopandas",
    group_name="feature_engineering",
    description=(
        "K=5 folds espaciales no contiguos (H3 res 5 + KMeans + buffer 1 km). "
        "Persiste un parquet por fold con los splits train/val/test. "
        "Garantiza sin leakage entre folds vecinos."
    ),
)
def parcel_splits_spatial_kfold(context: AssetExecutionContext) -> MaterializeResult:
    """Materializa ``data/splits/spatial_kfold_v1/fold_{0..4}/*.parquet``.

    Args:
        context: contexto Dagster.

    Returns:
        ``MaterializeResult`` con metadata por fold (sizes train/val/test +
        balance de clases).
    """
    import polars as pl

    from ml.features.spatial_split import build_spatial_kfold  # type: ignore[import-not-found]

    parcels_path = DEFAULT_PARCELS_FIXTURE
    out_dir = DATA_SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "parcel_splits_spatial_kfold.start",
        parcels_path=str(parcels_path),
        out_dir=str(out_dir),
    )
    parcels = _load_parcels_geodataframe(parcels_path)
    folds = build_spatial_kfold(parcels, k=5, h3_res=5, buffer_km=1.0)

    fold_sizes = []
    for fold in folds:
        fold_dir = out_dir / f"fold_{fold.fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"parcel_id": list(fold.train_ids)}).write_parquet(
            fold_dir / "train_parcel_ids.parquet"
        )
        pl.DataFrame({"parcel_id": list(fold.val_ids)}).write_parquet(
            fold_dir / "val_parcel_ids.parquet"
        )
        pl.DataFrame({"parcel_id": list(fold.test_ids)}).write_parquet(
            fold_dir / "test_parcel_ids.parquet"
        )
        sizes = {
            "fold": fold.fold_id,
            "train": len(fold.train_ids),
            "val": len(fold.val_ids),
            "test": len(fold.test_ids),
        }
        fold_sizes.append(sizes)
        log.info("parcel_splits_spatial_kfold.fold_written", **sizes)

    total_parcels = sum(s["train"] + s["val"] + s["test"] for s in fold_sizes) // len(folds)
    class_balance_summary = "n/a"
    if "crop_class" in parcels.columns:
        counts = parcels["crop_class"].value_counts().to_dict()
        class_balance_summary = ", ".join(f"{k}={v}" for k, v in counts.items())

    log.info(
        "parcel_splits_spatial_kfold.complete",
        n_folds=len(folds),
        total_parcels=total_parcels,
    )

    return MaterializeResult(
        metadata={
            "n_folds": MetadataValue.int(len(folds)),
            "h3_res": MetadataValue.int(5),
            "buffer_km": MetadataValue.float(1.0),
            "total_parcels": MetadataValue.int(total_parcels),
            "fold_sizes": MetadataValue.json(fold_sizes),
            "class_balance": MetadataValue.text(class_balance_summary),
            "output_dir": MetadataValue.path(str(out_dir.resolve())),
            "data_version": MetadataValue.text("spatial-kfold-italy-v1"),
            "code_version": MetadataValue.text(CODE_VERSION_LABEL),
        }
    )


@asset(
    deps=[
        AssetKey("parcel_features_fused"),
        AssetKey("parcel_splits_spatial_kfold"),
    ],
    compute_kind="sklearn",
    group_name="feature_engineering",
    description=(
        "StandardScaler ajustado solo sobre el split train del fold_0. "
        "Persistido con joblib (no pickle raw) en artifacts/scaler_v1.pkl. "
        "Sin leakage de val/test."
    ),
)
def parcel_features_scaler(context: AssetExecutionContext) -> MaterializeResult:
    """Materializa ``artifacts/scaler_v1.pkl``.

    Args:
        context: contexto Dagster.

    Returns:
        ``MaterializeResult`` con metadata ``feature_cols``, ``n_train``,
        ``mean_summary``, ``std_summary``.
    """
    import polars as pl

    from ml.features.scaler import fit_scaler_on_train  # type: ignore[import-not-found]

    features_path = DEFAULT_FUSED_PATH
    fold_train_path = DATA_SPLITS_DIR / "fold_0" / "train_parcel_ids.parquet"
    scaler_path = DEFAULT_SCALER_PATH
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "parcel_features_scaler.start",
        features_path=str(features_path),
        fold_train_path=str(fold_train_path),
        scaler_path=str(scaler_path),
    )
    if not features_path.exists():
        raise FileNotFoundError(
            f"features parquet ausente: {features_path}. Materializa primero "
            "parcel_features_fused."
        )
    if not fold_train_path.exists():
        raise FileNotFoundError(
            f"fold_0 train ids ausente: {fold_train_path}. Materializa primero "
            "parcel_splits_spatial_kfold."
        )

    df = pl.read_parquet(features_path)
    train_ids_df = pl.read_parquet(fold_train_path)
    train_ids = tuple(int(x) for x in train_ids_df["parcel_id"].to_list())

    feature_cols = tuple(c for c in df.columns if c not in ("parcel_id", "year"))
    scaler = fit_scaler_on_train(
        df,
        train_ids=train_ids,
        feature_cols=feature_cols,
        scaler_path=scaler_path,
        version="v1",
    )

    mean_summary = f"min={float(scaler.mean_.min()):.4f} max={float(scaler.mean_.max()):.4f}"
    std_summary = f"min={float(scaler.scale_.min()):.4f} max={float(scaler.scale_.max()):.4f}"
    log.info(
        "parcel_features_scaler.complete",
        n_train=len(train_ids),
        n_features=len(feature_cols),
        scaler_path=str(scaler_path),
    )

    return MaterializeResult(
        metadata={
            "feature_cols_count": MetadataValue.int(len(feature_cols)),
            "n_train": MetadataValue.int(len(train_ids)),
            "scaler_path": MetadataValue.path(str(scaler_path.resolve())),
            "mean_summary": MetadataValue.text(mean_summary),
            "std_summary": MetadataValue.text(std_summary),
            "version": MetadataValue.text("v1"),
            "data_version": MetadataValue.text("scaler-v1"),
            "code_version": MetadataValue.text(CODE_VERSION_LABEL),
        }
    )
