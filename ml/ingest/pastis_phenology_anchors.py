"""Construye anclas fenologicas calendario por parcela PASTIS-R.

PASTIS-R no expone DOY fenologicos calendario en su metadata: solo provee
`dates-S2` (timestamps de adquisicion Sentinel-2 por patch, primera fecha
~17-sep-2018, ultima ~oct-2019). El subset US-016 derivo
`sog_doy`/`peak_doy`/`senescence_doy` como **dias desde la primera imagen S2
del patch**, no como DOY calendario (1-365 del anio agronomico).

Este modulo convierte los DOY relativos del subset a DOY calendario del
anio de muestreo (default 2019), usando la primera fecha real de
`dates-S2` por patch como referencia. El parquet de salida es directamente
consumible por :func:`ml.ingest.s2_anchor_sampler.sample_s2_anchors_for_parcels`
como ``phenology_anchors_path`` y elimina el warning
``phenology_anchors_fallback_static``.

Decisiones canonicas (US-023-preview v2 fix S2 sampler):

- ``sog_doy`` que cae en 2018 (anio anterior al de muestreo): wrapeo al
  inicio del anio de muestreo con fallback agronomico ``SOS_BRITTANY=90``
  (DOY 90 = 31-mar; literatura Bretana coloca emergencia/stem-elongation
  de winter wheat entre DOY 90-110). Mantener el SOS antes del DOY=1 del
  anio de muestreo invalida la ventana del sampler (`+/- window_days`).
- ``peak_doy`` y ``senescence_doy`` que caen en 2019: conversion directa
  ``fecha_base + dias_relativos -> DOY calendario``.
- Parcelas sin patch_id, sin fecha base PASTIS o con DOY relativos NULL:
  fallback estatico Bretana ``(SOS=90, peak=180, senescence=220)``
  derivado de literatura (MDPI Brittany 2022).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["build_pastis_phenology_anchors"]


#: Fallback agronomico Bretana cuando la conversion falla por parcela.
#: SOS=90 (emergencia winter wheat / stem elongation, DOY 90-110 literatura).
#: peak=180 (floracion-llenado trigo + pico LAI cultivos verano).
#: senescence=220 (cosecha trigo invierno + senescencia maiz temprano).
FALLBACK_DOY_BRITTANY: dict[str, int] = {
    "sog_doy": 90,
    "peak_doy": 180,
    "senescence_doy": 220,
}

#: Limite inferior del DOY calendario aceptable. SOS antes del DOY=1 wrapea
#: al fallback Bretana para no producir DOY negativo.
MIN_VALID_DOY: int = 1

#: Limite superior del DOY calendario aceptable.
MAX_VALID_DOY: int = 365


def build_pastis_phenology_anchors(
    *,
    metadata_geojson_path: Path | str = Path("data/PASTIS-R/metadata.geojson"),
    features_subset_path: Path | str = Path(
        "data/test_fixtures/feature_selection_parcels_subset.parquet"
    ),
    output_path: Path | str = Path(
        "data/features/pastis_phenology_anchors_2019.parquet"
    ),
    target_year: int = 2019,
    overwrite: bool = False,
) -> Path:
    """Genera ``parcel_id, sog_doy, peak_doy, senescence_doy`` (DOY calendario).

    Lee el ``metadata.geojson`` de PASTIS-R, extrae la primera fecha
    Sentinel-2 por patch como fecha base, lee el subset US-016 con DOY
    relativos por parcela, y persiste un parquet con DOY calendario del
    ``target_year``. Las parcelas cuya conversion cae fuera del rango
    ``[1, 365]`` o cuyo patch base no es identificable usan el fallback
    Bretana documentado en :data:`FALLBACK_DOY_BRITTANY`.

    Args:
        metadata_geojson_path: Ruta al ``metadata.geojson`` de PASTIS-R.
        features_subset_path: Ruta al subset US-016 con DOY relativos.
        output_path: Parquet destino con esquema directamente consumible
            por :func:`ml.ingest.s2_anchor_sampler.sample_s2_anchors_for_parcels`.
        target_year: Anio de muestreo (default 2019; rango PASTIS-R).
        overwrite: Si True regenera aunque el parquet exista.

    Returns:
        Path del parquet generado con esquema
        ``parcel_id (Utf8), sog_doy (Int16), peak_doy (Int16), senescence_doy (Int16)``.

    Raises:
        FileNotFoundError: si el metadata.geojson o el subset no existen.
        ValueError: si el subset no expone ``patch_id`` o los DOY relativos.
    """
    out = Path(output_path)
    if out.exists() and not overwrite:
        logger.info("pastis_phenology_anchors_cache_hit", path=str(out))
        return out

    meta_path = Path(metadata_geojson_path)
    sub_path = Path(features_subset_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.geojson no encontrado en {meta_path}.")
    if not sub_path.exists():
        raise FileNotFoundError(f"subset features no encontrado en {sub_path}.")

    # 1. Extrae fecha base por patch desde metadata.geojson.
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)

    patch_base_date: dict[int, datetime] = {}
    for feat in meta["features"]:
        props = feat["properties"]
        dates_s2 = props.get("dates-S2", {})
        if not dates_s2:
            continue
        # Primera fecha cronologica (ordenamos por key int).
        first_key = min(dates_s2.keys(), key=int)
        first_yyyymmdd = str(dates_s2[first_key])
        try:
            patch_base_date[int(props["ID_PATCH"])] = datetime.strptime(
                first_yyyymmdd, "%Y%m%d"
            )
        except (KeyError, ValueError):
            continue

    if not patch_base_date:
        raise ValueError(
            "metadata.geojson no expone `dates-S2` ni `ID_PATCH` parseable. "
            "Verifica que el dataset PASTIS-R este completo."
        )

    logger.info(
        "pastis_dates_indexed",
        n_patches=len(patch_base_date),
        first_date_min=min(patch_base_date.values()).isoformat(),
    )

    # 2. Lee subset US-016 con DOY relativos + patch_id.
    df = pl.read_parquet(sub_path)
    required = {"parcel_id", "patch_id", "sog_doy", "peak_doy", "senescence_doy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"subset {sub_path} carece de columnas requeridas: {sorted(missing)}."
        )

    # 3. Convierte DOY relativo -> DOY calendario por parcela.
    rows: list[dict[str, int | str]] = []
    n_fallback_unknown_patch = 0
    n_fallback_null_doy = 0
    n_fallback_out_of_range = 0
    n_real = 0

    for row in df.iter_rows(named=True):
        parcel_id = str(row["parcel_id"])
        patch_id = row["patch_id"]
        base = patch_base_date.get(int(patch_id)) if patch_id is not None else None

        if base is None:
            n_fallback_unknown_patch += 1
            rows.append(
                {
                    "parcel_id": parcel_id,
                    "sog_doy": FALLBACK_DOY_BRITTANY["sog_doy"],
                    "peak_doy": FALLBACK_DOY_BRITTANY["peak_doy"],
                    "senescence_doy": FALLBACK_DOY_BRITTANY["senescence_doy"],
                }
            )
            continue

        converted: dict[str, int] = {}
        any_fallback = False
        for anchor in ("sog_doy", "peak_doy", "senescence_doy"):
            rel = row.get(anchor)
            if rel is None:
                any_fallback = True
                converted[anchor] = FALLBACK_DOY_BRITTANY[anchor]
                continue
            real_date = base + timedelta(days=int(rel))
            if real_date.year != target_year:
                # Cae en 2018 (winter sowing) o 2020 (no esperado): fallback.
                any_fallback = True
                converted[anchor] = FALLBACK_DOY_BRITTANY[anchor]
                continue
            cal_doy = real_date.timetuple().tm_yday
            if cal_doy < MIN_VALID_DOY or cal_doy > MAX_VALID_DOY:
                any_fallback = True
                converted[anchor] = FALLBACK_DOY_BRITTANY[anchor]
                continue
            converted[anchor] = cal_doy

        if any_fallback:
            # Si al menos un ancla cayo en fallback, lo contabilizamos
            # pero seguimos usando lo que SI fue real para los otros 2.
            if all(
                converted[a] == FALLBACK_DOY_BRITTANY[a]
                for a in ("sog_doy", "peak_doy", "senescence_doy")
            ):
                n_fallback_out_of_range += 1
            else:
                n_real += 1
                n_fallback_null_doy += 1
        else:
            n_real += 1

        rows.append({"parcel_id": parcel_id, **converted})

    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "parcel_id": pl.Utf8,
            "sog_doy": pl.Int16,
            "peak_doy": pl.Int16,
            "senescence_doy": pl.Int16,
        },
    ).write_parquet(out)

    logger.info(
        "pastis_phenology_anchors_persisted",
        path=str(out),
        n_total=len(rows),
        n_real_at_least_one_ancla=n_real,
        n_fallback_unknown_patch=n_fallback_unknown_patch,
        n_fallback_out_of_range=n_fallback_out_of_range,
        n_fallback_null_doy_partial=n_fallback_null_doy,
        target_year=target_year,
    )
    return out
