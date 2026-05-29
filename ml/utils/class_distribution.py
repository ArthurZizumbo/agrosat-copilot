"""Reporte de distribucion de clases para notebooks de baseline.

Sustituye el reporte ad-hoc "Clases con < 1000 parcelas: [...]" que aparece
en `notebooks/baseline/05_reencuadre_fenologico.ipynb` y produce informacion
util para decidir threshold de soporte, merge fenologico via
`PASTIS_R_GROUPINGS`, y stratificacion del CV espacial.

Funciones publicas:

- :func:`class_distribution_report` — DataFrame Polars con `class_id`,
  `class_name`, `n_parcels`, `share`, `support_band` (high/med/low/very_low),
  `agronomic_group`, `phenological_cycle`.
- :func:`recommend_threshold` — sugiere un threshold sensato basado en
  percentiles del soporte, en lugar del 1000 hardcoded que rompia el reporte.
- :func:`merge_to_phenological_groups` — agrupa class_ids segun
  `PASTIS_R_GROUPINGS["phenological_cycle"]` para reducir cardinalidad y
  habilitar baselines con clases balanceadas.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
import structlog

from ml.ingest.pastis_loader import PASTIS_R_CLASSES, PASTIS_R_GROUPINGS

logger = structlog.get_logger(__name__)

__all__ = [
    "SupportBand",
    "class_distribution_report",
    "merge_to_phenological_groups",
    "recommend_threshold",
]

SupportBand = Literal["high", "med", "low", "very_low"]


def class_distribution_report(
    df: pl.DataFrame,
    *,
    class_col: str = "class_id",
    thresholds: tuple[int, int, int] = (1000, 200, 30),
    drop_class_ids: tuple[int, ...] = (0, 19),
) -> pl.DataFrame:
    """Construye un reporte detallado de distribucion de clases.

    Resuelve el ruido que producia el reporte "Clases con < 1000 parcelas:
    [3, 8, ...]" sustituyendolo por una tabla con bandas de soporte y nombres
    legibles.

    Args:
        df: DataFrame Polars con la columna `class_col`.
        class_col: Nombre de la columna de clase. Default `"class_id"`.
        thresholds: Tupla `(high, med, low)` para clasificar en bandas de
            soporte. `n >= high` es "high"; `med <= n < high` es "med"; `low
            <= n < med` es "low"; `n < low` es "very_low". Default
            `(1000, 200, 30)`.
        drop_class_ids: Class IDs a descartar antes del conteo (PASTIS-R 0
            Background y 19 Void). Default `(0, 19)`.

    Returns:
        DataFrame con columnas `class_id`, `class_name`, `n_parcels`,
        `share` (proporcion), `support_band` (`high|med|low|very_low`),
        `agronomic_group`, `phenological_cycle`. Ordenado por `n_parcels`
        descendente.
    """
    if class_col not in df.columns:
        raise ValueError(f"`df` no contiene la columna `{class_col}`.")

    filtered = df.filter(
        pl.col(class_col).is_not_null()
        & ~pl.col(class_col).is_in(list(drop_class_ids))
    )
    counts = (
        filtered.group_by(class_col)
        .len()
        .rename({"len": "n_parcels", class_col: "class_id"})
        .with_columns(pl.col("class_id").cast(pl.Int64))
        .sort("n_parcels", descending=True)
    )
    total = counts["n_parcels"].sum()
    if total == 0:
        logger.warning("class_distribution_empty", n_total=0)
        return counts.with_columns(
            pl.lit(0.0).alias("share"),
            pl.lit("very_low").alias("support_band"),
            pl.lit(None, dtype=pl.Utf8).alias("class_name"),
            pl.lit(None, dtype=pl.Utf8).alias("agronomic_group"),
            pl.lit(None, dtype=pl.Utf8).alias("phenological_cycle"),
        )

    high_t, med_t, low_t = thresholds

    def _band(n: int) -> str:
        if n >= high_t:
            return "high"
        if n >= med_t:
            return "med"
        if n >= low_t:
            return "low"
        return "very_low"

    class_names = {int(k): v for k, v in PASTIS_R_CLASSES.items()}
    agronomic = PASTIS_R_GROUPINGS.get("agronomic_group", {})
    phenological = PASTIS_R_GROUPINGS.get("phenological_cycle", {})

    enriched = counts.with_columns(
        (pl.col("n_parcels") / total).alias("share"),
        pl.col("n_parcels")
        .map_elements(_band, return_dtype=pl.Utf8)
        .alias("support_band"),
        pl.col("class_id")
        .map_elements(
            lambda cid: class_names.get(int(cid), f"class_{int(cid)}"),
            return_dtype=pl.Utf8,
        )
        .alias("class_name"),
        pl.col("class_id")
        .map_elements(
            lambda cid: agronomic.get(int(cid), "unknown"),
            return_dtype=pl.Utf8,
        )
        .alias("agronomic_group"),
        pl.col("class_id")
        .map_elements(
            lambda cid: phenological.get(int(cid), "unknown"),
            return_dtype=pl.Utf8,
        )
        .alias("phenological_cycle"),
    )

    logger.info(
        "class_distribution_report",
        n_classes=enriched.height,
        n_total=int(total),
        n_high=int(enriched.filter(pl.col("support_band") == "high").height),
        n_med=int(enriched.filter(pl.col("support_band") == "med").height),
        n_low=int(enriched.filter(pl.col("support_band") == "low").height),
        n_very_low=int(enriched.filter(pl.col("support_band") == "very_low").height),
    )
    return enriched


def recommend_threshold(
    report: pl.DataFrame,
    *,
    n_count_col: str = "n_parcels",
    method: Literal["p25", "p50", "minmax_balance"] = "p25",
) -> int:
    """Sugiere un threshold de soporte sensato para reportes.

    El threshold hardcoded de 1000 que aparecia en notebooks producia el
    ruido "solo 1 clase cumple" porque PASTIS-R Italia esta muy
    desbalanceado (1 clase mayoritaria con ~30k parcelas, resto con <500).

    Args:
        report: DataFrame de `class_distribution_report`.
        n_count_col: Columna con el conteo por clase.
        method: Estrategia de calculo:

            - `"p25"`: percentil 25 del conteo (mas tolerante).
            - `"p50"`: mediana del conteo.
            - `"minmax_balance"`: media geometrica entre min y max.

    Returns:
        Threshold entero recomendado. Para Italia 18 clases tipicamente
        cae en el rango [30, 200].
    """
    counts = report[n_count_col].to_numpy()
    if counts.size == 0:
        return 0
    if method == "p25":
        import numpy as np

        return int(np.percentile(counts, 25))
    if method == "p50":
        import numpy as np

        return int(np.percentile(counts, 50))
    if method == "minmax_balance":
        import numpy as np

        return int(np.sqrt(counts.min() * counts.max()))
    raise ValueError(f"`method` no soportado: {method!r}.")


def merge_to_phenological_groups(
    df: pl.DataFrame,
    *,
    class_col: str = "class_id",
    grouping_name: str = "phenological_cycle",
    output_col: str = "pheno_group_id",
) -> pl.DataFrame:
    """Agrega una columna de grupo agronomico/fenologico para reducir cardinalidad.

    Usa `PASTIS_R_GROUPINGS` (cargado desde
    `data/reference/pastis_class_mapping.json`). Permite entrenar baselines
    sobre grupos balanceados cuando el set de 18 clases es demasiado escaso.

    Args:
        df: DataFrame con `class_col`.
        class_col: Columna con `class_id` PASTIS.
        grouping_name: Clave de `PASTIS_R_GROUPINGS`. Default
            `"phenological_cycle"` (cereales invernal/primavera/perenne/...).
        output_col: Nombre de la nueva columna.

    Returns:
        Una copia del DataFrame con la columna `output_col` adicional.
    """
    grouping = PASTIS_R_GROUPINGS.get(grouping_name)
    if not grouping:
        raise ValueError(
            f"Agrupacion `{grouping_name}` no disponible en "
            f"PASTIS_R_GROUPINGS. Opciones: {list(PASTIS_R_GROUPINGS)}."
        )

    return df.with_columns(
        pl.col(class_col)
        .map_elements(
            lambda cid: grouping.get(int(cid), "other"),
            return_dtype=pl.Utf8,
        )
        .alias(output_col)
    )
