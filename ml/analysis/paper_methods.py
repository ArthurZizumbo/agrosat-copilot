"""Metodos de EDA y feature engineering derivados de la literatura academica.

Este modulo traduce a codigo reproducible Polars/sklearn siete metodos
extraidos de la lectura completa de cuatro papers de teledeteccion agricola.
Cada funcion publica cita su fuente (autor + arXiv ID / DOI) en el docstring.

Papers de referencia
--------------------
- Paper A: Russwurm, M., Korner, M. (2018). *Multi-Temporal Land Cover
  Classification with Sequential Recurrent Encoders*. ISPRS International
  Journal of Geo-Information 7(4):129. arXiv:1802.02080. (Provisto por el
  sponsor.)
- Paper B: Tarasiou, M., Guler, R.A., Zafeiriou, S. (2021). *Context-self
  contrastive pretraining for crop type semantic segmentation*. IEEE TGRS.
  arXiv:2104.04310. (Provisto por el sponsor.)
- Paper C: *Phenology-Aware Transformer (PVM)* (2025). Remote Sensing
  17(14):2346. DOI 10.3390/rs17142346.
- Paper D: Qin, R. et al. (2025). *Spatiotemporal masked pre-training for
  advancing crop mapping on satellite image time series with limited labels
  (STCLN)*. International Journal of Applied Earth Observation and
  Geoinformation.

Convencion Polars
-----------------
Todas las funciones publicas reciben/devuelven :class:`polars.DataFrame` o
estructuras nativas de Python; ``numpy`` aparece unicamente de forma interna
en el borde tecnico de ``scipy``/``sklearn`` o al operar sobre los tensores
``.npy`` crudos de PASTIS-R. Se usa ``structlog`` para logging estructurado,
nunca ``print``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "aggregate_rare_classes",
    "boundary_interior_stats",
    "boundary_pixel_mask",
    "cloud_gap_robustness",
    "compute_boundary_ratio",
    "confusion_symmetry_analysis",
    "phenology_calendar_features",
    "temporal_sampling_stats",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical names of the 4 phenological stages (Paper C, PVM crop-growth
#: calendar). The order matches the ``growth_stage`` index 0..3.
_PHENOLOGY_STAGE_NAMES: tuple[str, ...] = (
    "dormant",
    "green_up",
    "peak",
    "senescence",
)

#: Tolerance in days to consider that a day of the year is "covered" by
#: a satellite observation (Paper A, irregular revisit analysis).
_DOY_COVERAGE_TOLERANCE_DAYS: int = 15


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_numpy(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Convierte una secuencia o array a ``np.ndarray`` 1D.

    Borde tecnico para entrar a ``scipy``/``sklearn``. Si ``values`` ya es
    un ``np.ndarray`` se devuelve sin copia.

    Args:
        values: Secuencia de Python o ``np.ndarray``.

    Returns:
        ``np.ndarray`` resultante (no necesariamente 1D si la entrada es
        multidimensional).
    """
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(list(values))


def _neighbourhood_varies(window: np.ndarray) -> bool:
    """Indica si una ventana NxN contiene mas de un valor distinto.

    Helper de :func:`boundary_pixel_mask`. Un pixel es frontera (Paper B,
    Tarasiou et al. 2021) cuando no todas las ground truths de su vecindario
    comparten el mismo valor.

    Args:
        window: Sub-array 2D de la mascara semantica.

    Returns:
        ``True`` si la ventana tiene al menos dos valores distintos.
    """
    return bool(np.unique(window).size > 1)


def _doy_from_yyyymmdd(date_int: int) -> int:
    """Convierte un entero ``YYYYMMDD`` a day-of-year (1..366).

    Args:
        date_int: Fecha como entero ``YYYYMMDD`` (formato ``dates-S2`` de
            PASTIS-R).

    Returns:
        Day-of-year en ``[1, 366]``. Si la fecha es invalida devuelve ``0``.
    """
    try:
        date_str = f"{int(date_int):08d}"
        dt = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        year_start = np.datetime64(f"{date_str[:4]}-01-01")
        return int((dt - year_start) / np.timedelta64(1, "D")) + 1
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# (B) boundary_pixel_mask — Paper B (Tarasiou et al. 2021, arXiv:2104.04310)
# ---------------------------------------------------------------------------


def boundary_pixel_mask(
    semantic: np.ndarray,
    *,
    neighbourhood: int = 3,
) -> np.ndarray:
    """Marca pixeles de frontera de parcela sobre una mascara semantica.

    Implementa la definicion de frontera de Tarasiou et al. 2021 (Context-self
    contrastive pretraining, arXiv:2104.04310): un pixel es frontera cuando
    **no todas** las ground truths de su vecindario ``NxN`` comparten el mismo
    valor. El paper muestra (Fig. 2) que la varianza espectral de una parcela
    proviene casi enteramente de estos pixeles de borde.

    Args:
        semantic: Mascara semantica 2D ``(H, W)`` con ids de clase por pixel
            (canal ``TARGET[0]`` de PASTIS-R).
        neighbourhood: Lado de la ventana cuadrada (default 3 -> vecindario
            3x3). Debe ser impar y >= 3.

    Returns:
        Array booleano ``(H, W)`` con ``True`` en los pixeles de frontera.

    Raises:
        ValueError: Si ``semantic`` no es 2D o ``neighbourhood`` es par o < 3.
    """
    arr = np.asarray(semantic)
    if arr.ndim != 2:
        raise ValueError(f"semantic debe ser 2D (H, W); recibido ndim={arr.ndim}")
    if neighbourhood < 3 or neighbourhood % 2 == 0:
        raise ValueError(
            f"neighbourhood debe ser impar y >= 3; recibido {neighbourhood}"
        )

    h, w = arr.shape
    radius = neighbourhood // 2
    mask = np.zeros((h, w), dtype=bool)

    # Edge-replication padding to avoid introducing false contours.
    padded = np.pad(arr, radius, mode="edge")
    for i in range(h):
        for j in range(w):
            window = padded[i : i + neighbourhood, j : j + neighbourhood]
            mask[i, j] = _neighbourhood_varies(window)

    logger.info(
        "boundary_pixel_mask_computed",
        shape=(h, w),
        neighbourhood=neighbourhood,
        n_boundary=int(mask.sum()),
        boundary_ratio=float(mask.mean()) if mask.size else 0.0,
    )
    return mask


# ---------------------------------------------------------------------------
# (A) boundary_interior_stats — Paper B (Tarasiou et al. 2021, Fig. 2)
# ---------------------------------------------------------------------------


def boundary_interior_stats(
    patch: dict[str, Any],
    *,
    band_index: int = 6,
    neighbourhood: int = 3,
) -> pl.DataFrame:
    """Estadisticos espectrales por grupo interior / frontera / exterior.

    Reproduce el analisis de la Figura 2 de Tarasiou et al. 2021
    (arXiv:2104.04310): clasifica cada pixel de un patch PASTIS-R en uno de
    tres grupos a partir de la mascara semantica y reporta descriptivos de la
    banda elegida. El paper demuestra que los pixeles interiores son
    homogeneos y que la dispersion espectral vive en la frontera.

    Definicion de grupos:
        - ``exterior``: pixeles de fondo (clase 0).
        - ``boundary``: pixeles cuyo vecindario ``NxN`` no es homogeneo
          (ver :func:`boundary_pixel_mask`).
        - ``interior``: pixeles de parcela (clase > 0) no fronterizos.

    Args:
        patch: Diccionario de patch PASTIS-R tal como lo devuelve
            ``ml.ingest.pastis_loader.load_pastis_patch`` (keys ``s2`` con
            shape ``(T, 10, H, W)`` y ``semantic`` con shape ``(H, W)``).
        band_index: Indice de banda Sentinel-2 a analizar (default 6 = B08
            NIR, la banda usada en la Fig. 2 del paper).
        neighbourhood: Lado de la ventana para detectar fronteras (default 3).

    Returns:
        :class:`polars.DataFrame` con una fila por grupo y columnas
        ``group, mean, std, p25, p50, p75, count``. La banda se promedia
        temporalmente sobre el eje ``T`` antes de calcular los descriptivos.

    Raises:
        ValueError: Si ``patch`` carece de ``s2``/``semantic`` o ``band_index``
            esta fuera de rango.
    """
    schema: dict[str, Any] = {
        "group": pl.Utf8,
        "mean": pl.Float64,
        "std": pl.Float64,
        "p25": pl.Float64,
        "p50": pl.Float64,
        "p75": pl.Float64,
        "count": pl.Int64,
    }

    s2 = patch.get("s2")
    semantic = patch.get("semantic")
    if s2 is None or semantic is None:
        logger.warning("boundary_interior_stats_missing_data")
        return pl.DataFrame(schema=schema)

    s2_arr = np.asarray(s2, dtype=np.float64)
    if s2_arr.ndim != 4:
        raise ValueError(
            f"patch['s2'] debe ser 4D (T, bands, H, W); recibido ndim={s2_arr.ndim}"
        )
    n_bands = s2_arr.shape[1]
    if not 0 <= band_index < n_bands:
        raise ValueError(
            f"band_index={band_index} fuera de rango [0, {n_bands - 1}]"
        )

    semantic_arr = np.asarray(semantic)
    # Temporal average of the chosen band -> 2D map (H, W).
    band_map = s2_arr[:, band_index, :, :].mean(axis=0)

    boundary = boundary_pixel_mask(semantic_arr, neighbourhood=neighbourhood)
    is_parcel = semantic_arr > 0
    exterior = ~is_parcel
    boundary_in_parcel = boundary & is_parcel
    interior = is_parcel & ~boundary

    groups: dict[str, np.ndarray] = {
        "interior": interior,
        "boundary": boundary_in_parcel,
        "exterior": exterior,
    }

    rows: list[dict[str, Any]] = []
    for name, group_mask in groups.items():
        vals = band_map[group_mask]
        if vals.size == 0:
            rows.append(
                {
                    "group": name,
                    "mean": None,
                    "std": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "count": 0,
                }
            )
            continue
        rows.append(
            {
                "group": name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "count": int(vals.size),
            }
        )

    logger.info(
        "boundary_interior_stats_computed",
        band_index=band_index,
        n_interior=int(interior.sum()),
        n_boundary=int(boundary_in_parcel.sum()),
        n_exterior=int(exterior.sum()),
    )
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# (compute_boundary_ratio) — Paper B (Tarasiou et al. 2021)
# ---------------------------------------------------------------------------


def compute_boundary_ratio(
    patch: dict[str, Any],
    *,
    neighbourhood: int = 3,
) -> dict[int, float]:
    """Calcula la fraccion de pixeles frontera por instancia de parcela.

    Feature nuevo por parcela motivado por Tarasiou et al. 2021
    (arXiv:2104.04310): dado que el pixel de borde concentra la senal
    discriminante, la razon ``pixeles_frontera / pixeles_totales`` de una
    parcela es un descriptor de su geometria (parcelas pequenas o irregulares
    tienen ratio alto; parcelas grandes y compactas ratio bajo).

    Se documenta aqui (y no en ``ml/features/selection.py``) porque depende
    directamente de :func:`boundary_pixel_mask` y opera sobre el tensor
    crudo del patch PASTIS-R, no sobre el DataFrame wide-format de features
    que consume ``selection.py``.

    Args:
        patch: Diccionario de patch PASTIS-R (``ml.ingest.pastis_loader.
            load_pastis_patch``) con keys ``semantic`` y ``instance``
            (canales ``TARGET[0]`` y ``TARGET[1]``).
        neighbourhood: Lado de la ventana para detectar fronteras (default 3).

    Returns:
        Diccionario ``{instance_id: boundary_ratio}`` con un float en
        ``[0, 1]`` por instancia de parcela. La instancia 0 (fondo) se
        excluye. Vacio si el patch no trae ``instance``.
    """
    semantic = patch.get("semantic")
    instance = patch.get("instance")
    if semantic is None or instance is None:
        logger.warning("compute_boundary_ratio_missing_instance")
        return {}

    semantic_arr = np.asarray(semantic)
    instance_arr = np.asarray(instance)
    boundary = boundary_pixel_mask(semantic_arr, neighbourhood=neighbourhood)

    ratios: dict[int, float] = {}
    for inst_id in np.unique(instance_arr):
        iid = int(inst_id)
        if iid == 0:
            continue  # background
        inst_mask = instance_arr == inst_id
        total = int(inst_mask.sum())
        if total == 0:
            continue
        n_boundary = int((inst_mask & boundary).sum())
        ratios[iid] = float(n_boundary / total)

    logger.info(
        "compute_boundary_ratio_computed",
        n_instances=len(ratios),
        mean_ratio=float(np.mean(list(ratios.values()))) if ratios else 0.0,
    )
    return ratios


# ---------------------------------------------------------------------------
# (C) temporal_sampling_stats — Paper A (Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def temporal_sampling_stats(dates: list[int]) -> dict[str, float | int]:
    """Caracteriza la irregularidad de la revisita satelital.

    Analisis derivado de Russwurm & Korner 2018 (arXiv:1802.02080), que
    documenta como Sentinel-2 entrega adquisiciones con espaciado no uniforme
    (huecos por cobertura nubosa) y trata estos huecos como ruido temporal.
    Esta funcion cuantifica esa irregularidad para una serie concreta.

    Args:
        dates: Lista de fechas de adquisicion como enteros ``YYYYMMDD``
            (formato ``dates-S2`` de PASTIS-R). El orden interno no importa;
            se ordena antes de calcular gaps.

    Returns:
        Diccionario con:
            - ``n_obs``: numero de observaciones.
            - ``mean_gap_days``: gap medio entre adquisiciones consecutivas.
            - ``max_gap_days``: gap maximo.
            - ``min_gap_days``: gap minimo.
            - ``std_gap_days``: desviacion estandar de los gaps.
            - ``doy_coverage``: fraccion del ano (0..1) con al menos una
              observacion dentro de +/- 15 dias.
        Si ``dates`` tiene menos de 2 fechas validas, los campos de gap son
        ``0.0`` y ``doy_coverage`` se calcula con las fechas disponibles.
    """
    valid = [int(d) for d in dates if int(d) > 0]
    n_obs = len(valid)
    base: dict[str, float | int] = {
        "n_obs": n_obs,
        "mean_gap_days": 0.0,
        "max_gap_days": 0.0,
        "min_gap_days": 0.0,
        "std_gap_days": 0.0,
        "doy_coverage": 0.0,
    }
    if n_obs == 0:
        return base

    # Day-of-year of each acquisition for the annual coverage.
    doys = sorted(d for d in (_doy_from_yyyymmdd(v) for v in valid) if d > 0)

    if n_obs >= 2:
        # Gaps in absolute calendar days (not DOY, to cross years).
        ordered = sorted(valid)
        days = np.array(
            [
                (
                    np.datetime64(f"{d:08d}"[:4] + "-" + f"{d:08d}"[4:6] + "-" + f"{d:08d}"[6:8])
                    - np.datetime64("2000-01-01")
                )
                / np.timedelta64(1, "D")
                for d in ordered
            ],
            dtype=np.float64,
        )
        gaps = np.diff(days)
        base["mean_gap_days"] = float(np.mean(gaps))
        base["max_gap_days"] = float(np.max(gaps))
        base["min_gap_days"] = float(np.min(gaps))
        base["std_gap_days"] = float(np.std(gaps))

    # Coverage: fraction of the 365 days of the year with an observation at <=15 days.
    if doys:
        doy_arr = np.array(doys, dtype=np.int64)
        all_days = np.arange(1, 366)
        covered = np.zeros(all_days.size, dtype=bool)
        for d in doy_arr:
            covered |= np.abs(all_days - d) <= _DOY_COVERAGE_TOLERANCE_DAYS
        base["doy_coverage"] = float(covered.mean())

    logger.info(
        "temporal_sampling_stats_computed",
        n_obs=n_obs,
        mean_gap_days=base["mean_gap_days"],
        doy_coverage=base["doy_coverage"],
    )
    return base


# ---------------------------------------------------------------------------
# (D) confusion_symmetry_analysis — Paper A (Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def confusion_symmetry_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Descompone la matriz de confusion en componentes simetrica/asimetrica.

    Russwurm & Korner 2018 (arXiv:1802.02080) distinguen dos tipos de
    confusion entre clases: las **simetricas** (p. ej. triticale<->centeno)
    delatan similitud espectral/fenologica, mientras que las **asimetricas**
    apuntan a factores externos (desbalance de clases, errores de
    anotacion). Para cada par de clases ``(i, j)`` esta funcion calcula:

        - componente simetrica: ``min(C[i, j], C[j, i])``.
        - componente asimetrica: ``abs(C[i, j] - C[j, i])``.

    Args:
        y_true: Vector de etiquetas verdaderas.
        y_pred: Vector de etiquetas predichas (misma longitud que ``y_true``).
        class_names: Mapeo opcional ``{class_id: nombre}`` para etiquetar el
            resultado de forma legible.

    Returns:
        :class:`polars.DataFrame` con columnas ``class_a, class_b, symmetric,
        asymmetric, interpretation`` ordenado por confusion total
        (``symmetric + asymmetric``) descendente. ``interpretation`` es
        ``"spectral_similarity"`` cuando la componente simetrica domina, o
        ``"external_factor"`` en caso contrario.

    Raises:
        ValueError: Si ``y_true`` e ``y_pred`` tienen longitudes distintas.
    """
    schema: dict[str, Any] = {
        "class_a": pl.Utf8,
        "class_b": pl.Utf8,
        "symmetric": pl.Int64,
        "asymmetric": pl.Int64,
        "interpretation": pl.Utf8,
    }
    yt = _to_numpy(y_true).astype(np.int64).ravel()
    yp = _to_numpy(y_pred).astype(np.int64).ravel()
    if yt.size != yp.size:
        raise ValueError(
            f"y_true y_pred deben tener igual longitud; {yt.size} != {yp.size}"
        )
    if yt.size == 0:
        return pl.DataFrame(schema=schema)

    classes = sorted(set(yt.tolist()) | set(yp.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(yt.tolist(), yp.tolist(), strict=True):
        cm[idx[t], idx[p]] += 1

    def _label(class_id: int) -> str:
        if class_names is not None and class_id in class_names:
            return str(class_names[class_id])
        return str(class_id)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = int(cm[i, j])
            cji = int(cm[j, i])
            if cij == 0 and cji == 0:
                continue
            symmetric = min(cij, cji)
            asymmetric = abs(cij - cji)
            interpretation = (
                "spectral_similarity"
                if symmetric >= asymmetric
                else "external_factor"
            )
            rows.append(
                {
                    "class_a": _label(classes[i]),
                    "class_b": _label(classes[j]),
                    "symmetric": symmetric,
                    "asymmetric": asymmetric,
                    "interpretation": interpretation,
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df = df.with_columns(
        (pl.col("symmetric") + pl.col("asymmetric")).alias("__total")
    ).sort("__total", descending=True).drop("__total")

    logger.info(
        "confusion_symmetry_analysis_computed",
        n_classes=n,
        n_pairs=df.height,
    )
    return df


# ---------------------------------------------------------------------------
# (E) aggregate_rare_classes — Paper A (Russwurm & Korner 2018)
# ---------------------------------------------------------------------------


def aggregate_rare_classes(
    y: pl.Series,
    *,
    min_count: int = 400,
    other_label: int = -1,
) -> tuple[pl.Series, dict[Any, Any]]:
    """Colapsa clases poco frecuentes en una unica etiqueta agregada.

    Russwurm & Korner 2018 (arXiv:1802.02080) reportan una distribucion de
    clases muy desbalanceada (maiz 919k px vs guisantes 6k px) y agregan las
    clases poco frecuentes aplicando un umbral de conteo ("classes occurring
    >= 400 times"), reduciendo ~200 etiquetas a 17. Esta funcion replica esa
    estrategia: las clases con menos de ``min_count`` ocurrencias se reasignan
    a ``other_label``.

    Args:
        y: Serie Polars con las etiquetas de clase (entera).
        min_count: Umbral minimo de ocurrencias para conservar una clase
            como categoria propia (default 400, valor del paper).
        other_label: Etiqueta destino para las clases agregadas (default -1).

    Returns:
        Tupla ``(remapped_series, report)`` donde:
            - ``remapped_series`` es la serie con las clases raras colapsadas
              (mismo ``name`` que ``y``).
            - ``report`` contiene ``{original_class: count}`` por cada clase
              original mas la key ``"aggregated"`` con la lista de clases
              colapsadas y ``"min_count"`` con el umbral usado.
    """
    # value_counts returns columns [<name>, "count"]; the first column
    # contains the distinct values, the second their frequency.
    vc = y.value_counts(sort=True)
    value_col = vc.columns[0]
    count_map: dict[int, int] = {
        int(v): int(c)
        for v, c in zip(
            vc.get_column(value_col).to_list(),
            vc.get_column("count").to_list(),
            strict=True,
        )
        if v is not None
    }

    aggregated = sorted(c for c, n in count_map.items() if n < min_count)
    aggregated_set = set(aggregated)

    original = y.to_list()
    remapped = [
        other_label if (v is not None and int(v) in aggregated_set) else v
        for v in original
    ]
    remapped_series = pl.Series(y.name or "class", remapped, dtype=pl.Int64)

    # The report mixes int keys (per-class count) with str keys
    # ("aggregated", "min_count", "other_label"), hence the type is dict[Any, Any].
    report: dict[Any, Any] = {int(c): int(n) for c, n in count_map.items()}
    report["aggregated"] = aggregated
    report["min_count"] = int(min_count)
    report["other_label"] = int(other_label)

    logger.info(
        "aggregate_rare_classes_done",
        n_classes_original=len(count_map),
        n_aggregated=len(aggregated),
        min_count=min_count,
    )
    return remapped_series, report


# ---------------------------------------------------------------------------
# (F) phenology_calendar_features — Paper C (PVM 2025, RS 17(14):2346)
# ---------------------------------------------------------------------------


def phenology_calendar_features(
    temporal_df: pl.DataFrame,
    *,
    doy_col: str = "peak_doy",
    n_stages: int = 4,
) -> pl.DataFrame:
    """Deriva una etapa de crecimiento categorica desde un day-of-year.

    Inspirado en el "crop-growth calendar" del Phenology-Aware Transformer
    (PVM, Remote Sensing 17(14):2346, 2025): el modelo codifica las etapas
    fenologicas (siembra/crecimiento/pico/cosecha) como un vector indexado por
    day-of-year y pondera la atencion temporal con esas pistas. Aqui se
    construye una version EDA del concepto: el day-of-year de una metrica
    fenologica se discretiza en ``n_stages`` etapas calendario.

    Args:
        temporal_df: DataFrame de features temporales que ya contiene la
            columna ``doy_col`` (tipicamente ``peak_doy``, ``sog_doy`` o
            ``senescence_doy`` producidas por
            ``ml.features.temporal_features.extract_temporal_features``).
        doy_col: Nombre de la columna de day-of-year a discretizar
            (default ``"peak_doy"``).
        n_stages: Numero de etapas fenologicas (default 4 ->
            ``dormant/green_up/peak/senescence``). El ano se parte en
            ``n_stages`` intervalos iguales de DOY.

    Returns:
        DataFrame original con dos columnas nuevas:
            - ``growth_stage`` (Int64): indice de etapa en ``[0, n_stages-1]``.
            - ``growth_stage_name`` (Utf8): nombre legible de la etapa.
        Las filas con ``doy_col`` nulo reciben ``growth_stage = -1`` y
        ``growth_stage_name = "unknown"``.

    Raises:
        ValueError: Si ``doy_col`` no existe en ``temporal_df`` o
            ``n_stages < 2``.
    """
    if doy_col not in temporal_df.columns:
        raise ValueError(
            f"doy_col {doy_col!r} no presente en temporal_df.columns; "
            f"disponibles: {temporal_df.columns}"
        )
    if n_stages < 2:
        raise ValueError(f"n_stages debe ser >= 2; recibido {n_stages}")

    # Stage names: use the canonical ones when n_stages == 4, otherwise
    # generate generic labels stage_0..stage_{n-1}.
    if n_stages == len(_PHENOLOGY_STAGE_NAMES):
        stage_names = list(_PHENOLOGY_STAGE_NAMES)
    else:
        stage_names = [f"stage_{i}" for i in range(n_stages)]

    stage_width = 366.0 / n_stages
    doy = temporal_df.get_column(doy_col).cast(pl.Float64).to_list()

    stage_idx: list[int] = []
    stage_lbl: list[str] = []
    for value in doy:
        if value is None or not np.isfinite(value):
            stage_idx.append(-1)
            stage_lbl.append("unknown")
            continue
        idx = int(min(n_stages - 1, max(0, int((value - 1.0) / stage_width))))
        stage_idx.append(idx)
        stage_lbl.append(stage_names[idx])

    out = temporal_df.with_columns(
        [
            pl.Series("growth_stage", stage_idx, dtype=pl.Int64),
            pl.Series("growth_stage_name", stage_lbl, dtype=pl.Utf8),
        ]
    )
    logger.info(
        "phenology_calendar_features_done",
        doy_col=doy_col,
        n_stages=n_stages,
        n_rows=out.height,
        n_unknown=sum(1 for s in stage_idx if s == -1),
    )
    return out


# ---------------------------------------------------------------------------
# (G) cloud_gap_robustness — Paper D (Qin et al. 2025, STCLN)
# ---------------------------------------------------------------------------


def cloud_gap_robustness(
    temporal_extractor_callable: Callable[[Any], pl.DataFrame],
    parcel_timeseries: Any,
    *,
    mask_fractions: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6),
    seed: int = 42,
) -> pl.DataFrame:
    """Mide la deriva de features al simular huecos de nubes en la serie.

    Inspirado en el spatiotemporal masking de Qin et al. 2025 (STCLN, Int. J.
    Applied Earth Obs. Geoinf.): el preentrenamiento enmascara parches
    temporales de la serie de imagenes para aprender representaciones
    robustas. Aqui se usa la misma idea como herramienta de EDA: se eliminan
    fracciones crecientes de timesteps de una parcela, se re-ejecuta el
    extractor de features temporales y se cuantifica cuanto se desplazan los
    valores respecto al baseline sin enmascarar.

    Args:
        temporal_extractor_callable: Funcion que recibe una serie temporal de
            parcela (``xarray.DataArray`` con dims ``(time, band)``) y
            devuelve un :class:`polars.DataFrame` de features (tipicamente
            ``ml.features.temporal_features.extract_temporal_features``).
        parcel_timeseries: Serie temporal de la parcela aceptada por
            ``temporal_extractor_callable``. Debe exponer una dimension
            ``time`` indexable via ``.isel(time=...)`` (contrato xarray).
        mask_fractions: Fracciones de timesteps a eliminar. Debe incluir
            ``0.0`` (baseline) como primer elemento para que la deriva sea
            relativa a la serie completa.
        seed: Semilla del generador aleatorio (reproducibilidad).

    Returns:
        :class:`polars.DataFrame` con columnas ``mask_fraction,
        n_timesteps_kept, feature_name, value, drift_from_baseline``. Una fila
        por (fraccion, feature numerico). ``drift_from_baseline`` es el valor
        absoluto de la diferencia respecto al run con ``mask_fraction == 0``.

    Notes:
        Si una fraccion deja menos de 2 timesteps, esa fraccion se omite (el
        extractor temporal requiere >= 2 puntos para interpolar). Solo se
        rastrean columnas numericas distintas de ``parcel_id``/``year``.
    """
    schema: dict[str, Any] = {
        "mask_fraction": pl.Float64,
        "n_timesteps_kept": pl.Int64,
        "feature_name": pl.Utf8,
        "value": pl.Float64,
        "drift_from_baseline": pl.Float64,
    }
    rng = np.random.default_rng(seed)

    # Total number of timesteps of the series.
    try:
        n_total = int(parcel_timeseries.sizes["time"])
    except (AttributeError, KeyError, TypeError):
        try:
            n_total = len(parcel_timeseries.coords["time"])
        except Exception:  # noqa: BLE001
            logger.warning("cloud_gap_robustness_no_time_dim")
            return pl.DataFrame(schema=schema)

    if n_total < 2:
        logger.warning("cloud_gap_robustness_series_too_short", n_total=n_total)
        return pl.DataFrame(schema=schema)

    baseline_values: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for fraction in mask_fractions:
        n_keep = round(n_total * (1.0 - fraction))
        n_keep = max(0, min(n_total, n_keep))
        if n_keep < 2:
            logger.info(
                "cloud_gap_robustness_skip_fraction",
                mask_fraction=fraction,
                n_keep=n_keep,
            )
            continue

        keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        masked = parcel_timeseries.isel(time=keep_idx)
        try:
            features = temporal_extractor_callable(masked)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cloud_gap_robustness_extractor_failed",
                mask_fraction=fraction,
                error=str(exc),
            )
            continue

        numeric_cols = [
            c
            for c in features.columns
            if c not in ("parcel_id", "year")
            and features.schema[c].is_numeric()
        ]
        if features.height == 0:
            continue
        feature_row = features.row(0, named=True)

        for col in numeric_cols:
            raw = feature_row[col]
            value = float(raw) if raw is not None and np.isfinite(raw) else float("nan")
            if abs(fraction) < 1e-9:
                baseline_values[col] = value
                drift = 0.0
            else:
                base_val = baseline_values.get(col)
                if base_val is None or not np.isfinite(base_val) or not np.isfinite(value):
                    drift = float("nan")
                else:
                    drift = abs(value - base_val)
            rows.append(
                {
                    "mask_fraction": float(fraction),
                    "n_timesteps_kept": int(n_keep),
                    "feature_name": col,
                    "value": value,
                    "drift_from_baseline": drift,
                }
            )

    logger.info(
        "cloud_gap_robustness_done",
        n_total_timesteps=n_total,
        mask_fractions=list(mask_fractions),
        n_rows=len(rows),
    )
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)
