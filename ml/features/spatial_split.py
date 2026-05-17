"""Spatial K-fold split por tessellation H3 + KMeans (US-016).

Construye particiones train/val/test estratificadas espacialmente para evitar
leakage entre vecinos cercanos. La estrategia:

1. Calcula el centroide de cada parcela en EPSG:4326.
2. Asigna cada parcela a una celda H3 de resolución ``h3_res`` (default 5,
   ~252 km² por celda).
3. Agrupa las celdas H3 únicas y clusteriza sus centroides con
   ``KMeans(n_clusters=k)`` — cada cluster define un fold.
4. Las parcelas heredan el fold de su celda H3 contenedora.
5. Aplica buffer de exclusión: las parcelas a < ``buffer_km`` de la frontera
   inter-fold se sacan del val/test del fold actual y se devuelven al train
   global del fold para evitar leakage de vecinos.
6. Dentro de cada train interno, se separa una fracción ``val_fraction`` con
   ``np.random.default_rng(random_state)``.

Devuelve K :class:`FoldAssignment` cada uno con ``train_ids``, ``val_ids``,
``test_ids`` disjuntos por construcción.

Referencias agronómicas:

- Lyons et al. 2018 — *A comparison of resampling methods for remote sensing
  classification and accuracy assessment*. RSE 208, 145-153. DOI
  10.1016/j.rse.2018.02.026 — justifica spatial CV en remote sensing.
- Roberts et al. 2017 — *Cross-validation strategies for data with temporal,
  spatial, hierarchical, or phylogenetic structure*. Ecography 40, 913-929.

Dependencia ``h3``
------------------
``h3-py`` 4.4.2 está declarado en ``pyproject.toml`` grupo ``geo`` desde
US-016 (aprobado por Arthur 2026-05-17). El import sigue siendo condicional
para que el módulo se cargue en entornos minimalistas que solo necesiten
el dataclass; si ``h3`` no está disponible, la API lanza ``ImportError``
con instrucciones explícitas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import structlog
from shapely.geometry import Point
from sklearn.cluster import KMeans

# Import diferido para no romper colectores que solo necesiten el dataclass.
try:
    import h3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - cubierto por test condicional
    h3 = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

__all__ = [
    "FoldAssignment",
    "build_spatial_kfold",
]


# ---------------------------------------------------------------------------
# Dataclass de salida.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldAssignment:
    """Asignación de parcel_ids por fold (train/val/test disjuntos).

    Attributes:
        fold_id: Identificador entero del fold ``[0, k)``.
        train_ids: Tupla de ``parcel_id`` en el split train (los demás folds
            menos los excluidos por buffer).
        val_ids: Subset del train interno reservado como validación.
        test_ids: Tupla de ``parcel_id`` cuyas celdas H3 pertenecen al
            cluster KMeans del fold (excluyendo los buffereados).
    """

    fold_id: int
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]


# ---------------------------------------------------------------------------
# API pública.
# ---------------------------------------------------------------------------


def build_spatial_kfold(
    parcels: gpd.GeoDataFrame,
    *,
    k: int = 5,
    h3_res: int = 5,
    buffer_km: float = 1.0,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> list[FoldAssignment]:
    """Construye K folds espaciales con tessellation H3 + KMeans.

    Args:
        parcels: GeoDataFrame con columnas ``parcel_id`` y ``geometry`` en
            EPSG:4326 (POLYGON o POINT). El centroide se calcula con
            ``GeoSeries.centroid`` interno.
        k: Número de folds (default 5). Debe ser ≥ 2 y ≤ número de celdas
            H3 únicas. Si hay menos celdas que ``k``, KMeans degrada
            asignando una celda por fold y rellenando con vacíos.
        h3_res: Resolución H3 (default 5 ≈ 252 km²). Valores válidos
            ``[0, 15]``; 5 da ~30 hex en Italia agrícola.
        buffer_km: Distancia mínima en km para excluir parcelas cercanas a
            la frontera entre folds. ``0.0`` desactiva la exclusión.
        val_fraction: Fracción del train interno usada como validación por
            fold. Default 0.2 (20% de train → val).
        random_state: Seed determinista para KMeans y val shuffle.

    Returns:
        Lista de K :class:`FoldAssignment` con ``train_ids``, ``val_ids``,
        ``test_ids`` disjuntos. Garantiza ``parcel_id`` no se repite en
        más de un fold (sumando train+val+test ⊆ parcels).

    Raises:
        ImportError: si ``h3-py`` no está instalado (la US-016 documenta
            ``h3 ^4.1.2`` como dependencia pendiente de aprobación).
        ValueError: si ``k < 2`` o si ``parcels`` no contiene las columnas
            requeridas.
    """
    if h3 is None:
        raise ImportError(
            "El paquete `h3` no está instalado. US-016 requiere `h3 ^4.1.2` "
            "(coordinar con Arthur antes del coding). Fallback documentado en "
            "`docs/us-planning/us-016.md` §2.4 (grid rectangular)."
        )
    if k < 2:
        raise ValueError(f"`k` debe ser ≥ 2; recibido {k}.")
    if "parcel_id" not in parcels.columns:
        raise ValueError("`parcels` debe contener la columna `parcel_id`.")
    if parcels.geometry.name not in parcels.columns:
        raise ValueError(
            "`parcels` debe tener geometría activa (parcels.set_geometry('geom'))."
        )
    if parcels.crs is None:
        logger.warning("spatial_kfold_crs_missing", note="Asumiendo EPSG:4326")
        parcels = parcels.set_crs("EPSG:4326")
    elif parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs("EPSG:4326")

    # Centroide en EPSG:3857 (métrico) para evitar UserWarning de geopandas
    # sobre operaciones geométricas en CRS geográfico. Re-proyectamos a 4326
    # para alimentar h3 (que espera lat/lng).
    centroids = parcels.geometry.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
    parcel_ids = parcels["parcel_id"].astype("int64").to_numpy()
    n_parcels = len(parcel_ids)
    if n_parcels == 0:
        return [FoldAssignment(fold_id=i, train_ids=(), val_ids=(), test_ids=()) for i in range(k)]

    # 1) Asigna cada parcela a una celda H3.
    h3_cells = np.array(
        [_assign_h3_cell(c, h3_res) for c in centroids],
        dtype=object,
    )

    # 2) Clusteriza las celdas únicas con KMeans.
    unique_cells, inv = np.unique(h3_cells, return_inverse=True)
    cell_centroids = np.array(
        [_cell_to_latlng(c) for c in unique_cells],
        dtype=np.float64,
    )
    effective_k = min(k, len(unique_cells))
    if effective_k < k:
        logger.warning(
            "spatial_kfold_k_clamped",
            requested=k,
            effective=effective_k,
            unique_h3_cells=len(unique_cells),
        )
    kmeans = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10,
    )
    cell_folds = kmeans.fit_predict(cell_centroids)
    parcel_folds = cell_folds[inv]

    # 3) Aplica buffer de exclusión sobre fronteras inter-fold.
    excluded_mask = _apply_buffer_exclusion(
        parcel_ids=parcel_ids,
        parcel_folds=parcel_folds,
        centroids=np.array([(c.x, c.y) for c in centroids], dtype=np.float64),
        buffer_km=buffer_km,
    )

    rng = np.random.default_rng(random_state)
    assignments: list[FoldAssignment] = []
    for fold_id in range(k):
        if fold_id >= effective_k:
            assignments.append(
                FoldAssignment(fold_id=fold_id, train_ids=(), val_ids=(), test_ids=())
            )
            continue
        test_mask = (parcel_folds == fold_id) & (~excluded_mask)
        train_mask = (parcel_folds != fold_id) & (~excluded_mask)

        train_pool = parcel_ids[train_mask]
        test_ids = tuple(int(x) for x in parcel_ids[test_mask])

        if len(train_pool) == 0:
            assignments.append(
                FoldAssignment(
                    fold_id=fold_id,
                    train_ids=(),
                    val_ids=(),
                    test_ids=test_ids,
                )
            )
            continue

        shuffled = train_pool.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(np.round(len(shuffled) * val_fraction))) if val_fraction > 0 else 0
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
        val_ids = tuple(int(x) for x in shuffled[:n_val])
        train_only_ids = tuple(int(x) for x in shuffled[n_val:])

        assignments.append(
            FoldAssignment(
                fold_id=fold_id,
                train_ids=train_only_ids,
                val_ids=val_ids,
                test_ids=test_ids,
            )
        )

    logger.info(
        "spatial_kfold_built",
        n_parcels=int(n_parcels),
        n_unique_h3=len(unique_cells),
        k=int(k),
        effective_k=int(effective_k),
        excluded=int(excluded_mask.sum()),
        buffer_km=float(buffer_km),
        h3_res=int(h3_res),
    )
    return assignments


# ---------------------------------------------------------------------------
# Helpers privados.
# ---------------------------------------------------------------------------


def _assign_h3_cell(centroid: Point, h3_res: int) -> str:
    """Devuelve la celda H3 que contiene el centroide.

    Compatible con h3-py 4.x (``latlng_to_cell``) y 3.x (``geo_to_h3``).
    Si la API 4.x no está disponible, hace fallback.
    """
    assert h3 is not None  # garantizado por guard en build_spatial_kfold
    lat = float(centroid.y)
    lng = float(centroid.x)
    fn: Any
    if hasattr(h3, "latlng_to_cell"):  # h3-py 4.x
        fn = h3.latlng_to_cell
    elif hasattr(h3, "geo_to_h3"):  # h3-py 3.x fallback
        fn = h3.geo_to_h3
    else:  # pragma: no cover
        raise ImportError(
            "h3-py expone una API desconocida; ni `latlng_to_cell` ni `geo_to_h3`."
        )
    return str(fn(lat, lng, h3_res))


def _cell_to_latlng(cell: str) -> tuple[float, float]:
    """Devuelve ``(lat, lng)`` del centroide de la celda H3.

    Compatible con h3-py 4.x (``cell_to_latlng``) y 3.x (``h3_to_geo``).
    """
    assert h3 is not None
    fn: Any
    if hasattr(h3, "cell_to_latlng"):  # h3-py 4.x
        fn = h3.cell_to_latlng
    elif hasattr(h3, "h3_to_geo"):  # h3-py 3.x fallback
        fn = h3.h3_to_geo
    else:  # pragma: no cover
        raise ImportError("h3-py expone una API desconocida; no se halló accessor centroide.")
    lat, lng = fn(cell)
    return float(lat), float(lng)


def _apply_buffer_exclusion(
    *,
    parcel_ids: np.ndarray,
    parcel_folds: np.ndarray,
    centroids: np.ndarray,
    buffer_km: float,
) -> np.ndarray:
    """Marca parcelas excluidas por estar < ``buffer_km`` de la frontera inter-fold.

    Implementación O(N) usando distancia haversine aproximada con conversión
    lon/lat → metros vía equirectangular (suficiente para Italia continental;
    error < 1% en latitudes 38-46°N). Para datasets > 10k parcelas, una
    implementación con KDTree en EPSG:3857 sería más rápida — diferido a
    una optimización futura.

    Args:
        parcel_ids: Vector ``(N,)`` de identificadores.
        parcel_folds: Vector ``(N,)`` de fold asignado por parcela.
        centroids: Matrix ``(N, 2)`` con ``(lon, lat)`` en EPSG:4326.
        buffer_km: Radio de exclusión. Si ``0.0`` no se excluye nada.

    Returns:
        Array boolean ``(N,)`` donde ``True`` marca parcela a excluir.
    """
    n = len(parcel_ids)
    excluded = np.zeros(n, dtype=bool)
    if buffer_km <= 0.0 or n == 0:
        return excluded

    # Conversión equirectangular (centro Italia ~ 43°N).
    deg_per_km_lat = 1.0 / 111.0
    mid_lat = float(np.mean(centroids[:, 1]))
    deg_per_km_lon = 1.0 / (111.320 * max(np.cos(np.radians(mid_lat)), 1e-3))
    buffer_deg_lat = buffer_km * deg_per_km_lat
    buffer_deg_lon = buffer_km * deg_per_km_lon
    # Norma adimensional: trabajamos en grados; bbox-prefilter + distancia.
    for i in range(n):
        if excluded[i]:
            continue
        lon_i, lat_i = centroids[i]
        dlat = centroids[:, 1] - lat_i
        dlon = centroids[:, 0] - lon_i
        bbox_mask = (np.abs(dlat) < buffer_deg_lat) & (np.abs(dlon) < buffer_deg_lon)
        cross_fold = bbox_mask & (parcel_folds != parcel_folds[i])
        if cross_fold.any():
            # Distancia métrica real con equirectangular.
            dist_km = np.sqrt(
                (dlon[cross_fold] / deg_per_km_lon) ** 2
                + (dlat[cross_fold] / deg_per_km_lat) ** 2
            )
            if (dist_km < buffer_km).any():
                # Excluimos la parcela actual y sus vecinas dentro del buffer.
                idxs = np.where(cross_fold)[0][dist_km < buffer_km]
                excluded[i] = True
                excluded[idxs] = True
    return excluded
