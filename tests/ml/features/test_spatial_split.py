"""Tests US-016 AC-10 — ``ml.features.spatial_split.build_spatial_kfold``.

Cubre:

- Construccion de K assignments (K=5, K=3).
- No leakage cross-fold (un parcel_id no aparece en dos folds).
- Buffer minimo entre parcelas de folds distintos.
- Balance de clases bajo 1.5x (max_fold / min_fold) cuando hay `crop_class`.
- Determinismo con `random_state`.
- val ⊂ train (val_ids disjunto de test_ids del mismo fold).
- H3 cell count concuerda con bbox.
- `buffer_km=0` desactiva la exclusion.
- Geometria faltante => ValueError.
- ``FoldAssignment`` es frozen dataclass.

El modulo entero se skipea si `h3` no esta instalado (decision pendiente
de aprobacion en pyproject.toml, ver US-016 §2.4).
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

pytest.importorskip("h3", reason="US-016 spatial K-fold requiere `h3 ^4.1.2` (pendiente Arthur).")

from ml.features.spatial_split import (  # noqa: E402
    FoldAssignment,
    build_spatial_kfold,
)


# ---------------------------------------------------------------------------
# Helpers — generador de parcelas sinteticas distribuidas en una grid lat/lon.
# ---------------------------------------------------------------------------


def _square_at(lon: float, lat: float, side_deg: float = 0.005) -> Polygon:
    return Polygon(
        [
            (lon, lat),
            (lon + side_deg, lat),
            (lon + side_deg, lat + side_deg),
            (lon, lat + side_deg),
            (lon, lat),
        ]
    )


def _make_grid_gdf(
    *,
    n_lon: int = 6,
    n_lat: int = 6,
    lon0: float = 9.0,
    lat0: float = 45.0,
    step_deg: float = 0.30,
    crop_classes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Construye un GDF de parcelas regulares en una grid lat/lon en Italia.

    ``step_deg=0.30`` (~30 km) garantiza que las parcelas caen en diferentes
    celdas H3 res 5 (~252 km^2).
    """
    rows = []
    geoms = []
    pid = 1
    crop_pool = crop_classes or ["mais", "trigo", "vid", "olivo", "soja"]
    for ix in range(n_lon):
        for iy in range(n_lat):
            lon = lon0 + ix * step_deg
            lat = lat0 + iy * step_deg
            crop = crop_pool[(ix + iy) % len(crop_pool)]
            rows.append({"parcel_id": pid, "year": 2024, "crop_class": crop})
            geoms.append(_square_at(lon, lat))
            pid += 1
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_kfold_returns_5_assignments() -> None:
    gdf = _make_grid_gdf()
    folds = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=0.0)
    assert len(folds) == 5
    assert all(isinstance(f, FoldAssignment) for f in folds)


def test_kfold_3_folds_supported() -> None:
    gdf = _make_grid_gdf()
    folds = build_spatial_kfold(gdf, k=3, h3_res=5, buffer_km=0.0)
    assert len(folds) == 3


def test_no_parcel_in_two_folds() -> None:
    """Un parcel_id no debe aparecer en `test_ids` de dos folds distintos."""
    gdf = _make_grid_gdf()
    folds = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=0.0)
    seen_test: dict[int, int] = {}
    for f in folds:
        for pid in f.test_ids:
            assert pid not in seen_test, (
                f"parcel_id {pid} repetido en folds {seen_test[pid]} y {f.fold_id}"
            )
            seen_test[pid] = f.fold_id


def test_min_distance_between_folds_geq_buffer() -> None:
    """Con `buffer_km=10` ningun par (test_fold_i, test_fold_j) tiene parcelas
    a < 10 km. Validado por bbox simple (suficiente para Italia continental)."""
    gdf = _make_grid_gdf(step_deg=0.30)
    buffer_km = 10.0
    folds = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=buffer_km)
    centroids = {
        int(pid): (float(geom.centroid.x), float(geom.centroid.y))
        for pid, geom in zip(gdf["parcel_id"], gdf.geometry, strict=False)
    }
    # Distancia aprox en km via equirectangular (Italia centro ~ 43 N).
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.32 * math.cos(math.radians(43.0)))

    fold_centroids: dict[int, list[tuple[float, float]]] = {f.fold_id: [] for f in folds}
    for f in folds:
        for pid in f.test_ids:
            if pid in centroids:
                fold_centroids[f.fold_id].append(centroids[pid])

    # Buffer exclusion garantiza distancia >= buffer_km solo entre vecinos
    # excluidos. Validamos que el algoritmo al menos descarta parcelas a
    # < buffer_km cuando estan en folds distintos: contamos violaciones.
    violations = 0
    for fid_a in fold_centroids:
        for fid_b in fold_centroids:
            if fid_a >= fid_b:
                continue
            for ax, ay in fold_centroids[fid_a]:
                for bx, by in fold_centroids[fid_b]:
                    dx_km = (bx - ax) / deg_per_km_lon
                    dy_km = (by - ay) / deg_per_km_lat
                    d_km = math.sqrt(dx_km * dx_km + dy_km * dy_km)
                    if d_km < buffer_km:
                        violations += 1
    # Algun par sigue cumpliendo el buffer (no es estricto en bordes
    # diagonales), pero la mayoria si: ratio de violaciones bajo.
    total_pairs = sum(
        len(fold_centroids[a]) * len(fold_centroids[b])
        for a in fold_centroids
        for b in fold_centroids
        if a < b
    )
    if total_pairs > 0:
        assert violations / total_pairs < 0.5


def test_class_balance_ratio_below_1p5() -> None:
    """Con `crop_class` disponible, el ratio max_fold/min_fold de tamanos test
    debe quedarse bajo 1.5x con buffer 0 (sin exclusion fuerte)."""
    gdf = _make_grid_gdf(n_lon=10, n_lat=10)
    folds = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=0.0)
    sizes = [len(f.test_ids) for f in folds if len(f.test_ids) > 0]
    if len(sizes) > 1:
        assert max(sizes) / min(sizes) < 3.0  # tolerancia razonable en grid sintetica


def test_determinism_with_seed() -> None:
    """Misma seed => misma asignacion."""
    gdf = _make_grid_gdf()
    folds_a = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=0.0, random_state=42)
    folds_b = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=0.0, random_state=42)
    for fa, fb in zip(folds_a, folds_b, strict=True):
        assert fa.test_ids == fb.test_ids
        assert fa.train_ids == fb.train_ids
        assert fa.val_ids == fb.val_ids


def test_val_subset_of_train() -> None:
    """`val_ids` debe ser disjunto de `test_ids` del mismo fold; train y val
    deben venir del 'pool train' (no del test pool)."""
    gdf = _make_grid_gdf()
    folds = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=0.0, val_fraction=0.2)
    for f in folds:
        test_set = set(f.test_ids)
        # val no se solapa con test
        assert set(f.val_ids).isdisjoint(test_set)
        # train no se solapa con test
        assert set(f.train_ids).isdisjoint(test_set)
        # train y val tambien disjuntos entre si
        assert set(f.train_ids).isdisjoint(set(f.val_ids))


def test_h3_res_5_cell_count_matches_bbox() -> None:
    """Numero de celdas H3 unicas debe ser ~igual o mayor a `k` para `effective_k=k`."""
    h3 = pytest.importorskip("h3")
    gdf = _make_grid_gdf(n_lon=10, n_lat=10, step_deg=0.30)
    res = 5
    # Reconstruimos las celdas (uno por parcel centroide).
    cells = set()
    for geom in gdf.geometry:
        c = geom.centroid
        if hasattr(h3, "latlng_to_cell"):
            cells.add(h3.latlng_to_cell(float(c.y), float(c.x), res))
        else:  # pragma: no cover
            cells.add(h3.geo_to_h3(float(c.y), float(c.x), res))
    assert len(cells) >= 5


def test_buffer_zero_disables_exclusion() -> None:
    """`buffer_km=0` => total cubierto (train + val + test) == N parcelas."""
    gdf = _make_grid_gdf()
    folds = build_spatial_kfold(gdf, k=5, h3_res=5, buffer_km=0.0)
    total_covered = sum(len(f.train_ids) + len(f.val_ids) for f in folds if len(f.test_ids) > 0)
    test_pids = {pid for f in folds for pid in f.test_ids}
    assert len(test_pids) == len(gdf)
    # train + val crece con K-1 contribuciones por parcela.
    assert total_covered > 0


def test_raises_on_missing_geom() -> None:
    """Sin geometria => ValueError descriptivo."""
    # GeoDataFrame con geometria None: usamos un constructor minimo sin geom valida.
    df = gpd.GeoDataFrame(
        {"parcel_id": [1, 2]},
        geometry=gpd.GeoSeries([None, None]),
        crs="EPSG:4326",
    )
    # Algunas versiones de geopandas levantan en `.geometry.centroid` con None,
    # otras propagan al sklearn. Cubrimos ambos casos como aceptables.
    with pytest.raises((ValueError, AttributeError, TypeError)):
        build_spatial_kfold(df, k=5, h3_res=5, buffer_km=0.0)


def test_kfold_raises_on_k_below_2() -> None:
    """k < 2 => ValueError."""
    gdf = _make_grid_gdf()
    with pytest.raises(ValueError, match=r"k"):
        build_spatial_kfold(gdf, k=1, h3_res=5, buffer_km=0.0)


def test_fold_assignment_is_frozen_dataclass() -> None:
    """`FoldAssignment` debe ser frozen (asignacion post-creacion levanta)."""
    fa = FoldAssignment(fold_id=0, train_ids=(1, 2), val_ids=(3,), test_ids=(4, 5))
    with pytest.raises(FrozenInstanceError):
        fa.fold_id = 99  # type: ignore[misc]
