"""Vectorize REAL PASTIS parcels with the MODEL'S PREDICTION (held-out fold).

Companion to ``build_demo_parcels_real.py`` (which paints ground truth). Here we
show how the model *recognises* crops: for a held-out **fold-5** patch (the model
is trained on folds 1-4, so the prediction is honest / out-of-sample), each real
parcel gets its predicted crop from the deployed XGBoost-AlphaEarth classifier
(``ml.agent.tools.classify._load_classifier``), joined to its geometry via the
per-patch instance id.

Output GeoJSON properties per parcel: ``crop_class`` (= predicted, so the map
paints predictions), ``pred_class``, ``true_class`` and ``correct`` (for a
hits/errors view like the notebook's panel d). ``metadata.accuracy`` carries the
parcel accuracy over the painted set.

Usage:
    poetry run python scripts/build_demo_parcels_prediction.py
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import rasterio.features
import structlog
import typer
from pyproj import Transformer
from rasterio.transform import from_bounds

from ml.data.pastis_filter import PASTIS_CLASS_NAMES, SEMANTIC18_CLASS_NAMES

log = structlog.get_logger(__name__)

_PASTIS_ROOT = Path("data/PASTIS-R")
_ANN = _PASTIS_ROOT / "ANNOTATIONS"
_INST = _PASTIS_ROOT / "INSTANCE_ANNOTATIONS"
_META = _PASTIS_ROOT / "metadata.geojson"
_FEATURES = Path("data/features/features_fused_pastis.parquet")
_OUT = Path("frontend/public/demo/parcelas_prediccion_francia.geojson")

_PATCH_SIDE = 128
_SOURCE_EPSG = 2154  # PASTIS metadata is RGF93 / Lambert-93
_HELD_OUT_FOLD = 5  # classifier trains on folds 1-4; predict on 5 (honest)
_MIN_PARCEL_PX = 12


def _patch_bounds_utm(geometry: dict) -> tuple[float, float, float, float]:
    coords = list(
        itertools.chain.from_iterable(
            itertools.chain.from_iterable(geometry["coordinates"])
        )
    )
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _reproject_ring(ring: list, transformer: Transformer) -> list:
    out = []
    for x, y in ring:
        lon, lat = transformer.transform(x, y)
        out.append([round(lon, 6), round(lat, 6)])
    return out


def _pick_patch(df_fold5: pl.DataFrame) -> int | None:
    """Pick the fold-5 patch with the most parcels that exists locally + diverse."""
    counts = (
        df_fold5.group_by("patch_id")
        .agg(pl.len().alias("n"), pl.col("class_id").n_unique().alias("k"))
        .sort("n", descending=True)
    )
    for row in counts.iter_rows(named=True):
        pid = int(row["patch_id"])
        if row["k"] < 3:
            continue
        if (_ANN / f"TARGET_{pid}.npy").exists() and (_INST / f"INSTANCES_{pid}.npy").exists():
            return pid
    return None


def main() -> None:
    """Build the prediction demo GeoJSON for one held-out fold-5 patch."""
    from ml.agent.tools.classify import _load_classifier

    df = pl.read_parquet(_FEATURES).filter(pl.col("fold") == _HELD_OUT_FOLD)
    dim_cols = sorted(c for c in df.columns if c.startswith("dim_"))
    patch_id = _pick_patch(df)
    if patch_id is None:
        raise typer.Exit(code=1)

    meta = json.loads(_META.read_text(encoding="utf-8"))
    feat = next(x for x in meta["features"] if int(x["properties"]["ID_PATCH"]) == patch_id)
    minx, miny, maxx, maxy = _patch_bounds_utm(feat["geometry"])
    transform = from_bounds(minx, miny, maxx, maxy, _PATCH_SIDE, _PATCH_SIDE)
    transformer = Transformer.from_crs(_SOURCE_EPSG, 4326, always_xy=True)

    instances = np.load(_INST / f"INSTANCES_{patch_id}.npy").astype(np.int32)
    classifier = _load_classifier()

    sub = df.filter(pl.col("patch_id") == patch_id)
    features: list[dict] = []
    n_correct = 0
    for row in sub.iter_rows(named=True):
        inst = int(row["instance_id"])
        mask = instances == inst
        if int(mask.sum()) < _MIN_PARCEL_PX:
            continue
        embedding = np.asarray([row[c] for c in dim_cols], dtype=np.float64)
        proba = classifier.predict_proba_18(embedding)
        pred_name = SEMANTIC18_CLASS_NAMES[int(np.argmax(proba))]
        true_name = PASTIS_CLASS_NAMES.get(int(row["class_id"]), "Unknown")
        correct = pred_name == true_name
        n_correct += int(correct)
        for geom, _ in rasterio.features.shapes(instances, mask=mask, transform=transform):
            rings = [_reproject_ring(r, transformer) for r in geom["coordinates"]]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": rings},
                    "properties": {
                        "parcel_id": str(row["parcel_id"]),
                        "patch_id": patch_id,
                        "crop_class": pred_name,  # map paints the prediction
                        "pred_class": pred_name,
                        "true_class": true_name,
                        "correct": correct,
                        "confidence": round(float(np.max(proba)), 3),
                    },
                }
            )
    if not features:
        raise typer.Exit(code=1)

    all_pts = [pt for f in features for ring in f["geometry"]["coordinates"] for pt in ring]
    lons = [pt[0] for pt in all_pts]
    lats = [pt[1] for pt in all_pts]
    n_parcels = len({f["properties"]["parcel_id"] for f in features})
    accuracy = round(n_correct / max(n_parcels, 1), 3)
    fc = {
        "type": "FeatureCollection",
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "metadata": {
            "source": "PASTIS-R fold-5 (held-out) -- model prediction vs ground truth",
            "model": "XGBoost-AlphaEarth (deployed perceiver), trained folds 1-4",
            "patch_id": patch_id,
            "n_parcels": n_parcels,
            "accuracy": accuracy,
            "pred_counts": dict(Counter(f["properties"]["pred_class"] for f in features)),
        },
        "features": features,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(fc), encoding="utf-8")
    log.info(
        "prediction_geojson_written",
        path=str(_OUT),
        patch=patch_id,
        n_parcels=n_parcels,
        accuracy=accuracy,
    )


if __name__ == "__main__":
    typer.run(main)
