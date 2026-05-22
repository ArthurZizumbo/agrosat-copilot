"""Fixture sintetico determinista para los tests del baseline (US-019).

Genera un DataFrame Polars con el mismo esquema minimo que el subset
PASTIS-R real (``parcel_id``, ``patch_id``, ``class_id``, ``fold`` +
features numericas) pero pequeno y con clases linealmente separables,
para que los tests de :mod:`ml.train.baseline` corran en segundos sin
depender del parquet de 76 MB.

El generador es determinista: misma ``seed`` produce el mismo frame
byte-a-byte, requisito de reproducibilidad de la suite de tests.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["make_baseline_dataset"]


def make_baseline_dataset(
    n: int = 240,
    *,
    n_classes: int = 4,
    n_features: int = 12,
    n_patches: int = 12,
    seed: int = 42,
) -> pl.DataFrame:
    """Construye un DataFrame Polars sintetico para el baseline.

    Cada clase recibe un centroide distinto en el espacio de features, de
    modo que un clasificador entrenado correctamente alcanza F1-macro alto
    (separabilidad garantizada). Las parcelas se agrupan en ``n_patches``
    patches para que el CV espacial tenga estructura geografica.

    Args:
        n: Numero total de parcelas (filas).
        n_classes: Numero de clases distintas. Las clases se etiquetan con
            ids no contiguos (saltan el 0 y el 19, como PASTIS-R real) para
            ejercitar el ``LabelEncoder``.
        n_features: Numero de columnas de feature numericas.
        n_patches: Numero de patches PASTIS-R sinteticos (estructura para
            el CV espacial).
        seed: Semilla determinista.

    Returns:
        DataFrame Polars con columnas ``parcel_id`` (str), ``patch_id``
        (int), ``year`` (int), ``class_id`` (int, no contiguo), ``fold``
        (int 1..5), ``n_pixels`` (int) y ``feat_000..feat_NNN`` (float).
    """
    rng = np.random.default_rng(seed)

    # Class ids no contiguos: imita PASTIS-R tras descartar 0 y 19.
    class_pool = [c for c in range(1, 19) if c not in (0, 19)][:n_classes]
    labels = rng.integers(0, n_classes, size=n)
    class_ids = np.array([class_pool[i] for i in labels], dtype=np.int64)

    # Features: centroide por clase + ruido gaussiano (separables).
    centroids = rng.normal(0.0, 5.0, size=(n_classes, n_features))
    features = centroids[labels] + rng.normal(0.0, 1.0, size=(n, n_features))

    patch_ids = rng.integers(10000, 10000 + n_patches, size=n).astype(np.int64)
    folds = (rng.integers(0, 5, size=n) + 1).astype(np.int64)

    data: dict[str, object] = {
        "parcel_id": [f"{int(p)}_{i}" for i, p in enumerate(patch_ids)],
        "patch_id": patch_ids,
        "year": np.full(n, 2019, dtype=np.int64),
        "class_id": class_ids,
        "fold": folds,
        "n_pixels": rng.integers(50, 500, size=n).astype(np.int64),
    }
    for j in range(n_features):
        data[f"feat_{j:03d}"] = features[:, j].astype(np.float64)

    return pl.DataFrame(data)
