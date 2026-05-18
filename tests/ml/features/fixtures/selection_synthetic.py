"""Fixtures sinteticos deterministas para la suite US-018 (selection).

Tres helpers reusables, todos con seed fijo (default 42) para reproducibilidad
byte-a-byte entre corridas:

- :func:`make_collinear_features` — clusters de columnas con ``|r| > 0.95``
  para probar :func:`ml.features.selection.drop_correlated_features`.
- :func:`make_skewed_distribution` — serie con sesgo positivo controlado para
  validar reglas de :func:`ml.features.selection.select_normalizer`.
- :func:`make_pastis_subset_synthetic` — frame ``(n, n_features)`` + clase +
  folds estratificados que mimica el subset PASTIS-R sin requerir descarga.
- :func:`make_categorical_fixture` — frame con columnas categoricas Utf8 +
  target numerico para validar :mod:`ml.features.encoding`.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def make_collinear_features(
    n_samples: int = 500,
    n_clusters: int = 4,
    cols_per_cluster: int = 3,
    noise_sigma: float = 0.05,
    seed: int = 42,
) -> pl.DataFrame:
    """Genera un frame con ``n_clusters`` grupos de columnas correlacionadas.

    Cada cluster comparte una senal base ``z_k ~ N(0, 1)`` mas ruido
    gaussiano de bajo sigma, garantizando ``|r| > 0.95`` intra-cluster e
    independencia inter-cluster.

    Args:
        n_samples: Filas del frame.
        n_clusters: Numero de clusters colineales.
        cols_per_cluster: Columnas por cluster (todas casi identicas).
        noise_sigma: Sigma del ruido aditivo por columna.
        seed: Semilla numpy.

    Returns:
        DataFrame Polars con ``parcel_id, year`` + ``n_clusters *
        cols_per_cluster`` columnas ``cluster{k}_col{c}``.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, list[float] | list[int]] = {
        "parcel_id": list(range(1, n_samples + 1)),
        "year": [2024] * n_samples,
    }
    for k in range(n_clusters):
        base = rng.normal(loc=0.0, scale=1.0, size=n_samples)
        for c in range(cols_per_cluster):
            noise = rng.normal(loc=0.0, scale=noise_sigma, size=n_samples)
            data[f"cluster{k}_col{c}"] = (base + noise).tolist()
    return pl.DataFrame(data)


def make_skewed_distribution(
    n: int = 500,
    skew: float = 2.0,
    seed: int = 42,
) -> pl.Series:
    """Genera una serie con sesgo positivo controlado (lognormal-like).

    Args:
        n: Numero de muestras.
        skew: Parametro de sesgo (no es exacto, pero scipy.stats.skew(result)
            queda cerca del valor solicitado para ``skew >= 1.0``).
        seed: Semilla.

    Returns:
        :class:`polars.Series` Float64 con valores positivos y sesgo >0.
    """
    rng = np.random.default_rng(seed)
    sigma = max(0.1, float(skew) / 2.0)
    raw = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    return pl.Series("skewed", raw.tolist(), dtype=pl.Float64)


def make_pastis_subset_synthetic(
    n_samples: int = 500,
    n_features: int = 187,
    n_classes: int = 20,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.Series, np.ndarray]:
    """Construye un subset sintetico estilo PASTIS-R 187-col / 20-clases.

    Diseno:

    - 3 features con senal real (separadora entre clases).
    - 5 features con redundancia (copia + ruido) -> activan filtro de
      correlacion.
    - 2 features constantes -> activan ``variance_threshold(0.01)``.
    - El resto: ruido gaussiano.
    - Clases estratificadas (mismo numero por clase, ``floor(n/n_classes)``).
    - Folds 1..5 distribuidos en orden round-robin sobre las filas
      (mantiene cada fold con muestras de cada clase, simula el split
      espacial estratificado).

    Args:
        n_samples: Numero de muestras. Se ajusta a multiplo de ``n_classes``.
        n_features: Numero de columnas feature (sin contar ``parcel_id,
            year``).
        n_classes: Numero de clases distintas (default 20 = PASTIS).
        seed: Semilla.

    Returns:
        Tupla ``(X, y, folds)``:

        - ``X``: :class:`polars.DataFrame` con ``parcel_id, year`` + ``n_features``
          columnas con nombres canonicos tipo ``NDVI_mean``, ``NDVI_fft_amp_0``,
          ``peak_doy``, etc.
        - ``y``: :class:`polars.Series` Int64 con la clase.
        - ``folds``: ``np.ndarray`` Int64 shape ``(n_samples,)`` con valores
          1..5.
    """
    if n_features < 10:
        raise ValueError(f"n_features >= 10 requerido; recibido {n_features}")
    rng = np.random.default_rng(seed)

    # Ajustamos a multiplo del numero de clases para estratificacion exacta.
    n_per_class = n_samples // n_classes
    if n_per_class == 0:
        raise ValueError(f"n_samples ({n_samples}) < n_classes ({n_classes})")
    total = n_per_class * n_classes
    y_arr = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    # Mezclamos el orden de muestras para que folds no caigan en bloques de clase.
    perm = rng.permutation(total)
    y_arr = y_arr[perm]

    # Folds round-robin sobre las muestras permutadas: cada fold tiene
    # aprox total/5 muestras de cada clase.
    folds = (np.arange(total) % 5 + 1).astype(np.int64)

    # Nombres canonicos: primeras 153 son stats (9 stats x 17 indices),
    # luego 24 FFT (8 x 3 indices), luego 8 fenologia, luego ruido.
    stat_suffixes = ("mean", "std", "min", "max", "p05", "p25", "p50", "p75", "p95")
    indices = (
        "NDVI",
        "NDWI",
        "EVI",
        "NDMI",
        "NBR",
        "MSAVI2",
        "NDRE",
        "MCARI",
        "CCCI",
        "GCVI",
        "PSRI",
        "NDCI",
        "FAPAR",
        "LAI",
        "RENDVI",
        "SAVI",
        "TSAVI",
    )
    feature_names: list[str] = []
    for idx in indices:
        for suf in stat_suffixes:
            feature_names.append(f"{idx}_{suf}")
    for idx in ("NDVI", "NDWI", "EVI"):
        for k in range(4):
            feature_names.append(f"{idx}_fft_amp_{k}")
            feature_names.append(f"{idx}_fft_phase_{k}")
    feature_names.extend(
        [
            "sog_doy",
            "peak_doy",
            "peak_value",
            "senescence_doy",
            "ndvi_auc",
            "ndvi_slope_pre_peak",
            "ndvi_slope_post_peak",
            "maturity_duration_days",
        ]
    )
    # Padding hasta n_features.
    extra = n_features - len(feature_names)
    if extra > 0:
        for j in range(extra):
            feature_names.append(f"noise_{j}")
    else:
        feature_names = feature_names[:n_features]

    # Matriz base: ruido N(0,1).
    matrix = rng.normal(loc=0.0, scale=1.0, size=(total, n_features)).astype(np.float64)

    # Senal real: las primeras 3 columnas (NDVI_mean, NDVI_std, NDVI_min)
    # codifican la clase con escala visible.
    class_offsets = (y_arr.astype(np.float64) - n_classes / 2.0) * 0.3
    matrix[:, 0] = class_offsets + rng.normal(0.0, 0.1, size=total)  # NDVI_mean
    matrix[:, 1] = -class_offsets + rng.normal(0.0, 0.1, size=total)  # NDVI_std
    matrix[:, 2] = class_offsets * 0.5 + rng.normal(0.0, 0.1, size=total)  # NDVI_min

    # Redundancia: cols 3..7 son copias de col 0 con ruido bajo (|r| ~ 0.99).
    for j in range(3, 8):
        matrix[:, j] = matrix[:, 0] + rng.normal(0.0, 0.02, size=total)

    # Constantes: ultimas 2 cols (noise_extra) tienen varianza ~ 0.
    matrix[:, -1] = 0.0
    matrix[:, -2] = 1.0

    # NDVI puede ser negativo: forzamos rango [-1, 1] en NDVI_mean para test
    # de Yeo-Johnson sobre negativos.
    matrix[:, 0] = np.clip(matrix[:, 0], -0.5, 1.0)

    columns: dict[str, list[float] | list[int]] = {
        "parcel_id": list(range(1, total + 1)),
        "year": [2024] * total,
    }
    for j, name in enumerate(feature_names):
        columns[name] = matrix[:, j].tolist()

    X = pl.DataFrame(columns)
    y = pl.Series("class_id", y_arr.tolist(), dtype=pl.Int64)
    return X, y, folds


def make_categorical_fixture(
    n: int = 100,
    n_classes: int = 4,
    seed: int = 42,
) -> pl.DataFrame:
    """Genera un DataFrame con columnas categoricas + target numerico.

    Diseno:

    - ``parcel_id``, ``year`` como en los demas fixtures.
    - ``season`` (Utf8) en ``{"winter", "spring", "summer", "autumn"}``.
    - ``crop_group`` (Utf8) con ``n_classes`` categorias
      ``cat_0..cat_{n_classes-1}`` distribuidas casi uniformemente.
    - ``yield_proxy`` (Float64) target con dependencia DETERMINISTA del
      ``crop_group`` (cada categoria suma un offset) para validar
      :func:`ml.features.encoding.encode_target_mean`.
    - ``peak_doy`` (Int64) en ``[1, 366]`` para validar
      :func:`ml.features.encoding.derive_season_from_doy`.

    Args:
        n: Numero de filas (>= n_classes).
        n_classes: Cardinalidad de ``crop_group``.
        seed: Semilla numpy.

    Returns:
        :class:`polars.DataFrame` con 6 columnas.
    """
    if n < n_classes:
        raise ValueError(f"n ({n}) debe ser >= n_classes ({n_classes})")
    rng = np.random.default_rng(seed)
    seasons = ["winter", "spring", "summer", "autumn"]
    season_col = rng.choice(seasons, size=n).tolist()
    crop_idx = rng.integers(0, n_classes, size=n)
    crop_col = [f"cat_{int(i)}" for i in crop_idx]
    # Target depende de crop_group: cada categoria tiene su propio offset.
    offsets = rng.normal(0.0, 2.0, size=n_classes)
    yield_proxy = offsets[crop_idx] + rng.normal(0.0, 0.5, size=n)
    peak_doy = rng.integers(1, 367, size=n).tolist()
    return pl.DataFrame(
        {
            "parcel_id": list(range(1, n + 1)),
            "year": [2024] * n,
            "season": season_col,
            "crop_group": crop_col,
            "yield_proxy": yield_proxy.tolist(),
            "peak_doy": peak_doy,
        }
    )
