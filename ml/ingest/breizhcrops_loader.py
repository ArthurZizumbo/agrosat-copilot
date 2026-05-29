"""Carga de series temporales BreizhCrops desde disco a estructuras Polars.

BreizhCrops (Russwurm et al., ISPRS Archives 2020 — sucesor mantenido del
dataset de Russwurm & Korner, ISPRS IJGI 2018) entrega series temporales
Sentinel-2 por parcela agricola de Bretaña (Francia). Cada parcela es una
secuencia ``(T, n_bands)`` almacenada en una base HDF5 ``<region>.h5``, con
un indice tabular ``<region>.csv`` y un mapeo de 9 clases en
``classmapping.csv``.

A diferencia de PASTIS-R (rejilla densa 128x128), BreizhCrops es una
coleccion de series por objeto: 1 vector temporal multibanda por parcela,
sin componente espacial. Esto lo hace el complemento natural para validar
que las features temporales (FFT / fenologia) generalizan cross-region.

Este modulo expone helpers ligeros que reutilizan el paquete oficial
``breizhcrops`` con descarga DESHABILITADA: si los archivos no estan en
disco con el layout esperado, las funciones publicas retornan DataFrames
Polars con esquema valido VACIO (modo degradado, espejo de
``pastis_loader.py``), de forma que cualquier notebook completa la
ejecucion sin error y sin tocar la red.

La descarga es manual y unica via ``scripts/download_breizhcrops.sh``.

Layout esperado (root = ``data/breizhcrops/``)::

    data/breizhcrops/classmapping.csv
    data/breizhcrops/codes.csv
    data/breizhcrops/2017/L2A/frh04.csv
    data/breizhcrops/2017/L2A/frh04.h5
    data/breizhcrops/2017/L2A/frh01.csv
    data/breizhcrops/2017/L2A/frh01.h5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Bandas conservadas por el paquete breizhcrops en nivel L2A. El orden
# replica SELECTED_BANDS["L2A"] de breizhcrops.datasets.breizhcrops: la
# columna 0 de cada serie es `doa` (fecha como entero) y las 10 siguientes
# son las bandas opticas; CLD/EDG/SAT (mascaras) se descartan en EDA.
BREIZHCROPS_L2A_BANDS: list[str] = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]
"""Orden canonico de las 10 bandas opticas Sentinel-2 L2A de BreizhCrops.

Mapeado a la nomenclatura del proyecto (B2->B02, etc.) para alinearse con
``PASTIS_S2_BANDS`` y permitir comparacion cross-dataset directa.
"""

# Indice posicional de cada banda dentro del array crudo que devuelve
# breizhcrops.BreizhCrops.load() para L2A: [doa, B2, B3, B4, B5, B6, B7,
# B8, B8A, B11, B12, CLD, EDG, SAT]. El indice 0 (doa) NO es banda.
_L2A_BAND_OFFSET: int = 1

BREIZHCROPS_CLASSES: dict[int, str] = {
    0: "barley",
    1: "wheat",
    2: "rapeseed",
    3: "corn",
    4: "sunflower",
    5: "orchards",
    6: "nuts",
    7: "permanent meadows",
    8: "temporary meadows",
}
"""Mapeo `class_id -> nombre` de las 9 clases canonicas BreizhCrops.

Fuente: ``classmapping.csv`` distribuido con el dataset (bucket S2 publico).
Se hardcodea para que el modulo exponga la taxonomia incluso en modo
degradado (sin dataset descargado).
"""

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "breizhcrops"

_PARCEL_INDEX_SCHEMA: dict[str, Any] = {
    "parcel_id": pl.Utf8,
    "region": pl.Utf8,
    "year": pl.Int64,
    "level": pl.Utf8,
    "code_cultu": pl.Utf8,
    "class_id": pl.Int16,
    "class_name": pl.Utf8,
    "sequence_length": pl.Int64,
}

_PIXEL_SERIES_SCHEMA: dict[str, Any] = {
    "parcel_id": pl.Utf8,
    "t": pl.Int64,
    "date": pl.Int64,
    "doy": pl.Int64,
    "band": pl.Utf8,
    "value": pl.Float64,
    "class_id": pl.Int16,
    "class_name": pl.Utf8,
}


def _required_paths(root: Path, region: str, year: int, level: str) -> dict[str, Path]:
    """Construye las rutas que el paquete breizhcrops espera para una region.

    Args:
        root: Raiz del dataset (``data/breizhcrops/``).
        region: Region BreizhCrops (ej. ``frh04``).
        year: Anio del ciclo (solo 2017 verificado).
        level: Nivel de procesamiento Sentinel-2 (``L2A``).

    Returns:
        Diccionario con keys ``classmapping``, ``codes``, ``index``, ``h5``.
    """
    level_dir = root / str(year) / level
    return {
        "classmapping": root / "classmapping.csv",
        "codes": root / "codes.csv",
        "index": level_dir / f"{region}.csv",
        "h5": level_dir / f"{region}.h5",
    }


def _dataset_available(root: Path, region: str, year: int, level: str) -> bool:
    """Verifica que TODOS los archivos requeridos existan en disco.

    Esta guarda es la que garantiza que jamas se dispare una descarga de
    red: solo instanciamos ``breizhcrops.BreizhCrops`` cuando el layout
    completo ya esta presente localmente.

    Args:
        root: Raiz del dataset.
        region: Region BreizhCrops.
        year: Anio del ciclo.
        level: Nivel de procesamiento.

    Returns:
        ``True`` si classmapping, codes, index y h5 existen y no estan
        vacios; ``False`` en caso contrario (activa modo degradado).
    """
    paths = _required_paths(root, region, year, level)
    return all(p.exists() and p.stat().st_size > 0 for p in paths.values())


def _open_dataset(root: Path, region: str, year: int, level: str) -> Any | None:
    """Instancia ``breizhcrops.BreizhCrops`` SIN descarga (offline-safe).

    El paquete no expone un flag ``download=False``: descarga si faltan
    archivos. Por eso solo construimos el dataset cuando
    :func:`_dataset_available` confirma que todo esta en disco. Si el
    paquete no esta instalado o la construccion falla, devolvemos ``None``
    para caer a modo degradado.

    Args:
        root: Raiz del dataset.
        region: Region BreizhCrops.
        year: Anio del ciclo.
        level: Nivel de procesamiento.

    Returns:
        Instancia de ``BreizhCrops`` o ``None`` si no es posible cargarla
        sin red.
    """
    if not _dataset_available(root, region, year, level):
        return None
    try:
        from breizhcrops import BreizhCrops  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        return BreizhCrops(
            region=region,
            root=str(root),
            year=year,
            level=level,
            load_timeseries=True,
            verbose=False,
        )
    except Exception:  # noqa: BLE001
        # Construccion fallida (h5 corrupto, indice incompatible, etc.):
        # degradamos a esquema vacio en vez de propagar. Sin log porque
        # breizhcrops es opcional y el notebook documenta el modo.
        return None


def breizhcrops_parcel_index(
    region: str = "frh04",
    year: int = 2017,
    level: str = "L2A",
    root: Path | None = None,
) -> pl.DataFrame:
    """Devuelve el indice plano de parcelas BreizhCrops de una region.

    Equivalente a ``pastis_patch_index`` pero para series por objeto: una
    fila por parcela con su clase y la longitud de su serie temporal. Util
    para muestreo estratificado por clase antes de cargar las series H5.

    Args:
        region: Region BreizhCrops (``frh01``..``frh04``, ``belle-ile``).
        year: Anio del ciclo agricola (solo 2017 verificado).
        level: Nivel de procesamiento Sentinel-2 (``L2A`` recomendado).
        root: Raiz del dataset. Si ``None``, usa ``data/breizhcrops/``
            relativo al repo.

    Returns:
        DataFrame Polars con columnas ``parcel_id, region, year, level,
        code_cultu, class_id, class_name, sequence_length``. Vacio (con
        esquema valido) si el dataset no esta descargado o el paquete
        ``breizhcrops`` no esta disponible.
    """
    root = root or _DEFAULT_ROOT
    ds = _open_dataset(root, region, year, level)
    if ds is None:
        return pl.DataFrame(schema=_PARCEL_INDEX_SCHEMA)

    idx = ds.index.reset_index()
    rows: list[dict[str, Any]] = []
    for _, r in idx.iterrows():
        class_id = int(r["classid"])
        rows.append(
            {
                "parcel_id": str(r["id"]),
                "region": region,
                "year": int(year),
                "level": level,
                "code_cultu": str(r["CODE_CULTU"]),
                "class_id": class_id,
                "class_name": BREIZHCROPS_CLASSES.get(class_id, str(r.get("classname", "unknown"))),
                "sequence_length": int(r["sequencelength"]),
            }
        )

    if not rows:
        return pl.DataFrame(schema=_PARCEL_INDEX_SCHEMA)
    return pl.DataFrame(rows, schema=_PARCEL_INDEX_SCHEMA)


def _doa_to_date_doy(doa_int: float) -> tuple[int, int]:
    """Convierte el campo `doa` (datetime64[ns] como int) a (YYYYMMDD, DOY).

    El paquete breizhcrops almacena la fecha de adquisicion como
    ``pd.to_datetime(...).astype(int)`` (nanosegundos desde epoch). Aqui la
    revertimos a un entero ``YYYYMMDD`` legible y al dia del anio.

    Args:
        doa_int: Valor crudo de la columna 0 (`doa`) de la serie.

    Returns:
        Tupla ``(date_yyyymmdd, doy)``. ``(0, 0)`` si el valor no es finito.
    """
    if not np.isfinite(doa_int):
        return 0, 0
    dt = np.datetime64(int(doa_int), "ns")
    day = dt.astype("datetime64[D]")
    year = day.astype("datetime64[Y]").astype(int) + 1970
    months = day.astype("datetime64[M]")
    month = months.astype(int) % 12 + 1
    day_of_month = (day - months).astype(int) + 1
    jan1 = np.datetime64(f"{year:04d}-01-01", "D")
    doy = int((day - jan1).astype(int)) + 1
    return year * 10000 + month * 100 + day_of_month, doy


def breizhcrops_pixel_series(
    region: str = "frh04",
    year: int = 2017,
    level: str = "L2A",
    sample_parcels: int | None = None,
    seed: int = 42,
    root: Path | None = None,
    only_parcel_ids: set[str] | None = None,
) -> pl.DataFrame:
    """Convierte series BreizhCrops a un ``pl.DataFrame`` long-format.

    Cada parcela aporta ``T`` pasos temporales x 10 bandas opticas. El
    formato long resultante es directamente comparable con la salida de
    ``pastis_to_polars`` (mismas columnas semanticas: ``band``, ``value``,
    ``class_id``), habilitando el analisis cross-dataset BreizhCrops vs
    PASTIS-R.

    El muestreo estratificado se hace por parcela (no por pixel, porque
    BreizhCrops no tiene rejilla espacial): de ``sample_parcels`` parcelas
    elegidas con semilla fija se expanden todas sus observaciones.

    Args:
        region: Region BreizhCrops.
        year: Anio del ciclo agricola.
        level: Nivel de procesamiento (``L2A``).
        sample_parcels: Si no ``None``, numero maximo de parcelas a
            samplear (reproducible con ``seed``). ``None`` carga todas.
        seed: Semilla para el muestreo de parcelas.
        root: Raiz del dataset. ``None`` usa ``data/breizhcrops/``.
        only_parcel_ids: Si no ``None``, restringe la extraccion a las
            parcelas cuyo ``id`` (como string) este en el conjunto. Es la
            via eficiente para extraer un subconjunto previamente muestreado
            sin expandir toda la region (la region completa son cientos de
            miles de parcelas). Se aplica ANTES de ``sample_parcels``.

    Returns:
        DataFrame Polars con columnas ``parcel_id, t, date, doy, band,
        value, class_id, class_name``. Vacio (esquema valido) si el
        dataset no esta descargado o el paquete no esta disponible.
    """
    root = root or _DEFAULT_ROOT
    ds = _open_dataset(root, region, year, level)
    if ds is None:
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)

    n_parcels = len(ds)
    if n_parcels == 0:
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)

    order = np.arange(n_parcels)
    if only_parcel_ids is not None:
        # Filtra por posiciones cuyo `id` esta en el conjunto pedido. El
        # indice de breizhcrops usa un RangeIndex posicional, asi que la
        # posicion en `order` coincide con `ds.index.iloc[pos]`.
        wanted = {str(p) for p in only_parcel_ids}
        id_series = ds.index["id"].astype(str).to_numpy()
        order = np.where(np.isin(id_series, list(wanted)))[0]
        if order.size == 0:
            return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)
    if sample_parcels is not None and sample_parcels < order.size:
        rng = np.random.default_rng(seed)
        order = rng.choice(order, size=sample_parcels, replace=False)

    band_names = BREIZHCROPS_L2A_BANDS
    n_bands = len(band_names)
    frames: list[pl.DataFrame] = []

    # Abrimos el HDF5 UNA sola vez para toda la extraccion: reabrir el archivo
    # por parcela domina el tiempo cuando `order` tiene cientos/miles de ids.
    try:
        h5_ctx = _h5_open(ds)
    except Exception:  # noqa: BLE001
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)

    with h5_ctx as h5:
        for i in order:
            try:
                row = ds.index.iloc[int(i)]
                raw = np.asarray(h5[row.path], dtype=np.float64)
            except Exception:  # noqa: BLE001, S112
                # Serie ilegible: la saltamos sin abortar la carga completa.
                # Sin log porque breizhcrops es opcional y el notebook documenta
                # el modo degradado (espejo de pastis_loader.py).
                continue
            if raw.ndim != 2 or raw.shape[0] == 0:
                continue

            class_id = int(row["classid"])
            class_name = BREIZHCROPS_CLASSES.get(class_id, "unknown")
            parcel_id = str(row["id"])

            t_steps = raw.shape[0]
            doa_col = raw[:, 0]
            dates = np.empty(t_steps, dtype=np.int64)
            doys = np.empty(t_steps, dtype=np.int64)
            for ti in range(t_steps):
                d, doy = _doa_to_date_doy(doa_col[ti])
                dates[ti] = d
                doys[ti] = doy

            for bi in range(n_bands):
                col = raw[:, _L2A_BAND_OFFSET + bi]
                frames.append(
                    pl.DataFrame(
                        {
                            "parcel_id": [parcel_id] * t_steps,
                            "t": np.arange(t_steps, dtype=np.int64),
                            "date": dates,
                            "doy": doys,
                            "band": [band_names[bi]] * t_steps,
                            "value": col.astype(np.float64),
                            "class_id": np.full(t_steps, class_id, dtype=np.int16),
                            "class_name": [class_name] * t_steps,
                        },
                        schema=_PIXEL_SERIES_SCHEMA,
                    )
                )

    if not frames:
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed")


def _h5_open(ds: Any) -> Any:
    """Abre el HDF5 de la instancia BreizhCrops en modo lectura.

    Aislado en su propia funcion para que el ``with`` del caller sea
    legible y para poder mockearlo en tests sin red ni h5py real.

    Args:
        ds: Instancia de ``breizhcrops.BreizhCrops``.

    Returns:
        Context manager de ``h5py.File`` sobre ``ds.h5path``.
    """
    import h5py  # type: ignore[import-untyped]

    return h5py.File(ds.h5path, "r")


__all__ = [
    "BREIZHCROPS_CLASSES",
    "BREIZHCROPS_L2A_BANDS",
    "breizhcrops_parcel_index",
    "breizhcrops_pixel_series",
]
