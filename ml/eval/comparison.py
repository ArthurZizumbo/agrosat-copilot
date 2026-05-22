"""Comparativa de escenarios del baseline tabular (US-022, EPIC 4).

Modulo de la ultima US del EPIC 4 (Avance 3). Entrena Random Forest y
XGBoost sobre **tres vistas** del mismo conjunto de parcelas PASTIS-R y
construye la tabla comparativa que sustenta el criterio "Metrica" del
Avance 3:

- **(a) AlphaEarth 64-dim** — el embedding ``dim_00..dim_63`` del
  Foundation Model AlphaEarth Foundations.
- **(b) Sentinel-2 crudo** — las 10 bandas ``B02..B12`` promediadas
  temporal y espacialmente por parcela (generado por
  ``scripts/build_s2_raw_parcels.py``).
- **(c) Vector combinado** — las 187 features espectro-temporales del
  EPIC 3 (US-018).

Decisiones canonicas (plan ``docs/us-planning/us-022.md`` 2.1):

- **D2**: las 3 vistas se alinean por ``parcel_id`` con un *inner join*
  para que la comparativa sea justa (mismo set de parcelas).
- **D3**: el CV espacial es 5-fold y se reusa para los 3 escenarios via
  :func:`ml.train.baseline._build_cv_splits` (cacheado). CERO ``KFold``
  aleatorio.
- **D4**: ``train_time_s`` es el wall-clock del ``fit`` final sobre todo
  el escenario.
- **D7**: la exportacion LaTeX usa ``df.to_pandas().to_latex()`` aislada
  en :func:`export_comparison_latex` — unico uso de pandas del modulo,
  encapsulado como I/O de presentacion.

``ml.train.baseline`` se importa de forma **diferida** dentro de las
funciones para romper el ciclo de imports: ``baseline`` importa de
``ml.eval.metrics`` y ``ml.eval.__init__`` re-exporta este modulo — un
import a nivel de modulo dispararia un circular import al cargar el
paquete ``ml.eval``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "ComparisonResult",
    "build_comparison_table",
    "export_comparison_latex",
]

# Orden canonico de los escenarios (etiqueta corta -> nombre legible).
_SCENARIO_LABELS: dict[str, str] = {
    "alphaearth": "AlphaEarth 64-dim",
    "s2_raw": "Sentinel-2 crudo (10 bandas)",
    "combined": "Vector combinado (187 feat)",
}

# Modelos de la comparativa, en orden de la tabla.
_MODEL_KINDS: tuple[str, ...] = ("rf", "xgb")

# Columnas de metadata que NO son features (se excluyen al contar
# `n_features`). Espejo de `ml.train.baseline._META_COLS` mas las columnas
# extra que arrastran los parquets de escenario.
_META_COLS: frozenset[str] = frozenset(
    {
        "parcel_id",
        "year",
        "patch_id",
        "instance_id",
        "class_id",
        "class_name",
        "fold",
        "n_pixels",
        "area_m2",
        "geometry",
    }
)

# Orden de columnas de la tabla comparativa final.
_TABLE_COLUMNS: tuple[str, ...] = (
    "scenario",
    "model",
    "n_features",
    "f1_macro",
    "f1_weighted",
    "miou",
    "train_time_s",
)


# ---------------------------------------------------------------------------
# Dataclass de salida.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Resultado de la comparativa de escenarios del baseline.

    Attributes:
        table: ``pl.DataFrame`` con 6 filas (3 escenarios x 2 modelos) y
            columnas ``(scenario, model, n_features, f1_macro,
            f1_weighted, miou, train_time_s)``, ordenado por ``f1_macro``
            descendente.
        best_scenario: Nombre legible del escenario con mayor ``f1_macro``
            (sobre el mejor de sus dos modelos).
        alphaearth_delta: Delta de ``f1_macro`` del escenario AlphaEarth
            puro frente al Sentinel-2 crudo (mejor modelo de cada uno).
            Cuantifica el valor incremental del embedding; positivo si
            AlphaEarth supera al S2 crudo.
        n_parcels: Numero de parcelas efectivas tras el *inner join* de
            los 3 escenarios (decision D2).
    """

    table: pl.DataFrame
    best_scenario: str
    alphaearth_delta: float
    n_parcels: int


# ---------------------------------------------------------------------------
# API publica.
# ---------------------------------------------------------------------------


def build_comparison_table(
    scenario_paths: dict[str, str | Path],
    *,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    max_samples: int = 0,
    random_state: int = 42,
) -> ComparisonResult:
    """Entrena RF+XGB sobre 3 escenarios con el mismo spatial CV.

    Carga los tres parquets de escenario, los alinea por ``parcel_id``
    con un *inner join* (decision D2) para que los tres modelos vean
    exactamente el mismo conjunto de parcelas, y entrena Random Forest y
    XGBoost sobre cada vista reusando :func:`ml.train.baseline.train_one_model`
    (mismo CV espacial 5-fold, decision D3).

    Args:
        scenario_paths: Mapa ``{"alphaearth": ruta, "s2_raw": ruta,
            "combined": ruta}``. Las tres claves son obligatorias; cada
            parquet debe contener ``parcel_id`` y ``class_id``.
        k_folds: Numero de folds del CV espacial (default 5).
        buffer_km: Buffer anti-leakage en km entre folds (default 1.0).
        max_samples: Subsample estratificado por clase para dev/CI
            (``0`` = dataset completo).
        random_state: Semilla determinista.

    Returns:
        Un :class:`ComparisonResult` con la tabla de 6 filas, el escenario
        ganador y el delta de ``f1_macro`` AlphaEarth vs S2 crudo.

    Raises:
        ValueError: si falta alguna de las tres claves de escenario o si
            el *inner join* deja el conjunto de parcelas vacio.
        FileNotFoundError: si alguno de los parquets no existe.
    """
    missing = set(_SCENARIO_LABELS) - set(scenario_paths)
    if missing:
        raise ValueError(
            f"`scenario_paths` debe contener las claves {sorted(_SCENARIO_LABELS)}; "
            f"faltan {sorted(missing)}."
        )

    raw_scenarios = {
        key: _load_scenario(scenario_paths[key]) for key in _SCENARIO_LABELS
    }
    aligned = _align_scenarios_by_parcel(raw_scenarios)
    n_parcels = next(iter(aligned.values())).height

    if max_samples > 0:
        aligned = _subsample_aligned(aligned, max_samples, random_state)
        logger.info(
            "comparison_subsampled",
            max_samples=max_samples,
            n_parcels=next(iter(aligned.values())).height,
        )

    logger.info(
        "comparison_table_start",
        n_parcels=n_parcels,
        n_effective=next(iter(aligned.values())).height,
        k_folds=k_folds,
    )

    rows: list[dict[str, object]] = []
    for scenario_key, label in _SCENARIO_LABELS.items():
        scenario_df = aligned[scenario_key]
        n_features = _count_features(scenario_df)
        for model_kind in _MODEL_KINDS:
            metrics, train_time_s = _train_scenario_model(
                scenario_df,
                model_kind=model_kind,
                k_folds=k_folds,
                buffer_km=buffer_km,
                random_state=random_state,
            )
            rows.append(
                {
                    "scenario": label,
                    "model": model_kind.upper(),
                    "n_features": n_features,
                    "f1_macro": round(metrics["f1_macro"], 4),
                    "f1_weighted": round(metrics["f1_weighted"], 4),
                    "miou": round(metrics["miou"], 4),
                    "train_time_s": round(train_time_s, 3),
                }
            )
            logger.info(
                "comparison_cell_done",
                scenario=scenario_key,
                model=model_kind,
                f1_macro=round(metrics["f1_macro"], 4),
                train_time_s=round(train_time_s, 2),
            )

    table = pl.DataFrame(rows, schema={c: _column_dtype(c) for c in _TABLE_COLUMNS})
    table = table.sort("f1_macro", descending=True)

    best_scenario = _best_scenario(table)
    alphaearth_delta = _alphaearth_delta(table)
    logger.info(
        "comparison_table_done",
        best_scenario=best_scenario,
        alphaearth_delta=round(alphaearth_delta, 4),
        n_parcels=n_parcels,
    )
    return ComparisonResult(
        table=table,
        best_scenario=best_scenario,
        alphaearth_delta=alphaearth_delta,
        n_parcels=n_parcels,
    )


def export_comparison_latex(
    result: ComparisonResult,
    path: str | Path,
) -> Path:
    """Exporta la tabla comparativa a LaTeX (booktabs) para el Paper Track.

    Genera un fragmento LaTeX ``booktabs``-style listo para ``\\input{}``
    en el paper. Polars no exporta LaTeX nativo: la conversion usa
    ``df.to_pandas().to_latex()`` — el **unico** uso de pandas del modulo,
    aislado aqui como I/O de presentacion (decision D7).

    Args:
        result: El :class:`ComparisonResult` a exportar.
        path: Ruta destino del fichero ``.tex``.

    Returns:
        El :class:`~pathlib.Path` del ``.tex`` escrito.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Unico uso de pandas del modulo (D7): conversion de presentacion.
    pandas_df = result.table.to_pandas()
    body = pandas_df.to_latex(
        index=False,
        escape=True,
        column_format="llrrrrr",
        caption=(
            "Comparativa del baseline tabular sobre tres escenarios de "
            f"features (PASTIS-R, {result.n_parcels} parcelas, CV espacial "
            "5-fold). Escenario ganador: "
            f"{result.best_scenario}; delta F1-macro AlphaEarth vs "
            f"Sentinel-2 crudo: {result.alphaearth_delta:+.4f}."
        ),
        label="tab:baseline-comparison",
        float_format="%.4f",
    )
    out_path.write_text(body, encoding="utf-8")
    logger.info("comparison_latex_written", path=str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Helpers privados — carga y alineacion.
# ---------------------------------------------------------------------------


def _load_scenario(path: str | Path) -> pl.DataFrame:
    """Carga un parquet de escenario desde disco.

    Args:
        path: Ruta al parquet del escenario.

    Returns:
        El DataFrame Polars crudo.

    Raises:
        FileNotFoundError: si el parquet no existe.
        ValueError: si carece de ``parcel_id`` o ``class_id``.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Parquet de escenario no encontrado en {resolved}. "
            "Genera el escenario (b) con `make s2-raw-parcels` o verifica "
            "las rutas de AlphaEarth / feature_selection."
        )
    df = pl.read_parquet(resolved)
    for col in ("parcel_id", "class_id"):
        if col not in df.columns:
            raise ValueError(
                f"El parquet {resolved} debe contener la columna `{col}`."
            )
    return df


def _align_scenarios_by_parcel(
    scenarios: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Alinea los escenarios por ``parcel_id`` con un *inner join* (D2).

    Calcula la interseccion de ``parcel_id`` presente en los tres
    escenarios y filtra cada DataFrame a ese conjunto comun, ordenado de
    forma determinista. Garantiza que los tres modelos se evaluen sobre
    exactamente el mismo set de parcelas y los mismos folds espaciales.

    Args:
        scenarios: Mapa ``{clave: DataFrame}`` de los escenarios crudos.

    Returns:
        Mapa ``{clave: DataFrame}`` con los tres frames filtrados al
        ``parcel_id`` comun y ordenados por ``parcel_id``.

    Raises:
        ValueError: si la interseccion de ``parcel_id`` es vacia.
    """
    common: set[str] | None = None
    for df in scenarios.values():
        ids = set(df.get_column("parcel_id").cast(pl.Utf8).to_list())
        common = ids if common is None else common & ids
    if not common:
        raise ValueError(
            "El inner join por `parcel_id` de los 3 escenarios quedo vacio: "
            "los parquets no comparten parcelas."
        )

    common_sorted = sorted(common)
    keep = pl.DataFrame({"parcel_id": common_sorted})
    aligned: dict[str, pl.DataFrame] = {}
    for key, df in scenarios.items():
        normalized = df.with_columns(pl.col("parcel_id").cast(pl.Utf8))
        # Inner join contra el conjunto comun + orden determinista: los tres
        # frames quedan con las mismas filas en el mismo orden.
        joined = (
            keep.join(normalized, on="parcel_id", how="inner")
            .unique(subset="parcel_id", keep="first")
            .sort("parcel_id")
        )
        aligned[key] = joined

    logger.info(
        "scenarios_aligned",
        n_common=len(common_sorted),
        scenarios=sorted(scenarios),
    )
    return aligned


def _subsample_aligned(
    aligned: dict[str, pl.DataFrame],
    max_samples: int,
    random_state: int,
) -> dict[str, pl.DataFrame]:
    """Submuestrea los escenarios alineados de forma estratificada por clase.

    Selecciona el mismo conjunto de ``parcel_id`` para los tres escenarios
    (mantiene la alineacion de D2). Util para dev/CI con ``max_samples``.

    Args:
        aligned: Mapa de escenarios ya alineados por ``parcel_id``.
        max_samples: Tamano objetivo del subsample.
        random_state: Semilla determinista.

    Returns:
        Mapa de escenarios submuestreados al mismo conjunto de parcelas.
    """
    reference = next(iter(aligned.values()))
    if reference.height <= max_samples:
        return aligned

    frac = max_samples / reference.height
    sampled_ids = (
        reference.select(["parcel_id", "class_id"])
        .filter(pl.col("class_id").is_not_null())
        .group_by("class_id", maintain_order=True)
        .map_groups(
            lambda g: g.sample(
                n=max(1, round(g.height * frac)),
                seed=random_state,
            )
        )
        .get_column("parcel_id")
    )
    keep = pl.DataFrame({"parcel_id": sampled_ids})
    return {
        key: keep.join(df, on="parcel_id", how="inner").sort("parcel_id")
        for key, df in aligned.items()
    }


# ---------------------------------------------------------------------------
# Helpers privados — entrenamiento y metricas.
# ---------------------------------------------------------------------------


def _train_scenario_model(
    scenario_df: pl.DataFrame,
    *,
    model_kind: str,
    k_folds: int,
    buffer_km: float,
    random_state: int,
) -> tuple[dict[str, float], float]:
    """Entrena un modelo sobre un escenario y mide el wall-clock del fit.

    Reusa :func:`ml.train.baseline.train_one_model` (mismo CV espacial que
    el resto del baseline, decision D3). El import es diferido para evitar
    el circular import con ``ml.eval.__init__``.

    Args:
        scenario_df: DataFrame del escenario ya alineado.
        model_kind: ``"rf"`` o ``"xgb"``.
        k_folds: Numero de folds del CV espacial.
        buffer_km: Buffer anti-leakage en km.
        random_state: Semilla determinista.

    Returns:
        Tupla ``(metrics, train_time_s)`` con las metricas out-of-fold del
        baseline y el wall-clock del ``fit`` final (decision D4).
    """
    # Import diferido: rompe el ciclo `baseline -> eval.metrics` /
    # `eval.__init__ -> comparison`.
    from ml.train.baseline import train_one_model

    start = time.perf_counter()
    result = train_one_model(
        scenario_df,
        model=model_kind,  # type: ignore[arg-type]
        k_folds=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )
    train_time_s = time.perf_counter() - start
    return result.metrics, train_time_s


def _count_features(scenario_df: pl.DataFrame) -> int:
    """Cuenta las columnas numericas usables como features de un escenario.

    Excluye la metadata (``_META_COLS``) y las columnas no numericas.

    Args:
        scenario_df: DataFrame del escenario.

    Returns:
        Numero de columnas de feature.
    """
    return sum(
        1
        for col in scenario_df.columns
        if col not in _META_COLS and scenario_df.schema[col].is_numeric()
    )


def _best_scenario(table: pl.DataFrame) -> str:
    """Devuelve el escenario con mayor ``f1_macro`` de la tabla.

    Args:
        table: Tabla comparativa de 6 filas.

    Returns:
        El nombre legible del escenario ganador (mejor de sus 2 modelos).
    """
    top_row = table.sort("f1_macro", descending=True).row(0, named=True)
    return str(top_row["scenario"])


def _alphaearth_delta(table: pl.DataFrame) -> float:
    """Calcula el delta de ``f1_macro`` AlphaEarth vs Sentinel-2 crudo.

    Compara el mejor modelo del escenario AlphaEarth con el mejor del
    Sentinel-2 crudo. Cuantifica el valor incremental del embedding.

    Args:
        table: Tabla comparativa de 6 filas.

    Returns:
        ``f1_macro(AlphaEarth) - f1_macro(S2 crudo)``; positivo si el
        embedding aporta. ``nan`` si falta alguno de los dos escenarios.
    """
    ae_label = _SCENARIO_LABELS["alphaearth"]
    s2_label = _SCENARIO_LABELS["s2_raw"]
    ae_rows = table.filter(pl.col("scenario") == ae_label)
    s2_rows = table.filter(pl.col("scenario") == s2_label)
    if ae_rows.height == 0 or s2_rows.height == 0:
        return float("nan")
    ae_best = float(ae_rows.get_column("f1_macro").max())  # type: ignore[arg-type]
    s2_best = float(s2_rows.get_column("f1_macro").max())  # type: ignore[arg-type]
    return ae_best - s2_best


def _column_dtype(column: str) -> pl.DataType:
    """Devuelve el dtype Polars de una columna de la tabla comparativa.

    Args:
        column: Nombre de la columna.

    Returns:
        El :class:`polars.DataType` correspondiente.
    """
    if column in ("scenario", "model"):
        return pl.Utf8()
    if column == "n_features":
        return pl.Int64()
    return pl.Float64()
