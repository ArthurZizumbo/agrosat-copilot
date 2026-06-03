"""Wrapper de alto nivel para materializar el bloque ``pheno_text_*``.

Este modulo orquesta la generacion de descripciones fenologicas con
Gemini 3.5 Flash sobre el dataset full (PASTIS-R, ~85951 parcelas;
``parcel_id`` formato ``10000_1``, no italiano) y persiste el resultado
como un parquet listo para ``LEFT JOIN`` en ``ml.features.fusion``.

Contrato (US-023-preview v2):

1. **Sin mocks ni skips silenciosos**: si falta API key, se levanta
   ``RuntimeError`` explicito con instrucciones de configuracion.
2. **Cache idempotente**: si el parquet ya existe y ``overwrite=False``,
   se reutiliza sin volver a invocar el LLM.
3. **Muestreo estratificado opcional**: ``balanced_by_class=True`` toma
   ``min_per_class`` filas por ``class_id`` (semilla fija para
   reproducibilidad).
4. **Budget tracking**: estima el costo Gemini al inicio del run
   (``COST_PER_DESCRIPTION_USD * N``) y lo loguea via structlog.
5. **Esquema canonico**: ``parcel_id`` siempre se persiste como
   ``pl.Utf8`` (ver :mod:`ml.utils.parcel_id`).

El wrapper deliberadamente NO acepta el flag ``skip_llm`` desde su
firma: notebooks y pipelines deben ejecutar Gemini real. Para tests
unitarios se inyecta un cliente mock via
``ml.features.phenology_description.set_llm_client``.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import structlog

from ml.features.phenology_description import (
    COST_PER_DESCRIPTION_USD,
    DEFAULT_TEXT_EMBED_DIM,
    _has_credentials,
    build_phenology_text_block,
)
from ml.utils.parcel_id import canonical_parcel_id

logger = structlog.get_logger(__name__)

__all__ = ["materialize_phenology_text"]


def _check_credentials_or_raise() -> None:
    """Verifica credenciales Gemini; levanta ``RuntimeError`` si faltan.

    Raises:
        RuntimeError: si no hay ``GEMINI_API_KEY`` ni ``GOOGLE_API_KEY``
            ni configuracion Vertex AI presente.
    """
    if not _has_credentials():
        raise RuntimeError(
            "Gemini no esta configurado. Define una de estas opciones en "
            ".env.local antes de invocar materialize_phenology_text:\n"
            "  1. GEMINI_API_KEY=...        (Google AI Studio)\n"
            "  2. GOOGLE_API_KEY=...        (alias historico)\n"
            "  3. GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT=...\n"
            "Si necesitas un dry-run, inyecta un cliente mock con "
            "ml.features.phenology_description.set_llm_client(callable)."
        )


def _stratified_sample(
    df: pl.DataFrame,
    *,
    class_col: str,
    min_per_class: int,
    seed: int,
) -> pl.DataFrame:
    """Muestrea hasta ``min_per_class`` filas por valor de ``class_col``.

    Para clases con menos parcelas que ``min_per_class`` se toman todas
    (no se hace upsampling). El resultado se mezcla globalmente con la
    misma ``seed`` para evitar ordenacion por clase residual.

    Args:
        df: DataFrame de entrada.
        class_col: Columna con el id de clase (Int / Utf8).
        min_per_class: Numero objetivo de filas por clase.
        seed: Semilla del muestreo.

    Returns:
        DataFrame muestreado y mezclado.
    """
    parts: list[pl.DataFrame] = []
    for class_value, sub in df.group_by(class_col, maintain_order=True):
        del class_value  # only for debugging in pdb if needed.
        n = min(sub.height, min_per_class)
        if sub.height <= min_per_class:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=n, seed=seed, with_replacement=False))
    if not parts:
        return df.clear()
    combined = pl.concat(parts, how="vertical_relaxed")
    return combined.sample(
        fraction=1.0, seed=seed, with_replacement=False, shuffle=True
    )


def materialize_phenology_text(
    parcels_features_path: Path | str,
    *,
    output_path: Path = Path("data/features/phenology_text_pastis.parquet"),
    max_parcels: int | None = None,
    balanced_by_class: bool = True,
    min_per_class: int = 30,
    seed: int = 42,
    model: str = "gemini-3.5-flash",
    overwrite: bool = False,
    enforce_api_key: bool = True,
    class_col: str = "class_id",
    parcel_id_col: str = "parcel_id",
    year_col: str = "year",
    progress_every: int = 100,
) -> Path:
    """Materializa el bloque ``pheno_text_*`` sobre las parcelas reales.

    Comportamiento canonico:

    1. Lee el dataset de features full (PASTIS-R, ~85951 parcelas) desde
       ``parcels_features_path``.
    2. Si ``enforce_api_key=True``: verifica credenciales Gemini con
       :func:`_check_credentials_or_raise`. Si no hay cliente inyectado
       ni env vars, levanta ``RuntimeError``.
    3. Si ``balanced_by_class=True``: estratifica por ``class_col``
       tomando ``min_per_class`` por clase (las que tengan menos usan
       todas las filas disponibles).
    4. Si ``max_parcels`` es ``int > 0``: aplica subsample posterior al
       muestreo balanceado.
    5. Cache: si ``output_path`` existe y ``overwrite=False`` retorna
       sin recomputar (logueando ``phenology_text_cache_hit``).
    6. Llama a :func:`build_phenology_text_block` con ``skip_llm=False``
       forzado.
    7. Persiste parquet con esquema canonico:
       ``parcel_id`` (Utf8) + ``year`` (Int16) +
       ``pheno_text_000..pheno_text_{D-1}`` (Float32, D=384 por default).

    Args:
        parcels_features_path: Path al parquet de features completas
            (PASTIS-R full, ~85951 parcelas).
        output_path: Path destino del parquet con embeddings textuales.
        max_parcels: Limite superior de parcelas a procesar tras el
            muestreo balanceado. ``None`` = sin limite.
        balanced_by_class: Si ``True`` estratifica por ``class_col``.
        min_per_class: Minimo de filas por clase tras la estratificacion.
        seed: Semilla del muestreo.
        model: Identificador del modelo Gemini (default
            ``"gemini-3.5-flash"``).
        overwrite: Si ``True`` recomputa aunque ``output_path`` exista.
        enforce_api_key: Si ``True`` y no hay credenciales ni cliente
            inyectado, levanta ``RuntimeError``. Solo ponerlo en
            ``False`` para tests con cliente mockeado.
        class_col: Columna con el id de clase (default ``"class_id"``).
        parcel_id_col: Columna identificadora de parcela.
        year_col: Columna del anio agronomico.
        progress_every: Frecuencia (en numero de parcelas) del log
            de progreso.

    Returns:
        ``Path`` apuntando al parquet generado o reutilizado.

    Raises:
        RuntimeError: si ``enforce_api_key=True`` y no hay credenciales
            ni cliente inyectado.
        FileNotFoundError: si ``parcels_features_path`` no existe.
    """
    from ml.features.phenology_description import _LLM_CLIENT

    parcels_path = Path(parcels_features_path)
    if not parcels_path.exists():
        raise FileNotFoundError(
            f"parcels_features_path no existe: {parcels_path}"
        )

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        cached = pl.read_parquet(output_path)
        logger.info(
            "phenology_text_cache_hit",
            output_path=str(output_path),
            n_parcels=cached.height,
        )
        return output_path

    # Credentials validation: the hard barrier lives in
    # build_phenology_text_block, but we bring the error forward here to
    # avoid reading the full parquet if we are going to fail.
    if enforce_api_key and _LLM_CLIENT is None:
        _check_credentials_or_raise()

    df = pl.read_parquet(parcels_path)
    df = canonical_parcel_id(df, col=parcel_id_col)
    logger.info(
        "phenology_text_input_loaded",
        path=str(parcels_path),
        n_rows=df.height,
        n_cols=len(df.columns),
    )

    sample = df
    if balanced_by_class:
        if class_col not in df.columns:
            raise KeyError(
                f"balanced_by_class=True requiere la columna {class_col!r}. "
                f"Columnas disponibles: {df.columns}"
            )
        sample = _stratified_sample(
            df, class_col=class_col, min_per_class=min_per_class, seed=seed
        )
        logger.info(
            "phenology_text_balanced_sample",
            n_after_balance=sample.height,
            min_per_class=min_per_class,
        )

    if max_parcels is not None and max_parcels > 0 and sample.height > max_parcels:
        sample = sample.sample(
            n=max_parcels, seed=seed, with_replacement=False
        )
        logger.info("phenology_text_subsampled", n=sample.height)

    n_total = sample.height
    est_cost_usd = n_total * COST_PER_DESCRIPTION_USD
    logger.info(
        "phenology_text_budget_estimate",
        n_total=n_total,
        cost_per_call_usd=COST_PER_DESCRIPTION_USD,
        est_total_cost_usd=round(est_cost_usd, 4),
        model=model,
    )

    t_start = time.monotonic()
    block = build_phenology_text_block(
        sample,
        parcel_id_col=parcel_id_col,
        year_col=year_col,
        model=model,
        skip_llm=False,
        progress_every=progress_every,
    )
    elapsed_s = time.monotonic() - t_start

    # Validation of the canonical output schema.
    if parcel_id_col in block.columns and block.schema[parcel_id_col] != pl.Utf8:
        block = canonical_parcel_id(block, col=parcel_id_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    block.write_parquet(output_path)
    logger.info(
        "phenology_text_materialized",
        output_path=str(output_path),
        n_parcels=block.height,
        n_text_cols=DEFAULT_TEXT_EMBED_DIM,
        elapsed_s=round(elapsed_s, 2),
        est_cost_usd=round(n_total * COST_PER_DESCRIPTION_USD, 4),
    )
    return output_path
