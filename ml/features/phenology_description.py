"""Rama semantica fenologica via descripcion + text-encoder (US-022b-D).

Implementa el metodo Wen et al. (2025) "Phenology description is all you
need!" §3.2-3.3: dada la curva NDVI temporal de una parcela, produce una
descripcion textual estructurada con un LLM y la codifica con un
text-encoder a un vector denso que se concatena al vector tabular del
baseline.

El generador usa **Gemini 3.5 Flash** (ADR-006 D4, decision aprobada por el
equipo) via Vertex AI / LiteLLM (sin SDK ``google-genai`` nuevo). El
text-encoder por defecto es ``sentence-transformers`` (ya en deps); el
text-encoder CLIP de FarSLIP queda como alternativa documentada (no
implementada aqui — propiedad de US-017).

Decisiones canonicas (plan ``docs/us-planning/us-022b.md`` §6.3):

- **D-ARQ-3**: la rama semantica es **componente de modelo**, no columna
  trivial de FE. Modulo propio + cache + tests.
- **D-ARQ-5**: Gemini se invoca via ``litellm`` (ya en deps) que rutea a
  Vertex AI. Sin SDK ``google-genai`` nuevo.
- **R7 mitigado**: ``temperature=0`` + cache por ``parcel_id`` + subset
  estratificado obligatorios. Test suite mockea Gemini en CI (cero
  llamadas de red).
- **Output del modulo**: ``pl.DataFrame`` con ``parcel_id`` + N columnas
  ``pheno_text_NNN``; el bloque entra a ``fusion.py`` via LEFT JOIN.

Prompt de 3 bloques (Wen Fig. 2):
  1. **General**: descripcion del problema y formato esperado.
  2. **Time-Series Curve**: los valores NDVI y dias del anio.
  3. **Restrictive Instruction**: salida estructurada, sin texto extra.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "PhenologyDescription",
    "build_phenology_text_block",
    "default_cache_dir",
    "encode_descriptions",
    "generate_phenology_description",
    "set_llm_client",
]

#: Numero de dimensiones del vector denso producido por
#: ``sentence-transformers/all-MiniLM-L6-v2`` (modelo default).
#: La constante existe para que ``fusion.py`` valide el contrato.
DEFAULT_TEXT_ENCODER: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_EMBED_DIM: int = 384

#: Prompt completo de 3 bloques estilo Wen et al. 2025 Fig. 2.
PROMPT_TEMPLATE: str = """[BLOQUE 1 - GENERAL]
Eres un agronomo experto en teledeteccion. Tu tarea es describir la
fenologia de una parcela agricola a partir de su curva NDVI anual derivada
de Sentinel-2. La descripcion sirve para alinearse, via un text-encoder
contrastivo, con una representacion visual del cultivo.

[BLOQUE 2 - SERIE TEMPORAL]
Curva NDVI muestreada en {n_points} puntos del anio:
{curve_serialization}

Pico NDVI: {peak_value:.3f} en el dia {peak_doy} del anio.
Inicio de crecimiento (SOG, umbral 0.3): dia {sog_doy}.
Senescencia (NDVI < 0.3 tras pico): dia {senescence_doy}.
Pista de cultivo (puede estar vacia): {crop_type_hint}.

[BLOQUE 3 - INSTRUCCION RESTRICTIVA]
Devuelve UNICAMENTE un parrafo de 3-4 frases en espanol descriptivo,
sin saltos de linea, sin viñetas, sin encabezados. Menciona: temporada
agronomica probable, comportamiento del crecimiento (rapido/lento/doble
pico/uniforme), nivel de pico (bajo/medio/alto), y duracion estimada de
la madurez (corta/media/larga). NO inventes valores numericos fuera de
los datos dados. NO repitas los numeros literales del bloque 2.
"""


# ---------------------------------------------------------------------------
# Dataclass + cache.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhenologyDescription:
    """Descripcion fenologica de una parcela + su embedding.

    Attributes:
        parcel_id: Identificador de la parcela (acepta int o string).
        description: Texto generado por el LLM (sin caracteres de control).
        embedding: Vector denso del text-encoder (shape ``(D,)``).
        crop_type_hint: Pista de cultivo suministrada al prompt (puede ser
            ``None``).
    """

    parcel_id: int | str
    description: str
    embedding: list[float]
    crop_type_hint: str | None


def default_cache_dir() -> Path:
    """Directorio default para el cache de descripciones (DVC-friendly).

    Returns:
        ``data/cache/phenology_descriptions/`` (crea el directorio si no
        existe; ``.gitignore`` lo excluye salvo el manifest).
    """
    repo_root = Path(__file__).resolve().parents[2]
    cache = repo_root / "data" / "cache" / "phenology_descriptions"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ---------------------------------------------------------------------------
# Cliente LLM (mockeable en tests).
# ---------------------------------------------------------------------------


#: Firma del cliente LLM: ``(prompt, *, model, temperature) -> str``.
LlmClient = Callable[..., str]


_LLM_CLIENT: LlmClient | None = None


def set_llm_client(client: LlmClient | None) -> None:
    """Inyecta un cliente LLM custom (uso principal: tests mockeados).

    Args:
        client: Callable ``(prompt: str, *, model: str, temperature: float)
            -> str``. Pasar ``None`` restaura el default LiteLLM/Vertex AI.
    """
    global _LLM_CLIENT
    _LLM_CLIENT = client


def _default_litellm_client(prompt: str, *, model: str, temperature: float) -> str:
    """Cliente Gemini 3.5 Flash via LiteLLM (provider-agnostic fallback).

    Importacion lazy: solo se requiere en runtime y solo si no hay cliente
    inyectado por ``set_llm_client``. En CI/tests SIEMPRE hay un mock.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "litellm no esta instalado. Ejecuta `poetry install --with ml`. "
            "Alternativamente inyecta un cliente con `set_llm_client(...)`."
        ) from exc

    response = litellm.completion(  # type: ignore[attr-defined]
        model=model if "/" in model else f"vertex_ai/{model}",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    # LiteLLM normaliza la respuesta a formato OpenAI.
    content = response["choices"][0]["message"]["content"]
    return str(content).strip()


def _default_google_genai_client(prompt: str, *, model: str, temperature: float) -> str:
    """Cliente nativo Gemini 3.5 Flash via google-genai SDK.

    Mas eficiente que LiteLLM para este caso: soporta ``thinking_level``
    (default ``"minimal"`` para latencia baja en descripciones cortas) y
    permite structured output con ``response_json_schema``. Por ahora
    devolvemos texto plano para compatibilidad con el contrato
    ``LlmClient``; el structured output queda preparado para
    ``build_phenology_text_block_structured`` (siguiente iteracion).

    Auth:
      - Vertex AI (preferido): exporta ``GOOGLE_GENAI_USE_VERTEXAI=true`` +
        ``GOOGLE_CLOUD_PROJECT=...`` + ``GOOGLE_CLOUD_LOCATION=us-central1``.
        Usa la SA del entorno (Cloud Run, Vertex AI Workbench, etc.).
      - Gemini API publica: exporta ``GEMINI_API_KEY=...`` (o ``GOOGLE_API_KEY``).

    Importacion lazy del SDK; tests usan ``set_llm_client`` con mock.

    Args:
        prompt: Texto del prompt 3-bloques (Wen et al. 2025 Fig. 2).
        model: Identificador del modelo, default ``"gemini-3.5-flash"``.
        temperature: Determinismo. ``0.0`` para cache estable por parcela.

    Returns:
        Texto plano de la descripcion fenologica generada.
    """
    try:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-genai no esta instalado. Ejecuta `poetry add google-genai`. "
            "Alternativamente usa el cliente LiteLLM (fallback) o inyecta un "
            "mock con `set_llm_client(...)`."
        ) from exc

    # `Client()` autodetecta Vertex AI vs API publica desde env vars.
    client = genai.Client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=512,
        # `thinking_level="minimal"` reduce latencia para descripciones
        # cortas; soportado por Gemini 3.x (incluido 3.5-flash). El
        # parametro `thinking_budget` numerico fue deprecado segun la docu
        # oficial de Gemini 3.5.
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return str(response.text or "").strip()


def _get_client() -> LlmClient:
    """Devuelve el cliente activo segun el orden:

    1. Cliente inyectado por ``set_llm_client`` (tests siempre lo usan).
    2. Cliente nativo ``google-genai`` si la env var ``AGROSAT_LLM_PROVIDER``
       es ``"google-genai"`` o si ``google-genai`` esta instalado y no se
       fuerza otro.
    3. Fallback LiteLLM (provider-agnostic).
    """
    import os

    if _LLM_CLIENT is not None:
        return _LLM_CLIENT

    provider = os.environ.get("AGROSAT_LLM_PROVIDER", "").strip().lower()
    if provider == "google-genai":
        return _default_google_genai_client
    if provider == "litellm":
        return _default_litellm_client

    # Auto: preferir google-genai si esta disponible (mas rapido); LiteLLM
    # como fallback transparente.
    try:
        import google.genai  # noqa: F401

        return _default_google_genai_client
    except ImportError:
        return _default_litellm_client


# ---------------------------------------------------------------------------
# API publica.
# ---------------------------------------------------------------------------


def generate_phenology_description(
    ndvi_curve: np.ndarray,
    doy: np.ndarray | None = None,
    *,
    parcel_id: int | str | None = None,
    crop_type_hint: str | None = None,
    model: str = "gemini-3.5-flash",
    temperature: float = 0.0,
    cache_dir: Path | None = None,
) -> str:
    """Genera la descripcion fenologica textual de una parcela.

    Implementa el prompt 3-bloques de Wen et al. (2025) Fig. 2:
    General + Time-Series Curve + Restrictive Instruction.

    Args:
        ndvi_curve: Vector NDVI ``(T,)`` (curva diaria interpolada o
            muestreo regular). Acepta NaN; los reemplaza por el promedio
            local antes de serializar.
        doy: Vector ``(T,)`` de dia del anio (1..366). Si es ``None`` se
            asume rejilla equiespaciada ``linspace(1, 365, T)``.
        parcel_id: Para nombrar el archivo de cache. Si es ``None`` la
            descripcion no se cachea (uso ad-hoc).
        crop_type_hint: Pista opcional ("trigo", "maiz", "viñedo", ...)
            que entra al bloque 2 del prompt.
        model: Identificador del modelo (default ``"gemini-3.5-flash"``).
            LiteLLM rutea a Vertex AI; otros valores OpenAI-compatibles
            tambien funcionan.
        temperature: Temperatura de muestreo. **Default 0.0 obligatorio**
            (determinismo + mitigacion de costos R7).
        cache_dir: Directorio del cache. Si es ``None`` usa
            :func:`default_cache_dir`. El cache cachea por hash de
            ``(parcel_id, ndvi_curve.tobytes(), prompt_template_version)``.

    Returns:
        Descripcion textual (3-4 frases) sin saltos de linea.

    Raises:
        ValueError: si ``ndvi_curve`` esta vacio o tiene shape no 1D, o
            si ``temperature`` no es 0.0 (R7: enforced para mitigar costo).
    """
    if ndvi_curve.ndim != 1:
        raise ValueError(f"`ndvi_curve` debe ser 1D; recibido shape {ndvi_curve.shape}.")
    if ndvi_curve.size == 0:
        raise ValueError("`ndvi_curve` no puede estar vacio.")
    if temperature != 0.0:
        raise ValueError("`temperature` debe ser 0.0 (R7 — determinismo + costo Gemini).")

    if doy is None:
        doy = np.linspace(1.0, 365.0, ndvi_curve.size, dtype=np.float64)
    elif doy.shape != ndvi_curve.shape:
        raise ValueError(
            f"`doy` shape {doy.shape} no coincide con `ndvi_curve` shape {ndvi_curve.shape}."
        )

    # Imputa NaN con la media de los finitos (no rompe el prompt).
    clean_curve = ndvi_curve.copy()
    if np.isnan(clean_curve).any():
        finite_mean = float(np.nanmean(clean_curve)) if np.any(~np.isnan(clean_curve)) else 0.0
        clean_curve = np.where(np.isnan(clean_curve), finite_mean, clean_curve)

    cache_dir_resolved = cache_dir if cache_dir is not None else default_cache_dir()
    cache_key = _hash_curve(parcel_id, clean_curve, model)
    cache_file = cache_dir_resolved / f"{cache_key}.json"
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            description = str(payload["description"]).replace("\n", " ").strip()
            logger.info("phenology_description_cache_hit", parcel_id=parcel_id)
            return description
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "phenology_description_cache_unreadable",
                cache_file=str(cache_file),
                error=str(exc),
            )

    prompt = _build_prompt(clean_curve, doy, crop_type_hint=crop_type_hint)
    client = _get_client()
    raw = client(prompt, model=model, temperature=temperature)
    description = " ".join(raw.split())

    payload = {
        "parcel_id": parcel_id,
        "model": model,
        "temperature": temperature,
        "description": description,
        "crop_type_hint": crop_type_hint,
    }
    try:
        cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        logger.warning(
            "phenology_description_cache_write_failed",
            cache_file=str(cache_file),
            error=str(exc),
        )
    return description


def encode_descriptions(
    descriptions: Sequence[str],
    *,
    encoder: str = "sentence-transformers",
    model_name: str = DEFAULT_TEXT_ENCODER,
) -> np.ndarray:
    """Codifica las descripciones a vectores densos.

    Args:
        descriptions: Secuencia de textos a codificar.
        encoder: ``"sentence-transformers"`` (default; ya en deps) o
            ``"farslip-clip"`` (alternativa documentada; consume el
            text-encoder del :class:`FarSLIPExtractor`, **no implementada
            aqui** — propiedad de US-017).
        model_name: Identificador del modelo sentence-transformers.

    Returns:
        Matriz ``(N, D)`` ``np.float32``. D = ``DEFAULT_TEXT_EMBED_DIM`` para
        el modelo default.

    Raises:
        NotImplementedError: si ``encoder == "farslip-clip"`` (TODO
        Paper Track / Fase 2 US-022-b).
        ValueError: si ``encoder`` no es uno de los valores soportados.
    """
    if encoder == "farslip-clip":
        raise NotImplementedError(
            "encoder='farslip-clip' aun no implementado; "
            "queda como TODO Paper Track. Usa 'sentence-transformers'."
        )
    if encoder != "sentence-transformers":
        raise ValueError(
            f"`encoder` debe ser 'sentence-transformers' o 'farslip-clip'; recibido {encoder!r}."
        )
    if not descriptions:
        return np.zeros((0, DEFAULT_TEXT_EMBED_DIM), dtype=np.float32)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers no esta instalado. Ejecuta `poetry install --with ml`."
        ) from exc
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(descriptions),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_phenology_text_block(
    parcel_ndvi_frame: pl.DataFrame,
    *,
    ndvi_col_prefix: str = "NDVI_t_",
    parcel_id_col: str = "parcel_id",
    year_col: str = "year",
    crop_hint_col: str | None = None,
    model: str = "gemini-3.5-flash",
    encoder: str = "sentence-transformers",
    cache_dir: Path | None = None,
    max_parcels: int | None = None,
    seed: int = 42,
    skip_llm: bool = False,
) -> pl.DataFrame:
    """Construye el bloque ``pheno_text_*`` listo para LEFT JOIN en ``fusion.py``.

    Para cada fila de ``parcel_ndvi_frame``:
      1. Reconstruye la curva NDVI desde columnas ``{ndvi_col_prefix}{i:02d}``
         (o columnas FFT si las primeras no existen — caer-back simetrico
         al de :class:`TemporalDataset`).
      2. Llama a :func:`generate_phenology_description` (mockeable en tests
         via :func:`set_llm_client`).
      3. Codifica todas las descripciones con :func:`encode_descriptions`.

    Args:
        parcel_ndvi_frame: DataFrame con al menos ``parcel_id``, ``year`` y
            curva NDVI / coeficientes FFT.
        ndvi_col_prefix: Prefijo de las columnas T-discretas (default
            ``"NDVI_t_"``).
        parcel_id_col: Nombre de la columna identificadora.
        year_col: Nombre de la columna de anio.
        crop_hint_col: Columna opcional con la pista de cultivo
            (string-like). Si es ``None`` se omite.
        model: Modelo LLM (default ``"gemini-3.5-flash"``).
        encoder: Text-encoder (``"sentence-transformers"`` o
            ``"farslip-clip"``).
        cache_dir: Directorio de cache (default
            :func:`default_cache_dir`).
        max_parcels: Subsample estratificado (D-5 de US-022b-D). ``None``
            = todas las filas (cuidado con costo Gemini).
        seed: Semilla del subsample.
        skip_llm: Si ``True`` salta la llamada al LLM y genera embeddings
            zeros — util para CI y dev sin credenciales Vertex AI; el
            bloque resultante tiene la forma correcta pero contenido nulo.

    Returns:
        ``pl.DataFrame`` con ``parcel_id``, ``year`` y
        ``pheno_text_000..pheno_text_{D-1}``.
    """
    df = parcel_ndvi_frame
    if max_parcels is not None and max_parcels > 0 and df.height > max_parcels:
        df = df.sample(n=max_parcels, seed=seed, with_replacement=False)
        logger.info("pheno_text_block_subsampled", n=df.height)

    parcel_ids = df.get_column(parcel_id_col).to_list()
    years = df.get_column(year_col).to_list() if year_col in df.columns else [None] * df.height
    crop_hints: list[str | None] = (
        df.get_column(crop_hint_col).to_list()
        if crop_hint_col is not None and crop_hint_col in df.columns
        else [None] * df.height
    )

    curves = _extract_ndvi_curves(df, prefix=ndvi_col_prefix)
    descriptions: list[str] = []
    if skip_llm:
        # Generamos descripciones placeholders deterministas para que el
        # encoder produzca embeddings reproducibles en CI sin red.
        descriptions = [f"placeholder_pheno_{pid}" for pid in parcel_ids]
    else:
        for pid, curve, hint in zip(parcel_ids, curves, crop_hints, strict=True):
            descriptions.append(
                generate_phenology_description(
                    curve,
                    parcel_id=pid,
                    crop_type_hint=hint,
                    model=model,
                    temperature=0.0,
                    cache_dir=cache_dir,
                )
            )

    if skip_llm:
        # En skip_llm devolvemos ceros para evitar descargar el modelo
        # sentence-transformers (test sin red ni dependencias pesadas).
        embeddings = np.zeros((len(descriptions), DEFAULT_TEXT_EMBED_DIM), dtype=np.float32)
    else:
        embeddings = encode_descriptions(descriptions, encoder=encoder)

    text_cols = [f"pheno_text_{i:03d}" for i in range(embeddings.shape[1])]
    block = {
        parcel_id_col: parcel_ids,
        year_col: years,
    }
    for j, col_name in enumerate(text_cols):
        block[col_name] = embeddings[:, j].tolist()
    schema: dict[str, pl.DataType] = {
        parcel_id_col: df.schema[parcel_id_col],
        year_col: df.schema.get(year_col, pl.Int16()),
    }
    for col_name in text_cols:
        schema[col_name] = pl.Float32()
    return pl.DataFrame(block, schema=schema)


# ---------------------------------------------------------------------------
# Helpers privados.
# ---------------------------------------------------------------------------


def _build_prompt(
    ndvi_curve: np.ndarray,
    doy: np.ndarray,
    *,
    crop_type_hint: str | None,
) -> str:
    """Construye el prompt completo Wen Fig. 2 (3 bloques)."""
    # Submuestreo del prompt: si la curva es muy larga, presentamos hasta
    # 24 puntos uniformemente espaciados (suficiente para captar la
    # estacionalidad y mantiene el token count razonable).
    max_points = 24
    if ndvi_curve.size > max_points:
        idx = np.linspace(0, ndvi_curve.size - 1, max_points, dtype=np.int64)
        curve_sub = ndvi_curve[idx]
        doy_sub = doy[idx]
    else:
        curve_sub = ndvi_curve
        doy_sub = doy

    serialization = ", ".join(
        f"(doy {int(d)}: {v:.3f})" for d, v in zip(doy_sub, curve_sub, strict=True)
    )
    peak_idx = int(np.argmax(ndvi_curve))
    peak_value = float(ndvi_curve[peak_idx])
    peak_doy = int(doy[peak_idx])
    sog_threshold = 0.3
    sog_indices = np.where(ndvi_curve >= sog_threshold)[0]
    sog_doy = int(doy[sog_indices[0]]) if sog_indices.size else -1
    post_peak_below = np.where(
        (np.arange(ndvi_curve.size) > peak_idx) & (ndvi_curve < sog_threshold)
    )[0]
    senescence_doy = int(doy[post_peak_below[0]]) if post_peak_below.size else -1

    return PROMPT_TEMPLATE.format(
        n_points=curve_sub.size,
        curve_serialization=serialization,
        peak_value=peak_value,
        peak_doy=peak_doy,
        sog_doy=sog_doy,
        senescence_doy=senescence_doy,
        crop_type_hint=crop_type_hint if crop_type_hint is not None else "(sin pista)",
    )


def _hash_curve(parcel_id: object, curve: np.ndarray, model: str) -> str:
    """Hash determinista de la combinacion (parcel_id, curva, modelo, prompt_v1)."""
    h = hashlib.sha256()
    h.update(repr(parcel_id).encode("utf-8"))
    h.update(curve.astype(np.float32).tobytes())
    h.update(model.encode("utf-8"))
    h.update(b"prompt_v1")
    return h.hexdigest()[:16]


def _extract_ndvi_curves(df: pl.DataFrame, *, prefix: str) -> list[np.ndarray]:
    """Extrae una curva por fila desde columnas ``{prefix}{i:02d}``.

    Si no encuentra esas columnas, reconstruye desde FFT (mismo helper que
    :class:`ml.train.phenology_models.TemporalDataset`).
    """
    candidate_cols = sorted(c for c in df.columns if c.startswith(prefix))
    if candidate_cols:
        matrix = df.select(candidate_cols).fill_null(0.0).to_numpy().astype(np.float32)
        return [matrix[i] for i in range(matrix.shape[0])]

    # Fallback: reconstruccion FFT.
    from ml.train.phenology_models import _reconstruct_curve as _rc

    matrix = _rc(df, index_name="NDVI", sequence_length=72)
    return [matrix[i] for i in range(matrix.shape[0])]
