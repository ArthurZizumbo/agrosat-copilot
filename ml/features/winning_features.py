"""Seleccion canonica del conjunto de features ganador para downstream EPIC 5.

Cierra el ciclo del baseline US-023-preview: una vez ejecutadas las
ablaciones (FarSLIP, pheno_text, spectral_signature, geom_only), esta
funcion decide cuales bloques se promueven y persiste el parquet final
`data/features/features_fused_winning_italy.parquet` que consumen los
modelos densos de EPIC 5 (U-Net, U-TAE, TSViT, Swin-UNETR) y los ensambles
de EPIC 6.

Regla de decision (alineada con el plan US-023-preview):

- Bloque base obligatorio: `phenology_only` (8 cols) + `indices_stats` (85
  cols del subset US-018: NDVI/NDWI/EVI x stats + FFT NDVI).
- Bloque AlphaEarth (`ae_*`): siempre incluir si esta presente (Foundation
  Model gratis vía GEE).
- Bloque ERA5 (`era5_*`) + SRTM (`srtm_*`): incluir salvo que ablation
  muestre que aportan negativamente.
- `geom_*`: descartar por defecto (US-022-b decision: leakage espacial).
- Bloques opcionales (`farslip`, `pheno_text`, `spectral_signature`):
  incluir solo si la ablacion los promueve (delta >= +0.005 vs `full`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "WinningFeatureSet",
    "persist_winning_features",
    "select_winning_features",
]


@dataclass(frozen=True)
class WinningFeatureSet:
    """Resultado de la seleccion del conjunto ganador.

    Attributes:
        name: Etiqueta corta (e.g. `"phenology+ae+farslip"`).
        feature_cols: Tupla ordenada de columnas seleccionadas.
        decisions: Mapping `{bloque: bool}` con las decisiones promover/descartar.
        rationale: Texto en lenguaje accesible explicando la seleccion.
        delta_vs_full: Delta de F1-macro reportado por la ablacion para
            cada bloque opcional promovido.
    """

    name: str
    feature_cols: tuple[str, ...]
    decisions: dict[str, bool]
    rationale: str
    delta_vs_full: dict[str, float]


def select_winning_features(
    ablation_table: pl.DataFrame,
    available_cols: Sequence[str],
    *,
    promote_threshold: float = 0.005,
    discard_geom: bool = True,
) -> WinningFeatureSet:
    """Selecciona el conjunto ganador en base a la tabla de ablation.

    Args:
        ablation_table: DataFrame con columnas `feature_set`, `model`,
            `f1_macro`, `delta_vs_full`. Tipicamente el output de
            :func:`ml.utils.baseline_notebook_helpers.run_ablation_and_persist`.
        available_cols: Lista de columnas disponibles en el dataset fused.
        promote_threshold: Delta minimo para promover un bloque opcional.
        discard_geom: Si True, descarta `geom_*` por leakage espacial.

    Returns:
        `WinningFeatureSet` con la decision y la lista de columnas.
    """
    deltas = _read_deltas(ablation_table)
    decisions: dict[str, bool] = {
        "geom": not discard_geom,
        "farslip": deltas.get("with_farslip", float("nan")) >= promote_threshold,
        "pheno_text": deltas.get("with_pheno_text", float("nan")) >= promote_threshold,
        "spectral_signature": (
            deltas.get("with_spectral_signature", float("nan")) >= promote_threshold
        ),
    }

    base_cols = _base_block_cols(available_cols, include_geom=not discard_geom)
    optional_cols: list[str] = []
    for block, promoted in decisions.items():
        if not promoted:
            continue
        optional_cols.extend(_optional_block_cols(available_cols, block))
    selected = tuple(sorted(set(base_cols + optional_cols)))

    promoted_blocks = [b for b, ok in decisions.items() if ok and b != "geom"]
    name_parts = ["phenology", "ae", "indices"]
    name_parts.extend(promoted_blocks)
    name = "+".join(name_parts)

    rationale_lines = [
        f"Conjunto ganador: `{name}` con {len(selected)} columnas.",
        (
            "Bloque base: AlphaEarth, indices espectrales x stats, fenologia "
            "y FFT NDVI."
        ),
    ]
    if discard_geom:
        rationale_lines.append(
            "`geom_*` descartado (leakage espacial confirmado en US-022-b)."
        )
    for block, promoted in decisions.items():
        if block == "geom":
            continue
        if promoted:
            rationale_lines.append(
                f"`{block}` promovido (delta={deltas.get(f'with_{block}', float('nan')):+.4f})."
            )
        else:
            rationale_lines.append(
                f"`{block}` descartado (delta={deltas.get(f'with_{block}', float('nan')):+.4f} "
                f"< {promote_threshold:+.3f})."
            )

    rationale = "\n".join(rationale_lines)
    logger.info(
        "winning_features_selected",
        name=name,
        n_cols=len(selected),
        decisions=decisions,
    )
    return WinningFeatureSet(
        name=name,
        feature_cols=selected,
        decisions=decisions,
        rationale=rationale,
        delta_vs_full={k: v for k, v in deltas.items() if k.startswith("with_")},
    )


def persist_winning_features(
    winning: WinningFeatureSet,
    fused_df: pl.DataFrame,
    *,
    output_path: Path | str = Path(
        "data/features/features_fused_winning_italy.parquet"
    ),
    overwrite: bool = False,
) -> Path:
    """Persiste el subset de features ganadoras del dataset fused.

    Mantiene las columnas de metadata (`parcel_id`, `year`, `class_id`,
    `patch_id`) ademas de las features seleccionadas.

    Args:
        winning: Resultado de :func:`select_winning_features`.
        fused_df: DataFrame Polars con todas las features.
        output_path: Path destino del parquet.
        overwrite: Si False y el archivo existe, no escribe.

    Returns:
        Path del parquet escrito.
    """
    output = Path(output_path)
    if output.exists() and not overwrite:
        logger.info("winning_features_already_exists", path=str(output))
        return output

    meta_cols = [
        c for c in ("parcel_id", "year", "class_id", "patch_id", "fold")
        if c in fused_df.columns
    ]
    feature_cols_present = [c for c in winning.feature_cols if c in fused_df.columns]
    keep = meta_cols + feature_cols_present
    subset = fused_df.select(keep)

    output.parent.mkdir(parents=True, exist_ok=True)
    subset.write_parquet(output)

    manifest_path = output.with_suffix(".manifest.json")
    import json

    manifest_path.write_text(
        json.dumps(
            {
                "name": winning.name,
                "n_features": len(feature_cols_present),
                "feature_cols": feature_cols_present,
                "meta_cols": meta_cols,
                "decisions": winning.decisions,
                "delta_vs_full": winning.delta_vs_full,
                "rationale": winning.rationale,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "winning_features_persisted",
        parquet=str(output),
        manifest=str(manifest_path),
        n_features=len(feature_cols_present),
    )
    return output


# ---------------------------------------------------------------------------
# Helpers privados.
# ---------------------------------------------------------------------------


def _read_deltas(ablation_table: pl.DataFrame) -> dict[str, float]:
    """Extrae el mapping `{feature_set: delta_vs_full}` (modelo de referencia)."""
    if "delta_vs_full" not in ablation_table.columns:
        return {}
    # Si hay varios modelos, tomar el con mejor f1_macro en `full`.
    ref_model: str | None = None
    if "model" in ablation_table.columns:
        full_rows = ablation_table.filter(pl.col("feature_set") == "full")
        if full_rows.height > 0:
            ref_model = (
                full_rows.sort("f1_macro", descending=True).select("model").row(0)[0]
            )
    filtered = ablation_table
    if ref_model is not None:
        filtered = filtered.filter(pl.col("model") == ref_model)
    out: dict[str, float] = {}
    for row in filtered.iter_rows(named=True):
        delta = row.get("delta_vs_full")
        if delta is None:
            continue
        out[row["feature_set"]] = float(delta)
    return out


def _base_block_cols(available_cols: Sequence[str], *, include_geom: bool) -> list[str]:
    """Devuelve las columnas base obligatorias presentes en `available_cols`."""
    pheno_known = {
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
    }
    cols: list[str] = []
    for c in available_cols:
        if c in pheno_known:
            cols.append(c)
        elif "_fft_" in c:
            cols.append(c)
        elif c.startswith("ae_") or c.startswith("dim_") or c.startswith("emb_") or c.startswith("alphaearth_"):
            cols.append(c)
        elif c.startswith("era5_") or c.startswith("srtm_"):
            cols.append(c)
        elif c.startswith("s1_"):
            cols.append(c)
        elif any(
            c.startswith(f"{idx.lower()}_")
            for idx in ("NDVI", "NDWI", "EVI", "MSAVI2", "MCARI", "CCCI", "NDRE")
        ):
            # Indices espectrales x stats (e.g. ndvi_mean, ndwi_p95, ...).
            cols.append(c)
        elif include_geom and c.startswith("geom_"):
            cols.append(c)
    return cols


def _optional_block_cols(available_cols: Sequence[str], block: str) -> list[str]:
    """Devuelve las cols del bloque opcional indicado."""
    prefix_map = {
        "farslip": "farslip_",
        "pheno_text": "pheno_text_",
        "spectral_signature": "spectral_signature_",
    }
    prefix = prefix_map.get(block)
    if prefix is None:
        return []
    return [c for c in available_cols if c.startswith(prefix)]
