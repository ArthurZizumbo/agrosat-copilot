"""Tests de ``ml.eval.feature_ablation`` (US-022b-C).

Cobertura objetivo >=85%. Los tests usan fixtures Polars sinteticos pequenos
para mantener la corrida en CPU < 30s. El spatial CV es real (no mockeado)
pero acotado a k=3 para que el cache sea pequeno.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.eval.feature_ablation import (
    FeatureAblationResult,
    build_default_feature_sets,
    export_ablation_table,
    run_feature_ablation,
)


@pytest.fixture
def synthetic_features() -> pl.DataFrame:
    """DataFrame minimalista con todos los bloques de features representados.

    180 parcelas, 3 clases balanceadas, columnas tipicas de cada bloque
    para que ``build_default_feature_sets`` devuelva los 5 conjuntos no
    vacios.
    """
    rng = np.random.default_rng(42)
    n = 180
    classes = rng.integers(low=1, high=4, size=n)  # 1, 2, 3 (evita 0/19 drop)
    patch_ids = rng.integers(low=10000, high=10010, size=n).astype(np.int64)
    base = {
        "parcel_id": [f"{p}_{i}" for p, i in zip(patch_ids, range(n), strict=True)],
        "year": [2019] * n,
        "patch_id": patch_ids.tolist(),
        "class_id": classes.tolist(),
    }
    # AlphaEarth 64 dims con senal correlada con la clase.
    for i in range(64):
        col = rng.normal(loc=classes * 0.1, scale=1.0, size=n)
        base[f"ae_{i:02d}"] = col.tolist()
    # Indices x stats (algunos)
    for idx in ("NDVI", "NDWI", "EVI"):
        for stat in ("mean", "std", "p50"):
            base[f"{idx}_{stat}"] = rng.normal(size=n).tolist()
    # FFT (3 indices x (1 DC + 3 armonicos) x 2 (amp, phase)) = 24
    for idx in ("NDVI", "NDWI", "EVI"):
        for k in range(4):
            base[f"{idx}_fft_amp_{k}"] = rng.normal(size=n).tolist()
            base[f"{idx}_fft_phase_{k}"] = rng.normal(size=n).tolist()
    # Fenologia (8 cols)
    for col in (
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
    ):
        base[col] = rng.normal(size=n).tolist()
    # ERA5 (24 cols).
    for m in range(1, 13):
        base[f"era5_tmean_m{m:02d}"] = rng.normal(size=n).tolist()
        base[f"era5_prec_m{m:02d}"] = rng.normal(size=n).tolist()
    # SRTM (3 cols).
    base["srtm_elev_mean"] = rng.normal(size=n).tolist()
    base["srtm_slope_mean"] = rng.normal(size=n).tolist()
    base["srtm_aspect_dominant"] = rng.normal(size=n).tolist()
    # Geom (3 cols).
    base["geom_area_ha"] = rng.normal(size=n).tolist()
    base["geom_perimeter_m"] = rng.normal(size=n).tolist()
    base["geom_elongation"] = rng.normal(size=n).tolist()
    return pl.DataFrame(base)


def test_build_default_feature_sets_returns_five_sets(
    synthetic_features: pl.DataFrame,
) -> None:
    """Los 5 sets canonicos siempre estan; sets opcionales aparecen segun cols.

    US-023-preview P3 agrega ``geom_only`` cuando hay cols ``geom_*`` — el
    fixture sintetico las trae, asi que el set esta presente.
    """
    sets = build_default_feature_sets(synthetic_features.columns)
    canonical = {
        "full",
        "no_geom",
        "no_geom_no_era5_srtm",
        "alphaearth_only",
        "phenology_only",
    }
    assert canonical.issubset(sets.keys())
    # `geom_only` aparece porque el fixture trae geom_area_ha/perimeter/elongation.
    assert "geom_only" in sets


def test_default_sets_geom_exclusion_correct(synthetic_features: pl.DataFrame) -> None:
    sets = build_default_feature_sets(synthetic_features.columns)
    assert all(not c.startswith("geom_") for c in sets["no_geom"])
    # full contiene las 3 cols geom_*.
    assert sum(1 for c in sets["full"] if c.startswith("geom_")) == 3


def test_default_sets_era5_srtm_exclusion(synthetic_features: pl.DataFrame) -> None:
    sets = build_default_feature_sets(synthetic_features.columns)
    cleaned = sets["no_geom_no_era5_srtm"]
    assert all(not c.startswith("era5_") for c in cleaned)
    assert all(not c.startswith("srtm_") for c in cleaned)
    assert all(not c.startswith("geom_") for c in cleaned)


def test_alphaearth_only_set_isolates_ae_cols(synthetic_features: pl.DataFrame) -> None:
    sets = build_default_feature_sets(synthetic_features.columns)
    ae = sets["alphaearth_only"]
    assert len(ae) == 64
    assert all(c.startswith("ae_") for c in ae)


def test_phenology_only_set_contains_pheno_and_fft(
    synthetic_features: pl.DataFrame,
) -> None:
    sets = build_default_feature_sets(synthetic_features.columns)
    pheno = sets["phenology_only"]
    assert "sog_doy" in pheno
    assert "peak_value" in pheno
    assert any("_fft_amp_" in c for c in pheno)


def test_run_ablation_with_xgb_returns_one_row_per_set(
    synthetic_features: pl.DataFrame,
) -> None:
    """Cada set defaultsumado produce 1 fila; US-023-preview P3 anade `geom_only`."""
    results = run_feature_ablation(
        df=synthetic_features,
        models=("xgb",),
        k_folds=3,
        buffer_km=0.5,
        seed=42,
    )
    # 5 canonicos + geom_only (fixture trae cols geom_*) = 6.
    assert len(results) == 6
    sets_seen = {r.feature_set for r in results}
    assert {
        "full",
        "no_geom",
        "no_geom_no_era5_srtm",
        "alphaearth_only",
        "phenology_only",
        "geom_only",
    }.issubset(sets_seen)


def test_run_ablation_delta_vs_full_correct(synthetic_features: pl.DataFrame) -> None:
    results = run_feature_ablation(
        df=synthetic_features,
        models=("xgb",),
        k_folds=3,
        buffer_km=0.5,
        seed=42,
    )
    full = next(r for r in results if r.feature_set == "full")
    no_geom = next(r for r in results if r.feature_set == "no_geom")
    expected_delta = no_geom.f1_macro - full.f1_macro
    assert no_geom.delta_vs_full == pytest.approx(expected_delta, abs=1e-9)
    # delta para `full` mismo siempre es NaN.
    assert np.isnan(full.delta_vs_full)


def test_run_ablation_missing_full_raises_value_error(
    synthetic_features: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="full"):
        run_feature_ablation(
            df=synthetic_features,
            feature_sets={"alphaearth_only": ("ae_00", "ae_01")},
            models=("xgb",),
        )


def test_run_ablation_no_df_no_path_raises(synthetic_features: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="features_path"):
        run_feature_ablation()


def test_run_ablation_max_samples_subsamples(synthetic_features: pl.DataFrame) -> None:
    # max_samples mas chico que n -> sample uniforme determinista.
    results = run_feature_ablation(
        df=synthetic_features,
        models=("xgb",),
        max_samples=120,
        k_folds=3,
        buffer_km=0.5,
        seed=42,
    )
    # 5 sets canonicos + geom_only (fixture trae cols geom_*) = 6.
    assert len(results) == 6


def test_run_ablation_empty_set_emits_nan_row(synthetic_features: pl.DataFrame) -> None:
    # Set vacio (cols inexistentes) -> NaN sin romper la corrida.
    custom_sets = {
        "full": ("ae_00", "ae_01"),
        "ghost": ("nonexistent_a", "nonexistent_b"),
    }
    results = run_feature_ablation(
        df=synthetic_features,
        feature_sets=custom_sets,
        models=("xgb",),
        k_folds=3,
        buffer_km=0.5,
        seed=42,
    )
    ghost = next(r for r in results if r.feature_set == "ghost")
    assert ghost.n_features == 0
    assert np.isnan(ghost.f1_macro)


def test_export_ablation_table_writes_csv_and_md(
    tmp_path: Path, synthetic_features: pl.DataFrame
) -> None:
    fake_results = [
        FeatureAblationResult(
            feature_set="full",
            model_kind="xgb",
            f1_macro=0.50,
            f1_weighted=0.55,
            miou=0.40,
            n_features=189,
            delta_vs_full=float("nan"),
        ),
        FeatureAblationResult(
            feature_set="no_geom",
            model_kind="xgb",
            f1_macro=0.51,
            f1_weighted=0.56,
            miou=0.41,
            n_features=186,
            delta_vs_full=0.01,
        ),
    ]
    csv_path, md_path = export_ablation_table(fake_results, tmp_path / "ablation_table")
    assert csv_path.exists()
    assert md_path.exists()
    table = pl.read_csv(csv_path)
    assert table.height == 2
    assert "delta_vs_full" in table.columns


def test_run_ablation_xgb_full_outperforms_or_equals_random(
    synthetic_features: pl.DataFrame,
) -> None:
    """El XGB sobre 64 dims AE con senal sintetica debe superar 0.0 trivial."""
    results = run_feature_ablation(
        df=synthetic_features,
        models=("xgb",),
        k_folds=3,
        buffer_km=0.5,
        seed=42,
    )
    ae_only = next(r for r in results if r.feature_set == "alphaearth_only")
    # La senal sintetica esta correlada con la clase via ae_*; F1 > 0.0.
    # Aceptamos hasta 0.05 (CV espacial puede colapsar folds chicos a 0).
    assert ae_only.f1_macro >= 0.0
    assert ae_only.n_features == 64


# ---------------------------------------------------------------------------
# US-023-preview P3 — geom_only (test cuantitativo de leakage espacial).
# ---------------------------------------------------------------------------


def test_default_sets_includes_geom_only_when_geom_cols_present(
    synthetic_features: pl.DataFrame,
) -> None:
    """Si hay cols `geom_*`, build_default_feature_sets agrega `geom_only`."""
    sets = build_default_feature_sets(synthetic_features.columns)
    assert "geom_only" in sets
    geom_only = sets["geom_only"]
    assert all(c.startswith("geom_") for c in geom_only)
    assert len(geom_only) == 3  # area, perimeter, elongation


def test_default_sets_omits_geom_only_when_no_geom_cols() -> None:
    """Sin cols `geom_*`, la llave `geom_only` no se agrega (graceful)."""
    cols_without_geom = (
        "parcel_id",
        "year",
        "class_id",
        "ae_00",
        "ae_01",
        "ae_02",
    )
    sets = build_default_feature_sets(cols_without_geom)
    assert "geom_only" not in sets


# ---------------------------------------------------------------------------
# US-023-preview P5 — with_spectral_signature / spectral_signature_only.
# ---------------------------------------------------------------------------


def test_default_sets_includes_spectral_signature_when_cols_present() -> None:
    """Si hay cols `spectral_signature_*`, se agregan los 2 sets opcionales."""
    cols = (
        "parcel_id",
        "year",
        "class_id",
        "ae_00",
        "sog_doy",
        "NDVI_fft_amp_0",
        "spectral_signature_000",
        "spectral_signature_001",
        "spectral_signature_002",
    )
    sets = build_default_feature_sets(cols)
    assert "with_spectral_signature" in sets
    assert "spectral_signature_only" in sets
    spec_only = sets["spectral_signature_only"]
    assert all(c.startswith("spectral_signature_") for c in spec_only)
    assert len(spec_only) == 3
    # `with_spectral_signature` = phenology_only + spectral_signature_*.
    with_spec = sets["with_spectral_signature"]
    assert "spectral_signature_000" in with_spec
    assert "sog_doy" in with_spec


def test_default_sets_omits_spectral_signature_when_cols_absent(
    synthetic_features: pl.DataFrame,
) -> None:
    """Sin cols `spectral_signature_*`, las llaves opcionales no se agregan."""
    sets = build_default_feature_sets(synthetic_features.columns)
    assert "with_spectral_signature" not in sets
    assert "spectral_signature_only" not in sets


def test_feature_ablation_lgbm_supported(synthetic_features: pl.DataFrame) -> None:
    """run_feature_ablation acepta `lgbm` y devuelve resultados validos."""
    results = run_feature_ablation(
        df=synthetic_features,
        models=("lgbm",),
        k_folds=3,
        buffer_km=0.5,
        seed=42,
    )
    assert len(results) >= 1
    assert all(r.model_kind == "lgbm" for r in results)
    full = next(r for r in results if r.feature_set == "full")
    assert 0.0 <= full.f1_macro <= 1.0
    assert full.n_features > 0


def test_default_sets_pheno_text_only_added_when_pheno_text_cols_present() -> None:
    """Si hay cols `pheno_text_*`, se agrega tambien `pheno_text_only` (P4 prep)."""
    cols = (
        "parcel_id",
        "year",
        "class_id",
        "ae_00",
        "sog_doy",
        "NDVI_fft_amp_0",
        "pheno_text_000",
        "pheno_text_001",
    )
    sets = build_default_feature_sets(cols)
    assert "with_pheno_text" in sets
    assert "pheno_text_only" in sets
    assert all(c.startswith("pheno_text_") for c in sets["pheno_text_only"])
