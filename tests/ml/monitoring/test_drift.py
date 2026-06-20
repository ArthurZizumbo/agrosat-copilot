"""Tests del pipeline puro de drift (US-060) sobre datos REALES del repo.

Todos los DataFrames provienen del parquet real
``data/farslip/embeddings_pastis.parquet`` (embeddings FarSLIP + ``class_id`` en
el espacio 18-clase US-030). No se fabrican datos sinteticos: cuando se necesita
contraste con/sin drift se usan filas reales identicas (sin drift) o
subpoblaciones reales distintas / un desplazamiento explicito de una banda real
(con drift). Si el parquet no esta disponible (DVC sin pull), los tests se
skipean en lugar de inventar datos.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.monitoring.drift import (
    DRIFT_SCORE_THRESHOLD,
    DriftSummary,
    build_drift_report,
    embedding_columns,
    exceeds_threshold,
    extract_drift_score,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FARSLIP_PARQUET = _REPO_ROOT / "data" / "farslip" / "embeddings_pastis.parquet"

#: Filas por lado: lo suficiente para que KS/Chi2 sean estables, pero pequeno
#: para que cada test corra en pocos segundos.
_N = 400


@pytest.fixture(scope="module")
def real_corpus() -> pl.DataFrame:
    """Carga el parquet real FarSLIP o skipea si no esta disponible (DVC)."""
    if not _FARSLIP_PARQUET.exists():
        pytest.skip(f"Parquet real ausente (DVC sin pull): {_FARSLIP_PARQUET}")
    df = pl.read_parquet(_FARSLIP_PARQUET)
    if df.height < 2 * _N or "class_id" not in df.columns:
        pytest.skip("Parquet real no cumple el contrato minimo (filas/class_id).")
    return df


def _bands(corpus: pl.DataFrame) -> list[str]:
    """Primeras 3 dimensiones de embedding usadas como bandas numericas KS."""
    return embedding_columns(list(corpus.columns))[:3]


def _embeddings(corpus: pl.DataFrame) -> list[str]:
    """Bloque de 16 dimensiones de embedding para el test MMD."""
    return embedding_columns(list(corpus.columns))[3:19]


def test_embedding_dim_count(real_corpus: pl.DataFrame) -> None:
    """El parquet real expone >= 64 dimensiones de embedding (AlphaEarth 64-dim)."""
    cols = embedding_columns(list(real_corpus.columns))
    assert len(cols) >= 64, f"esperadas >=64 dims de embedding, hay {len(cols)}"


def test_no_drift_same_distribution(real_corpus: pl.DataFrame) -> None:
    """reference == current -> drift_score 0.0 y 0 columnas con drift."""
    reference = real_corpus.head(_N)
    current = reference.clone()
    _, summary = build_drift_report(
        reference,
        current,
        band_columns=_bands(real_corpus),
        embedding_cols=_embeddings(real_corpus),
        class_column="class_id",
    )
    assert isinstance(summary, DriftSummary)
    assert summary.drift_score == 0.0
    assert summary.n_columns_drifted == 0
    assert not exceeds_threshold(summary.drift_score)
    # Embeddings identicos -> MMD por debajo del umbral (no drift).
    assert summary.embedding_drift is False


def test_drift_detected_shifted_bands(real_corpus: pl.DataFrame) -> None:
    """Bandas desplazadas en el current -> KS marca drift y score supera 0.3."""
    bands = _bands(real_corpus)
    reference = real_corpus.head(_N)
    # Desplazo explicito de las bandas reales (suma de +5 sigmas aprox).
    shift_exprs = [(pl.col(b) + 10.0).alias(b) for b in bands]
    current = real_corpus.head(_N).with_columns(shift_exprs)
    _, summary = build_drift_report(
        reference,
        current,
        band_columns=bands,
        class_column=None,
    )
    assert summary.n_columns_drifted >= len(bands)
    assert summary.drift_score > DRIFT_SCORE_THRESHOLD
    assert exceeds_threshold(summary.drift_score)
    # Todas las bandas desplazadas con p-value muy bajo.
    for b in bands:
        assert summary.column_pvalues[b] is not None
        assert summary.column_pvalues[b] < 0.05


def test_class_chi2_drift(real_corpus: pl.DataFrame) -> None:
    """Dos distribuciones reales de clases distintas -> Chi-cuadrado detecta drift."""
    # reference: clase mayoritaria excluida; current: solo esa clase mayoritaria.
    majority = (
        real_corpus.group_by("class_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .row(0)[0]
    )
    reference = real_corpus.filter(pl.col("class_id") != majority).head(_N)
    current = real_corpus.filter(pl.col("class_id") == majority).head(_N)
    _, summary = build_drift_report(
        reference,
        current,
        band_columns=[],
        embedding_cols=[],
        class_column="class_id",
    )
    assert "class_id" in summary.column_drift
    assert summary.column_drift["class_id"] is True
    assert summary.column_pvalues["class_id"] is not None
    assert summary.column_pvalues["class_id"] < 0.05


def test_threshold_trigger() -> None:
    """``exceeds_threshold`` es estricto en 0.3 (alerta solo si > 0.3)."""
    assert DRIFT_SCORE_THRESHOLD == 0.3
    assert exceeds_threshold(0.31) is True
    assert exceeds_threshold(0.30) is False
    assert exceeds_threshold(0.0) is False
    # extract_drift_score devuelve el escalar del summary.
    summary = DriftSummary(drift_score=0.5, n_columns=4, n_columns_drifted=2)
    assert extract_drift_score(summary) == 0.5
    assert exceeds_threshold(extract_drift_score(summary)) is True


def test_html_report_generated(real_corpus: pl.DataFrame) -> None:
    """``build_drift_report`` retorna HTML no vacio con marcado <html."""
    reference = real_corpus.head(_N)
    current = real_corpus.tail(_N)
    html, _ = build_drift_report(
        reference,
        current,
        band_columns=_bands(real_corpus),
        embedding_cols=_embeddings(real_corpus),
        class_column="class_id",
    )
    assert isinstance(html, str)
    assert len(html) > 0
    assert "<html" in html.lower()


def test_requires_some_monitored_column(real_corpus: pl.DataFrame) -> None:
    """Sin bandas, sin embeddings y sin clase -> ValueError explicito."""
    reference = real_corpus.head(10)
    with pytest.raises(ValueError, match="No monitorable columns"):
        build_drift_report(
            reference,
            reference.clone(),
            band_columns=[],
            embedding_cols=[],
            class_column=None,
        )
