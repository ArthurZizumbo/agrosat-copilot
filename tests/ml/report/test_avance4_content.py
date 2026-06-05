"""Tests para ml.report.avance4_content (segmentacion Avance 4).

Verifica que las fichas del Avance 4 expongan KPIs y conclusiones, que las
metricas de las arquitecturas sean coherentes (mIoU/F1 en [0, 1]) y que el
ganador declarado sea TSViT-pheno.
"""

from __future__ import annotations

from ml.report.avance4_content import (
    A4_CARDS,
    SEGMENTATION_FIGURE_KINDS,
    SEGMENTATION_MODELS,
    SegmentationModel,
)


def test_a4_cards_expose_kpis_and_conclusions() -> None:
    """Cada ficha del Avance 4 tiene KPIs y conclusiones no vacios."""
    assert A4_CARDS, "A4_CARDS no puede estar vacio"
    for card in A4_CARDS:
        assert card.kpis, f"sin KPIs en {card.notebook_id}"
        assert card.conclusions, f"sin conclusiones en {card.notebook_id}"
        for kpi in card.kpis:
            assert kpi.label.strip() and kpi.value.strip() and kpi.delta.strip()


def test_segmentation_models_metrics_in_range() -> None:
    """mIoU, F1-macro y pixel-accuracy estan en el rango [0, 1]."""
    assert len(SEGMENTATION_MODELS) >= 6, "Se esperaban al menos 6 arquitecturas"
    for model in SEGMENTATION_MODELS:
        assert isinstance(model, SegmentationModel)
        assert 0.0 <= model.miou <= 1.0, f"mIoU fuera de rango en {model.slug}"
        assert 0.0 <= model.f1_macro <= 1.0, f"F1-macro fuera de rango en {model.slug}"
        if model.pixel_accuracy is not None:
            assert 0.0 <= model.pixel_accuracy <= 1.0


def test_tsvit_pheno_is_winner() -> None:
    """El modelo con mayor mIoU es TSViT-pheno (ganador declarado)."""
    best = max(SEGMENTATION_MODELS, key=lambda m: m.miou)
    assert best.slug == "tsvit-pheno", f"Ganador inesperado: {best.slug}"
    assert best.miou > 0.6, "El ganador deberia superar mIoU 0,6"


def test_figure_kinds_are_unique() -> None:
    """Los tipos de figura por arquitectura no se repiten."""
    slugs = [kind for kind, _ in SEGMENTATION_FIGURE_KINDS]
    assert len(slugs) == len(set(slugs)), "Tipos de figura duplicados"
