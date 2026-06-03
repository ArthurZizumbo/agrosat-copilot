"""Resolucion de figuras de segmentacion del Avance 4.

Centraliza la logica de localizar la figura de un modelo por tipo
(``curves``, ``per_class_iou``, ``confusion``, ``samples``), aceptando el
nombre exacto, variantes con sufijo (``anysat`` -> ``anysat_fast``) y un mapa
de fallback para las figuras de DeepLab/TSViT publicadas en ``paper/figures/
us-025/`` con nombres propios. Mantener esto fuera del notebook evita repetir
rutas y el mapa hardcodeado en cada celda de galeria.
"""

from __future__ import annotations

from pathlib import Path

# Figure types per model and their readable label (presentation order).
FIGURE_TYPES: tuple[tuple[str, str], ...] = (
    ("curves", "Curvas de entrenamiento"),
    ("per_class_iou", "IoU por clase"),
    ("confusion", "Matriz de confusion"),
    ("samples", "RGB / verdad / prediccion"),
)

# Fallback us-025: DeepLab/TSViT figures with their real names, which
# complement (not replace) the figures from the team's shared Drive.
_US025_DIR = Path("paper/figures/us-025")
_US025_MAP: dict[tuple[str, str], str] = {
    ("confusion", "deeplabv3plus"): "deeplab_confusion_semantic18.png",
    ("samples", "deeplabv3plus"): "deeplab_semantic18_pred_example_0.png",
    ("confusion", "tsvit"): "tsvit_confusion_tsvit-pheno.png",
    ("samples", "tsvit"): "tsvit_pred_example_0.png",
}


def find_figure(figures_dir: Path, key: str, model: str) -> Path | None:
    """Localiza la figura ``key`` del ``model`` o ``None`` si no existe.

    Args:
        figures_dir: Directorio principal de figuras de segmentacion.
        key: Tipo de figura (``"confusion"``, ``"samples"``, ...).
        model: Slug del modelo (``"unet"``, ``"tsvit"``, ...).

    Returns:
        Path a la figura encontrada (nombre exacto, variante con sufijo o
        fallback us-025), o ``None`` si ninguna existe.
    """
    exact = figures_dir / f"{key}_{model}.png"
    if exact.exists():
        return exact
    variants = sorted(figures_dir.glob(f"{key}_{model}_*.png"))
    if variants:
        return variants[0]
    fallback_name = _US025_MAP.get((key, model))
    if fallback_name:
        fallback = _US025_DIR / fallback_name
        if fallback.exists():
            return fallback
    return None
