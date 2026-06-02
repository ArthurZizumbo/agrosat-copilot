"""Contenido estructurado del Avance 4 (segmentacion semantica densa).

Provee la metadata editorial (KPIs, conclusiones e indice de arquitecturas)
del Avance 4, donde se entrenan y comparan seis arquitecturas de segmentacion
sobre PASTIS-R: U-Net, DeepLabv3+, SegFormer-B2, U-TAE, TSViT (Paper 1) y
Swin-UNETR/AnySAT. Las cifras provienen de los parquet reales de
``reports/segmentation/metrics/model_comparison_avance4_*.parquet``; no se
inventan: el mejor modelo (TSViT-pheno) alcanza mIoU 0.625 y F1-macro 0.75.

El texto vive aqui en lugar de embebido en el dashboard para que se renderice
identico en los distintos canales y sea testeable sin levantar Streamlit.
Reutiliza ``KPI`` y ``NotebookCard`` (mismo patron que el Avance 1 y 2).
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.report.notebook_content import KPI, NotebookCard


@dataclass(frozen=True)
class SegmentationModel:
    """Resumen de una arquitectura de segmentacion del Avance 4.

    Attributes:
        slug: Identificador del archivo de figura (e.g. ``"tsvit"``).
        name: Nombre legible de la arquitectura.
        miou: mIoU flat (18 clases) sobre el conjunto de validacion.
        f1_macro: F1-macro flat sobre el conjunto de validacion.
        pixel_accuracy: Exactitud por pixel (o ``None`` si no se reporto).
        note: Nota corta sobre la corrida (rol, particularidad o limitacion).
    """

    slug: str
    name: str
    miou: float
    f1_macro: float
    pixel_accuracy: float | None
    note: str


# Metricas reales extraidas de los parquet de
# reports/segmentation/metrics/model_comparison_avance4_*.parquet.
SEGMENTATION_MODELS: tuple[SegmentationModel, ...] = (
    SegmentationModel(
        "tsvit-pheno",
        "TSViT-pheno (Paper 1)",
        0.6253,
        0.7500,
        0.8759,
        "Ganador del Avance 4: encoder temporal con reencuadre fenologico.",
    ),
    SegmentationModel(
        "tsvit",
        "TSViT",
        0.6215,
        0.7473,
        0.8724,
        "Encoder temporal puro; base del ganador.",
    ),
    SegmentationModel(
        "utae",
        "U-TAE",
        0.4742,
        0.6087,
        None,
        "Atencion temporal U-Net; segundo en mIoU.",
    ),
    SegmentationModel(
        "anysat",
        "AnySAT / Swin-UNETR",
        0.4459,
        0.5716,
        0.7501,
        "Backbone multimodal ligero (30k parametros entrenables).",
    ),
    SegmentationModel(
        "deeplabv3plus",
        "DeepLabv3+",
        0.2709,
        0.3864,
        0.6743,
        "Baseline denso 2D sin componente temporal.",
    ),
    SegmentationModel(
        "unet",
        "U-Net",
        0.2423,
        0.3463,
        0.6918,
        "Baseline denso 2D de referencia.",
    ),
    SegmentationModel(
        "segformer",
        "SegFormer-B2",
        0.2325,
        0.3423,
        None,
        "Transformer jerarquico; corrida sobre 3 bandas RGB.",
    ),
)

# Sufijos de figura disponibles por arquitectura en
# reports/segmentation/figures/.
SEGMENTATION_FIGURE_KINDS: tuple[tuple[str, str], ...] = (
    ("samples", "Predicciones sobre patches de validacion"),
    ("confusion", "Matriz de confusion entre clases de cultivo"),
    ("per_class_iou", "IoU por clase (deteccion del desbalance)"),
    ("curves", "Curvas de entrenamiento y validacion"),
)


SEGMENTATION_CARD = NotebookCard(
    notebook_id="segmentation-avance4",
    notebook_path="notebooks/segmentation/Avance4.Equipo17.ipynb",
    title="Segmentacion semantica densa - Avance 4",
    subtitle=(
        "Entrenamiento y comparativa de seis arquitecturas de segmentacion "
        "sobre los patches PASTIS-R de 128x128 pixeles con etiquetas reales de "
        "agricultores franceses. El objetivo es pasar de la clasificacion por "
        "parcela (Avances 3) a la prediccion densa pixel a pixel, evaluando "
        "que familia de modelos captura mejor la firma espectro-temporal de "
        "cada cultivo. Las arquitecturas cubren tres familias: convolucional "
        "densa 2D (U-Net, DeepLabv3+), transformer jerarquico (SegFormer-B2) y "
        "encoder temporal con atencion (U-TAE, TSViT, AnySAT/Swin-UNETR). El "
        "modelo ganador (TSViT con reencuadre fenologico) se ajusto con Optuna."
    ),
    sections=(
        "1. Carga de patches PASTIS-R y split espacial oficial",
        "2. U-Net y DeepLabv3+ (baselines densos 2D)",
        "3. SegFormer-B2 (transformer jerarquico)",
        "4. U-TAE (atencion temporal U-Net)",
        "5. TSViT y TSViT-pheno (encoder temporal, Paper 1)",
        "6. AnySAT / Swin-UNETR (backbone multimodal)",
        "7. Ajuste fino con Optuna sobre los top-2",
        "8. Comparativa final y conclusiones",
    ),
    figures_dir="",
    kpis=(
        KPI("Arquitecturas", "6", "U-Net, DeepLabv3+, SegFormer, U-TAE, TSViT, AnySAT"),
        KPI("Mejor mIoU", "0,625", "TSViT-pheno (18 clases flat)"),
        KPI("Mejor F1-macro", "0,750", "TSViT-pheno"),
        KPI("Mejor pixel-acc", "0,876", "TSViT-pheno"),
    ),
    conclusions=(
        (
            "Los encoders temporales dominan a los baselines densos 2D",
            "La separacion es clara: TSViT-pheno (mIoU 0,625) y TSViT (0,622) "
            "superan a U-TAE (0,474) y muy por encima de los baselines densos "
            "U-Net (0,242) y DeepLabv3+ (0,271). La leccion es consistente con "
            "todo el proyecto: la senal que separa cultivos vive en la "
            "evolucion temporal del pixel, no en una sola imagen. Los modelos "
            "que consumen la serie completa de Sentinel-2 aprovechan esa senal; "
            "los densos 2D, que ven un compuesto estatico, no.",
        ),
        (
            "El reencuadre fenologico aporta un margen pequeno pero consistente",
            "TSViT-pheno mejora a TSViT plano en mIoU (0,6253 vs 0,6215), "
            "F1-macro (0,750 vs 0,747) y pixel-accuracy (0,876 vs 0,872). El "
            "delta es modesto pero estable en las tres metricas, coherente con "
            "el hallazgo del Avance 3: codificar el calendario fenologico ayuda "
            "al modelo a desambiguar cultivos con ciclos parecidos.",
        ),
        (
            "El mejor mIoU flat queda bajo el target de rubrica, pero la lectura es matizada",
            "El objetivo de rubrica es mIoU >= 0,70 sobre segmentacion densa y "
            "el mejor flat alcanza 0,625. No se maquilla: hay brecha. Tres "
            "matices la contextualizan. Primero, la pixel-accuracy del ganador "
            "es 0,876, alta para 18 clases con desbalance ~31x. Segundo, al "
            "agrupar clases en familias agronomicas el mIoU agrupado sube. "
            "Tercero, las clases minoritarias (legumbres, vinedos) son las que "
            "arrastran el mIoU macro hacia abajo, como muestran las figuras de "
            "IoU por clase. El cierre de la brecha pasa por loss ponderada y "
            "mas epocas en H100, no por cambiar de familia de modelo.",
        ),
        (
            "SegFormer parte en desventaja por usar solo 3 bandas RGB",
            "La corrida de SegFormer-B2 (mIoU 0,232) se ejecuto sobre 3 bandas "
            "RGB en lugar de las 10 bandas Sentinel-2, por lo que su resultado "
            "no es comparable de igual a igual con el resto. Se mantiene en la "
            "comparativa por transparencia, pero su bajo desempeno refleja la "
            "perdida de informacion espectral, no una limitacion de la "
            "arquitectura transformer en si.",
        ),
        (
            "Optuna confirma el ganador sin sobreajustar el presupuesto",
            "El ajuste fino con Optuna sobre los dos mejores candidatos "
            "(TSViT-pheno y U-TAE) convergio en pocas iteraciones, como muestra "
            "la curva de convergencia. El estudio warm-start partio del mejor "
            "checkpoint para no gastar ventana H100 en reexplorar el espacio. "
            "El resultado confirma a TSViT-pheno como referencia para el "
            "ensamble del EPIC 6 (stacking U-TAE + TSViT + Swin-UNETR).",
        ),
        (
            "Lo que sigue",
            "Con la comparativa cerrada, el siguiente paso es el ensamble del "
            "EPIC 6 (voting / bagging / stacking / blending) usando TSViT-pheno "
            "y U-TAE como base learners, y la corrida full en H100 con loss "
            "ponderada para atacar la brecha de mIoU en clases minoritarias. "
            "El ganador alimenta ademas la capa conversacional (Gemma 4) como "
            "herramienta de segmentacion del agente.",
        ),
    ),
)

A4_CARDS: tuple[NotebookCard, ...] = (SEGMENTATION_CARD,)
