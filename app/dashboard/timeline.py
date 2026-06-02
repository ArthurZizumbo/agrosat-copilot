"""Landing narrativo: Historia del proyecto (A0 -> A4).

Cuenta la evolucion del proyecto como una linea de tiempo de seis hitos, cada
uno con sus metricas clave que mejoran de fase en fase (F1 baseline 0.32 ->
0.41 fenologico -> mIoU segmentacion 0.625). Es la seccion por defecto del
dashboard: orienta al lector antes de entrar al detalle por Avance.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Milestone:
    """Hito de la linea de tiempo del proyecto.

    Attributes:
        phase: Etiqueta corta de la fase (e.g. ``"A1"``).
        label: Etiqueta del tab.
        title: Titulo del hito.
        body: Parrafo narrativo del hito.
        kpis: Pares (label, value) con las metricas clave del hito.
    """

    phase: str
    label: str
    title: str
    body: str
    kpis: tuple[tuple[str, str], ...]


MILESTONES: tuple[Milestone, ...] = (
    Milestone(
        "A0",
        "A0 · Setup",
        "Punto de partida: datos y plataforma",
        "El proyecto arranca eligiendo PASTIS-R (Francia, etiquetas reales de "
        "agricultores) como dataset ancla y AlphaEarth Foundations como modelo "
        "base de features. Se monta la plataforma reproducible (Polars, DVC, "
        "MLflow, Dagster) y se fija la hipotesis central: la senal que separa "
        "cultivos vive en la evolucion temporal del pixel, no en una sola imagen.",
        (("Dataset ancla", "PASTIS-R"), ("Clases", "18 cultivos"), ("FM EO", "AlphaEarth v2.1")),
    ),
    Milestone(
        "A1",
        "A1 · EDA",
        "Analisis Exploratorio de Datos",
        "Seis notebooks de EDA caracterizan la calidad del dato, el desbalance "
        "de clases (~31x), la nubosidad por region y la separabilidad de los "
        "embeddings AlphaEarth. Un Random Forest crudo sobre los 64 embeddings "
        "ya alcanza OOB 0,83-0,89 sin feature engineering, dejando el piso del "
        "baseline muy por encima del minimo de rubrica.",
        (("Notebooks EDA", "6"), ("OOB AlphaEarth", "0,83-0,89"), ("Desbalance", "~31x")),
    ),
    Milestone(
        "A2",
        "A2 · FE",
        "Ingenieria de Caracteristicas",
        "Tres notebooks construyen y seleccionan features sobre Sentinel-2, "
        "PASTIS-R y AlphaEarth. Los filtros reducen la matriz hasta un 55,7% "
        "sin perder senal, y la comparativa confirma que AlphaEarth (F1 0,52) "
        "y las features espectro-temporales manuales (F1 0,54) rinden casi "
        "igual: conviene fusionarlas en lugar de elegir una.",
        (("Notebooks FE", "3"), ("Reduccion features", "hasta 55,7%"), ("Fusion F1", "0,54")),
    ),
    Milestone(
        "A3",
        "A3 · Baseline",
        "Baseline tabular y reencuadre fenologico",
        "El baseline closed-set tabular tiene techo bajo (F1 0,32) sobre 18 "
        "clases; ese resultado motiva el reencuadre fenologico. El XGBoost "
        "sobre el conjunto ganador sube a F1-macro 0,41 (+0,09). Se confirma "
        "que descartar las columnas geometricas no degrada el modelo (aprende "
        "fenologia, no geografia) y se descartan pheno_text y firma espectral.",
        (("Baseline v1", "F1 0,32"), ("Reencuadre", "F1 0,41"), ("Hipotesis", "H-1..H-4")),
    ),
    Milestone(
        "A4",
        "A4 · Segmentacion",
        "Segmentacion semantica densa",
        "Seis arquitecturas compiten en prediccion densa pixel a pixel. Los "
        "encoders temporales dominan: TSViT-pheno gana con mIoU 0,625 y "
        "F1-macro 0,75, muy por encima de los baselines densos U-Net (0,24) y "
        "DeepLabv3+ (0,27). El mejor mIoU flat queda bajo el target 0,70, pero "
        "la pixel-accuracy llega a 0,876 y el mIoU agrupado sube.",
        (("Arquitecturas", "6"), ("Mejor mIoU", "0,625"), ("Mejor F1-macro", "0,75")),
    ),
    Milestone(
        "Sig",
        "Lo que sigue",
        "Ensamble, fine-tune y capa conversacional",
        "Con el modelo denso ganador, el proyecto avanza al ensamble del EPIC 6 "
        "(voting / bagging / stacking / blending) usando TSViT-pheno y U-TAE "
        "como base learners, la corrida full en H100 con loss ponderada para "
        "cerrar la brecha de mIoU, y la integracion en el agente conversacional "
        "(Gemma 4 + Qwen3.5) que cierra el producto AgroSatCopilot.",
        (("Ensambles", "4"), ("VLM principal", "Gemma 4"), ("Target final", "F1 >= 0,80")),
    ),
)

TIMELINE_TAB_LABELS: tuple[str, ...] = tuple(m.label for m in MILESTONES)


def _render_milestone(milestone: Milestone) -> None:
    """Renderiza un hito: tarjeta narrativa + KPIs de la fase."""
    st.markdown(
        f'<div class="timeline-milestone">'
        f'<div class="milestone-phase">{milestone.phase}</div>'
        f'<div class="milestone-title">{milestone.title}</div>'
        f'<p class="milestone-body">{milestone.body}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
    kpi_html = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
        for label, value in milestone.kpis
    )
    st.markdown(f'<div class="kpi-row">{kpi_html}</div>', unsafe_allow_html=True)


def render_timeline_section() -> None:
    """Renderiza la landing narrativa con la linea de tiempo A0 -> A4."""
    st.markdown(
        '<h2 style="margin-top:0.5rem;color:#1E293B;font-weight:700;">'
        "Historia del proyecto - de los datos a la segmentacion</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#475569;font-size:1rem;line-height:1.6;">'
        "Cada pestana es un hito del proyecto con las metricas que evolucionan "
        "de fase en fase. La historia es una sola: la senal agronomica vive en "
        "el tiempo, y cada Avance acerca el modelo a capturarla mejor.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(list(TIMELINE_TAB_LABELS))
    for tab, milestone in zip(tabs, MILESTONES, strict=True):
        with tab:
            _render_milestone(milestone)
