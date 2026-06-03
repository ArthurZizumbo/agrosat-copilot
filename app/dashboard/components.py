"""Componentes de render reutilizables del dashboard.

Agrupa los renderers genericos que comparten todas las secciones por Avance:
fila de KPIs, encabezado de ficha, figura con narrativa, divisor de seccion,
tablas y conclusiones. Tambien expone helpers para artefactos opcionales
(figura o tabla parquet) que degradan con ``st.warning`` cuando el archivo no
existe, en lugar de romper el render.

La fuente editorial de cada ficha es ``ml.report.notebook_content.NotebookCard``
y las narrativas por figura viven en ``ml.report.figure_narratives``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.dashboard.loaders import list_csvs, load_csv, load_parquet
from ml.report.figure_narratives import FigureNarrative, get_narrative
from ml.report.notebook_content import NotebookCard, list_figures

# Defensive cap on rows to render in ``st.dataframe`` to avoid saturating the
# client with very large tables.
DATAFRAME_HEAD_ROWS = 200


def render_kpi_row(card: NotebookCard) -> None:
    """Renderiza una fila de KPI cards para la ficha dada."""
    if not card.kpis:
        return
    cards_html = "".join(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{kpi.label}</div>'
        f'<div class="kpi-value">{kpi.value}</div>'
        f'<div class="kpi-delta">{kpi.delta}</div>'
        f"</div>"
        for kpi in card.kpis
    )
    st.markdown(f'<div class="kpi-row">{cards_html}</div>', unsafe_allow_html=True)


def render_section_divider(label: str, badge: str | None = None) -> None:
    """Renderiza un divisor de seccion con label y badge opcional."""
    badge_html = f'<span class="section-divider-badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-divider">{badge_html}<h3>{label}</h3></div>',
        unsafe_allow_html=True,
    )


def render_card_header(card: NotebookCard) -> None:
    """Renderiza titulo, subtitulo, pill del notebook fuente y KPIs."""
    st.markdown(
        f'<h2 style="margin-top:0.5rem;color:#1E293B;font-weight:700;">{card.title}</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#475569;font-size:1rem;line-height:1.6;'
        f'margin-bottom:0.6rem;">{card.subtitle}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="source-pill">Notebook fuente: {card.notebook_path}</div>',
        unsafe_allow_html=True,
    )
    render_kpi_row(card)

    if card.sections:
        with st.expander("Indice del notebook (secciones)", expanded=False):
            for section in card.sections:
                st.markdown(f"- {section}")


def render_figure_with_narrative(png_path: Path, narrative: FigureNarrative | None) -> None:
    """Renderiza una figura PNG con su narrativa interpretativa al lado."""
    title = narrative.title if narrative is not None else png_path.stem.replace("_", " ").title()

    st.markdown(
        f'<div class="figure-card"><div class="figure-title">{title}</div></div>',
        unsafe_allow_html=True,
    )

    col_img, col_text = st.columns([3, 2], gap="medium")
    with col_img:
        st.image(str(png_path), use_container_width=True)
    with col_text:
        if narrative is not None:
            st.markdown(
                f'<div class="narrative-block">{narrative.narrative}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="method-block">'
                f"<strong>Como se construyo:</strong> {narrative.method}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="method-block">'
                f"<strong>Figura:</strong> {png_path.name}. "
                f"Narrativa interpretativa pendiente de redaccion."
                f"</div>",
                unsafe_allow_html=True,
            )


def render_card_figures(card: NotebookCard, figures_root: Path) -> None:
    """Renderiza las figuras de la ficha con narrativa por figura.

    Args:
        card: Ficha del notebook con ``figures_dir`` y ``notebook_id``.
        figures_root: Raiz que contiene los subdirectorios de figuras.
    """
    pngs = list_figures(card, figures_root)
    csvs = list_csvs(figures_root / card.figures_dir) if card.figures_dir else []

    if not pngs and not csvs:
        if card.figures_dir:
            st.info(
                f"Pendiente: no se encontraron figuras en "
                f"`paper/figures/{card.figures_dir}/`. Ejecuta el notebook "
                f"fuente para poblarlas."
            )
        return

    if pngs:
        render_section_divider("Figuras del analisis", badge=f"{len(pngs)} figuras")
        for png in pngs:
            narrative = get_narrative(card.notebook_id, png.name)
            render_figure_with_narrative(png, narrative)

    if csvs:
        render_section_divider("Tablas asociadas", badge=f"{len(csvs)} tablas")
        for csv_path in csvs:
            st.markdown(
                f'<div class="figure-card"><div class="figure-title">{csv_path.name}</div></div>',
                unsafe_allow_html=True,
            )
            df = load_csv(csv_path)
            if df.is_empty():
                st.caption("Tabla vacia o ilegible.")
                continue
            st.dataframe(df.head(DATAFRAME_HEAD_ROWS).to_pandas(), use_container_width=True)


def render_card_conclusions(card: NotebookCard) -> None:
    """Renderiza las conclusiones interpretadas como cards alternadas."""
    if not card.conclusions:
        return
    render_section_divider(
        "Conclusiones e interpretacion",
        badge=f"{len(card.conclusions)} hallazgos",
    )
    for idx, (heading, body) in enumerate(card.conclusions):
        css_class = "conclusion-card"
        if idx % 3 == 1:
            css_class += " accent"
        elif idx % 3 == 2:
            css_class += " success"
        st.markdown(
            f'<div class="{css_class}">'
            f'<div class="conclusion-heading">{heading}</div>'
            f'<p class="conclusion-body">{body}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_card(card: NotebookCard, figures_root: Path) -> None:
    """Renderiza una ficha completa: header + KPIs + figuras + conclusiones.

    Args:
        card: Ficha del notebook a renderizar.
        figures_root: Raiz de figuras (``paper/figures/``).
    """
    render_card_header(card)
    render_card_figures(card, figures_root)
    render_card_conclusions(card)


def render_optional_figure(png_path: Path, caption: str, missing_hint: str) -> None:
    """Renderiza una figura si existe, con ``st.warning`` graceful si falta.

    Args:
        png_path: Ruta absoluta al PNG.
        caption: Texto descriptivo bajo la imagen.
        missing_hint: Sugerencia (comando ``make``) para regenerar el artefacto.
    """
    if not png_path.exists():
        st.warning(f"Figura no disponible (`{png_path.name}`). {missing_hint}")
        return
    st.image(str(png_path), caption=caption, use_container_width=True)


def render_parquet_table(parquet_path: Path, caption: str, missing_hint: str) -> None:
    """Renderiza una tabla parquet con cache lazy y warning graceful.

    Args:
        parquet_path: Ruta absoluta al parquet.
        caption: Texto descriptivo bajo la tabla.
        missing_hint: Sugerencia (comando ``make``) para regenerar el artefacto.
    """
    if not parquet_path.exists():
        st.warning(f"Tabla no disponible (`{parquet_path.name}`). {missing_hint}")
        return
    df = load_parquet(parquet_path)
    if df.is_empty():
        st.caption(f"Tabla vacia o ilegible: `{parquet_path.name}`.")
        return
    st.dataframe(df.head(DATAFRAME_HEAD_ROWS).to_pandas(), use_container_width=True)
    if caption:
        st.caption(caption)
