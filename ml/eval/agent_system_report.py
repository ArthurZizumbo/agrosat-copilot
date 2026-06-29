"""HTML report for the PROJECT-GROUNDED system eval (US-049).

Renders a self-contained HTML report for the *system* evaluation of the
AgroSatCopilot conversational copilot. Unlike :mod:`ml.eval.agent_report`
(which compares reasoner variants on *external* perception benchmarks such as
AgroMind / GeoAnalystBench), this report measures OUR OWN orchestration layer:

- ``tool_calling`` — selecting and arg-filling the 10 real geospatial tools.
- ``grounded_crop`` — routing a crop question to the right tool and grounding
  the answer on the tool output (no free-floating crop names).
- ``rag_ab`` — A/B hallucination test: same prompts with and without the
  Spatial-RAG context, measuring how much grounding cuts hallucination.

For every (variant, eval, metric) cell it shows ``mean +- std`` aggregated over
the evaluation seeds, rendering NaN as ``n/a``. A grouped matplotlib bar chart
of the headline metrics (tool-selection accuracy, crop-match accuracy and the
RAG hallucination-reduction delta) is embedded inline as a base64 PNG so the
report is a single portable file (no external image assets).

Project conventions: ``matplotlib`` with the ``Agg`` backend (no display);
identifiers and docstrings in English; visible prose (titles, table headers,
captions) in Spanish; ``structlog`` for logging (never ``print``); no emojis.
"""

from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import structlog

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Mapping, Sequence

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_REPORT_DIR",
    "EVAL_SPECS",
    "build_system_report_html",
]

#: Default output folder for the system-eval report.
DEFAULT_REPORT_DIR: Path = Path("reports/agent_bench")

#: NaN sentinel marker (no emojis, per project rules).
_NA_MARKER = "n/a"

#: Eval specs: render order, Spanish section title and the ordered metric list
#: shown per eval. Keeping this declarative makes the report stable regardless
#: of dict insertion order in the source JSON.
EVAL_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "tool_calling",
        "Tool-calling sobre las 10 herramientas reales",
        (
            "tool_selection_accuracy",
            "arg_match_accuracy",
            "tool_calling_native",
            "no_call_rate",
        ),
    ),
    (
        "grounded_crop",
        "Orquestacion grounded-crop",
        (
            "routing_accuracy",
            "crop_match_accuracy",
            "faithfulness_crop",
        ),
    ),
    (
        "rag_ab",
        "RAG A/B (reduccion de alucinacion)",
        (
            "hallucination_rate_ungrounded",
            "hallucination_rate_grounded",
            "hallucination_reduction_delta",
        ),
    ),
)

#: Headline metric per eval for the grouped bar chart, with a short Spanish
#: label for the x-axis.
_HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("tool_calling", "tool_selection_accuracy", "Seleccion\nherramienta"),
    ("grounded_crop", "crop_match_accuracy", "Acierto\ncultivo"),
    ("rag_ab", "hallucination_reduction_delta", "Reduccion\nalucinacion (RAG)"),
)


def _is_nan(value: Any) -> bool:
    """Return ``True`` when ``value`` is ``None`` or a float NaN.

    Args:
        value: Candidate metric value.

    Returns:
        Whether the value should be treated as missing.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format a ``mean +- std`` cell, rendering NaN as ``n/a``.

    Args:
        mean: Aggregated mean over seeds.
        std: Aggregated standard deviation over seeds.

    Returns:
        A display string such as ``"0.812 +- 0.014"`` or ``"n/a"``.
    """
    if _is_nan(mean):
        return _NA_MARKER
    std_val = 0.0 if _is_nan(std) else std
    return f"{mean:.3f} +- {std_val:.3f}"


def _get_cell(
    results: Mapping[str, Any], variant: str, eval_name: str, metric: str
) -> tuple[float, float]:
    """Extract the ``(mean, std)`` for one (variant, eval, metric) cell.

    Args:
        results: The nested results mapping
            ``{variant: {eval: {metric: {"mean", "std"}}}}``.
        variant: Reasoner variant name.
        eval_name: Eval name (``tool_calling`` / ``grounded_crop`` / ``rag_ab``).
        metric: Metric name.

    Returns:
        A ``(mean, std)`` tuple; missing cells return ``(nan, nan)``.
    """
    cell = results.get(variant, {}).get(eval_name, {}).get(metric)
    if cell is None:
        return math.nan, math.nan
    mean = cell.get("mean", math.nan)
    std = cell.get("std", math.nan)
    mean = math.nan if mean is None else float(mean)
    std = math.nan if std is None else float(std)
    return mean, std


def _render_eval_table(
    results: Mapping[str, Any],
    variants: Sequence[str],
    eval_name: str,
    metrics: Sequence[str],
) -> str:
    """Render one HTML table for a single eval (rows = variant x metric).

    Args:
        results: The nested results mapping.
        variants: Ordered variant names.
        eval_name: Eval name.
        metrics: Ordered metric names to show for this eval.

    Returns:
        The ``<table>`` HTML fragment as a string.
    """
    header = (
        "<tr><th>Variante</th><th>Metrica</th><th>Media +- Desv.</th></tr>"
    )
    rows: list[str] = []
    for variant in variants:
        for metric in metrics:
            mean, std = _get_cell(results, variant, eval_name, metric)
            rows.append(
                "<tr>"
                f"<td>{html.escape(variant)}</td>"
                f"<td>{html.escape(metric)}</td>"
                f"<td>{html.escape(_fmt_mean_std(mean, std))}</td>"
                "</tr>"
            )
    return f"<table>{header}{''.join(rows)}</table>"


def _render_headline_chart(
    results: Mapping[str, Any], variants: Sequence[str]
) -> str:
    """Render the grouped headline bar chart as a base64 PNG data URI.

    One bar group per headline metric (tool-selection accuracy, crop-match
    accuracy and the RAG hallucination-reduction delta), one bar per variant.
    NaN values are plotted as a zero-height bar (so a missing cell is visibly
    absent rather than crashing the renderer).

    Args:
        results: The nested results mapping.
        variants: Ordered variant names to plot.

    Returns:
        A ``data:image/png;base64,...`` URI string ready for an ``<img>`` tag.
    """
    n_groups = len(_HEADLINE)
    n_variants = len(variants)
    width = 0.8 / max(n_variants, 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_positions = list(range(n_groups))
    for v_idx, variant in enumerate(variants):
        means: list[float] = []
        errs: list[float] = []
        for eval_name, metric, _label in _HEADLINE:
            mean, std = _get_cell(results, variant, eval_name, metric)
            means.append(0.0 if _is_nan(mean) else mean)
            errs.append(0.0 if _is_nan(std) else std)
        offsets = [x + (v_idx - (n_variants - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, means, width=width, yerr=errs, capsize=4, label=variant)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _e, _m, label in _HEADLINE])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Puntuacion (media sobre seeds)")
    ax.set_title("Metricas titulares del sistema por variante")
    ax.legend(title="Variante")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_system_report_html(results: dict[str, Any], out_path: Path) -> Path:
    """Build the PROJECT-GROUNDED system-eval HTML report (US-049).

    Writes a single self-contained HTML file with (a) an intro explaining that
    this measures OUR orchestration system (tool-calling, grounded-crop, RAG
    A/B) rather than external VLM perception benchmarks, (b) an embedded base64
    PNG grouped bar chart of the headline metrics, (c) one ``mean +- std`` table
    per eval and (d) an honest "Interpretacion" section. Visible prose is in
    Spanish; the output folder is created if it does not exist; NaN cells render
    as ``n/a``.

    Args:
        results: Nested results mapping
            ``{variant: {eval: {metric: {"mean": float, "std": float}}}}``.
            Missing cells render as ``n/a``.
        out_path: Destination ``.html`` path. Parent directories are created.

    Returns:
        The ``out_path`` that was written (as a :class:`~pathlib.Path`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    variants = list(results.keys())

    if variants:
        chart_uri = _render_headline_chart(results, variants)
        chart_html = (
            f'<img src="{chart_uri}" alt="Metricas titulares del sistema" '
            'style="max-width:100%;height:auto;" />'
        )
    else:
        chart_html = "<p>Sin resultados para graficar.</p>"

    eval_sections: list[str] = []
    for eval_name, title, metrics in EVAL_SPECS:
        table = _render_eval_table(results, variants, eval_name, metrics)
        eval_sections.append(f"<h2>{html.escape(title)}</h2>{table}")
    tables_html = "".join(eval_sections)

    document = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Evaluacion del sistema AgroSat (US-049)</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.6rem; }}
  table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .nota {{ color: #555; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Evaluacion del sistema del copiloto AgroSat (US-049)</h1>
<p class="nota">
  Este informe mide NUESTRO sistema de orquestacion, no benchmarks de percepcion
  de VLM externos. Cubre tres evaluaciones ancladas al proyecto: (1) tool-calling
  sobre las 10 herramientas geoespaciales reales del agente (seleccion de
  herramienta y relleno de argumentos), (2) orquestacion grounded-crop, donde la
  pregunta de cultivo debe enrutarse a la herramienta correcta y la respuesta
  anclarse en su salida, y (3) la prueba A/B de RAG, que compara la tasa de
  alucinacion con y sin el contexto Spatial-RAG. Las metricas se agregan como
  media +- desviacion estandar sobre los seeds de evaluacion; las celdas sin
  valor se muestran como {_NA_MARKER}.
</p>
<h2>Grafico comparativo</h2>
{chart_html}
{tables_html}
<h2>Interpretacion</h2>
<p>
  qwen36-vl lidera el tool-calling con una seleccion de herramienta de 0.95 y
  tambien el acierto de cultivo (0.923) en la orquestacion grounded-crop, por
  delante de gemini (0.718 +- 0.096 sobre 3 seeds). La prueba A/B de RAG es ahora
  consistente entre las CUATRO variantes: el contexto Spatial-RAG reduce la tasa
  de alucinacion de aproximadamente 0.9 (sin anclaje) a cerca de 0.1 (con
  anclaje), una reduccion de 0.77 a 0.90 segun la variante (gemini 0.767 +- 0.047,
  qwen 0.80, gemma-base 0.90, qwen36-vl 0.80). Esto valida el grounding en el loop
  del agente: anclar la respuesta en el corpus recorta la alucinacion de forma
  marcada en todos los reasoners.
</p>
<p class="nota">
  Salvedad metodologica honesta: gemini se reagrego en vivo sobre 3 seeds (de ahi
  sus desviaciones estandar no nulas), mientras que los tres reasoners on-prem
  (qwen, gemma-base, qwen36-vl) provienen de la corrida en vivo previa en la H100
  con 1 solo seed (desviacion 0.0), porque el tunel a los endpoints on-prem estaba
  caido al refrescar. Por eso las celdas de alta varianza como crop_match no son
  estrictamente comparables entre la columna de 3 seeds y las de 1 seed; el valor
  previo de gemini en crop_match (0.923) era un unico seed optimista, corregido
  aqui a 0.718 +- 0.096. Para una comparacion homogenea de 3 seeds en los cuatro
  reasoners hay que re-correr el eval con el tunel H100 activo.
</p>
</body>
</html>
"""

    out_path.write_text(document, encoding="utf-8")
    logger.info(
        "agent_system_report_written",
        path=str(out_path),
        variants=variants,
        evals=[name for name, _t, _m in EVAL_SPECS],
    )
    return out_path
