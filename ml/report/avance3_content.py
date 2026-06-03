"""Contenido estructurado del Avance 3 (baseline tabular y fenologico).

Modela los cinco bloques del baseline saneado post-Avance 3 (US-023-preview)
como datos en lugar de codigo: cada tab es un ``BaselineTab`` con su ficha
editorial (``NotebookCard``) y la lista de artefactos (figuras y tablas) que
debe renderizar. El dashboard recorre esta estructura con un unico renderer
generico, de modo que agregar o cambiar un tab no implica escribir render.

Las decisiones H-1..H-4 provienen de
``reports/baseline/Avance3/decision_table.parquet``; las metricas del
reencuadre fenologico (XGBoost F1-macro 0.4094, hipotesis C-2 confirmada con
delta 0.0) provienen del cierre de US-022-b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ml.report.notebook_content import KPI, NotebookCard

# Canonical hint when a Baseline artifact does not yet exist on disk.
BASELINE_MISSING_HINT = (
    "Artefacto pendiente - ejecuta `make reencuadre-notebook-full && make baseline-v2-full`"
)


@dataclass(frozen=True)
class BaselineArtifact:
    """Artefacto (figura o tabla) que un tab del Avance 3 debe renderizar.

    Attributes:
        kind: ``"figure"`` para PNG o ``"table"`` para parquet.
        relpath: Ruta relativa a la raiz del repositorio.
        caption: Texto descriptivo bajo el artefacto.
        fallbacks: Rutas alternativas en orden de preferencia. La primera que
            exista en disco se usa; replica los ``if exists()`` historicos.
    """

    kind: Literal["figure", "table"]
    relpath: str
    caption: str
    fallbacks: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineTab:
    """Tab del Avance 3: etiqueta + ficha editorial + artefactos.

    Attributes:
        label: Etiqueta visible del tab (en espanol).
        card: Ficha editorial con titulo, prosa, KPIs y conclusiones.
        artifacts: Artefactos a renderizar en orden.
    """

    label: str
    card: NotebookCard
    artifacts: tuple[BaselineArtifact, ...] = field(default_factory=tuple)


_ABLATION_CARD = NotebookCard(
    notebook_id="a3-ablation",
    notebook_path="notebooks/baseline/05_reencuadre_fenologico.ipynb",
    title="Ablation de conjuntos de features",
    subtitle=(
        "Comparativa cuantitativa de los conjuntos de features evaluados sobre "
        "spatial CV 5-fold (mismo splitter de US-022b). Cada fila reporta "
        "n_features, F1-macro, F1-weighted, mIoU y delta vs el conjunto full. "
        "Los conjuntos opcionales solo aparecen cuando el bloque correspondiente "
        "esta materializado en disco. Un delta positivo indica que el bloque "
        "aporta senal; cercano a cero sugiere redundancia con AlphaEarth; "
        "negativo es bandera roja de ruido o leakage."
    ),
    sections=(),
    figures_dir="",
    kpis=(
        KPI("Conjuntos evaluados", "5+2", "full, no_geom, alphaearth_only, ..."),
        KPI("Mejor F1-macro", "0,409", "XGBoost full / no_geom"),
        KPI("Splitter", "Spatial CV", "5-fold, buffer 1 km"),
    ),
    conclusions=(
        (
            "Descartar geom_* no degrada el modelo",
            "El conjunto no_geom rinde identico al full (delta = 0.0): el "
            "modelo aprende fenologia, no geografia. Lo mismo al quitar ERA5 + "
            "SRTM, confirmando que AlphaEarth ya codifica clima y topografia.",
        ),
    ),
)

_LEAKAGE_CARD = NotebookCard(
    notebook_id="a3-leakage",
    notebook_path="notebooks/baseline/05_reencuadre_fenologico.ipynb",
    title="Leakage geografico - columnas geom_*",
    subtitle=(
        "Las 3 columnas geom_area, geom_perimeter y geom_elongation actuan en "
        "la practica como proxy de la region: parcelas de la misma zona "
        "comparten distribucion de tamano y forma. Un modelo entrenado solo "
        "sobre esas 3 features alcanza F1-macro < 0.10 (sin senal de clase real) "
        "pero al combinarse con el bloque base activa un atajo regional que "
        "infla la metrica en entrenamiento y falla al transferir. Decision: "
        "excluir geom_* del baseline definitivo, mantener solo como metadato."
    ),
    sections=(),
    figures_dir="",
    kpis=(
        KPI("geom_only F1-macro", "< 0,10", "Sin senal de clase real"),
        KPI("Delta no_geom vs full", "0,0", "Hipotesis C-2 confirmada"),
        KPI("Decision", "Excluir", "Metadato de auditoria"),
    ),
    conclusions=(),
)

_OPTIONAL_CARD = NotebookCard(
    notebook_id="a3-optional",
    notebook_path="notebooks/baseline/05_reencuadre_fenologico.ipynb",
    title="Bloques opcionales evaluados",
    subtitle=(
        "Tres bloques de features opcionales se evaluaron por separado: "
        "embeddings FarSLIP (US-022-c), descripcion fenologica textual via "
        "Gemini Flash 3.5 y descriptor de firma espectral Red Edge Position "
        "(Frampton et al. 2013). Cada bloque tiene su plot, tabla y decision "
        "documentada. La regla general: promover al baseline si el delta supera "
        "el umbral; dejar como base learner del stacking EPIC 6 si es marginal; "
        "descartar con justificacion si es negativo."
    ),
    sections=(),
    figures_dir="",
    kpis=(
        KPI("Bloques evaluados", "3", "FarSLIP, pheno_text, firma espectral"),
        KPI("pheno_text", "Descartado", "delta -0,145"),
        KPI("FarSLIP", "Pendiente", "Sin matching full"),
    ),
    conclusions=(),
)

_MODELS_CARD = NotebookCard(
    notebook_id="a3-models-v2",
    notebook_path="notebooks/baseline/05_reencuadre_fenologico.ipynb",
    title="Modelos baseline v2 reentrenados",
    subtitle=(
        "Los 3 modelos canonicos del Avance 3 (XGBoost tabular, TempCNN e "
        "InceptionTime temporales) se reentrenan sobre el conjunto de features "
        "ganador post-ablation. El splitter es spatial CV 5-fold con buffer de "
        "1 km, mismo de US-022b para garantizar comparabilidad v1 vs v2. El "
        "baseline v1 reporto F1-macro 0.4094 (XGBoost), 0.143-0.146 (TempCNN) y "
        "0.187 (InceptionTime). La v2 se decide por F1-macro; los empates se "
        "rompen por F1-weighted y luego por mIoU."
    ),
    sections=(),
    figures_dir="",
    kpis=(
        KPI("XGBoost v2", "0,409", "Ganador tabular"),
        KPI("Splitter", "Spatial CV", "5-fold, buffer 1 km"),
        KPI("Base learners EPIC 6", "2", "TempCNN + InceptionTime"),
    ),
    conclusions=(),
)

_CONCLUSIONS_CARD = NotebookCard(
    notebook_id="a3-conclusions",
    notebook_path="(sintesis del Avance 3)",
    title="Conclusiones y siguientes pasos",
    subtitle=(
        "Resumen ejecutivo de los hallazgos del baseline saneado post-A3 y la "
        "transicion a EPIC 5 (modelado denso con TSViT / U-TAE / DeepLabv3+ / "
        "SegFormer-B2 / Swin-UNETR). Las cuatro hipotesis H-1..H-4 resumen las "
        "decisiones tomadas sobre cada bloque de features."
    ),
    sections=(),
    figures_dir="",
    kpis=(
        KPI("Hipotesis evaluadas", "4", "H-1 a H-4"),
        KPI("Conjunto ganador", "no_geom", "AlphaEarth + fenologia"),
        KPI("Techo a batir A4", "0,409", "F1-macro tabular"),
    ),
    conclusions=(
        (
            "H-1 - geom_* introduce leakage regional",
            "Las 3 columnas geometricas no aportan senal de clase real "
            "(F1-macro < 0.10) pero activan un atajo espacial. Decision: "
            "excluir del baseline, mantener como metadato de auditoria.",
        ),
        (
            "H-2 - FarSLIP aporta solo con matching de parcelas",
            "Los embeddings FarSLIP cubren 30173 de 85951 parcelas. El delta "
            "es interpretable solo sobre el subset matched; la extraccion full "
            "queda pendiente. Decision: pendiente (sin datos full).",
        ),
        (
            "H-3 - pheno_text via Gemini: descartado",
            "La descripcion fenologica textual con LLM no aporto senal "
            "(delta = -0.145, por debajo del umbral). Decision: descartar del "
            "baseline; se reconsidera como deuda de investigacion.",
        ),
        (
            "H-4 - firma espectral REP: descartado",
            "El descriptor Red Edge Position (Frampton 2013) no aporto senal "
            "sobre el conjunto base (delta = -0.148). Decision: descartar; "
            "queda documentado para ciclos posteriores.",
        ),
        (
            "Lo que sigue en EPIC 5 (Avance 4)",
            "Con el baseline saneado y el conjunto de features decidido, el "
            "Avance 4 arranca con punto de partida limpio. El XGBoost ganador "
            "(F1-macro 0.409) sirve de techo a batir para la segmentacion "
            "densa; los modelos temporales alimentan el stacking del EPIC 6.",
        ),
    ),
)


A3_TABS: tuple[BaselineTab, ...] = (
    BaselineTab(
        label="Ablation de features",
        card=_ABLATION_CARD,
        artifacts=(
            BaselineArtifact(
                "figure",
                "paper/figures/us-023-preview/ablation_optional_blocks.png",
                "Comparativa visual de los conjuntos opcionales contra el conjunto full.",
            ),
            BaselineArtifact(
                "table",
                "reports/baseline/feature_ablation/ablation_table.parquet",
                "Tabla ablation_table.parquet - conjuntos evaluados con metricas y delta vs full.",
                fallbacks=("reports/baseline/reencuadre_fenologico/ablation_table.parquet",),
            ),
        ),
    ),
    BaselineTab(
        label="Leakage geográfico",
        card=_LEAKAGE_CARD,
        artifacts=(
            BaselineArtifact(
                "figure",
                "paper/figures/us-023-preview/ablation_geom_comparison.png",
                "Dos barras full vs no_geom con anotacion del delta de F1-macro.",
                fallbacks=("paper/figures/reencuadre_fenologico/ablation_geom_comparison.png",),
            ),
            BaselineArtifact(
                "table",
                "reports/baseline/feature_ablation/ablation_geom_table.parquet",
                "Fila geom_only vs full - test cuantitativo de leakage espacial.",
            ),
        ),
    ),
    BaselineTab(
        label="Bloques opcionales",
        card=_OPTIONAL_CARD,
        artifacts=(
            BaselineArtifact(
                "figure",
                "paper/figures/us-023-preview/ablation_optional_blocks.png",
                "FarSLIP, pheno_text y firma espectral contra el conjunto full.",
            ),
            BaselineArtifact(
                "table",
                "reports/baseline/feature_ablation/ablation_table_pheno_text_v2.parquet",
                "Metricas pheno_text (Gemini Flash 3.5, 1080 parcelas balanceadas).",
                fallbacks=(
                    "reports/baseline/feature_ablation/ablation_table_pheno_text.parquet",
                    "reports/baseline/feature_ablation/ablation_pheno_text_table.parquet",
                ),
            ),
        ),
    ),
    BaselineTab(
        label="Modelos baseline v2",
        card=_MODELS_CARD,
        artifacts=(
            BaselineArtifact(
                "figure",
                "paper/figures/us-023-preview/model_comparison_v2.png",
                "XGBoost, TempCNN e InceptionTime con overlay de deltas vs baseline v1.",
            ),
            BaselineArtifact(
                "table",
                "reports/baseline/model_comparison_v2/model_comparison_v2.parquet",
                "3 modelos x 6 metricas: F1-macro, F1-weighted, mIoU, accuracy, kappa, train_time.",
            ),
        ),
    ),
    BaselineTab(
        label="Conclusiones",
        card=_CONCLUSIONS_CARD,
        artifacts=(
            BaselineArtifact(
                "table",
                "reports/baseline/Avance3/decision_table.parquet",
                "Tabla de decisiones H-1..H-4 con descripcion y veredicto por hipotesis.",
            ),
        ),
    ),
)
