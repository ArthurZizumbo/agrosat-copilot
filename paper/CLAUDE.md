# Paper — AgroSatCopilot

Scope sub-agente `paper/`. Hereda el orquestador root ([../AGENTS.md](../AGENTS.md)) — no se repiten aqui los NON-NEGOTIABLE (idioma, secrets, DVC/MLflow, sin emojis).

## Estado

ESQUELETO. Paper Track es **opcional** y arranca **post-presentacion** (semanas 10-11). Hoy no compromete ningun Avance del curso.

- `sections/` y `bib/` solo tienen `.gitkeep` — **no existen** `main.tex` ni `refs.bib`. No hay target LaTeX ni compilacion.
- Unica tabla escrita a mano: [tables/us-023-preview/baseline_v2_comparison.tex](tables/us-023-preview/baseline_v2_comparison.tex) (3 modelos canonicos US-023-preview).
- Todo lo demas en `figures/` es **generado** (extraido de notebooks o copiado de `reports/`), mas `avance1_eda_report.html`.

## Comandos

```bash
poetry install --with paper            # deps opcionales del paper (grupo poetry "paper")
make avance2-figures                   # extrae figuras inline de los 3 nb FE -> figures/feature-engineering/
make eda-figures-paper-methods         # copia 5 PNG de reports/paper_methods/ -> figures/paper-methods/
make paper-methods-notebook            # regenera + ejecuta notebooks/eda/02e_eda_metodos_paper.ipynb (papermill)
```

## Stack local

- `ml/analysis/paper_methods.py` — 8 funciones que materializan metodos de papers REALES sobre el dato del proyecto: `boundary_pixel_mask`, `boundary_interior_stats`, `compute_boundary_ratio`, `temporal_sampling_stats`, `confusion_symmetry_analysis`, `aggregate_rare_classes`, `phenology_calendar_features`, `cloud_gap_robustness`.
- Methods cita los modelos que **de hecho** se entrenan: TSViT, U-TAE, AnySat, DeepLabv3+, SegFormer. **No** Gemma 4 LoRA aqui.
- Atribucion AlphaEarth: `Satellite Embedding V1 Annual, data version 1.1, CC-BY-4.0` (no "v2.1").

## Convenciones

- ✅ Funciones de `paper_methods.py` retornan estructuras Polars / dicts; logging via `structlog`.
- ✅ Cita por-funcion al paper origen (Russwurm & Korner, Tarasiou et al., Phenology-Aware Transformer, Qin et al. STCLN).
- ✅ Figuras reproducibles solo desde los targets `make` de arriba; nunca editar el PNG a mano.
- ❌ No inventar `main.tex`/`refs.bib` ni un pipeline LaTeX sin acordarlo con el equipo (Paper Track no arrancado).
- ❌ No atribuir AlphaEarth como "v2.1"; no listar Gemma 4 LoRA en Methods.
- ❌ No sacrificar entregables del curso por trabajo del paper.

## No tocar

- Figuras generadas (`figures/**`) y `avance1_eda_report.html` — son artefactos; regenerar via target, no editar.
- El notebook `notebooks/eda/02e_eda_metodos_paper.ipynb` se edita en su **builder** (`scripts/build_paper_methods_notebook.py`), no a mano.
- `ml/analysis/paper_methods.py` — al cambiar firmas, sincronizar `tests/ml/analysis/test_paper_methods.py`.

## Tests

```bash
poetry run pytest tests/ml/analysis/test_paper_methods.py            # unit (datos sinteticos)
poetry run pytest tests/ml/analysis/test_paper_methods.py -m empirical   # valida sobre dato real (skip si falta)
```

## Skills

- [agrosat-ml-evaluation](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — benchmarks y figuras interpretadas.
- [agrosat-dvc-mlflow](../.claude/skills/agrosat-dvc-mlflow/SKILL.md) — reproducibilidad de datos/runs.
