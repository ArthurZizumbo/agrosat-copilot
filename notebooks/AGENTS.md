# Notebooks — AgroSatCopilot

> Sub-agente de `notebooks/`. Sobreescribe al root solo en su scope. Las reglas NON-NEGOTIABLE (idioma, sin emojis, DVC/MLflow, sin scripts ad-hoc) viven en [`../CLAUDE.md`](../CLAUDE.md) — aquí solo lo operativo del directorio.

## Estado

Cuatro carpetas, una por fase. **Solo los Avances 1-4 son notebooks**; los Avances 0, 5, 6 y 7 son PDFs (no hay notebook para ellos — no crearlos).

- `eda/` — `02a_eda_sentinel2`, `02b_eda_alphaearth`, `02c_eda_bivariado_temporal`, `02c_eda_pastis`, `02d_eda_breizhcrops`, `02e_eda_metodos_paper`, `Avance1.Equipo17` (+ `Avance1.Equipo17.pdf`).
- `feature_engineering/` — `03a_fe_sentinel2`, `03b_fe_spectral_temporal_pastis`, `03c_fe_alphaearth_pastis`, `Avance2.Equipo17`.
- `baseline/` — `04_baseline`, `04b_baseline`, `04c_baseline`, `04_farslip_eval_pastis`, `05_reencuadre_fenologico`, `Avance3.Equipo17`.
- `segmentation/` — `04d_segmentation_unet`, `04e_segmentation_anysat`, `04g_segmentation_unet_fast`, `04h_segmentation_anysat_fast`, `04i_segmentation_segformer_b0`, `04j_segmentation_utae`, `5a_deeplabv3plus`, `5b_tsvit`, `Avance4.Equipo17`.

Los `.ipynb` se commitean **ejecutados con outputs poblados** (tablas HTML, PNG inline) — son entregable visual. Cada carpeta tiene un `html/` con el render exportado. Cachés GEE (`*/data/cache/gee/*.parquet`) NO se committean (se regeneran deterministas).

## Comandos

```bash
make notebooks-check        # papermill smoke 02a+02b+02c en modo degradado (~3 min) — gate de CI
make notebooks-strip        # nbstripout ON-DEMAND solo (NUNCA en quality gates)

# Reconstruir notebooks GENERADOS desde su fuente (ver "No tocar"):
make feature-selection-build   # -> 03b   (scripts/build_us018_notebook.py)
make feature-fusion-build      # -> 03c   (scripts/build_fusion_notebook.py)
make baseline-notebook         # -> 04_baseline (reconstruye + ejecuta)
make reencuadre-notebook       # -> 05_reencuadre_fenologico (reconstruye + ejecuta)
make paper-methods-notebook    # -> 02e_eda_metodos_paper
make eda-notebook-avance1      # -> Avance1.Equipo17 (notebook_content.py + figure_narratives.py)
make avance2-build             # -> Avance2.Equipo17 (figuras embebidas)
poetry run python scripts/build_avance3_notebook.py   # -> Avance3.Equipo17
poetry run python scripts/build_avance4_notebook.py   # -> Avance4.Equipo17

# Figuras del Avance 4 (regenera curves/per_class/confusion):
poetry run python -m ml.eval.avance4_figures

# Papermill de un notebook puntual (regenera sus outputs):
poetry run papermill notebooks/eda/02b_eda_alphaearth.ipynb \
    notebooks/eda/02b_eda_alphaearth.ipynb -p sample_size 6000
```

## Stack local

- **Polars** para DataFrames (`import polars as pl`). pandas solo en `.to_pandas()` puntual para libs que lo exijan.
- Bootstrap de repo via `from ml.utils.notebook_setup import find_repo_root, configure_ee_from_env` — nunca paths absolutos.
- Utilidades ya existentes (verificar antes de duplicar): `ml/ingest/gee_sampler.py` (`sample_s2_roi`, `sample_alphaearth_roi`, `sample_alphaearth_at_coords`), `ml/ingest/pastis_loader.py` (`pastis_patch_index`, `pastis_pixel_labels`, `PASTIS_R_CLASSES`), `ml/analysis/embeddings.py` + `ml/analysis/visualization.py` (t-SNE/UMAP, heatmaps), `ml/utils/sampling.py`.
- Hot-reload: celda de imports lleva `%load_ext autoreload` + `%autoreload 2` para captar cambios en `ml/*.py` sin reiniciar kernel.

## Convenciones (✅ / ❌)

- ✅ Celda 2 con tag `parameters` (papermill) — defaults reducidos para el smoke de CI.
- ✅ `import polars as pl`; preferir `display(df.head())` y `display(Markdown(...))` sobre `print(df)`.
- ✅ `find_repo_root()` para resolver rutas; `plt.close(fig)` tras `display(fig)` (evita doble render).
- ✅ Markdown / prints / títulos de plot en español con acentos y ñ. Identificadores, comentarios técnicos y cache keys en inglés ASCII.
- ✅ Sección final "Conclusiones" en lenguaje accesible: números reales del output + "Lo que sigue". Sin US-XXX/EPIC/AC-X.
- ❌ Lógica >5 líneas inline en celda → extraer a `ml/` (ingest/analysis/features/utils) y llamar.
- ❌ Paths absolutos, `print()` para DataFrames, emojis o separadores ASCII (`==`, `Step 1`) decorativos.
- ❌ Scripts ad-hoc `scripts/_smoke_*.py` / `_debug_*.py` — validar inline con `assert`/`display()` o en `tests/ml/`.

## No tocar (GENERADOS — no editar a mano, se sobrescriben)

Estos notebooks se reconstruyen desde código; editarlos a mano se pierde en el próximo build. **Editá la fuente, luego rebuild**:

| Notebook generado | Fuente a editar |
|-------------------|-----------------|
| Todos los `Avance{1,2,3,4}.Equipo17.ipynb` | `ml/report/notebook_content.py`, `figure_narratives.py`, `avance{2,3,4}_content.py`, `notebook_conclusions.py` |
| `03b_fe_spectral_temporal_pastis`, `03c_fe_alphaearth_pastis` | `scripts/build_us018_notebook.py`, `build_fusion_notebook.py` + módulos `ml/features/` |
| `04_baseline` | `scripts/build_baseline_notebook.py` + módulos `ml/` |
| `05_reencuadre_fenologico` | `scripts/build_reencuadre_notebook.py` + módulos `ml/` |
| `02e_eda_metodos_paper` | `scripts/build_paper_methods_notebook.py` |
| Figuras de `Avance4` | `ml/eval/avance4_figures.py` |

Los demás (`02a`, `02b`, `02c_*`, `02d`, `03a`, `04b/04c`, `04d`-`04j`, `5a`, `5b`) se editan directo.

## Tests

- Lógica reutilizable → `tests/ml/*.py` (pytest + monkeypatch del módulo `ee`). `poetry run pytest tests/ml/ -q`.
- Reproducibilidad end-to-end → `make notebooks-check` (papermill, replicado en CI).
- Si tocaste una función de `ml/` usada por un notebook, actualizá su test en `tests/ml/`.

## Skills

- [agrosat-ml-features](../.claude/skills/agrosat-ml-features/SKILL.md) — índices espectrales, features temporales (Polars).
- [agrosat-ml-baseline](../.claude/skills/agrosat-ml-baseline/SKILL.md) — baseline tabular XGBoost.
- [agrosat-ml-segmentation](../.claude/skills/agrosat-ml-segmentation/SKILL.md) — 6 arquitecturas EPIC 5.
- [agrosat-ml-evaluation](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — plots interpretados, benchmarks.
- [agrosat-gee-alphaearth](../.claude/skills/agrosat-gee-alphaearth/SKILL.md) — sampling GEE / AlphaEarth.
