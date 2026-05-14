# Notebooks Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root para trabajo en notebooks de los Avances del curso.

**Rol**: Notebooks Jupyter secuenciales ejecutables con papermill que documentan los Avances del curso (Avances 1-5 son notebooks entregables; Avances 0, 6, 7 son PDFs).

## Skills References

- [agrosat-ml-features](../.claude/skills/agrosat-ml-features/SKILL.md) — Polars, índices espectrales
- [agrosat-ml-baseline](../.claude/skills/agrosat-ml-baseline/SKILL.md) — Baseline XGBoost
- [agrosat-ml-segmentation](../.claude/skills/agrosat-ml-segmentation/SKILL.md) — 6 modelos EPIC 5
- [agrosat-ml-ensemble](../.claude/skills/agrosat-ml-ensemble/SKILL.md) — Ensambles EPIC 6
- [agrosat-ml-evaluation](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — Plots interpretados

## Estructura Canónica (mapeada a Avances del curso)

| Notebook | Avance | Fecha entrega | Rúbrica |
|----------|--------|---------------|---------|
| `01_avance0_propuesta.ipynb` | Avance 0 (PDF derivado) | 26-abr-2026 | Propuesta |
| `02a_eda_sentinel2.ipynb` | Avance 1 | 3-may-2026 | Univariado |
| `02b_eda_alphaearth.ipynb` | Avance 1 | 3-may-2026 | AlphaEarth caracterización |
| `02c_eda_bivariado_temporal.ipynb` | Avance 1 | 3-may-2026 | Bivariado + temporal |
| `03_feature_engineering.ipynb` | Avance 2 | 17-may-2026 | FE 30+30+30+10 pts |
| `04_baseline_xgboost_alphaearth.ipynb` | Avance 3 | 20-may-2026 | Baseline |
| `05_alt_models.ipynb` | Avance 4 | 24-may-2026 | 6 arquitecturas |
| `06_final_gemma4_ensembles.ipynb` | Avance 5 | 31-may-2026 | Gemma 4 + ensambles |
| `07_agent_eval.ipynb` | Avance 6 (PDF) | 7-jun-2026 | Conclusiones |

## Critical Rules

- **ALWAYS**: Ejecutable end-to-end con `papermill notebook.ipynb output.ipynb -p param value`
- **ALWAYS**: Imports y configs al inicio (`%load_ext autoreload`, `%autoreload 2`)
- **ALWAYS**: Polars para DataFrames (no pandas salvo conversión final a `.to_pandas()` para libs que lo requieran)
- **ALWAYS**: Reutilizar funciones de `ml/` — el notebook llama, no implementa lógica
- **ALWAYS**: Cada notebook tiene sección "Conclusiones" mapeada 1:1 a rúbrica del Avance
- **ALWAYS**: Plots exportados a `paper/figures/` con alta resolución si van al paper
- **ALWAYS**: Notebooks se commitean **ejecutados end-to-end con outputs poblados** (tablas HTML, PNG inline, plots). Reproducibilidad se valida con `make notebooks-check` (papermill). NO usar `nbstripout` salvo on-demand explícito.
- **NEVER**: Implementar lógica nueva en notebook — refactorizar a `ml/` y llamar
- **NEVER**: Hardcodear paths absolutos — usar `pathlib` y env vars
- **NEVER**: Crear scripts ad-hoc `scripts/_*.py` para smoke / debug / verificación — validar con `display()` inline en el notebook o con tests pytest en `tests/ml/`
- **NEVER**: Notebook que requiere intervención manual entre celdas

---

## Idioma — regla crítica

El proyecto separa estrictamente texto del lector vs símbolos del intérprete:

| Elemento | Idioma | Encoding | Ejemplo |
|----------|--------|----------|---------|
| **Markdown cells** | Español con acentos y ñ | UTF-8 | `## Sección 1 — Análisis exploratorio` |
| **Strings dentro de `display(Markdown(...))`** | Español con acentos | UTF-8 | `display(Markdown("**Distribución de clases**"))` |
| **Strings dentro de `print(...)`** | Español con acentos | UTF-8 | `print(f"Cargados {n} píxeles en {dt:.1f}s")` |
| **Títulos de plots (`title=`, `xlabel=`)** | Español con acentos | UTF-8 | `tsne_scatter(..., "t-SNE Italia × Dynamic World")` |
| **Nombres de variables / funciones / parámetros** | Inglés ASCII puro | ASCII | `df_italia`, `sample_alphaearth_roi`, `roi_name` |
| **Comentarios técnicos en código** | Inglés o español SIN acentos | ASCII | `# fallback if filter empty` |
| **Docstrings** | Español neutro estilo Google | UTF-8 | Ver §"Docstrings" más abajo |
| **Claves de cache / IDs lógicos** | Inglés ASCII puro | ASCII | `cache_key="italia"`, `pianura_padana` |

**Razón**: el código se intercambia con tooling (linters, mypy, papermill, IDEs) que no siempre maneja bien caracteres no-ASCII en identificadores. El texto que ve el lector del notebook (markdown, prints, displays, títulos de figura) sí lleva ortografía correcta porque es entregable visual del curso.

### Ejemplo concreto

```python
# Comentario técnico en ASCII puro
# Bboxes reducidos (~50x50 km) para evitar timeout del compute graph
ROI_BBOXES: dict[str, list[float]] = {
    "pianura_padana": [10.0, 45.0, 11.0, 45.5],
}

for roi_name, bbox in ROI_BBOXES.items():
    df_roi = sample_alphaearth_roi(roi=ee.Geometry.BBox(*bbox), roi_name=roi_name)
    # print con ortografia correcta en espanol -> visible al lector
    print(f"{roi_name}: {df_roi.height} píxeles cargados")

# display con Markdown UTF-8 -> visible al lector
display(Markdown(f"**Muestras de Italia**: `{df_italia.height:,}` filas en {len(ROI_BBOXES)} ROIs"))
```

---

## Patrón DRY — funciones reutilizables, no código inline

### Regla

Si una función se usa en 2+ notebooks **o** un patrón se repite 3+ veces dentro del mismo notebook → extraer a `ml/utils/`, `ml/ingest/`, `ml/analysis/` o `ml/features/`.

### Catálogo actual de utilidades (verificar antes de duplicar)

```python
# ml/utils/notebook_setup.py — bootstrap repo + EE auth
from ml.utils.notebook_setup import (
    find_repo_root,            # localiza pyproject.toml subiendo niveles
    load_env_local,            # carga .env.local sin python-dotenv
    configure_ee_from_env,     # devuelve (gee_project, sa_json_path)
)

# ml/utils/sampling.py
from ml.utils.sampling import stratified_sample

# ml/ingest/gee_sampler.py — Earth Engine sampling con cache parquet
from ml.ingest.gee_sampler import (
    init_ee,                          # inicializa EE (no interactive_auth por defecto)
    sample_s2_roi,                    # Sentinel-2 sobre ROI
    sample_alphaearth_roi,            # AlphaEarth 64-dim sobre ROI
    sample_alphaearth_at_coords,      # AlphaEarth en (lon, lat) explicitas, con batching
    sample_dynamic_world_at,          # DW class id/name en coords, con batching
    fetch_s2_ndvi_rgb_for_parcel,     # RGB + NDVI para visualizacion de parcela
)

# ml/ingest/pastis_loader.py
from ml.ingest.pastis_loader import (
    pastis_patch_coords,        # reproyecta patches EPSG:2154 -> 4326
    pastis_patch_index,         # indice plano de metadata.geojson
    pastis_pixel_labels,        # labels por pixel con stratified sample
    PASTIS_R_GROUPINGS,         # mapping fenologico / agronomico
    PASTIS_R_CLASSES,           # id -> nombre legible
)

# ml/analysis/embeddings.py
from ml.analysis.embeddings import (
    DIM_COLS,                   # ['dim_00', ..., 'dim_63']
    correlation_matrix,         # 64x64 long-form
    qq_test_dims,               # normalidad por dim
    tsne_2d, umap_2d,           # reduccion dimensional con dropna
    rf_feature_importance,      # Random Forest + OOB score
    temporal_stability,         # cosine_sim inter-anual
    cross_region_consistency,   # interseccion top-k Italia <-> Francia
    compare_alphaearth_vs_ndvi, # plot 1x3 RGB | NDVI | pseudo-RGB
)

# ml/analysis/visualization.py
from ml.analysis.visualization import (
    qq_grid,
    correlation_heatmap,
    tsne_scatter, umap_scatter,
    cross_region_scatter,
)
```

### Anti-patrón: lógica inline en notebook

```python
# MAL — implementacion inline
import json
schema = {"patch_id": pl.Utf8, "TILE": pl.Utf8, "Fold": pl.Int64}
with (REPO / "data/PASTIS-R/metadata.geojson").open() as fh:
    gj = json.load(fh)
rows = [...]  # parser de 17 lineas
df_idx = pl.DataFrame(rows, schema=schema)

# BIEN — funcion en ml/ingest/
from ml.ingest.pastis_loader import pastis_patch_index
df_idx = pastis_patch_index()
```

---

## Programación Orientada a Objetos — transformers sklearn

Para feature engineering en `ml/features/`, usar clases que hereden de `BaseEstimator` y `TransformerMixin`:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl


class TemporalEmbeddingFeatures(BaseEstimator, TransformerMixin):
    """Genera features temporales a partir de embeddings AlphaEarth multi-anio.

    Args:
        year_cols: Lista de columnas anuales (e.g. ['emb_2022', ..., 'emb_2025']).
        compute_delta: Si True agrega columnas `delta_YYYY_YYYY` y `mean_cosine`.
    """

    def __init__(self, year_cols: list[str], compute_delta: bool = True) -> None:
        self.year_cols = year_cols
        self.compute_delta = compute_delta

    def fit(self, X: pl.DataFrame, y: object | None = None) -> "TemporalEmbeddingFeatures":
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        ...
```

**Beneficios**:
- Encaja en `sklearn.pipeline.Pipeline` con `StandardScaler`, `XGBRegressor`, etc.
- `fit_transform` y `transform` separados — soporta train/val sin leakage.
- Testeable aislado en `tests/ml/test_features.py`.

### Anti-patrón: scaler suelto

```python
# MAL — fit en train y transform en test estan desacoplados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
# (mas tarde, en otra celda...)
X_test_scaled = scaler.transform(X_test)  # facil de olvidar

# BIEN — Pipeline atrapa todo el preprocesamiento
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor()),
])
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

---

## Estructura estándar de notebook

Orden canónico de celdas:

```
1. Markdown — Título y descripción (en español, sin US-XXX/EPIC/AC-X)
2. Code — Parámetros papermill (tag `parameters`)
3. Code — Imports + bootstrap repo root + configuración matplotlib/polars
4. Code — Init Earth Engine / cargar datasets externos
5. Markdown — Sección 1 — descripción corta
6-N. Code/Markdown alternados — análisis paso a paso
N+1. Markdown — Conclusiones (en lenguaje accesible, interpretando los datos)
```

### Parámetros papermill — celda 2

```python
sample_size = 6_000
year = 2024
pastis_year = 2019
n_pastis_patches = 10
tsne_subsample = 5_000
figures_dir = "paper/figures/us-XXX"
```

Tag `parameters` en VS Code: clic en la celda → "More Actions" → "Add Cell Tag" → `parameters`.

### Bootstrap + configuración — celda 3

```python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from IPython.display import Markdown, display

# Bootstrap sys.path para que funcione desde notebooks/eda/
_REPO_BOOTSTRAP = Path.cwd().resolve()
for _candidate in (_REPO_BOOTSTRAP, *_REPO_BOOTSTRAP.parents):
    if (_candidate / "pyproject.toml").is_file():
        _REPO_BOOTSTRAP = _candidate
        break
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from ml.utils.notebook_setup import find_repo_root, configure_ee_from_env
# ... resto de imports de ml.*

# Polars: rendering rico HTML en Jupyter
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_rows(20)
pl.Config.set_fmt_str_lengths(60)

# matplotlib inline para display(fig) y plt.show() en celda
%matplotlib inline
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

# Autoreload para captar cambios en ml/*.py sin restart de kernel
%load_ext autoreload
%autoreload 2

REPO = find_repo_root()
FIGURES = REPO / figures_dir
FIGURES.mkdir(parents=True, exist_ok=True)
```

### Conclusiones — última celda markdown

Reglas obligatorias para la sección de cierre:

1. **NO mencionar** US-XXX, EPIC, AC-X, "rúbrica", "sponsor", "papermill", "CI".
2. **Lenguaje accesible** — explica qué es cada cosa antes de citarla (la primera vez que aparezca un término).
3. **Citar números reales** que aparecen en los outputs (OOB scores, n filas, medianas, etc.) — no genéricos.
4. **Interpretar el hallazgo**, no solo enumerarlo. Decir *por qué* importa para la próxima fase del proyecto.
5. **Cerrar con "Lo que sigue"** — siguientes pasos concretos basados en lo encontrado.

---

## Renderizado rico — display sobre print

Los notebooks son entregable visual. Preferir `display()` sobre `print()` para DataFrames:

```python
# BIEN — tabla HTML formateada con scroll horizontal
display(df_italia.head(8))

# BIEN — Markdown con encabezado en negrita
display(Markdown(f"**Muestras de Italia**: `{df_italia.height:,}` filas en {len(ROI_BBOXES)} ROIs"))

# BIEN — figura matplotlib con doble render protegido
fig = correlation_heatmap(corr_italia, threshold=0.7)
if fig is not None:
    display(fig)
    plt.close(fig)  # evita auto-show de %matplotlib inline

# Aceptable — print solo para logging de progreso en loops
for idx, (roi_name, bbox) in enumerate(ROI_BBOXES.items(), start=1):
    df_roi = sample_alphaearth_roi(roi=ee.Geometry.BBox(*bbox), ...)
    print(f"  [{idx}/{len(ROI_BBOXES)}] {roi_name}: {df_roi.height} px en {dt:.1f}s")
```

### Antipatrón: `print(df)` para DataFrames

```python
# MAL — pierde formato HTML, columnas truncadas en consola
print(df_italia.head())

# MAL — string con HTML inline (no se ve en dark mode)
display(Markdown("<div style='background:#f0f8ff'>Setup OK</div>"))
```

---

## Anti-patrones de IA — código natural

El código debe leerse como escrito por un humano, no por un asistente. **Evitar marcadores típicos de generación automática**:

### Sin emojis decorativos en código o comentarios

```python
# MAL
# 🔍 Cargar datos
# ✅ Procesar resultados
# 🚀 Entrenar modelo
print("🎯 Iniciando entrenamiento...")

# BIEN
# Cargar datos
print("Iniciando entrenamiento...")
```

Excepciones aceptadas (marcadores semánticos, no decorativos):

```python
# OK — marcador de estado discreto en display Markdown del lector
status = "✓" if n_consistent >= 5 else "⚠"
display(Markdown(f"{status} **Dimensiones consistentes**: `{n_consistent}/10`"))
```

### Sin separadores ASCII decorativos

```python
# MAL
print("=" * 70)
print("SECTION 2: DATA LOADING")
print("=" * 70)

# MAL
# ============================================
# DATA LOADING
# ============================================

# BIEN — usar markdown cell entre code cells
## Sección 2 — Carga de datos
```

### Comentarios concisos, no narrativos

```python
# MAL — narrativo, redundante con el código
# Step 1: Initialize the connection to Earth Engine
# This is an important step that authenticates against GEE.
# Step 2: Filter the collection by date and bounds.
collection = ee.ImageCollection(...).filterBounds(roi)

# BIEN — explica el "por que", no el "que"
# mosaic() vs first(): la coleccion tiene ~10k tiles/anio. first() devuelve
# un tile arbitrario con footprint limitado.
image = collection.mosaic().select(band_names)
```

### Mensajes de logging simples

```python
# MAL
logger.info("=" * 70)
logger.info("🎯 Starting training process...")
logger.info("=" * 70)

# BIEN
logger.info(f"Cargados {len(df)} registros")
logger.info(f"Entrenamiento completado en {elapsed:.2f}s")
```

---

## Type hints y docstrings

### Type hints obligatorios en funciones `ml/*.py`

```python
# MAL
def process_data(df, threshold=0.5):
    ...

# BIEN
def process_data(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    ...
```

### Docstrings estilo Google en español

```python
def sample_dynamic_world_at(
    coords: pl.DataFrame,
    year: int,
    cache_path: Path | None = None,
    cache_key: str = "coords",
    batch_size: int = 500,
) -> pl.DataFrame:
    """Extrae la clase moda Dynamic World del anio dado para cada (lon, lat).

    Procesa coords en lotes de `batch_size` puntos para evitar timeouts del
    compute graph server-side de GEE.

    Args:
        coords: DataFrame con columnas `px_id, lon, lat` en EPSG:4326.
        year: Anio para filtrar la coleccion Dynamic World.
        cache_path: Carpeta cache parquet.
        cache_key: Identificador logico para el cache.
        batch_size: Numero maximo de puntos por request `reduceRegions`.

    Returns:
        DataFrame con columnas `px_id, dw_class_id, dw_class_name, dw_confidence`.
    """
```

**Nota**: docstrings van en `ml/*.py` (modulos), no como markdown encima de celdas de notebook.

---

## Hot-reload de módulos `ml/*` en kernel Jupyter

Al editar `ml/ingest/gee_sampler.py` mientras el kernel está vivo, Python NO recarga el módulo automáticamente — `sys.modules` lo tiene cacheado.

**Solución estándar**: agregar al inicio del notebook (celda 3, junto a imports):

```python
%load_ext autoreload
%autoreload 2
```

Con esto, cada vez que ejecutes una celda, Jupyter detecta cambios en archivos `ml/*` y recarga los módulos. Sin esto, los fixes en `gee_sampler.py` no se ven hasta `Restart Kernel`.

---

## Tests inline vs `tests/ml/`

Donde validar comportamiento:

| Tipo de validación | Dónde | Cómo |
|--------------------|-------|------|
| Lógica de función | `tests/ml/test_*.py` | pytest + monkeypatch del módulo `ee` |
| Smoke / debug puntual | Inline en notebook | `display(df.head())`, `assert df.height > 0` |
| Reproducibilidad end-to-end | CI | `make notebooks-check` ejecuta papermill |

**Prohibido**: `scripts/_smoke_*.py`, `scripts/_debug_*.py`. La validación va al notebook o a `tests/ml/`. `scripts/` solo aloja operativos permanentes (`azure_h100_*.sh`, `cost_audit.sh`, etc.).

---

## Versionado de datos

### Datos pequeños (referencia)

`data/reference/*.json`, `data/reference/rois/*.geojson` — committeados directo a Git.

### Datos grandes (datasets, modelos, cachés grandes)

DVC + GCS remote (`gs://agrosat-dvc-remote`):

```bash
dvc add data/PASTIS-R/
git add data/PASTIS-R.dvc
git commit -m "feat(EX): add PASTIS-R dataset versioned via DVC"
dvc push
```

### Caché GEE local (`data/cache/gee/*.parquet`)

**No committear** — se regenera en cada máquina con las mismas funciones `sample_*_at_coords` (deterministas con seed). Listado en `.gitignore`.

---

## Plantilla de encabezado mínima

Para notebooks nuevos, esta es la celda 1 (markdown):

```markdown
# Título del análisis — sin US-XXX, sin EPIC

Este notebook responde una pregunta concreta: ¿X? Lo hacemos sampleando Y de
Z en N ROIs y aplicando A, B, C.

## Requisitos para ejecución end-to-end

- `earthengine authenticate` ejecutado y vigente.
- `data/DATASET/` descomprimido.
- Dependencias instaladas vía `poetry install --with ml,geo,paper`.

Si EE o los datasets no están disponibles, las funciones de `ml/ingest/`
retornan DataFrames vacíos con esquema válido y los plots se generan en modo
placeholder, de forma que el notebook completa la ejecución sin error.
```

---

## QA Checklist Notebooks

- [ ] Ejecuta end-to-end con papermill sin intervención
- [ ] Imports y configs al inicio (`%load_ext autoreload`, `%autoreload 2`)
- [ ] Polars en lugar de pandas
- [ ] Lógica reutilizable refactorizada a `ml/` (catálogo §"Patrón DRY")
- [ ] Sin código duplicado entre notebooks (US-010 ↔ US-011 ↔ US-012)
- [ ] Strings al lector (markdown / display / print / títulos) con acentos y ñ
- [ ] Código (variables / funciones / comentarios técnicos / cache keys) en ASCII puro
- [ ] Sin emojis decorativos, sin separadores `=`/`-`/`*`, sin "Step 1/Step 2"
- [ ] `display()` para DataFrames y figuras, `print()` solo para logging de progreso
- [ ] `plt.close(fig)` después de `display(fig)` para evitar doble render
- [ ] Sección "Conclusiones" en lenguaje accesible — sin US-XXX/EPIC/AC-X, con números reales y "Lo que sigue"
- [ ] Plots con alta resolución exportados a `paper/figures/` si aplican
- [ ] Notebook ejecutado end-to-end con papermill (`make notebooks-check`) y commiteado con outputs poblados
- [ ] Tests de papermill pasan en CI
- [ ] Si tocó función reutilizable: tests en `tests/ml/` actualizados

---

## Comandos útiles

```bash
# Ejecutar notebook end-to-end (regenera outputs)
poetry run papermill notebooks/eda/02b_eda_alphaearth.ipynb \
    notebooks/eda/02b_eda_alphaearth.ipynb \
    -p sample_size 6000

# Validar reproducibilidad de todos los notebooks
make notebooks-check

# Tests del módulo ml
poetry run pytest tests/ml/ -q

# Lint + secrets-scan + i18n-check (obligatorio antes de PR)
make check

# Limpiar cache GEE local (forzar re-fetch en proxima ejecucion)
rm -rf data/cache/gee/
```
