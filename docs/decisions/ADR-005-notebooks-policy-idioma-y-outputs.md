# ADR-005 — Política de idioma y outputs para notebooks Jupyter

**Status**: Propuesta · pendiente visto bueno equipo
**Fecha**: 2026-05-14
**Decisores**: Arthur Zizumbo (MLOps Lead), Aaron Bocanegra, Isaac Ávila
**US relacionada**: US-010 (EDA Sentinel-2), US-011 (EDA AlphaEarth), US-012 (EDA bivariado-temporal), prefigura US-014 (FE), US-017 (Baseline XGBoost)
**Avance**: A1 (2026-05-03 entrega original · recovery sprint S2)

---

## Contexto

Durante las múltiples pasadas de QA sobre `notebooks/eda/02b_eda_alphaearth.ipynb`
emergieron tres clases de problemas no técnicos que requieren política
explícita para evitar regresiones en US-012, US-014, US-017 y los Avances
posteriores:

### Problema 1 — Inconsistencia idiomática en strings al lector

El notebook ejecutado mostraba al lector textos como
`"Distribucion de clases"`, `"Seccion 1"`, `"vacio"`, `"anios"` — todos sin
acentos ni ñ. Mala presentación dado que **los notebooks son entregable
visual del curso** (rúbricas oficiales).

La causa fue la aplicación uniforme de la convención "evitar caracteres
no-ASCII" pensada originalmente para el código fuente, extendida por inercia
al texto que ve el lector.

### Problema 2 — Política de outputs ambigua

El [`notebooks/CLAUDE.md`](../../notebooks/CLAUDE.md) original tenía dos
reglas contradictorias heredadas del proyecto inicial:

```
ALWAYS: make notebooks-strip antes de commitear (nbstripout en CI, sin pre-commit)
NEVER:  Commitear notebook con outputs — make notebooks-strip o CI bloquea
```

Pero los notebooks son entregable visual del curso: revisar un `.ipynb` con
cells vacías obliga al sponsor a re-ejecutarlo (requiere auth GEE, datasets,
poetry, ~10 min) para ver el resultado. Eso rompe la rúbrica de
"reproducibilidad demostrada".

### Problema 3 — Conclusiones con jerga interna

Cell-22 de 02b originalmente listaba "alimentar US-014 y US-017",
referenciaba "AC-2..AC-7", mencionaba "Avance" y "rúbrica". El lector
externo (sponsor, panel evaluador) no tiene contexto para esa jerga interna
del backlog SCRUM y debe interpretar los hallazgos del análisis.

### Problema 4 — Tooling y caracteres no-ASCII

Identificadores Python con caracteres no-ASCII (variables `año`, funciones
`distribución`, etc.) rompen partes del pipeline:

- `mypy` los reporta como `invalid character in identifier` en algunos modos.
- `papermill` con `nbconvert --execute` puede fallar al inyectar parámetros
  cuyos nombres tengan acentos en algunos backends de kernel.
- Algunos IDEs (Cursor, ciertos plugins de VS Code) no autocompletan bien
  identificadores con UTF-8.
- Linters como `ruff` tienen reglas (`PLW1514`, encoding) que se confunden
  con strings UTF-8 sin `# coding: utf-8` explícito en archivos legacy.

---

## Decisión

### 1. Separación estricta idioma-encoding

| Categoría | Idioma | Encoding | Ejemplo |
|-----------|--------|----------|---------|
| **Markdown cells** | Español con acentos y ñ | UTF-8 | `## Sección 1 — Análisis exploratorio` |
| **Strings en `display(Markdown(...))`** | Español con acentos | UTF-8 | `display(Markdown("**Distribución de clases**"))` |
| **Strings en `print(...)`** | Español con acentos | UTF-8 | `print(f"Cargados {n} píxeles en {dt:.1f}s")` |
| **Títulos de plots (`title=`, `xlabel=`, `ylabel=`)** | Español con acentos | UTF-8 | `tsne_scatter(..., "t-SNE Italia × Dynamic World")` |
| **Nombres de variables / funciones / parámetros** | Inglés ASCII puro | ASCII | `df_italia`, `sample_alphaearth_roi`, `roi_name` |
| **Comentarios técnicos en código** | Inglés o español SIN acentos | ASCII | `# fallback if filter empty` |
| **Docstrings** | Español neutro estilo Google | UTF-8 | (regla de CLAUDE.md root) |
| **Claves de cache / IDs lógicos / nombres de archivo** | Inglés ASCII puro | ASCII | `cache_key="italia"`, `pianura_padana` |
| **Conventional Commits** | Inglés ASCII puro | ASCII | `feat(E2): fix DW batching for >1k coords` |

**Racional**: el código se intercambia con tooling que no siempre maneja
caracteres no-ASCII en identificadores (mypy, ruff, papermill, autocomplete).
El texto que el lector ve renderizado (markdown, prints, displays, títulos
de figura) sí lleva ortografía correcta porque es entregable visual del
curso.

### 2. Notebooks se committean ejecutados end-to-end con outputs poblados

Cambio de política respecto al template inicial:

- `nbstripout` queda **fuera** de `make check` y CI.
- Los `.ipynb` se committean con **todas sus salidas** (tablas HTML, PNG
  inline, plots interactivos).
- Reproducibilidad se valida con `make notebooks-check` (papermill end-to-end)
  en CI — sin enforcement de "cells limpias".
- Strip de outputs solo on-demand explícito (e.g. para un commit puntual
  donde Isaac pida limpieza).

**Racional**: la rúbrica del curso evalúa el notebook como artefacto visual
poblado. Stripear los outputs convierte la entrega en "instrucciones para
regenerar la entrega" — fricción innecesaria para el sponsor.

### 3. Conclusiones en lenguaje accesible

La última celda markdown de cada notebook EDA debe:

1. **NO mencionar** US-XXX, EPIC, AC-X, "rúbrica", "sponsor", "papermill",
   "CI", "Patrón A", o cualquier jerga del backlog interno.
2. **Lenguaje accesible** — explicar qué es cada concepto técnico la primera
   vez que aparezca (e.g. "AlphaEarth es un modelo de Google que comprime
   cada pixel satelital en 64 números").
3. **Citar números reales** que aparecen en los outputs (OOB scores, n filas,
   medianas) — no genéricos.
4. **Interpretar el hallazgo**, no solo enumerarlo. Decir *por qué* importa.
5. **Cerrar con "Lo que sigue"** en términos accesibles.

### 4. Hot-reload de módulos `ml/*` obligatorio

Todo notebook que importe de `ml/` debe incluir al inicio (cell-3 de
imports):

```python
%load_ext autoreload
%autoreload 2
```

**Racional**: sin esto, cualquier edición a `ml/ingest/gee_sampler.py` o
similar requiere `Restart Kernel` para verse reflejada en el notebook (Python
cachea módulos en `sys.modules`). Confunde al developer y hace pensar que un
fix no funcionó cuando sí funcionó pero el kernel sigue con la versión vieja.

### 5. Validación inline > scripts de smoke

Para debug/validación puntual de funciones de `ml/*` desde un notebook:

- **OK**: `display(df.head())`, `assert df.height > 0`, prints en celda.
- **OK**: tests pytest en `tests/ml/test_*.py` (regresión guard).
- **PROHIBIDO**: scripts `scripts/_smoke_*.py`, `scripts/_debug_*.py`,
  `scripts/_diagnose_*.py`.

`scripts/` solo aloja operativos permanentes (`azure_h100_*.sh`,
`cost_audit.sh`, `verify_structure.sh`).

### 6. Sin marcadores de generación automática

Patrones típicos de IA prohibidos en código y comentarios:

- Emojis decorativos en código o comentarios (`# 🔍 Cargar...`, `# 🚀 Train...`).
  Excepción discreta: `✓` / `⚠` como marcador semántico en `display(Markdown(...))`.
- Separadores ASCII (`print("=" * 70)`, `# ===== SECTION =====`). Usar
  markdown cells entre code cells.
- Comentarios narrativos tipo `# Step 1: Initialize` / `# Step 2: Process`.
  Comentarios deben explicar el *por qué*, no el *qué*.
- Mensajes de logging decorativos (`logger.info("=" * 70)`,
  `logger.info("🎯 Starting...")`).

---

## Consecuencias

### Positivas

- Entregable visual del curso (notebook con outputs) listo para sponsor sin
  re-ejecución.
- Texto al lector con ortografía correcta — alineado con expectativa del
  curso (idioma neutro español documentado en regla #2 de CLAUDE.md root).
- Identificadores Python en ASCII puro — compatible con todo el tooling sin
  ajustes per-tool.
- Conclusiones interpretables por el panel evaluador sin contexto del
  backlog SCRUM interno.
- Hot-reload elimina la clase de bug "edit `.py` → no se ve el cambio →
  developer cree que falló el fix".

### Negativas / trade-offs

- Notebooks committeados ocupan más espacio en Git (outputs HTML/PNG inline
  pueden sumar MBs por notebook). Mitigación: imágenes via DVC cuando
  excedan 5 MB; PNG comprimido vía `plt.savefig(dpi=200, bbox_inches='tight')`.
- Diffs de notebook en PR son ruidosos (cambios en outputs aparecen como
  cambios en el JSON del `.ipynb`). Mitigación: GitHub renderiza `.ipynb`
  como HTML lo cual ayuda; en reviews críticos pedir versión `nbconvert` lado
  a lado.
- Doble enforcement (`ruff` no detecta acentos en strings visibles vs `mypy`
  en identificadores): aceptable porque la regla es visual y se verifica en
  PR review.

### Neutrales

- `make notebooks-strip` sigue existiendo como target opcional para casos
  específicos. No se invoca por defecto en `make check`.
- Acentos en docstrings de `ml/*.py` siguen permitidos (regla heredada del
  CLAUDE.md root: docstrings/docs en español neutro).

---

## Alternativas consideradas

### A) ASCII puro en todo (status quo previo)

**Descartado**: rompe la presentación al lector. Los entregables del curso
no son código fuente sino artefactos analíticos.

### B) UTF-8 en todo (incluyendo identificadores Python)

**Descartado**: rompe compatibilidad con `mypy`, autocompletado de IDEs, y
algunos backends de papermill. El beneficio (escribir `año = 2024` en vez de
`year = 2024`) es marginal vs el costo de fricción con tooling.

### C) Stripear notebooks como antes + entregar PDFs aparte

**Descartado**: duplica trabajo (ejecutar → exportar PDF → mantener
sincronizado con el notebook). Aleja del flujo de un solo artefacto.

### D) Conclusiones con referencias a US/EPIC (status quo previo)

**Descartado**: el panel evaluador no tiene visibilidad al backlog interno.
Mantener referencias en `docs/us-handoff/` y `docs/us-resolved/`, no en
notebooks que ven externos.

---

## Implementación

### Cambios aplicados (2026-05-14, sesión Arthur)

- `notebooks/CLAUDE.md`: documentada §"Idioma — regla crítica" con tabla
  explícita, §"Estructura estándar de notebook" con reglas de Conclusiones,
  §"Hot-reload de módulos" con `autoreload`, §"Anti-patrones de IA".
- `notebooks/eda/02b_eda_alphaearth.ipynb`: aplicada políticas 1, 2, 3 a
  todas las celdas. Strings visibles con acentos correctos; código con ASCII
  puro; cell-22 reescrita en lenguaje accesible.
- `CLAUDE.md` y `AGENTS.md` (root): regla #12 ya actualizada en sesión previa
  para reflejar política de commit con outputs poblados.
- `Makefile`: `notebooks-strip` removido de `make check`; agregado
  `notebooks-check` (papermill).
- `.github/workflows/ci.yml`: removido enforcement de nbstripout.

### Cambios pendientes (otros notebooks)

- `notebooks/eda/02a_eda_sentinel2.ipynb`: pasar revisión idiomática
  (acentos en displays/prints/títulos de plots) y conclusiones accesibles.
- `notebooks/eda/02c_eda_pastis.ipynb`: idem.
- Notebooks futuros (`03_feature_engineering.ipynb`, etc.): seguir plantilla
  de `notebooks/CLAUDE.md` desde el inicio.

---

## Referencias

- [`notebooks/CLAUDE.md`](../../notebooks/CLAUDE.md) — implementación operativa
  de las 6 reglas.
- [`docs/us-handoff/us-011.md`](../us-handoff/us-011.md) §"Bugs corregidos
  sesión 2026-05-14" — BUG-15, BUG-16, BUG-17 que motivaron este ADR.
- [`CLAUDE.md`](../../CLAUDE.md) §"Reglas Globales NON-NEGOTIABLE" — regla
  #2 (idioma) y #12 (notebooks con outputs).
- Plantilla original: `Energy Optimization Copilot — AGENTS.md` (proyecto
  hermano del equipo, sección "Notebooks" y "Código Natural").

---

## Decisión final

**Aceptado en sesión 2026-05-14** por Arthur Zizumbo (MLOps Lead). Pendiente
ratificación asíncrona de Aaron e Isaac via PR review del notebook 02b
(`us-011` branch).

Próxima revisión: tras Avance 1 (3-may-2026, ya pasada) → reevaluar si las
reglas se sostienen al integrar US-012 y al cerrar el primer baseline en
US-017.
