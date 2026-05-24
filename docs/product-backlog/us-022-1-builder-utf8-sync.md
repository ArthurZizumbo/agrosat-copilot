# Backlog · US-022.1 — Sincronizar builder del notebook 04_baseline con ADR-005 §1 (UTF-8)

**Origen**: durante la validacion manual de los manual-tests de US-019 a US-022
(2026-05-22) se detecto que el notebook `04_baseline.ipynb` tenia 13 markdowns
sin acentos (`"Analisis SHAP"`, `"seccion"`, `"caracteristicas"`, etc.),
violando el ADR-005 §1 "Markdown cells: español con acentos y ñ, UTF-8".

El `.ipynb` ya quedo corregido in-place (cells 0, 1, 7, 8, 12, 19, 22, 25, 33,
34, 38, 39, 46 reescritas con acentos correctos + caracteres UTF-8 como `≥`,
`±`, `→` donde aplica). Tambien se reescribieron los 5 markdowns AUTO-INTERP
inyectados durante la limpieza.

**El builder `scripts/build_baseline_notebook.py` quedo desincronizado** —
sus bloques `_md("...")` siguen con los textos ASCII originales. Si alguien
regenera el notebook desde el builder (`make baseline-notebook`), los acentos
del `.ipynb` actual se perderian.

**Estado**: pendiente — no bloquea el Avance 3 (ya entregado) ni la rubrica
(el `.ipynb` commiteado es el entregable visual, no el builder).

**Prioridad**: media — bloquea solo escenarios de regeneracion completa del
notebook (raros una vez cerrado el Avance 3).

**Estimacion**: 1-2 horas. La complejidad esta en que el builder tiene los
textos `_md(...)` partidos en strings concatenados con wraps especificos
(line continuations); una sustitucion ingenua por `str.replace` no acerto
porque el wrap difiere entre el codigo fuente y un parrafo plano. Tres
opciones de implementacion:

- (a) **Edits manuales con Read + Edit** sobre cada bloque `_md(...)` del
  builder, copiando el texto exacto con sus saltos de linea. ~13 bloques
  + 5 prints/labels visibles = ~18 edits.
- (b) **Regenerar el builder** desde el `.ipynb` corregido: parsear las
  cells de notebook y emitir un nuevo `build_baseline_notebook.py` con
  los `_md(...)` correctos. Mas robusto pero requiere preservar las
  cells `_code(...)` y `_params_code(...)` que NO se tocaron.
- (c) **Marcar el builder como deprecado** y declarar que el `.ipynb` es la
  fuente de verdad post-Avance 3. Coherente con ADR-005 §2 "notebooks se
  committean ejecutados end-to-end con outputs poblados".

---

## Alcance del fix (cuando se ejecute)

### 13 markdowns en `CELLS = [...]`

Lista de `_md(...)` a reescribir, con el texto correcto disponible en el
`.ipynb` (`notebooks/04_baseline.ipynb`, cells 0, 1, 7, 8, 12, 19, 22, 25,
33, 34, 38, 39, 46).

### Strings visibles al lector en `_code(...)`

Segun ADR-005 §1, los `print(...)`, `ax.set_title(...)`, `ax.set_xlabel(...)`
deben ir con acentos. Ya se actualizaron 5 titulos de plot en el builder
durante la verificacion del 2026-05-22:

- `Curva de validacion — RF max_depth (accuracy)` → `validación` (pendiente)
- `Curva de validacion — XGB n_estimators (accuracy)` → `validación` (pendiente)
- `Curva de aprendizaje — {kind} (accuracy)` ✓ ya correcto
- `Importancia nativa top-20 — {kind}` ✓ ya correcto
- `Comparativa del baseline — 3 escenarios de features` → `escenarios de características` (pendiente)

Otros prints con texto sin acentos (~10 lineas) localizables con:
```bash
grep -nE "print\(.*'.*(caracteristicas|seccion|validacion|...).*'" scripts/build_baseline_notebook.py
```

### Comentarios `# ...` en codigo

ADR-005 §1 dice "Comentarios tecnicos en codigo: ingles o español SIN
acentos (ASCII)". Los comentarios `#` actuales **se mantienen sin acentos**
— no se tocan.

---

## Criterios de aceptacion

- [ ] **AC-1**: `grep -E "caracteristicas|imagenes|arboles|seccion" scripts/build_baseline_notebook.py` solo encuentra ocurrencias en comentarios `# ...` o en nombres de columnas/identificadores (ASCII), nunca en strings de `_md(...)` o `print(...)`.
- [ ] **AC-2**: `poetry run python scripts/build_baseline_notebook.py --out /tmp/test_baseline.ipynb` genera un notebook cuyo diff de markdowns vs `notebooks/04_baseline.ipynb` actual es vacio en texto (la diferencia debe estar solo en outputs/execution_count, nunca en `source` de cells markdown).
- [ ] **AC-3**: no se rompe el contrato del builder con `make baseline-notebook` (target sigue ejecutable).

---

## Dependencias

- **Upstream**: ninguna — el `.ipynb` corregido ya existe en `main` como
  fuente de verdad.
- **Downstream**: cualquier US futura que necesite regenerar el notebook
  desde cero (e.g. cambio estructural en el orden de secciones, integracion
  con US-022-b).

---

## Notas

- Identificado durante la sesion de cierre 2026-05-22, mismo lote que el
  fix del MissingGreenlet en Dagster (saldado), el push del `breizhcrops.dvc`
  (saldado), y el embed de PNGs + interpretaciones en el notebook (saldado).
- El builder tambien tiene un fix de titulo aplicado en la misma sesion:
  `lc_fig.suptitle(...)` reemplazado por `ax.set_title(...)` in-place para
  los 3 bloques de curvas (learning, validation RF, validation XGB). Esos
  fixes SI estan en el builder y son los unicos cambios sincronizados al
  2026-05-22.
- Si se elige la opcion (c) (deprecar el builder), actualizar tambien
  `notebooks/CLAUDE.md` y `scripts/README.md` para reflejar la politica.
