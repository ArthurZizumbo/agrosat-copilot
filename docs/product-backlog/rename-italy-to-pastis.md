# Rename `_italy` -> `_pastis` — migración diferida de notebooks y datos

**Estado**: PARCIAL (código migrado 31-may; datos en disco y notebooks DIFERIDOS)
**Owner**: Arthur Zizumbo · **Epic**: E5 (transversal con US-025)
**Prioridad**: baja (deuda de nomenclatura, no bloquea funcionalidad)

## Contexto

Los artefactos de features se nombraron con sufijo `_italy` (nombre heredado de la
fase exploratoria), pero su **contenido es PASTIS-R francés**: los `parcel_id` tienen
formato `{patch_id}_{instance_id}` (ej. `10000_1`), correspondientes a patches del
benchmark PASTIS-R de Bretaña, Francia — no parcelas italianas. La nomenclatura
correcta es `_pastis`.

El sufijo engañoso confunde al equipo (se asume que falta dato italiano cuando no
existe tal cosa). Detectado en la auditoría US-023-preview-v2 y confirmado abriendo
los parquets (`docs/audit/us-023-preview-v2-audit.md`).

## Lo que YA se hizo (31-may, US-025)

Migración del **código ML** a la nomenclatura canónica `_pastis`, con estrategia de
**alias de compatibilidad** para no romper nada:

- **Helper nuevo**: `ml/utils/dataset_paths.py`
  - `resolve_dataset_path(path)`: prefiere `_pastis`, cae al `_italy` legacy si el
    primero no existe en disco. Usado al **leer** artefactos por default.
  - `to_pastis_name(path)` / `to_legacy_name(path)`: conversión de sufijo.
- **Módulos migrados** (defaults a `_pastis`; lectura con fallback, escritura directa):
  - `ml/features/fusion.py` — constantes `_DEFAULT_*_PATH` + 3 use-sites
  - `ml/features/winning_features.py`
  - `ml/utils/baseline_notebook_helpers.py`
  - `ml/utils/phenology_text.py`
  - `ml/ingest/s2_anchor_sampler.py`
  - docstrings de `ml/features/fusion.py`, `ml/extractors/farslip_extractor.py`,
    `ml/farslip/extract_embeddings.py`
- **Regla aplicada**: ESCRITURA de artefactos nuevos -> nombre `_pastis` directo
  (el dato nuevo nace con nombre correcto). LECTURA de artefactos existentes ->
  `resolve_dataset_path` (encuentra el `_italy` legacy en disco sin renombrarlo).

**Verificado**: el código pide `_pastis`, encuentra los datos `_italy` legacy, todos
los módulos importan. Los notebooks ejecutados siguen funcionando sin cambios.

## Lo que se DIFIERE (no se hace ahora — decisión 31-may)

1. **Renombrar los datos en disco** `*_italy.parquet` -> `*_pastis.parquet` (+ sus
   `.dvc`) con `dvc move`. Hoy resuelven vía fallback; el rename físico es cosmético.
   - Archivos: `features_fused_italy`, `features_fused_winning_italy`,
     `phenology_text_italy`, `embeddings_italy*_PLACEHOLDER`, manifest JSON.
2. **Migrar los notebooks** que referencian `_italy`. Los notebooks
   (`04_baseline`, `04c_baseline`, `05_reencuadre`, `Avance3.Equipo17`) se
   **conservan ejecutados con todas sus salidas pobladas** (entregable visual del
   curso). NO se re-ejecutan ni se tocan sus outputs. Cuando se regeneren desde
   el builder en una sesión futura, el builder ya migrado producirá rutas `_pastis`.
3. **Builders de notebooks** (`scripts/build_baseline_notebooks_v2.py`,
   `build_reencuadre_notebook.py`, `us023_p4_pheno_text_ablation.py`): conservan
   referencias `_italy` legibles; al llamar a los módulos migrados, el fallback los
   cubre. Migrarlos a `_pastis` explícito queda para cuando se regeneren notebooks.
4. **Asset Dagster** `farslip_embeddings_italy`: NO se renombra (rompería el lineage
   declarativo). Solo se documentó en su docstring que el contenido es PASTIS-R.

## Criterio de cierre (cuando se retome)

- [ ] `dvc move` de los 6 parquets `_italy` -> `_pastis` + commit de los `.dvc`
- [ ] regenerar notebooks desde builders migrados a `_pastis` (con outputs poblados)
- [ ] migrar los 3 builders a `_pastis` explícito
- [ ] eliminar el fallback legacy de `resolve_dataset_path` (o dejarlo como red de
      seguridad permanente — decidir)
- [ ] `make notebooks-check` verde tras la migración
