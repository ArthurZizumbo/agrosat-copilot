# Backlog · Fase B — Dashboard EDA Nuxt + FastAPI + ECharts

**Origen**: US-013 Fase A entregó el dashboard EDA en Streamlit
(`app/eda_dashboard.py`) + reporte PDF (`ml/report/export_pdf.py`) + notebook
integrador (`notebooks/eda/Avance1.Equipo17.ipynb`). Fase B migra el
dashboard a la stack oficial del proyecto (FastAPI + Nuxt 4 + Apache
ECharts) para coherencia con el resto del producto.

**Estado**: pendiente — programado para post-A1, antes del Avance 5.

**Prioridad**: media — el dashboard Streamlit funciona como entregable de
A1; la migración es una mejora de coherencia, no bloquea ninguna entrega
formal.

**Estimación**: 10–15 horas distribuidas en 3 PRs encadenados.

---

## 1. Motivación

### Por qué migrar

1. **Coherencia de stack**: el proyecto declaró FastAPI 3.12 + Nuxt 4 SSR
   como única UI (`CLAUDE.md` §"Stack plataforma"). Streamlit fue
   excepción justificada para entregar rápido el A1.
2. **Reutilización post-A1**: el dashboard Nuxt sobrevive como módulo de
   visualización para el Avance 5 (modelo final), Avance 6 (conclusiones)
   y la defensa oral 2026-06-21.
3. **Interactividad real**: ECharts ofrece zoom, tooltip y drilldown
   nativos sobre los datos numéricos (correlaciones, PCA, distribuciones)
   que en Streamlit son figuras estáticas PNG.
4. **Deploy unificado**: una sola pipeline (Cloud Run) para backend +
   frontend en vez de gestionar Streamlit Community Cloud por separado.

### Por qué NO empezar ahora

- Fase A ya cubre el entregable A1 (dashboard interno + PDF + notebook).
- La pareja FastAPI+Nuxt requiere coordinación con el resto del backend
  (rutas, auth, i18n) — mejor consolidar primero el agente conversacional
  del EPIC 7.

---

## 2. Scope funcional

La página Nuxt `/eda` reproduce las 5 fichas del dashboard Streamlit
(Sentinel-2 univariado, AlphaEarth, bivariado/temporal, PASTIS-R
consolidado, conclusiones globales) más un tab de mapa espacial.

### Diferencias vs Fase A

| Elemento | Fase A (Streamlit) | Fase B (Nuxt + ECharts) |
|---|---|---|
| Figuras estáticas | PNG inline | PNG + ECharts interactivo cuando hay datos numéricos |
| KPIs | HTML cards inyectados | Componente `<KpiCard>` Vue reactivo |
| Mapa espacial | folium (Leaflet wrapper) | MapLibre GL + deck.gl (stack oficial) |
| i18n | Solo español | it / es / en simultáneo |
| Auth | Sin auth (local) | Clerk OAuth + RLS por session_id |
| Theme | Light fijo | Light / dark con preferencia del sistema |
| Conclusiones | Markdown via `st.markdown` | Componente `<ConclusionCard>` con Nuxt Content |

### Mapa de datos

```
ml/report/notebook_content.py       <-- fuente única de verdad (DRY)
ml/report/figure_narratives.py      <-- ya existe (Fase A)
        │
        ├── app/eda_dashboard.py (Streamlit)       Fase A
        ├── ml/report/export_pdf.py (PDF)          Fase A
        ├── scripts/build_avance1_notebook.py      Fase A
        └── backend/app/routers/eda.py (REST)      Fase B
                │
                └── frontend/pages/eda.vue + composables   Fase B
```

---

## 3. Plan de implementación

### PR 1 — Backend (~3 h)

**Branch**: `feature/E2-US-EDA-FASE-B-backend`

Crear `backend/app/routers/eda.py` con 3 endpoints:

```python
GET /eda/notebooks
    → 200 OK: list[NotebookCardOut]
    Devuelve las 5 fichas (notebook_id, title, subtitle, sections,
    figures_dir, conclusions, kpis).
    Consume `ml.report.notebook_content.CARDS` directo.

GET /eda/notebooks/{notebook_id}/figures
    → 200 OK: list[FigureOut]
    Lista los PNGs disponibles para la ficha + narrativa por figura.
    Consume `list_figures` + `get_narrative`.

GET /eda/figures/{notebook_id}/{filename}
    → 200 OK: image/png
    Sirve el binario PNG con cache headers agresivos (1 semana).
    Validación: filename debe estar en la lista de figuras de la ficha
    (evita path traversal).
```

**Pydantic models** en `backend/app/schemas/eda.py`:

```python
class FigureNarrativeOut(BaseModel):
    filename: str
    title: str
    narrative: str
    method: str

class FigureOut(BaseModel):
    url: str  # /eda/figures/<id>/<filename>
    narrative: FigureNarrativeOut | None

class KpiOut(BaseModel):
    label: str
    value: str
    delta: str

class ConclusionOut(BaseModel):
    heading: str
    body: str

class NotebookCardOut(BaseModel):
    notebook_id: str
    title: str
    subtitle: str
    sections: list[str]
    figures_dir: str
    kpis: list[KpiOut]
    conclusions: list[ConclusionOut]
```

**Tests** en `backend/tests/integration/test_eda_router.py`:
- `test_list_notebooks_returns_5_cards`
- `test_get_figures_returns_narratives_for_known_card`
- `test_get_figure_binary_serves_png`
- `test_get_figure_404_when_notebook_unknown`
- `test_get_figure_403_on_path_traversal` (`../../etc/passwd`)

**No incluido en PR1**: SSE, auth, rate limiting (lo agrega PR3).

### PR 2 — Frontend Nuxt + ECharts (~6 h)

**Branch**: `feature/E2-US-EDA-FASE-B-frontend`

Crear:

```
frontend/
├── pages/
│   └── eda.vue                    # página con tabs por ficha
├── components/
│   └── eda/
│       ├── EdaTabBar.vue          # tabs horizontal con 5 fichas
│       ├── EdaCardHeader.vue      # titulo + subtitulo + source pill
│       ├── EdaKpiRow.vue          # 4 KpiCards en grid
│       ├── KpiCard.vue            # 1 card individual
│       ├── EdaFigureBlock.vue     # img PNG + narrativa + método
│       ├── EdaConclusions.vue     # lista de ConclusionCard
│       ├── ConclusionCard.vue     # 1 card con heading + body
│       └── EdaSpatialMap.vue      # MapLibre + deck.gl (NO folium)
├── composables/
│   └── useEdaNotebooks.ts         # fetch + cache de /eda/notebooks
└── i18n/locales/
    ├── es.json + clave nueva "eda.*"
    ├── it.json + clave nueva "eda.*"
    └── en.json + clave nueva "eda.*"
```

**ECharts donde**:
- Tab AlphaEarth — gráfico de barras OOB Italia vs Francia, scatter t-SNE
  con tooltip por clase
- Tab Bivariado — heatmap interactivo de correlaciones (en vez de PNG
  estático), barplot VIF con drilldown
- Tab PASTIS — pie de distribución por tamaño de parcela, barras de
  varianza acumulada PCA

Las figuras matplotlib actuales se conservan como fallback / vista
"académica" cuando ECharts no agrega valor (ej: pairplots).

**Tests**:
- `frontend/tests/components/eda/KpiCard.spec.ts` (vitest)
- `frontend/tests/e2e/eda.spec.ts` (Playwright) — abre /eda, valida los
  6 tabs, screenshot diff vs baseline

### PR 3 — Auth + SSE + observabilidad (~3 h)

**Branch**: `feature/E2-US-EDA-FASE-B-ops`

- Clerk OAuth guard en `/eda` (solo equipo + sponsor)
- Rate limiting por `session_id` (slowapi)
- Structlog en endpoints `/eda/*` con `event`, `session_id`, `duration_ms`
- Métricas Prometheus opcionales si el agente conversacional ya las tiene

---

## 4. Tareas para retomar (checklist)

Cuando se arranque esta US:

- [ ] Crear branch `feature/E2-US-EDA-FASE-B-backend` desde `develop`
- [ ] Eliminar dashboard Streamlit (`app/eda_dashboard.py`,
  `deploy/streamlit/`, `tests/app/`) **solo cuando la página Nuxt esté
  mergeada en main y aceptada por el sponsor** — no antes
- [ ] Marcar `agrosat-backend-api` skill como auto-invoke al tocar
  `backend/app/routers/eda.py`
- [ ] Marcar `agrosat-frontend-components` skill al tocar
  `frontend/pages/eda.vue` y `frontend/components/eda/`
- [ ] Actualizar `docs/orchestration/auto-invoke.md` con el mapeo nuevo
- [ ] Eliminar `requirements.txt`, `runtime.txt`, `packages.txt` de
  `deploy/streamlit/` al cerrar la migración
- [ ] Mover `deploy/streamlit/` → `docs/decisions/ADR-EDA-streamlit-deprecated.md`
  con histórico

---

## 5. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| ECharts pesa ~900 KB en bundle | Lazy load con `defineAsyncComponent` cuando se entra a `/eda` |
| MapLibre + deck.gl incompatibles con SSR | Wrappear con `<ClientOnly>` |
| `ml.report.notebook_content` se modifica y rompe contrato API | Tests de contrato en `backend/tests/integration/test_eda_router.py` que validan shape del response |
| Sponsor pide nuevas figuras durante el sprint | Cambios solo tocan `notebook_content.py` y `figure_narratives.py` — los endpoints se autoactualizan |
| Streamlit ya quedó deployado en Streamlit Cloud | Mantener ambas durante 2 semanas antes de redirigir |

---

## 6. Criterios de éxito

- [ ] `GET /eda/notebooks` devuelve 5 fichas con shape Pydantic validado
- [ ] Página `/eda` en Nuxt renderiza los 6 tabs (5 fichas + mapa)
- [ ] Las narrativas y conclusiones que aparecen en Nuxt son **bit-identical**
  a las del dashboard Streamlit (mismo `notebook_content.py` como fuente)
- [ ] Cobertura tests backend ≥ 70 % sobre `backend/app/routers/eda.py`
- [ ] Cobertura tests frontend ≥ 50 % sobre `components/eda/`
- [ ] Lighthouse score ≥ 90 en `/eda` (mobile + desktop)
- [ ] i18n: claves `eda.*` traducidas en `it.json`, `es.json`, `en.json`
- [ ] Streamlit deprecado pero accesible hasta la presentación final
  (2026-06-21)

---

## 7. Referencias

- Fase A handoff: [`docs/us-handoff/us-013.md`](../us-handoff/us-013.md)
- Fase A manual test: [`docs/manual-test/us-013.md`](../manual-test/us-013.md)
- Fuente DRY: [`ml/report/notebook_content.py`](../../ml/report/notebook_content.py)
- Skills involucrados: `agrosat-backend-api` · `agrosat-frontend-components`
  · `agrosat-frontend-composables` · `agrosat-maplibre-geo` · `agrosat-testing`
