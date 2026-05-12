# Agent Sub-Agent — AgroSatCopilot (Google ADK)

> Sobreescribe al sub-agente `ml/` cuando trabajes en el agente conversacional.

**Rol**: Construir el agente Google ADK Plan-and-React con 9 tools geoespaciales y Spatial-RAG híbrido PostGIS + pgvector. Soporta dos variantes de LLM intercambiables con switch A/B desde la UI: variante A Gemini 3.1 Pro (Vertex AI cloud), variante B Qwen3.5-35B-A3B (vLLM self-hosted en H100). Tracing built-in de ADK, deploy nativo a Vertex AI Agent Engine.

## Skills References

- [agrosat-google-adk-agent](../../.claude/skills/agrosat-google-adk-agent/SKILL.md) — ADK Plan-and-React, FunctionTool, tracing
- [agrosat-spatial-rag](../../.claude/skills/agrosat-spatial-rag/SKILL.md) — PostGIS ST_DWithin + pgvector híbrido
- [agrosat-llm-finetuning](../../.claude/skills/agrosat-llm-finetuning/SKILL.md) — Backends Gemini + vLLM Qwen3.5
- [agrosat-ml-evaluation](../../.claude/skills/agrosat-ml-evaluation/SKILL.md) — AgroMind, GeoAnalystBench, GeoBenchX

## Auto-Invoke

| Acción | Skill |
|--------|-------|
| Crear tool ADK nuevo | `agrosat-google-adk-agent` |
| Implementar Plan-and-React | `agrosat-google-adk-agent` |
| Spatial-RAG híbrido | `agrosat-spatial-rag` |
| Switch backend Gemini ↔ vLLM | `agrosat-google-adk-agent` + `agrosat-llm-finetuning` |
| Memory store por session_id | `agrosat-google-adk-agent` |
| Evaluar agente vs benchmarks | `agrosat-ml-evaluation` |
| Deploy a Vertex AI Agent Engine | `agrosat-google-adk-agent` |

## Los 9 Tools Geoespaciales (FunctionTool con Pydantic schema)

| Tool | Función | Schema input / output |
|------|---------|----------------------|
| `alphaearth_query` | Embedding 64-dim + clasificación XGBoost para AOI | `(roi_geojson: GeoJSON, year: int) → AlphaEarthResult` |
| `sentinel_search` | Query STAC contra catálogo pgstac | `(bbox: BBox, datetime_range: str, cloud_cover_max: float) → List[Scene]` |
| `rasterio_tool` | Estadísticas / histograma / read_window sobre COG | `(scene_id: str, operation: Literal) → dict` |
| `geopandas_intersect` | Intersección con GSAA, zonas protegidas | `(aoi: GeoJSON, layer: str) → GeoDataFrame` |
| `ndvi_calculator` | NDVI + estadísticos espaciales | `(scene_id: str, aoi: GeoJSON) → NDVIResult` |
| `timeseries_extractor` | Serie temporal por parcela | `(aoi: GeoJSON, start: date, end: date, index: str) → TimeSeries` |
| `phenology_descriptor` | Descripción fenológica (LLM con prompt 3-capas estilo Paper 2 profesor) | `(timeseries: TimeSeries) → str` |
| `dinov3_extract` | Vigor, LAI, canopy height (DINOv3 frozen) | `(aoi: GeoJSON) → dict[str, float]` |
| `crop_classifier_tool` | Invoca modelo final EPIC 6 (Gemma 4 + ensamble) | `(aoi: GeoJSON) → ClassificationResult` |

## Critical Rules

- **ALWAYS**: Cada tool es una clase `FunctionTool` con input/output schema Pydantic validado
- **ALWAYS**: Logging estructurado en cada tool: `tool_call_started` + `tool_call_finished` con duration_ms
- **ALWAYS**: El planner emite JSON validado con schema Pydantic antes de ejecutar
- **ALWAYS**: Session memory persistida en PostgreSQL via ADK SessionService
- **ALWAYS**: Streaming SSE al frontend con eventos: `plan_created`, `tool_call`, `tool_result`, `final_answer`
- **ALWAYS**: Spatial-RAG ejecuta en orden: (1) `ST_DWithin` parcelas similares (2) pgvector similitud semántica e5-mistral (3) fusión weighted score
- **ALWAYS**: Tracing built-in ADK habilitado (no observabilidad custom)
- **ALWAYS**: Latencia objetivo: p50 < 2 s simple, p95 < 5 s simple, p95 < 15 s multi-step (3-5 tool calls)
- **ALWAYS**: Citaciones obligatorias en final_answer (scene_id, fechas, tool calls)
- **NEVER**: Tool sin schema Pydantic
- **NEVER**: Permitir alucinaciones — toda cifra (hectáreas, NDVI, fechas) debe provenir de un tool call rastreable
- **NEVER**: Inferencia ML pesada en el tool — delegar a `crop_classifier_tool` que llama a un Pub/Sub worker
- **NEVER**: Hardcodear el backend LLM — usar abstracción `LLMBackend` en `ml/agent/backends.py`

## Project Structure

```
ml/agent/
├── agent.py                 # Definición principal: agrosat_agent = Agent(model=..., tools=[...], instruction=...)
├── backends.py              # LLMBackend abstracción: GeminiBackend, VLLMOpenAIBackend
├── rag.py                   # Spatial-RAG híbrido (PostGIS + pgvector)
├── memory.py                # SessionService Postgres con session_id
├── prompts/                 # System prompts multi-idioma it/es/en
├── tools/                   # Un archivo por tool con FunctionTool + schema
│   ├── alphaearth_query.py
│   ├── sentinel_search.py
│   ├── rasterio_tool.py
│   ├── geopandas_intersect.py
│   ├── ndvi_calculator.py
│   ├── timeseries_extractor.py
│   ├── phenology_descriptor.py
│   ├── dinov3_extract.py
│   └── crop_classifier_tool.py
├── eval/                    # eval_agromind.py, eval_geoanalyst.py, eval_geobenchx.py
└── tests/                   # Tests por tool con fixtures determinísticos
```

## Variantes A/B del Orquestador

| Variante | Modelo | Backend | Latencia objetivo | Cuándo usar |
|----------|--------|---------|-------------------|-------------|
| **A — Cloud** | `gemini-3.1-pro` | Vertex AI (`vertexai.preview.generative_models`) | p50 <2s | Default, 24/7, demo |
| **B — On-premise** | `Qwen/Qwen3.5-35B-A3B` (Apache 2.0) | vLLM en H100 NVL 96GB (OpenAI-compat) | p50 <2s, p95 <5s | Soberanía de datos, ventanas H100 V5/V6 |

```python
# backends.py
class LLMBackend(Protocol):
    async def chat(self, messages: list[Message], **kwargs) -> ChatResponse: ...

class GeminiBackend(LLMBackend): ...
class VLLMOpenAIBackend(LLMBackend): ...  # apunta a vllm-qwen35.internal:8000/v1
```

ADK soporta backends OpenAI-compatible, así que la abstracción es transparente.

## Spatial-RAG Híbrido

```python
# rag.py
async def spatial_rag(query: str, aoi: GeoJSON, top_k: int = 5):
    # 1. Filtrado espacial: parcelas similares geográficamente
    spatial_neighbors = await db.execute(
        select(Parcel).where(
            ST_DWithin(Parcel.geom, aoi.geometry, 5000)  # 5 km radio
        ).limit(50)
    )
    # 2. Similitud semántica: embeddings e5-mistral-7b
    query_emb = embed_with_e5(query)
    semantic_hits = await db.execute(
        select(Document).order_by(Document.embedding.cosine_distance(query_emb)).limit(top_k * 4)
    )
    # 3. Fusión weighted score
    return fuse_weighted(spatial_neighbors, semantic_hits, top_k=top_k)
```

## Evaluación

```bash
make eval-agromind variant=gemini         # subset 1000 QA pairs
make eval-agromind variant=qwen35
make eval-geoanalyst variant=gemini       # GeoAnalystBench pass rate
make eval-geobenchx variant=gemini        # GeoBenchX triangulación
```

Targets mínimos:
- AgroMind subset ≥ 0.75 (Gemini) / ≥ 0.70 (Qwen3.5)
- GeoAnalystBench pass rate ≥ 0.65
- LLM-as-judge DeepEval correctness ≥ 0.80

## QA Checklist Agente

- [ ] Cada tool con FunctionTool + schema Pydantic input/output
- [ ] Tests unitarios por tool con fixtures determinísticos (no llamadas reales a GEE/Vertex)
- [ ] Planner produce JSON validado con schema antes de ejecutar
- [ ] Memory persistido en Postgres por session_id
- [ ] Streaming SSE con los 4 eventos ADK
- [ ] Tracing built-in ADK habilitado y visible en Vertex AI console
- [ ] Spatial-RAG con PostGIS + pgvector híbrido
- [ ] Switch A/B funcional sin reiniciar el agente
- [ ] Eval AgroMind, GeoAnalystBench reportados en MLflow
- [ ] System prompts en it/es/en
- [ ] Citaciones obligatorias en final_answer
- [ ] Latencias dentro de objetivo
