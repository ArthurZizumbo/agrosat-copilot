# Resumen del Plan AgroSatCopilot v6

> Destilado operativo del [plan completo v6](RefinamientoPlaneacionAgroSatCopilot_v6.md) (2723 líneas → 200). Mantiene lo accionable: alcance, EPICs, calendario, presupuesto, decisiones técnicas, riesgos, métricas de éxito. **Para detalle de cada US, abrir el plan completo en la sección correspondiente.**

---

## 1. Identidad

**AgroSatCopilot** es un SaaS conversacional open-source que permite a agrónomos consultar imágenes satelitales en italiano/español/inglés con razonamiento trazable y switch A/B entre Gemini 3.1 Pro cloud y Qwen3.5-35B-A3B on-prem.

- **Curso**: MNA Tec de Monterrey · 20-abr → 3-jul-2026.
- **Capacidad**: 3 devs × 12 h/sem × 10 sem ≈ **150 SP**.
- **Regiones piloto**: Pianura Padana, Toscana, Apulia (Italia) + PASTIS-R Francia (control).

## 2. Diferenciadores (vs Google Earth AI)

| Eje | Google Earth AI | AgroSatCopilot |
|-----|-----------------|----------------|
| LLM orquestador | Gemini closed | Gemini 3.1 Pro **+ Qwen3.5-35B-A3B switch A/B** |
| VLM | RS-FM Google interno | **Gemma 4 26B-MoE fine-tune agrícola (Apache 2.0)** |
| Idioma nativo | Inglés | **Italiano + Español + Inglés** |
| Despliegue | Solo GCP | GCP + Azure H100 + on-prem |
| Benchmark agrícola IT/ES | No existe | **AgroMind-IT/ES (Zenodo DOI)** |
| Costo operativo | Alto | **~$115 USD/mes con scale-to-zero** |
| Licencia | Propietaria | Apache 2.0 / MIT |

## 3. Stack ML (decisiones irrevocables — ver §A.1-A.7 del v6)

| Capa | Modelo | Notas |
|------|--------|-------|
| FM EO | AlphaEarth v2.1 | GEE gratis · 64-dim · 10 m/píxel · NO entrenar FM propio |
| Feature self-sup | DINOv3-satellite | `facebook/dinov3-vitl16-pretrain-sat493m` · frozen |
| VLM principal | Gemma 4 26B-MoE | Apache 2.0 · 256K ctx · LoRA rank 16 BF16 · ~82 GB H100 |
| VLM comparativo | Qwen3-VL-30B-A3B | Tabla comparativa Gemma 4 vs Qwen3-VL (paper opcional) |
| LLM cloud | Gemini 3.1 Pro | Vertex AI · 2M ctx · $2/$12 por 1M tokens |
| LLM on-prem | Qwen3.5-35B-A3B | vLLM en H100 · 128K ctx · Apache 2.0 · sin `-Instruct` |
| Framework agente | Google ADK | Tracing built-in + Vertex AI Agent Engine + OpenAI-compat |

**Modelos descartados (no reactivar)**: Prithvi-EO-2.0, MiniMax-M2.7 (115 GB no cabe), Kimi K2.6 (1T no cabe), Llama 3.3-70B QLoRA, LangGraph, Prefect, Alembic, DuckDB principal, PWA+Tauri.

## 4. Arquitecturas obligatorias por rúbrica

- **EPIC 5 — 6 segmentación**: U-Net ResNet-50 · DeepLabv3+ MobileNetV3 · SegFormer-B2 (con cabezal FarSLIP) · U-TAE · **TSViT (Paper 1)** · Swin-UNETR.
- **EPIC 6 — 4 ensambles**: Voting top-3 · Bagging XGB+AlphaEarth · Stacking heterogéneo (+ Gemma 4 base, meta XGB) · Blending Optuna.

## 5. Épicas (12) y Story Points

| # | Épica | Avance | Sprint | SP |
|---|-------|--------|--------|-----|
| E0 | Infra + Cookiecutter + MLOps (Dagster + dbmate) | A0 | S1 | 10 |
| E1 | Ingesta (AlphaEarth, Sentinel, DINOv3, pgstac) | A0-A1 | S1-S2 | 12 |
| E2 | EDA Polars (univariado/bivariado/temporal) | A1 | S2-S3 | 14 |
| E3 | Feature Eng + 17 índices + FarSLIP | A2 | S3-S4 | 18 |
| E4 | Baseline AlphaEarth + XGBoost/RF | A3 | S4-S5 | 10 |
| E5 | 6 arquitecturas segmentación | A4 | S5-S6 | 21 |
| E6 | Gemma 4 26B-MoE LoRA + 4 ensambles | A5 | S6 | 20 |
| E7 | Agente Google ADK (Gemini + Qwen3.5) | A5-A6 | S7 | 14 |
| E8 | Backend FastAPI + Pub/Sub + TiTiler | A6 | S7 | 9 |
| E9 | Frontend Nuxt 4 + switch A/B | A6 | S8 | 10 |
| E10 | Observabilidad + Drift + FinOps + Seguridad | A6-A7 | S8 | 8 |
| **MVP** | **E0-E10** | **A0-A7 + Pres** | **9 sem** | **146** |
| E11 | Paper Track (post-presentación, opcional) | — | S10-S11 | 28 |

**Stretch (sacrificables si hay atraso)**: US-045 Switch A/B UI, US-047 Evidently auto, US-040 Pub/Sub. Decisión equipo: mantener los 3 en MVP.

## 6. Secuenciación semanal

```
S1  (20-26 abr): E0 Setup + Avance 0           → Avance 0 dom 26-abr
S2  (27-abr a 3-may): E1 Ingesta + E2 EDA       → Avance 1 dom 3-may
S3  (4-10 may):  E2 + arrancar E3 FE
S4  (11-17 may): E3 FE + arrancar E4 baseline  → Avance 2 dom 17-may
S5  (18-24 may): E4 + E5 modelos 1-3            → Avance 3 mié 20, Avance 4 dom 24
S6  (25-31 may): E5 modelos 4-6 + ensambles + Gemma 4 LoRA → Avance 5 dom 31-may
S7  (1-7 jun):   E6 VLM + E7 ADK + E8 backend  → Avance 6 dom 7-jun
S8  (8-14 jun):  E9 frontend + E10 + Avance 7  → Avance 7 dom 14-jun
S9  (15-21 jun): Pulido + dry-runs + demo       → Presentación dom 21-jun
S10-S11 (22-jun a 3-jul): Buffer + Paper Track
```

## 7. Presupuesto H100 (80 h en 6 ventanas)

| Ventana | Fecha | Horas | Uso | VRAM |
|---------|-------|-------|-----|------|
| V1 | 18-20 may | 8 h | Baselines + preliminar TSViT | ~40 GB |
| V2 | 25-27 may | 12 h | U-TAE + TSViT + Swin-UNETR | ~60 GB |
| V3 | 28-30 may | 24 h | **Gemma 4 26B-MoE LoRA** (3 epochs AgroMind+IT/ES) | ~82 GB |
| V4 | 1-3 jun | 12 h | Qwen3-VL LoRA comparación + ensambles | ~92 GB |
| V5 | 5-7 jun | 16 h | Qwen3.5-35B-A3B vLLM serving + LoRA traces | ~91 GB |
| V6 | 18-20 jun | 8 h | Warm vLLM para demo presentación | ~91 GB |

**Costos**: training único **$262 spot — $602 on-demand USD**. Operativo mensual **~$115 USD** con scale-to-zero. L4 GCP spot: ~50 h para baselines, DINOv3 extraction, Gemma 4 E4B fallback.

## 8. Riesgos top-5 (de 14 totales — ver §13 del v6)

| ID | Riesgo | Prob/Imp | Mitigación |
|----|--------|----------|------------|
| R01 | Ventanas H100 (80 h) insuficientes | M/A | Plan B: Gemma 4 E4B QLoRA 4-bit L4 + Qwen3.5-9B fallback |
| R02 | Alucinaciones agente en respuestas factuales | A/A | Tool-calling + Spatial-RAG + citaciones + ADK tracing + LLM-as-judge |
| R03 | Sprint 6 sobrecomprometido (30 SP vs 15 cap) | A/A | Gemma 4 overnight + ensambles paralelos + reporte preliminar 24-may |
| R04 | Validación nativa AgroMind-IT/ES | A/M | Semilla Gemini + reviewer Scuola Sant'Anna + reviewer ES equipo |
| R05 | Cuota Azure H100 spot agotada | M/M | Gate día 4 S1 + fallback on-demand o GCP A100 |

Categorías rúbrica Avance 7: **datos** (R01,R03,R05-R08,R13), **ataques** (R14), **confianza** (R02,R04,R12), **cumplimiento** (R09-R11).

## 9. Métricas de éxito MVP

**Técnicas**: Baseline F1-macro ≥ 0.60 · Modelo final F1-macro ≥ 0.80 (Gemma 4 LoRA + ensambles) · mIoU ≥ 0.70 segmentación densa · Latencia chat p95 < 3 s simple, < 15 s multi-step · AgroMind ≥ 0.70 (Qwen3.5) / ≥ 0.75 (Gemini) · Cobertura ≥ 70 % backend, ≥ 50 % frontend · Cero alertas drift Evidently en semana previa a presentación.

**Producto**: dibujar polígono → respuesta < 10 s · overlays NDVI/NDWI/AlphaEarth interactivos · ≥ 10 tipos preguntas it/es/en · switch A/B funcional · video demo 3 min grabado.

**MLOps**: 100 % datasets DVC · 100 % experimentos MLflow · 100 % pipelines Dagster · 100 % commits a main desplegables · 100 % infra en Terraform · reproducibilidad GCP + Azure.

## 10. Papers de referencia

| # | Paper | Aporte | Aplicación |
|---|-------|--------|------------|
| 1 | **TSViT** (Tarasiou, arXiv:2301.04944, 2023) — Vision Transformers for SITS | Factorización temporal + espacial, SOTA en PASTIS | EPIC 5 US-026 obligatorio rúbrica |
| 2 | **Phenology Description** (Wen, ISPRS 2025) | CLIP + ViT + GPT-4 prompt 3-layer + GCN + LoRA contrastivo | EPIC 7 tool `phenology_descriptor` (Gemma 4 sustituye GPT-4) |
| 3 | Be My Eyes (2025) | Multi-agent perceiver-reasoner | EPIC 7 arquitectura del agente |
| 4 | FarSLIP (2025) | CLIP fine-grained RS adaptation | EPIC 5 US-024 SegFormer + cabezal FarSLIP |

## 11. Datasets y licencias

- **AlphaEarth Foundations v2.1** (GEE, Google DeepMind 2025, atribución).
- **Sentinel-2 L2A / Sentinel-1 SAR** (Copernicus CDSE, free).
- **PASTIS-R** (Francia, control benchmark).
- **GSAA Italia / LPIS** (administrativo agrícola europeo).
- **Dynamic World** (Google, labels ruidosos).
- **AgroMind** (28482 Q&A, eval).
- **AgroMind-IT/ES** (500 pares, **contribución original equipo**, Zenodo CC-BY-4.0).

Todas las atribuciones en [`docs/licenses/DATA_LICENSE.md`](../docs/licenses/DATA_LICENSE.md).

## 12. Gates Sprint 1 (20-26 abr)

| Día | Gate | Owner |
|-----|------|-------|
| Lun 20 | Verificar HF IDs: Gemma 4 26B-MoE, Gemma 4 E4B, Qwen3.5-35B-A3B, Qwen3-VL-30B-A3B, DINOv3 | Isaac |
| Mar 21 | Vertex AI Gemini 3.1 Pro accesible service account | Arthur |
| Mié 22 | Export dummy AlphaEarth 100 km² Toscana | Isaac |
| Jue 23 | Sponsor: confirmar región Azure H100 + booking spot + hello-world | Arthur |
| Vie 24 | Google ADK tutorial con FunctionTool dummy (Gemini + OpenAI-compat mock) | Aaron |
| Sáb 25 | Dagster + MLflow hello-world con asset Parquet dummy | Arthur |
| **Dom 26** | **Entrega Avance 0 PDF** | Equipo |

## 13. Apéndices y secciones clave del v6 (para navegación)

| Tema | Sección v6 | Líneas |
|------|------------|--------|
| Resumen ejecutivo y flujo de uso | §1 | 67-122 |
| Diferenciadores vs Earth AI | §2 | 123-162 |
| Antecedentes académicos (4 papers) | §3 | 163-232 |
| Estado del arte 2025-2026 | §4 | 233-330 |
| Google Earth AI (referencia industrial) | §5 | 331-360 |
| Stack tecnológico detallado | §6 | 361-446 |
| Arquitectura del sistema | §7 | 447-547 |
| FinOps (costos training + operativo) | §8 | 548-590 |
| Datasets y licenciamiento | §9 | 591-635 |
| Mapa épicas + SP + secuenciación | §10 | 636-686 |
| **EPIC 0-11 detallados (US-001 a US-055)** | §EPIC | 688-2280 |
| Roadmap sprints semanales | §11 | 2281-2438 |
| Gestión de riesgos (14) | §13 | 2440-2476 |
| Criterios éxito MVP | §14 | 2479-2515 |
| Alineación rúbricas curso | §15 | 2518-2556 |
| Decisiones técnicas clave (A.1-A.12) | §16 | 2559-2664 |
| Catálogo preguntas copiloto | Anexo A | 2667-2681 |
| Glosario IT/ES/EN | Anexo B | 2684-2716 |

---

**Mantenedor**: Arthur Zizumbo. **Plan completo**: [RefinamientoPlaneacionAgroSatCopilot_v6.md](RefinamientoPlaneacionAgroSatCopilot_v6.md).
