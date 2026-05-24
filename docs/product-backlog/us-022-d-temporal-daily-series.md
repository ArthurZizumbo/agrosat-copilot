# US-022-d — Investigacion serie diaria temporal vs FFT-reconstruccion (TempCNN / InceptionTime)

**Estado**: backlog
**Origen**: US-022-c P3 FULL CUDA (2026-05-24)
**Epic**: E5 (Modelos finales + ensambles)
**Sprint**: post-Avance 5 (estimado 1-7 jun)
**SP estimado**: 5
**Owner**: Isaac (ML)
**Trigger**: cuando se planee profundizacion en clasificacion temporal vs tabular para EPIC 6 ensemble

## Contexto

US-022-c P3 (FULL CUDA 200 epochs, 85951 parcelas, 4 fixes ML aplicados) confirmo D-7
del plan: **los modelos temporales TempCNN + InceptionTime NO superan el baseline tabular
XGBoost** sobre la representacion FFT-reconstruida actual.

**Resultados ML agent 2026-05-24 (5 corridas v1..v5 RTX 4070, batch=128)**:

| Modelo | F1-macro v1 (22-may, 30 ep, bug XGB) | F1-macro v4 (24-may, 200 ep + 4 fixes) | Delta |
|--------|--------------------------------------|----------------------------------------|-------|
| XGBoost (winner set, tabular) | 0.4094 | 0.4094 (sin cambio esperado) | — |
| TempCNN (CUDA, batch 128) | 0.0448 | **0.1430-0.1456** (3 runs) | +0.10 |
| InceptionTime (CUDA, batch 128) | 0.0985 | **0.1865** | +0.09 |

Los 4 fixes ML (`class_weights`, `weighted_sampler`, `lr_scheduler`, `early_stopping`) triplicaron
TempCNN y casi duplicaron Inception, pero ambos siguen ~2.5x debajo del XGB tabular 0.4094.

## Hipotesis (paper Wen et al. 2025 + revision agente)

El bottleneck NO es el modelo temporal en si — es la representacion de entrada:

- Actualmente: vector tabular de 189 columnas por parcela (AlphaEarth 64 + indices stats 85 +
  S1 backscatter 10 + SRTM 3 + ERA5 mensual 24 + geom 3), agregado por parcela.
- Para TempCNN/InceptionTime, el codigo actual **reconstruye T=72 muestras a 5 dias via FFT
  inverse** desde los harmonicos del vector tabular. Esta sintetizacion pierde la dinamica
  fina (eventos meteorologicos abruptos, transiciones fenologicas).
- Hipotesis: con **serie diaria real** (T=365 o T=183 muestras a 2 dias) desde Sentinel-2 L2A
  + S1 GRD agregado por parcela, TempCNN/InceptionTime tendrian materia prima rica suficiente
  para superar al XGB (que opera sobre stats agregados perdiendo dinamica).

## Alcance propuesto

| Bloque | SP | Descripcion |
|--------|----|--------------:|
| D1 | 1 | Ingesta serie diaria Sentinel-2 L2A NDVI/NDWI/EVI por parcela, 365 dias 2024 (GEE export ~2-3 GB) |
| D2 | 1 | Ingesta serie diaria S1 VV/VH backscatter por parcela, 365 dias 2024 (~1 GB) |
| D3 | 1 | Fusion temporal `ml/features/daily_series.py` (Polars LazyFrame): parcela_id, day_of_year, 6 features por dia. DVC tag `daily-series-italy-v1`. |
| D4 | 1 | Refactor `ml/train/phenology_models.py` para aceptar input (B, T=365, F=6) en lugar de FFT-reconstruccion. Spatial K-fold preservado. |
| D5 | 1 | Comparacion TempCNN/InceptionTime daily vs FFT-reconstruido + XGB tabular (3 modelos en mismo split). Resultados en `docs/us-resolved/us-022-d.md`. |

## Criterios de aceptacion

| AC | Criterio | Verificacion |
|----|----------|--------------|
| AC-1 | Serie diaria S2 + S1 ingestada para subset 85951 parcelas | `data/features/daily_series_italy.parquet` shape (85951, 8) con 6 dim feature por dia + parcel_id + year |
| AC-2 | TempCNN o InceptionTime con input diario supera F1=0.4094 (baseline XGB tabular) | `model_comparison_daily.parquet` con max(f1_macro_daily) > 0.4094 |
| AC-3 | Hallazgo honesto si AC-2 falla | seccion §"Hallazgo honesto: serie diaria tampoco supera tabular" + hipotesis alternativa (pretraining, archtectura DL distinta) |
| AC-4 | DVC tag `daily-series-italy-v1` | `git tag -l 'daily-series-italy-v1'` |
| AC-5 | MLflow runs registrados con `data_version` (DVC short hash) + `code_version` (git sha) | UI MLflow con runs `tempcnn-daily-v1`, `inception-daily-v1` |
| AC-6 | Costo GEE export <= $5 USD | `gcloud billing` |

## Decisiones tecnicas

- **D-1**: Serie a T=365 (no T=183 a 2 dias) para preservar Nyquist sobre eventos fenologicos
  cortos (floracion ~10 dias). Trade-off: 4x mas tokens vs FFT-reconstruido (T=72).
- **D-2**: 6 features por dia: NDVI, NDWI, EVI (S2 L2A QA-mask Cloud Score+), VV, VH, VV-VH
  (S1 GRD interpolado lineal a daily). Sin pretraining, sin downsample.
- **D-3**: Padding con interpolacion lineal cuando S2 falta (nubosidad >70%). Mascarado en
  forward pass con padding mask para evitar leak.
- **D-4**: GPU local RTX 4070 8 GB (batch 32-64 vs 128 actual). Wall clock estimado ~2h por
  ronda full data + 200 ep + spatial K-fold 5.

## Riesgos

- R1: GEE export 365 dias * 85951 parcelas excede free tier (250k requests/dia) -> chunks de 7 dias.
- R2: TempCNN con T=365 vs T=72 puede no caber en 8 GB VRAM con batch>=32 -> reducir batch o usar grad checkpoint.
- R3: AC-2 puede seguir fallando (serie diaria igual no supera tabular). Aceptable: contribuye a
  hallazgo honesto + abre puerta a investigar pretraining (DINO time-series, MoCo temporal).

## Referencias

- US-022-c P3 resultados: [`docs/us-handoff/us-022b.md`](../us-handoff/us-022b.md) §"Resultados FULL CUDA"
- D-7 plan: [`docs/us-planning/us-022-c.md`](../us-planning/us-022-c.md) seccion 11
- Paper-faro Wen et al. 2025: [`docs/general/papers/paper.pdf`](../general/papers/paper.pdf)
- Skill agrosat-ml-features (serie temporal): [`.claude/skills/agrosat-ml-features/SKILL.md`](../../.claude/skills/agrosat-ml-features/SKILL.md)
