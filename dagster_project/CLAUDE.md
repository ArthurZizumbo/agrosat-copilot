# Dagster Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root para trabajo en Dagster.

**Rol**: Orquestación asset-oriented (Dagster 1.9+) con lineage declarativo entre datasets, features y modelos. Integra DVC para versionado de datos y MLflow para tracking de experimentos.

## Skills References

- [agrosat-dagster-mlops](../.claude/skills/agrosat-dagster-mlops/SKILL.md) — Assets, jobs, schedules
- [agrosat-dvc-mlflow](../.claude/skills/agrosat-dvc-mlflow/SKILL.md) — Versionado y tracking
- [agrosat-gee-alphaearth](../.claude/skills/agrosat-gee-alphaearth/SKILL.md) — Asset ingesta AlphaEarth
- [agrosat-evidently-drift](../.claude/skills/agrosat-evidently-drift/SKILL.md) — Asset drift_check semanal

## Auto-Invoke

| Acción | Skill |
|--------|-------|
| Definir asset Dagster | `agrosat-dagster-mlops` |
| Asset de ingesta AlphaEarth | `agrosat-gee-alphaearth` + `agrosat-dagster-mlops` |
| Asset Sentinel-2 vía CDSE | `agrosat-dagster-mlops` |
| Asset training de modelo | `agrosat-dagster-mlops` + `agrosat-dvc-mlflow` |
| Asset drift Evidently | `agrosat-evidently-drift` |
| Schedule semanal | `agrosat-dagster-mlops` |

## Critical Rules

- **ALWAYS**: Cada asset es una función decorada con `@asset` con `deps=[...]` explícitas
- **ALWAYS**: Resources externos (GCS, Earth Engine, MLflow) declarados en `dagster_project/resources/`
- **ALWAYS**: Reintentos con `RetryPolicy(max_retries=3, delay=2.0)` para llamadas a GEE/CDSE
- **ALWAYS**: Asset materialization devuelve `MaterializeResult` con metadata útil (size, rows, hash)
- **ALWAYS**: Particionado temporal por año (`StaticPartitionsDefinition([2022, 2023, 2024, 2025])`)
- **ALWAYS**: Cada asset que produce data versiona output en DVC (`subprocess.run(['dvc', 'add', ...])`)
- **NEVER**: Lógica de negocio en assets — delegar a funciones de `ml/` o `backend/`
- **NEVER**: Hardcodear credenciales — usar `EnvVar` o resources

## Project Structure

```
dagster_project/
├── __init__.py
├── definitions.py           # Definitions(assets=[...], resources={...}, schedules=[...])
├── assets/                  # Un archivo por dominio
│   ├── alphaearth.py        # alphaearth_annual
│   ├── sentinel.py          # sentinel2_scenes, sentinel1_grd
│   ├── dinov3.py            # dinov3_features
│   ├── features.py          # spectral_indices, parcel_features
│   ├── models.py            # baseline_model, alt_models, final_vlm, ensemble
│   ├── drift.py             # drift_check (Evidently)
│   └── stac.py              # pgstac_catalog
├── resources/               # GEE, GCS, MLflow, Postgres, Azure Blob clients
├── jobs/                    # Selecciones de assets a ejecutar juntos
└── schedules/               # daily_ingest, weekly_drift, on_demand_train
```

## Lineage Esperado

```
alphaearth_annual ─┐
                   ├─→ spectral_indices ─→ parcel_features ─┐
sentinel2_scenes ──┤                                          ├─→ baseline_model
                   └─→ dinov3_features ────────────────────────┤
                                                                ├─→ alt_models ─→ final_vlm ─→ ensemble
pastis_r ──────────────────────────────────────────────────────┘                                        │
                                                                                                          ▼
                                                                                            drift_check (weekly)
```

## Comandos

```bash
make dagster-ui                          # Dagster UI puerto 3001
dagster asset materialize -m dagster_project.definitions --select alphaearth_annual
dagster asset list -m dagster_project.definitions
dagster dev -m dagster_project.definitions
```

## QA Checklist Dagster

- [ ] Cada asset con `deps=[...]` explícitas (no implícitas)
- [ ] Resources externos en `resources/`, no inline
- [ ] RetryPolicy en assets que tocan APIs externas
- [ ] Particionado temporal donde aplique
- [ ] MaterializeResult con metadata
- [ ] DVC versionado del output cuando aplique
- [ ] Schedule definida si el asset es recurrente
- [ ] Lineage visible en Dagster UI
- [ ] Tests unitarios con `materialize([asset], resources=mock_resources)`
