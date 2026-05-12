---
name: agrosat-evidently-drift
description: Monitor data drift on Sentinel-2 bands and AlphaEarth embeddings with Evidently AI, generating weekly HTML reports for AgroSatCopilot. Use for EPIC 10 US-047 drift pipeline as Dagster asset triggered weekly.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Evidently Drift Skill

## Rules — NON-NEGOTIABLE

- Reportes HTML semanales en `gs://agrosat-artifacts/drift/{week}/report.html`
- Reference set: ingest del año base 2022
- Current set: ingest del trimestre actual
- Métricas: PSI (Population Stability Index), KS test, Wasserstein distance
- Alerta cuando drift PSI >0.2 en cualquier banda o dimensión embedding
- Disparador: schedule Dagster `0 6 * * 1` (lunes 6am)

## Pipeline

```python
# ml/eval/drift.py
import polars as pl
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def generate_drift_report(reference: pl.DataFrame, current: pl.DataFrame, out_html: str):
    """Reporta drift sobre bandas Sentinel-2 y embeddings AlphaEarth."""
    ref_pd = reference.to_pandas()
    cur_pd = current.to_pandas()
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref_pd, current_data=cur_pd)
    report.save_html(out_html)

    # Extraer métricas accionables
    drift_summary = report.as_dict()
    return drift_summary
```

## Dagster Asset

```python
# dagster_project/assets/drift.py
@asset(
    deps=[alphaearth_annual, sentinel2_scenes],
    automation_condition=AutomationCondition.on_cron("0 6 * * 1"),  # Mon 6am
)
def drift_check(context, gcs: GCSResource):
    week = context.partition_key or datetime.now().isocalendar().week
    ref = pl.read_parquet("gs://agrosat-data/reference/2022_baseline.parquet")
    cur = pl.read_parquet(f"gs://agrosat-data/current/{week}.parquet")
    out_html = f"/tmp/drift_{week}.html"
    summary = generate_drift_report(ref, cur, out_html)

    remote_path = f"drift/{week}/report.html"
    gcs.upload(out_html, remote_path, bucket="agrosat-artifacts")

    if summary["overall_drift_share"] > 0.2:
        send_alert(f"Drift detected (week {week}): {summary['overall_drift_share']}")

    return MaterializeResult(
        metadata={
            "report_url": MetadataValue.url(f"https://storage.googleapis.com/agrosat-artifacts/{remote_path}"),
            "drift_share": MetadataValue.float(summary["overall_drift_share"]),
        }
    )
```

## Métricas Reportadas

| Métrica | Threshold alerta |
|---------|------------------|
| PSI (Population Stability Index) | > 0.2 |
| Wasserstein distance per band | > 0.3 |
| KS test p-value | < 0.01 |
| Embedding cosine drift (AlphaEarth 64-dim) | > 0.15 promedio |

## QA Checklist Drift

- [ ] Reference set bien definido (año base)
- [ ] Schedule semanal Dagster activado
- [ ] Reportes HTML accesibles vía URL pública firmada
- [ ] Alertas si PSI > 0.2
- [ ] Histórico de reportes en `gs://agrosat-artifacts/drift/`
