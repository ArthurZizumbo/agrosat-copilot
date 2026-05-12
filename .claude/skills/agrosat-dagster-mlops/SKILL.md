---
name: agrosat-dagster-mlops
description: Build Dagster 1.9+ asset-oriented pipelines for AgroSatCopilot with declarative lineage between datasets, features, and models, integrated with DVC versioning and MLflow tracking. Use when defining assets, jobs, schedules, or sensors.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Dagster MLOps Skill

## Rules — NON-NEGOTIABLE

- Cada asset con `@asset` y `deps=[...]` explícitas
- Resources externos en `dagster_project/resources/`
- RetryPolicy en assets que tocan APIs externas (GEE, CDSE)
- Particionado temporal con `StaticPartitionsDefinition([2017..2025])`
- `MaterializeResult` con metadata útil
- DVC versionado del output cuando aplique

## Assets Principales

```python
# dagster_project/assets/alphaearth.py
from dagster import asset, RetryPolicy, MaterializeResult, MetadataValue, StaticPartitionsDefinition

years_partition = StaticPartitionsDefinition([str(y) for y in range(2017, 2026)])

@asset(
    partitions_def=years_partition,
    retry_policy=RetryPolicy(max_retries=3, delay=60.0),
)
def alphaearth_annual(context, gee: GEEResource, postgres: PostgresResource):
    year = int(context.partition_key)
    rois = load_rois_config()
    results = []
    for roi in rois:
        task_id = gee.export_alphaearth(roi, year)
        results.append({"roi": roi["name"], "year": year, "task_id": task_id})
        postgres.upsert_alphaearth_tile(roi["name"], year, task_id)
    return MaterializeResult(
        metadata={"num_exports": MetadataValue.int(len(results))}
    )
```

```python
# dagster_project/assets/models.py
@asset(deps=[parcel_features])
def baseline_model(context, mlflow: MLflowResource):
    from ml.train.train_baseline import train_baseline_xgb
    model = train_baseline_xgb("data/processed/parcel_features.parquet")
    return MaterializeResult(
        metadata={
            "mlflow_run_id": MetadataValue.text(mlflow.active_run().info.run_id),
            "f1_macro": MetadataValue.float(model.f1_score),
        }
    )
```

## Resources

```python
# dagster_project/resources/gee.py
from dagster import ConfigurableResource, EnvVar
import ee

class GEEResource(ConfigurableResource):
    service_account_path: str
    project_id: str

    def setup_for_execution(self, context):
        creds = ee.ServiceAccountCredentials(None, self.service_account_path)
        ee.Initialize(creds, project=self.project_id)

    def export_alphaearth(self, roi, year):
        ...

# dagster_project/resources/mlflow.py
class MLflowResource(ConfigurableResource):
    tracking_uri: str = EnvVar("MLFLOW_TRACKING_URI")

    def setup_for_execution(self, context):
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow = mlflow

    def active_run(self):
        return self._mlflow.active_run()
```

## Definitions

```python
# dagster_project/definitions.py
from dagster import Definitions
from .assets import alphaearth_annual, sentinel2_scenes, dinov3_features, \
                    spectral_indices, parcel_features, \
                    baseline_model, alt_models, final_vlm, ensemble, drift_check
from .resources import gee, mlflow, postgres, gcs

defs = Definitions(
    assets=[
        alphaearth_annual, sentinel2_scenes, dinov3_features,
        spectral_indices, parcel_features,
        baseline_model, alt_models, final_vlm, ensemble, drift_check,
    ],
    resources={
        "gee": gee,
        "mlflow": mlflow,
        "postgres": postgres,
        "gcs": gcs,
    },
)
```

## Schedules

```python
# dagster_project/schedules/__init__.py
from dagster import ScheduleDefinition, define_asset_job

ingest_job = define_asset_job("ingest_daily", selection=["alphaearth_annual", "sentinel2_scenes"])
drift_job = define_asset_job("drift_weekly", selection=["drift_check"])

ingest_schedule = ScheduleDefinition(job=ingest_job, cron_schedule="0 3 * * *")
drift_schedule = ScheduleDefinition(job=drift_job, cron_schedule="0 6 * * 1")
```

## Comandos

```bash
make dagster-ui                          # UI puerto 3001
dagster dev -m dagster_project.definitions
dagster asset materialize -m dagster_project.definitions --select alphaearth_annual --partition 2025
dagster asset list -m dagster_project.definitions
```
