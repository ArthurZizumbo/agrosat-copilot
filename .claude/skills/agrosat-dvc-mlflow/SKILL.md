---
name: agrosat-dvc-mlflow
description: Version data with DVC 3.48+ (GCS remote) and track experiments with MLflow 2.16 for AgroSatCopilot reproducibility. Use when adding datasets to DVC, decorating training functions with @track_experiment, or accessing the MLflow Model Registry.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot DVC + MLflow Skill

## Rules — NON-NEGOTIABLE

- Datasets >10 MB versionados con DVC (`dvc add data/raw/...`)
- DVC remote: `gs://agrosat-dvc-remote` con service account
- Cada experiment run tagged: `data_version` (hash DVC) + `code_version` (git sha)
- Artefactos (checkpoints, plots, configs) loggeados al MLflow Artifact Store
- Model Registry: `agrosat-baseline`, `agrosat-segmentation-{u-net,...}`, `agrosat-final-gemma4`, `agrosat-ensemble-final`

## DVC Setup

```bash
# Init (una vez)
dvc init
dvc remote add -d gcs-remote gs://agrosat-dvc-remote
dvc remote modify gcs-remote credentialpath .env/gcp-service-account.json

# Versionar dataset
dvc add data/raw/pastis/
git add data/raw/pastis.dvc .gitignore
git commit -m "data(E1): track PASTIS-R dataset"
dvc push
```

## MLflow Setup

```python
# ml/utils/mlflow_utils.py
import mlflow
import functools
import subprocess
from pathlib import Path

def track_experiment(experiment_name: str):
    """Decorator: logs data_version + code_version automáticamente."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                try:
                    data_version = subprocess.check_output(
                        ["dvc", "status", "--json"], text=True
                    ).strip()
                except Exception:
                    data_version = "unknown"
                code_version = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
                mlflow.set_tag("data_version", data_version)
                mlflow.set_tag("code_version", code_version)
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Uso en Training

```python
@track_experiment("baseline-xgb-alphaearth")
def train(...):
    mlflow.log_params({"lr": 0.05, "max_depth": 6})
    # ... train ...
    mlflow.log_metric("f1_macro", 0.72)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.xgboost.log_model(model, "model", registered_model_name="agrosat-baseline")
```

## Model Registry

```python
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="agrosat-baseline",
    version=3,
    stage="Production",
)
```

## Comandos

```bash
make dvc-push                # dvc push
make dvc-pull                # dvc pull
make mlflow-ui               # MLflow UI :5000
make mlflow-server           # mlflow server con backend Postgres + artifact GCS
```
