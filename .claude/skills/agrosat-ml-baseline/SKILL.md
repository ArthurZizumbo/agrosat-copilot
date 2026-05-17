---
name: agrosat-ml-baseline
description: Train tabular baseline classifiers (XGBoost, LightGBM, Random Forest) on AlphaEarth 64-dim embeddings + spectral indices + DINOv3 features for AgroSatCopilot crop classification. Use when training, tuning or evaluating tabular baselines for crop classification.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot ML Baseline Skill

## Rules — NON-NEGOTIABLE

- AlphaEarth embedding 64-dim como feature principal (Apache 2.0, gratis vía GEE)
- Concatenar índices espectrales + DINOv3 1024-dim opcionalmente
- **Spatial CV** con bloques disjuntos por geometría (no random)
- Stratified por clase Y por región (Pianura/Toscana/Apulia/PASTIS)
- Calibración isotonic o sigmoid sobre probas
- Target F1-macro ≥ 0.60 para considerarse aceptable
- Registrar todo en MLflow (params, metrics, plots, model)

## Pipeline Baseline

```python
import polars as pl
import xgboost as xgb
import mlflow
from sklearn.metrics import f1_score, classification_report
from ml.utils.spatial_cv import SpatialBlockCV
from ml.utils.mlflow_utils import track_experiment

@track_experiment("baseline-xgb-alphaearth")
def train_baseline_xgb(features_path: str, target_col: str = "crop_class"):
    df = pl.read_parquet(features_path)
    X = df.drop([target_col, "parcel_id", "year", "geometry"]).to_numpy()
    y = df[target_col].to_numpy()

    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_samples", X.shape[0])

    cv = SpatialBlockCV(n_splits=5, buffer_m=1000)
    f1_scores = []

    for fold, (tr, va) in enumerate(cv.split(X, y, geom=df["geometry"])):
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", device="cuda",
        )
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        pred = model.predict(X[va])
        f1 = f1_score(y[va], pred, average="macro")
        f1_scores.append(f1)
        mlflow.log_metric(f"f1_macro_fold_{fold}", f1)

    mlflow.log_metric("f1_macro_cv_mean", np.mean(f1_scores))
    mlflow.log_metric("f1_macro_cv_std", np.std(f1_scores))
    mlflow.xgboost.log_model(model, "model")
    return model
```

## Spatial Block CV

```python
# ml/utils/spatial_cv.py
import numpy as np
import geopandas as gpd
from sklearn.model_selection import KFold

class SpatialBlockCV:
    def __init__(self, n_splits=5, buffer_m=1000, block_size_m=5000):
        self.n_splits = n_splits
        self.buffer_m = buffer_m
        self.block_size_m = block_size_m

    def split(self, X, y, geom):
        gdf = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326").to_crs("EPSG:32633")
        bounds = gdf.total_bounds
        x_blocks = int((bounds[2] - bounds[0]) / self.block_size_m) + 1
        y_blocks = int((bounds[3] - bounds[1]) / self.block_size_m) + 1
        block_id = (
            ((gdf.geometry.centroid.x - bounds[0]) // self.block_size_m).astype(int) * y_blocks +
            ((gdf.geometry.centroid.y - bounds[1]) // self.block_size_m).astype(int)
        )
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        unique_blocks = np.unique(block_id)
        for tr_b, va_b in kf.split(unique_blocks):
            tr = np.where(np.isin(block_id, unique_blocks[tr_b]))[0]
            va = np.where(np.isin(block_id, unique_blocks[va_b]))[0]
            yield tr, va
```

## Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)
```

## MLflow Helper

```python
# ml/utils/mlflow_utils.py
import mlflow, subprocess, functools

def track_experiment(experiment_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                data_version = subprocess.check_output(["dvc", "rev-parse"], text=True).strip()
                code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                mlflow.set_tag("data_version", data_version)
                mlflow.set_tag("code_version", code_version)
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

## QA Checklist

- [ ] Spatial CV (no random)
- [ ] Stratified por clase + región
- [ ] F1-macro ≥ 0.60 mínimo
- [ ] Calibración isotonic
- [ ] MLflow run con data_version + code_version
- [ ] Matriz confusión + classification_report logged
