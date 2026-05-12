---
name: agrosat-ml-ensemble
description: Build 4 ensemble strategies for AgroSatCopilot final model (EPIC 6, Avance 5) — Voting (homogeneous top-3), Bagging (XGB AlphaEarth), Stacking (U-TAE + TSViT + Swin-UNETR + XGB + Gemma 4), Blending (Optuna-optimized weights). Use for final model selection.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot ML Ensemble Skill

## Rules — NON-NEGOTIABLE

- 4 ensambles obligatorios por rúbrica Avance 5
- Out-of-fold predictions con spatial CV (no leakage)
- Base learners de EPIC 5 + baseline XGB + Gemma 4 (EPIC 6)
- Optuna para optimizar pesos blending
- MLflow run por ensamble con tag `ensemble_type`
- Comparar contra mejor modelo individual

## Los 4 Ensambles

```python
# ml/ensemble/voting.py
import numpy as np

def voting_homogeneous(probs_list: list[np.ndarray]) -> np.ndarray:
    """Majority vote sobre top-3 modelos temporales (U-TAE + TSViT + Swin-UNETR)."""
    return np.mean(probs_list, axis=0)  # promedio de probas → argmax
```

```python
# ml/ensemble/bagging.py
from sklearn.utils import resample
import xgboost as xgb

def bagging_xgb_alphaearth(X, y, n_bags=10):
    """10 XGB sobre bootstraps del training set."""
    models = []
    for i in range(n_bags):
        Xb, yb = resample(X, y, random_state=i)
        m = xgb.XGBClassifier(n_estimators=500, max_depth=6, objective="multi:softprob")
        m.fit(Xb, yb)
        models.append(m)
    return models

def bagging_predict(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)
```

```python
# ml/ensemble/stacking.py
from sklearn.model_selection import KFold

def stacking_heterogeneous(base_learners, meta_learner, X, y, X_test):
    """U-TAE + TSViT + Swin-UNETR + XGB + Gemma 4 → meta XGB."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X), len(base_learners) * num_classes))
    for i, (tr, va) in enumerate(kf.split(X)):
        for j, learner in enumerate(base_learners):
            learner.fit(X[tr], y[tr])
            oof[va, j*num_classes:(j+1)*num_classes] = learner.predict_proba(X[va])
    meta_learner.fit(oof, y)
    # Para test, refit base learners en train completo
    test_probs = np.concatenate([l.fit(X, y).predict_proba(X_test) for l in base_learners], axis=1)
    return meta_learner.predict(test_probs)
```

```python
# ml/ensemble/blending.py
import optuna

def blending_optuna(probs_list: list[np.ndarray], y_val) -> tuple[list[float], np.ndarray]:
    """Pesos optimizados minimizando gap F1 train-val."""
    n_models = len(probs_list)

    def objective(trial):
        weights = np.array([trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)])
        weights = weights / weights.sum()
        blended = sum(w * p for w, p in zip(weights, probs_list))
        return f1_score(y_val, blended.argmax(1), average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    weights = np.array([study.best_params[f"w_{i}"] for i in range(n_models)])
    weights = weights / weights.sum()
    return weights.tolist(), sum(w * p for w, p in zip(weights, probs_list))
```

## Tabla Comparativa Final

```python
import polars as pl

results = pl.DataFrame({
    "model": ["XGB baseline", "U-TAE", "TSViT", "Swin-UNETR", "Gemma 4 LoRA",
              "Voting top-3", "Bagging XGB", "Stacking", "Blending"],
    "f1_macro": [...],
    "mIoU": [...],
    "inference_ms": [...],
    "params_M": [...],
})
results.write_parquet("paper/figures/ensemble_comparison.parquet")
```

## QA Checklist Ensemble

- [ ] 4 ensambles implementados
- [ ] Out-of-fold predictions con spatial CV
- [ ] Optuna ≥200 trials para blending
- [ ] MLflow tag `ensemble_type`
- [ ] Tabla comparativa vs mejor individual
- [ ] Selección final justificada con trade-offs
