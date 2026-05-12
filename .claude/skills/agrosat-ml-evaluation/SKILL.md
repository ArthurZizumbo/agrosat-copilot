---
name: agrosat-ml-evaluation
description: Evaluate models against benchmarks for AgroSatCopilot — AgroMind (28482 QA pairs), GeoAnalystBench, GeoBenchX, GEO-Bench-2, AgroMind-IT/ES (own benchmark), with LLM-as-judge using DeepEval. Generate interpreted plots (confusion matrix, ROC, PR, spatial residuals, UMAP).
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot ML Evaluation Skill

## Rules — NON-NEGOTIABLE

- Métricas técnicas: mIoU, F1-macro, accuracy, Cohen kappa
- Benchmarks agente: AgroMind subset (1000 pares), GeoAnalystBench, GeoBenchX
- LLM-as-judge con DeepEval (correctness, faithfulness, relevance)
- Plots interpretados: matriz confusión + ROC + PR + residuos espaciales + UMAP
- Reportado en MLflow + paper/figures/
- Targets mínimos: F1-macro ≥0.80 final, mIoU ≥0.70, AgroMind ≥0.75 Gemini, ≥0.70 Qwen3.5

## Métricas Técnicas

```python
from sklearn.metrics import (
    f1_score, accuracy_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
from torchmetrics.classification import MulticlassJaccardIndex

def evaluate_segmentation(y_true, y_pred, num_classes):
    return {
        "f1_macro": f1_score(y_true.flatten(), y_pred.flatten(), average="macro"),
        "f1_weighted": f1_score(y_true.flatten(), y_pred.flatten(), average="weighted"),
        "accuracy": accuracy_score(y_true.flatten(), y_pred.flatten()),
        "kappa": cohen_kappa_score(y_true.flatten(), y_pred.flatten()),
        "miou": MulticlassJaccardIndex(num_classes=num_classes)(
            torch.tensor(y_pred), torch.tensor(y_true)
        ).item(),
    }
```

## Plots Interpretados

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Verdadero")
    return fig

def plot_spatial_residuals(gdf, y_true, y_pred):
    gdf["error"] = (y_true != y_pred).astype(int)
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(column="error", cmap="RdYlGn_r", ax=ax, legend=True)
    return fig
```

## AgroMind Benchmark

```python
from datasets import load_dataset
import deepeval
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

def eval_agromind(agent, subset_size=1000):
    ds = load_dataset("AgroMind/AgroMind", split="test")[:subset_size]
    metrics = {"correct": 0, "relevancy": [], "faithfulness": []}

    for sample in ds:
        response = agent.run(query=sample["question"], image=sample["image"])
        # exact match (rough proxy)
        if normalize(response) == normalize(sample["answer"]):
            metrics["correct"] += 1
        # LLM-as-judge
        test_case = LLMTestCase(input=sample["question"], actual_output=response, expected_output=sample["answer"])
        rel = AnswerRelevancyMetric(threshold=0.7).measure(test_case)
        fai = FaithfulnessMetric(threshold=0.7).measure(test_case)
        metrics["relevancy"].append(rel)
        metrics["faithfulness"].append(fai)

    return {
        "accuracy": metrics["correct"] / len(ds),
        "relevancy_mean": np.mean(metrics["relevancy"]),
        "faithfulness_mean": np.mean(metrics["faithfulness"]),
    }
```

## GeoAnalystBench

```python
def eval_geoanalyst(agent):
    """GeoAnalystBench pass rate sobre tareas analíticas geoespaciales."""
    # 280 tareas oficiales; cargar desde HF
    ds = load_dataset("GeoAnalystBench/v1", split="test")
    pass_count = 0
    for task in ds:
        result = agent.run(task["query"], context=task["context"])
        if validate_against_groundtruth(result, task["groundtruth"]):
            pass_count += 1
    return {"pass_rate": pass_count / len(ds)}
```

## AgroMind-IT/ES (propio)

```python
# Construir desde semilla sintética con Gemini 3.1 Pro
# Validar con reviewer nativo italiano (Scuola Sant'Anna) y reviewer ES equipo
# Publicar en Zenodo CC-BY-4.0 con DOI

def build_agromind_it_es(seed_pairs=500, seed_query_examples=...):
    pairs = []
    for ex in seed_query_examples:
        for lang in ["it", "es"]:
            translated = gemini.translate(ex.question, target_lang=lang)
            pairs.append({"question": translated, "answer_lang": lang, ...})
    return pairs
```

## Targets Mínimos

| Métrica | Target | Modelo |
|---------|--------|--------|
| F1-macro segmentación | ≥ 0.80 | Gemma 4 + ensamble |
| mIoU | ≥ 0.70 | Ensamble final |
| AgroMind subset | ≥ 0.75 | Variante Gemini |
| AgroMind subset | ≥ 0.70 | Variante Qwen3.5 |
| GeoAnalystBench pass rate | ≥ 0.65 | Cualquier variante |
| Latencia chat p95 | < 5 s simple, < 15 s multi-step | Sistema |

## QA Checklist Eval

- [ ] mIoU + F1-macro + accuracy + kappa
- [ ] 5 plots interpretados con párrafo cada uno
- [ ] AgroMind subset evaluado en ambas variantes
- [ ] GeoAnalystBench evaluado
- [ ] LLM-as-judge con DeepEval
- [ ] MLflow tags `benchmark` + `variant`
- [ ] Targets alcanzados o desviación documentada
