"""Homologa las 4 figuras de evaluacion de los modelos del Avance 4.

El integrador ``notebooks/segmentation/Avance4.Equipo17.ipynb`` muestra por
modelo cuatro figuras: curvas de entrenamiento, IoU por clase, matriz de
confusion y ejemplos RGB/verdad/prediccion. No todos los integrantes exportaron
las cuatro, asi que aqui se **regeneran las que faltan** a partir de los datos
crudos que si existen:

- **Curvas de entrenamiento** (``curves_<modelo>.png``): se leen del servidor
  MLflow local (Docker, experimento ``agrosat-segmentation``) que guarda
  ``train_loss`` y ``val_miou`` por epoca para los modelos us-025 (DeepLabv3+,
  TSViT). :func:`curves_from_mlflow`.
- **Matriz de confusion** (``confusion_<modelo>.png``) e **IoU por clase**
  (``per_class_iou_<modelo>.png``): requieren re-evaluar el checkpoint sobre el
  fold de validacion (inferencia). :func:`confusion_and_per_class_from_ckpt`.

Cada funcion escribe el PNG con el nombre que consume ``_find_fig`` del
integrador, de modo que ``show_model_figs`` las recoja sin tocar el notebook.

Operativo permanente (reproducible), no un script de smoke/debug.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "confusion_from_cm",
    "curves_from_mlflow",
    "optuna_convergence_figure",
    "per_class_iou_figure",
    "regen_deeplab_tsvit",
    "regen_isaac_model",
    "samples_grid",
]

#: Integrator model mapping -> MLflow run (experiment 7) with its history.
#: Each run logs ``train_loss`` and ``val_miou`` per epoch (step = epoch).
_MLFLOW_RUNS = {
    "deeplabv3plus": "alt-deeplabv3plus-mobilenet-v1",
    "tsvit": "alt-tsvit-pheno-v1",  # the phenological variant is the top-2 candidate
}


def _fetch_epoch_history(
    run_name: str, *, experiment: str, tracking_uri: str
) -> dict[str, np.ndarray]:
    """Lee las metricas por epoca de un run MLflow por nombre.

    Args:
        run_name: Nombre del run (``mlflow.runName``).
        experiment: Nombre del experimento MLflow.
        tracking_uri: URI del servidor MLflow.

    Returns:
        Dict con las series por epoca disponibles: ``train_loss``, ``val_miou``,
        ``val_f1_macro`` (las que el run haya logueado; ausentes -> array vacio).

    Raises:
        RuntimeError: si no se encuentra el run o no tiene ``train_loss``.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise RuntimeError(f"Experimento MLflow {experiment!r} no existe en {tracking_uri}.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=5,
    )
    finished = [r for r in runs if r.info.status == "FINISHED"]
    if not finished:
        raise RuntimeError(f"Run {run_name!r} (FINISHED) no encontrado en {experiment!r}.")
    run_id = finished[0].info.run_id

    def _series(metric: str) -> np.ndarray:
        hist = sorted(client.get_metric_history(run_id, metric), key=lambda m: m.step)
        return np.asarray([m.value for m in hist], dtype=np.float64)

    series = {
        "train_loss": _series("train_loss"),
        "val_miou": _series("val_miou"),
        "val_f1_macro": _series("val_f1_macro"),
    }
    if series["train_loss"].size == 0:
        raise RuntimeError(f"Run {run_name!r} sin historial `train_loss` por epoca.")
    return series


def curves_from_mlflow(
    model: str,
    *,
    out_dir: Path = Path("reports/segmentation/figures"),
    experiment: str = "agrosat-segmentation",
    tracking_uri: str = "http://localhost:5010",
    run_name: str | None = None,
) -> Path:
    """Genera la figura de curvas de entrenamiento de un modelo desde MLflow.

    Layout 1x3 (Loss | mIoU | F1-Macro) leyendo del servidor MLflow local las
    series que el run logueo por epoca (train_loss, val_miou, val_f1_macro). El
    panel mIoU y F1 marcan el mejor epoch (el del checkpoint). Escribe
    ``curves_<model>.png`` en ``out_dir`` con el nombre que consume el integrador.

    Args:
        model: Clave del modelo en el integrador (``deeplabv3plus`` / ``tsvit``).
        out_dir: Carpeta de salida de las figuras.
        experiment: Experimento MLflow.
        tracking_uri: URI del servidor MLflow (Docker local en :5010).
        run_name: Override del nombre de run; si es ``None`` usa ``_MLFLOW_RUNS``.

    Returns:
        Ruta del PNG escrito.

    Raises:
        KeyError: si ``model`` no esta en el mapeo y no se pasa ``run_name``.
    """
    name = run_name or _MLFLOW_RUNS[model]
    h = _fetch_epoch_history(name, experiment=experiment, tracking_uri=tracking_uri)
    train_loss, val_miou, val_f1 = h["train_loss"], h["val_miou"], h["val_f1_macro"]

    # Layout 1x3 (Loss | mIoU | F1-macro), aligned with the team style.
    # Our us-025 runs logged train_loss + val_miou + val_f1_macro per
    # epoch (not train_miou/val_loss), so each panel plots the series
    # actually recorded, without inventing curves.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(
        np.arange(train_loss.size), train_loss, color="#2b6cb0", marker="o", ms=3, label="Train"
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    def _val_panel(ax, series: np.ndarray, title: str, ylabel: str) -> None:
        if series.size == 0:
            ax.text(0.5, 0.5, "no registrado", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        x = np.arange(series.size)
        ax.plot(x, series, color="#dd6b20", marker="s", ms=3, label="Val")
        best = int(np.argmax(series))
        ax.axvline(best, color="#dd6b20", ls="--", lw=1, alpha=0.7)
        ax.scatter([best], [series[best]], color="#dd6b20", s=60, zorder=5, edgecolor="white")
        ax.annotate(
            f"best ep {best}\n{series[best]:.4f}",
            xy=(best, series[best]),
            xytext=(-6, -26),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="#9c4221",
        )
        ax.set_title(title)
        ax.set_xlabel("Epoca")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    _val_panel(axes[1], val_miou, "mIoU", "mIoU (val)")
    _val_panel(axes[2], val_f1, "F1-Macro", "F1-macro (val)")

    fig.suptitle(f"Curvas de entrenamiento - {model}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"curves_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "curves_written", model=model, run=name, epochs=int(train_loss.size), path=str(out_path)
    )
    return out_path


def per_class_iou_figure(
    per_class_iou: dict[int, float] | list[float] | np.ndarray,
    model: str,
    *,
    class_names: dict[int, str] | None = None,
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Genera el barplot de IoU por clase y lo escribe como ``per_class_iou_<model>.png``.

    Args:
        per_class_iou: IoU por clase (dict ``{id: iou}``, lista o array).
        model: Clave del modelo en el integrador.
        class_names: Mapa ``{id: nombre}`` para rotular el eje; ``None`` usa ``C{id}``.
        out_dir: Carpeta de salida.

    Returns:
        Ruta del PNG escrito.
    """
    if isinstance(per_class_iou, dict):
        ids = sorted(per_class_iou)
        ious = [per_class_iou[i] for i in ids]
    else:
        ious = list(per_class_iou)
        ids = list(range(len(ious)))
    labels = [(class_names.get(i, f"C{i}") if class_names else f"C{i}") for i in ids]

    fig, ax = plt.subplots(figsize=(8, max(4, len(ids) * 0.32)))
    ax.barh(labels[::-1], list(ious)[::-1], color="#2b6cb0")
    ax.set_xlabel("IoU")
    ax.set_xlim(0, 1)
    ax.set_title(f"IoU por clase - {model}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_class_iou_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("per_class_iou_written", model=model, n_classes=len(ids), path=str(out_path))
    return out_path


def confusion_from_cm(
    cm: np.ndarray,
    model: str,
    *,
    class_names: dict[int, str] | None = None,
    ignore_index: int | None = None,
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Genera la matriz de confusion (normalizada por fila) desde una cm acumulada.

    Reusa :func:`ml.eval.metrics.confusion_matrix_figure` (mismo estilo visual
    que el resto del proyecto). Escribe ``confusion_<model>.png``.

    Args:
        cm: Matriz de confusion ``(K, K)`` acumulada (filas=verdad, cols=pred).
        model: Clave del modelo en el integrador.
        class_names: Mapa ``{id: nombre}`` para rotular ejes.
        ignore_index: Si se da, descarta esa fila/columna del plot.
        out_dir: Carpeta de salida.

    Returns:
        Ruta del PNG escrito.
    """
    keep = np.ones(cm.shape[0], dtype=bool)
    if ignore_index is not None and 0 <= ignore_index < cm.shape[0]:
        keep[ignore_index] = False
    cm_k = cm[np.ix_(keep, keep)].astype(np.float64)
    ids = [i for i in range(cm.shape[0]) if keep[i]]
    labels = [(class_names.get(i, f"C{i}") if class_names else f"C{i}") for i in ids]

    # Row-wise normalization (per-class recall); rows without support stay at 0.
    row_sum = cm_k.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_k, row_sum, out=np.zeros_like(cm_k), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.5), max(5, len(ids) * 0.5)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdad")
    ax.set_title(f"Matriz de confusión - {model}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall por clase")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"confusion_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("confusion_written", model=model, k=cm_k.shape[0], path=str(out_path))
    return out_path


def regen_deeplab_tsvit(
    model: str,
    *,
    checkpoint: Path,
    num_classes: int = 18,
    ignore_index: int = 255,
    val_folds: tuple[int, ...] = (4,),
    n_timesteps: int = 10,
    device: str = "auto",
    out_dir: Path = Path("reports/segmentation/figures"),
) -> tuple[Path, Path]:
    """Regenera per_class_iou + confusion de un modelo us-025 (deeplab/tsvit).

    Reusa el flujo de los notebooks 5*: ``load_segmentation_model`` +
    ``evaluate_checkpoint`` sobre el fold de validacion.

    Args:
        model: ``"deeplabv3plus"`` o ``"tsvit"`` (clave del integrador).
        checkpoint: Ruta al ``best.pt``.
        num_classes: 18 (semantico) o 6 (HCAT).
        ignore_index: Etiqueta ignorada (255 en us-025).
        val_folds: Folds de validacion.
        n_timesteps: T para el temporal.
        device: ``auto`` / ``cuda`` / ``cpu``.
        out_dir: Carpeta de salida.

    Returns:
        Tupla ``(path_per_class, path_confusion)``.
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.eval.segmentation_inference import evaluate_checkpoint, load_segmentation_model
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    model_kind = "tsvit-pheno" if model == "tsvit" else model
    collapse = "median" if model == "deeplabv3plus" else None
    ds = PASTISSegmentationDataset(
        folds=val_folds, collapse_time=collapse, n_timesteps=n_timesteps, target="semantic18"
    )
    net = load_segmentation_model(
        checkpoint,
        model_kind=model_kind,
        num_classes=num_classes,
        n_timesteps=n_timesteps,
        device=device,
    )
    metrics, cm = evaluate_checkpoint(
        net, ds, model_kind=model_kind, num_classes=num_classes, ignore_index=ignore_index
    )
    p_iou = per_class_iou_figure(
        metrics["per_class_iou"], model, class_names=PASTIS_R_CLASSES, out_dir=out_dir
    )
    p_cm = confusion_from_cm(cm, model, class_names=PASTIS_R_CLASSES, out_dir=out_dir)
    return p_iou, p_cm


def regen_isaac_model(
    model: str,
    *,
    checkpoint: Path,
    pastis_root: Path = Path("data/PASTIS-R"),
    num_classes: int = 20,
    ignore_index: int = 19,
    val_folds: tuple[int, ...] = (4,),
    n_timesteps: int = 10,
    device: str = "auto",
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Regenera la matriz de confusion de un modelo de Isaac (utae / segformer).

    Carga el checkpoint con la arquitectura correcta (U-TAE portado o SegFormer
    HF), evalua el fold de validacion con el dataset multi-temporal (utae) o 2D
    (segformer) y escribe ``confusion_<model>.png``.

    Args:
        model: ``"utae"`` o ``"segformer"``.
        checkpoint: ``best_model.pt`` (utae) o carpeta/archivo del modelo.
        pastis_root: Raiz de PASTIS-R.
        num_classes: 20 (convencion de Isaac).
        ignore_index: 19 (void).
        val_folds: Folds de validacion.
        n_timesteps: T para utae.
        device: ``auto`` / ``cuda`` / ``cpu``.
        out_dir: Carpeta de salida.

    Returns:
        Ruta del PNG de confusion escrito.

    Raises:
        ValueError: si ``model`` no es ``utae`` ni ``segformer``.
    """
    import torch

    from ml.eval.metrics import dense_confusion_matrix
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    dev = torch.device(
        "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    if model == "utae":
        from ml.models.utae import build_utae
        from ml.tune.optuna_segmentation import PASTISMultiTempDataset

        net = build_utae(num_classes=num_classes, input_dim=10).to(dev).eval()
        ck = torch.load(checkpoint, map_location=dev, weights_only=False)
        net.load_state_dict(ck.get("model_state_dict", ck))
        ds = PASTISMultiTempDataset(
            pastis_root, list(val_folds), t_steps=n_timesteps, augment=False
        )
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                x = s["pixel_values"].unsqueeze(0).to(dev).float()
                pos = s["positions"].unsqueeze(0).to(dev)
                pred = net(x, pos).argmax(dim=1).squeeze(0).cpu().numpy()
                cm += dense_confusion_matrix(
                    pred, s["labels"].numpy(), n_classes=num_classes, ignore_index=ignore_index
                )
    elif model == "segformer":
        import json

        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from transformers import SegformerForSemanticSegmentation

        # Isaac's SegFormer (notebook 04i): 3 RGB bands (temporal median,
        # first 3 S2 bands normalized with S2_MEAN/STD), img 256px.
        seg_mean = np.array([1158.0, 1244.7, 1416.3], dtype=np.float32)[:, None, None]
        seg_std = np.array([671.7, 698.1, 761.3], dtype=np.float32)[:, None, None]
        seg_size = 256

        net = (
            SegformerForSemanticSegmentation.from_pretrained(
                str(Path(checkpoint).parent / "hf_model")
            )
            .to(dev)
            .eval()
        )
        root = Path(pastis_root)
        meta = json.loads((root / "metadata.geojson").read_text())
        pids = [
            f["properties"]["ID_PATCH"]
            for f in meta["features"]
            if f["properties"]["Fold"] in val_folds
        ]
        import torch

        with torch.no_grad():
            for pid in pids:
                s2 = np.load(root / "DATA_S2" / f"S2_{pid}.npy")  # (T, C, H, W)
                img = np.median(s2, axis=0)[:3].astype(np.float32)  # RGB composite
                img = (img - seg_mean) / (seg_std + 1e-6)
                mask = np.load(root / "ANNOTATIONS" / f"TARGET_{pid}.npy")
                if mask.ndim == 3:
                    mask = mask[0]
                t_img = (
                    TF.resize(
                        torch.from_numpy(img),
                        [seg_size, seg_size],
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                    .unsqueeze(0)
                    .to(dev)
                )
                t_mask = (
                    TF.resize(
                        torch.from_numpy(mask.astype(np.int64)).unsqueeze(0),
                        [seg_size, seg_size],
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    .squeeze(0)
                    .numpy()
                )
                logits = net(pixel_values=t_img).logits
                logits = F.interpolate(
                    logits, size=(seg_size, seg_size), mode="bilinear", align_corners=False
                )
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                cm += dense_confusion_matrix(
                    pred, t_mask, n_classes=num_classes, ignore_index=ignore_index
                )
    else:
        raise ValueError(f"model {model!r} no soportado; usa 'utae' o 'segformer'.")

    return confusion_from_cm(
        cm, model, class_names=PASTIS_R_CLASSES, ignore_index=ignore_index, out_dir=out_dir
    )


def optuna_convergence_figure(
    metrics_dir: Path = Path("reports/segmentation/metrics"),
    *,
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Grafica la convergencia de los estudios Optuna (un panel por modelo).

    Por cada ``tuning_<modelo>.parquet`` traza el mIoU de cada trial COMPLETE
    (puntos) y la curva *best-so-far* (escalonada), que muestra como Optuna fue
    encontrando mejores hiperparametros a lo largo de los trials. Los trials
    podados (PRUNED) se marcan distinto. Usa la columna ``value`` (mIoU val) o
    ``miou_grouped`` segun el esquema del parquet.

    Args:
        metrics_dir: Carpeta con los ``tuning_<modelo>.parquet``.
        out_dir: Carpeta de salida de la figura.

    Returns:
        Ruta del PNG ``optuna_convergence.png`` escrito.

    Raises:
        FileNotFoundError: si no hay ningun ``tuning_*.parquet``.
    """
    import polars as pl

    parts = sorted(metrics_dir.glob("tuning_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"Sin tuning_*.parquet en {metrics_dir}.")

    n = len(parts)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for idx, p in enumerate(parts):
        ax = axes[idx // ncols][idx % ncols]
        df = pl.read_parquet(p)
        model = df["model"][0] if "model" in df.columns else p.stem.replace("tuning_", "")
        metric_col = "value" if "value" in df.columns else "miou_grouped"
        df = df.sort("trial")
        trials = df["trial"].to_list()
        vals = df[metric_col].to_list()
        states = df["state"].to_list() if "state" in df.columns else ["COMPLETE"] * len(trials)

        comp_x = [t for t, s in zip(trials, states, strict=False) if s == "COMPLETE"]
        comp_y = [
            v for v, s in zip(vals, states, strict=False) if s == "COMPLETE" and v is not None
        ]
        pruned_x = [t for t, s in zip(trials, states, strict=False) if s == "PRUNED"]

        ax.scatter(comp_x, comp_y, color="#2b6cb0", s=28, label="trial (COMPLETE)", zorder=3)
        for px in pruned_x:
            ax.axvline(px, color="#cbd5e0", lw=0.6, alpha=0.6, zorder=1)

        # best-so-far over the COMPLETE ones in trial order.
        if comp_y:
            order = np.argsort(comp_x)
            cx = np.asarray(comp_x)[order]
            cy = np.asarray(comp_y)[order]
            best_so_far = np.maximum.accumulate(cy)
            ax.step(cx, best_so_far, where="post", color="#dd6b20", lw=1.8, label="best-so-far")
            bi = int(np.argmax(cy))
            ax.scatter([cx[bi]], [cy[bi]], color="#dd6b20", s=80, zorder=5, edgecolor="white")
            ax.annotate(
                f"mejor: {cy[bi]:.4f}",
                xy=(cx[bi], cy[bi]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#9c4221",
            )
        ax.set_title(f"{model}  ({len(comp_x)} COMPLETE, {len(pruned_x)} PRUNED)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("mIoU")
        ax.legend(fontsize=7, loc="lower right")

    # Turn off the leftover axes of the grid.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Convergencia del ajuste fino (Optuna) por modelo")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optuna_convergence.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("optuna_convergence_written", n_models=n, path=str(out_path))
    return out_path


def samples_grid(
    model: str,
    *,
    checkpoint: Path,
    num_classes: int = 18,
    ignore_index: int = 255,
    val_folds: tuple[int, ...] = (4,),
    n_timesteps: int = 10,
    n_examples: int = 4,
    device: str = "auto",
    out_dir: Path = Path("reports/segmentation/figures"),
) -> Path:
    """Grilla de ``n_examples`` parches (RGB | verdad | prediccion) + leyenda de clases.

    Para los modelos us-025 (deeplab/tsvit): carga el checkpoint, predice
    ``n_examples`` parches del fold de validacion y arma una grilla
    ``n_examples x 3`` con una leyenda de clases (nombres PASTIS) como pie de
    imagen. Escribe ``samples_<model>.png``.

    Args:
        model: ``"deeplabv3plus"`` o ``"tsvit"`` (clave del integrador).
        checkpoint: Ruta al ``best.pt``.
        num_classes: 18 (semantico) o 6 (HCAT).
        ignore_index: Etiqueta ignorada.
        val_folds: Folds de validacion.
        n_timesteps: T para el temporal.
        n_examples: Numero de parches a mostrar.
        device: ``auto`` / ``cuda`` / ``cpu``.
        out_dir: Carpeta de salida.

    Returns:
        Ruta del PNG escrito.
    """
    from matplotlib import colors
    from matplotlib.patches import Patch

    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.eval.segmentation_inference import (
        load_segmentation_model,
        predict_patch,
        rgb_from_patch,
    )
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    model_kind = "tsvit-pheno" if model == "tsvit" else model
    collapse = "median" if model == "deeplabv3plus" else None
    ds = PASTISSegmentationDataset(
        folds=val_folds, collapse_time=collapse, n_timesteps=n_timesteps, target="semantic18"
    )
    net = load_segmentation_model(
        checkpoint,
        model_kind=model_kind,
        num_classes=num_classes,
        n_timesteps=n_timesteps,
        device=device,
    )

    # Equispaced patches along the split (not the first 4 in a row).
    n = len(ds)  # type: ignore[arg-type]
    idxs = np.linspace(0, n - 1, num=min(n_examples, n), dtype=int).tolist()

    cmap = plt.get_cmap("tab20", num_classes)
    norm = colors.Normalize(vmin=0, vmax=num_classes - 1)
    rows = len(idxs)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    present: set[int] = set()
    titles = ("Entrada (RGB)", "Verdad", "Prediccion")
    for r, idx in enumerate(idxs):
        x, y = ds[idx]
        x_np = x.numpy()
        rgb = rgb_from_patch(np.median(x_np, axis=0) if x_np.ndim == 4 else x_np)
        pred = predict_patch(net, x, model_kind=model_kind)
        yt = np.where(y.numpy() == ignore_index, np.nan, y.numpy().astype(float))
        axes[r][0].imshow(np.clip(rgb, 0, 1))
        axes[r][1].imshow(yt, cmap=cmap, norm=norm, interpolation="nearest")
        axes[r][2].imshow(pred.astype(float), cmap=cmap, norm=norm, interpolation="nearest")
        for col in range(3):
            axes[r][col].axis("off")
            if r == 0:
                axes[r][col].set_title(titles[col])
        present.update(int(v) for v in np.unique(y.numpy()) if v != ignore_index)
        present.update(int(v) for v in np.unique(pred))

    # Legend of present classes (figure caption).
    handles = [
        Patch(color=cmap(norm(c)), label=f"{c}: {PASTIS_R_CLASSES.get(c, f'C{c}')}")
        for c in sorted(present)
        if c < num_classes
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"Ejemplos de prediccion - {model}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"samples_{model}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("samples_grid_written", model=model, n=len(idxs), path=str(out_path))
    return out_path
