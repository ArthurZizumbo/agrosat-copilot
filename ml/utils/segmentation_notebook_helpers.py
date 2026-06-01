"""Helpers DRY para los notebooks de segmentacion ``notebooks/segmentation/5*.ipynb``.

Centraliza los patrones que se repiten en ``5a_deeplabv3plus.ipynb`` y
``5b_tsvit.ipynb`` para que cada celda quede como una composicion de llamadas +
markdown + display, sin codigo inline. Espeja el patron de
:mod:`ml.utils.baseline_notebook_helpers`.

Cubre:

- :func:`run_training_or_load` — atajo skip-if-trained: si existe el ``best.pt``
  entrenado, lee sus metricas en vez de re-entrenar; si no, lanza el CLI de
  :mod:`ml.train.train_segmentation` por subprocess y parsea su log.
- :func:`training_results_table` — DataFrame Polars de uno o varios resultados.
- :func:`build_variant_comparison` — tabla base-vs-variante con delta (5b).
- :func:`segmentation_eval_table` — tabla de metricas de evaluacion por checkpoint.
- :func:`per_class_table` / :func:`per_class_comparison_table` — IoU/F1 por clase.
- :func:`plot_confusion_matrix` — matriz de confusion normalizada por fila.
- :func:`read_segmentation_lineage` — lectura robusta del lineage MLflow.
- :func:`pastis_class_names` — mapa indice de entrenamiento ``[0..17]`` -> nombre.
- Re-exports de :mod:`ml.eval.segmentation_inference`:
  :func:`load_segmentation_model`, :func:`evaluate_checkpoint`,
  :func:`predict_examples` (la logica vive alli; aqui solo se reexpone para que
  el notebook importe todo desde un unico modulo).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, SupportsFloat, cast

import polars as pl
import structlog

# Re-export: la implementacion permanece en ml.eval.segmentation_inference.
# La dependencia es unidireccional (utils -> eval), sin ciclo.
from ml.eval.segmentation_inference import (
    evaluate_checkpoint,
    load_segmentation_model,
    predict_examples,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

__all__ = [
    "TrainingResult",
    "build_variant_comparison",
    "evaluate_checkpoint",
    "load_segmentation_model",
    "pastis_class_names",
    "per_class_comparison_table",
    "per_class_table",
    "plot_confusion_matrix",
    "predict_examples",
    "read_segmentation_lineage",
    "run_training_or_load",
    "segmentation_eval_table",
    "training_results_table",
]

#: run_name MLflow por defecto segun la arquitectura (debe coincidir con el CLI).
_DEFAULT_RUN_NAMES = {
    "deeplabv3plus": "alt-deeplabv3plus-mobilenet-v1",
    "tsvit": "alt-tsvit-v1",
    "tsvit-pheno": "alt-tsvit-pheno-v1",
}

#: Subdirectorio de checkpoint segun arquitectura + target.
_CKPT_SUBDIR = {
    ("deeplabv3plus", "semantic18"): "deeplab-18",
    ("deeplabv3plus", "hcat6"): "deeplab-6",
    ("tsvit", "semantic18"): "tsvit-v1",
    ("tsvit", "hcat6"): "tsvit-v1",
    ("tsvit-pheno", "semantic18"): "tsvit-pheno-v1",
    ("tsvit-pheno", "hcat6"): "tsvit-pheno-v1",
}

#: Arquitecturas temporales (reciben ``--n-timesteps`` y serie temporal).
_TEMPORAL_KINDS = frozenset({"tsvit", "tsvit-pheno"})


@dataclass(frozen=True)
class TrainingResult:
    """Resultado de :func:`run_training_or_load` para una variante.

    Attributes:
        model: Arquitectura (``deeplabv3plus`` / ``tsvit`` / ``tsvit-pheno``).
        miou: mIoU del mejor epoch, o ``None`` si la corrida fallo.
        f1_macro: F1-macro del mejor epoch, o ``None``.
        pixel_acc: Exactitud por pixel del mejor epoch, o ``None``.
        returncode: Codigo de retorno del subprocess (``0`` si se reuso el
            checkpoint), o ``None`` si ni siquiera se lanzo.
        error: Mensaje de error en modo degradado, o ``None``.
        from_checkpoint: ``True`` si se reuso ``best.pt`` sin re-entrenar.
        best_epoch: Epoch del mejor checkpoint, o ``None``.
        cli_command: Comando CLI documentado (para mostrar en el notebook).
    """

    model: str
    miou: float | None
    f1_macro: float | None
    pixel_acc: float | None
    returncode: int | None
    error: str | None
    from_checkpoint: bool
    best_epoch: int | None
    cli_command: str


def _resolve_run_name(model_kind: str, run_name: str | None) -> str:
    return run_name or _DEFAULT_RUN_NAMES.get(model_kind, model_kind)


def _documented_cli(
    model_kind: str, target: str, run_name: str, epochs: int, batch_size: int
) -> str:
    """Construye el texto del comando CLI documentado (para display)."""
    cmd = (
        f"python -m ml.train.train_segmentation --model {model_kind} "
        f"--epochs {epochs} --batch-size {batch_size} --target {target} "
        f"--run-name {run_name}"
    )
    if model_kind in _TEMPORAL_KINDS:
        cmd += " --n-timesteps 10"
    return cmd


def run_training_or_load(
    model_kind: Literal["deeplabv3plus", "tsvit", "tsvit-pheno"],
    *,
    n_epochs: int,
    target: Literal["semantic18", "hcat6"] = "semantic18",
    run_name: str | None = None,
    checkpoint_dir: Path | str = Path("checkpoints/segmentation"),
    repo_root: Path | None = None,
    batch_size: int = 4,
    n_timesteps: int = 10,
    device: str = "auto",
    run_full: bool = False,
    documented_epochs: int | None = None,
    documented_batch_size: int | None = None,
    python_executable: str | None = None,
    on_message: Callable[[str], None] | None = None,
) -> TrainingResult:
    """Reusa el checkpoint entrenado o lanza el CLI de entrenamiento.

    Atajo skip-if-trained: si ``run_full`` es ``False`` y existe el ``best.pt``
    de la variante (``checkpoint_dir/<sub>/best.pt``), carga sus ``best_metrics``
    sin re-entrenar. En otro caso construye y ejecuta el comando CLI por
    subprocess (la corrida queda en MLflow) y parsea su log structlog buscando la
    ultima linea ``cli_done``. Modo degradado robusto: errores de subprocess,
    ``returncode != 0`` o metricas no parseables devuelven un
    :class:`TrainingResult` con campos ``None`` sin romper la ejecucion.

    Args:
        model_kind: Arquitectura a entrenar.
        n_epochs: Epochs a entrenar si se re-entrena.
        target: ``semantic18`` (18 clases) o ``hcat6`` (6 grupos HCAT).
        run_name: Nombre del run MLflow; si ``None`` usa el default por kind.
        checkpoint_dir: Raiz de checkpoints de segmentacion.
        repo_root: Raiz del repo para ``cwd`` del subprocess (default: CWD).
        batch_size: Tamano de batch del entrenamiento.
        n_timesteps: Pasos temporales (solo arquitecturas temporales).
        device: ``auto`` / ``cuda`` / ``cpu``.
        run_full: Si ``True``, ignora el atajo y fuerza el entrenamiento.
        documented_epochs: Epochs a mostrar en el comando documentado (default
            ``n_epochs``); util para reflejar la corrida real (30/15).
        documented_batch_size: Batch a mostrar en el comando documentado.
        python_executable: Interprete a usar (default ``sys.executable``).
        on_message: Callback que recibe el markdown a mostrar en el notebook.

    Returns:
        :class:`TrainingResult` con las metricas del mejor epoch.
    """
    run_name = _resolve_run_name(model_kind, run_name)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    py = python_executable or sys.executable
    doc_epochs = documented_epochs if documented_epochs is not None else n_epochs
    doc_batch = documented_batch_size if documented_batch_size is not None else batch_size
    cli_doc = _documented_cli(model_kind, target, run_name, doc_epochs, doc_batch)

    def _emit(msg: str) -> None:
        if on_message is not None:
            on_message(msg)

    # 1. Atajo skip-if-trained.
    if not run_full:
        sub = _CKPT_SUBDIR.get((model_kind, target))
        if sub is not None:
            ckpt = Path(checkpoint_dir) / sub / "best.pt"
            ckpt_abs = ckpt if ckpt.is_absolute() else repo / ckpt
            if ckpt_abs.is_file():
                import torch

                ck = torch.load(ckpt_abs, map_location="cpu", weights_only=False)
                bm = ck.get("best_metrics", {})
                _emit(
                    f"Checkpoint entrenado ya presente (`{ckpt_abs.relative_to(repo)}`, "
                    f"mejor epoch {bm.get('best_epoch')}): se omite el re-entrenamiento. "
                    f"Comando CLI documentado:\n\n`{cli_doc}`"
                )
                logger.info(
                    "training_loaded_from_checkpoint",
                    model_kind=model_kind,
                    target=target,
                    best_epoch=bm.get("best_epoch"),
                )
                return TrainingResult(
                    model=model_kind,
                    miou=float(bm["miou"]) if "miou" in bm else None,
                    f1_macro=float(bm["f1_macro"]) if "f1_macro" in bm else None,
                    pixel_acc=float(bm["pixel_acc"]) if "pixel_acc" in bm else None,
                    returncode=0,
                    error=None,
                    from_checkpoint=True,
                    best_epoch=int(bm["best_epoch"]) if "best_epoch" in bm else None,
                    cli_command=cli_doc,
                )

    # 2. Lanzar el CLI de entrenamiento.
    cmd = [
        py, "-m", "ml.train.train_segmentation",
        "--model", model_kind,
        "--epochs", str(n_epochs),
        "--batch-size", str(batch_size),
        "--target", target,
        "--device", device,
        "--run-name", run_name,
    ]
    if model_kind in _TEMPORAL_KINDS:
        cmd += ["--n-timesteps", str(n_timesteps)]
    _emit(f"`{' '.join(cmd)}`")

    miss = TrainingResult(
        model=model_kind, miou=None, f1_macro=None, pixel_acc=None,
        returncode=None, error=None, from_checkpoint=False, best_epoch=None,
        cli_command=cli_doc,
    )
    try:
        proc = subprocess.run(  # noqa: S603 - cmd se arma con literales controlados, no input externo
            cmd, cwd=str(repo), capture_output=True, text=True, check=False
        )
    except OSError as exc:
        _emit(f"> Subprocess no disponible: `{exc}`. Modo degradado.")
        logger.warning("training_subprocess_failed", model_kind=model_kind, error=str(exc))
        return TrainingResult(**{**miss.__dict__, "error": f"subprocess: {exc}"})

    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-12:])
        _emit(f"> Entrenamiento fallido (returncode={proc.returncode}).\n\n```\n{tail}\n```")
        logger.warning("training_returncode_nonzero", model_kind=model_kind, rc=proc.returncode)
        return TrainingResult(
            **{
                **miss.__dict__,
                "returncode": proc.returncode,
                "error": f"returncode={proc.returncode}",
            }
        )

    parsed = _parse_cli_done(log)
    logger.info(
        "training_completed",
        model_kind=model_kind,
        **{k: parsed.get(k) for k in ("miou", "f1_macro", "pixel_acc")},
    )
    return TrainingResult(
        model=model_kind,
        miou=parsed.get("miou"),
        f1_macro=parsed.get("f1_macro"),
        pixel_acc=parsed.get("pixel_acc"),
        returncode=0,
        error=None if parsed.get("miou") is not None else "metricas no parseables",
        from_checkpoint=False,
        best_epoch=None,
        cli_command=cli_doc,
    )


def _parse_cli_done(log: str) -> dict[str, float | None]:
    """Extrae miou/f1_macro/pixel_acc de la ultima linea ``cli_done`` del log.

    El CLI loguea con structlog (formato ``key=value``). Busca la ultima linea
    que contenga ``cli_done`` y parsea los tres tokens de metrica.

    Args:
        log: Texto combinado de stdout + stderr del subprocess.

    Returns:
        Dict con ``miou``, ``f1_macro``, ``pixel_acc`` (``None`` si no se hallan).
    """
    result: dict[str, float | None] = {"miou": None, "f1_macro": None, "pixel_acc": None}
    for line in reversed(log.splitlines()):
        if "cli_done" not in line:
            continue
        for key in result:
            token = f"{key}="
            if token in line:
                raw = line.split(token, 1)[1].split()[0].rstrip(",")
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = None
        break
    return result


def training_results_table(
    results: TrainingResult | Sequence[TrainingResult],
) -> pl.DataFrame:
    """Convierte uno o varios :class:`TrainingResult` en un DataFrame Polars.

    Args:
        results: Un resultado o secuencia de resultados.

    Returns:
        DataFrame con columnas ``model, miou, f1_macro, pixel_acc, returncode``.
    """
    rows = [results] if isinstance(results, TrainingResult) else list(results)
    return pl.DataFrame(
        [
            {
                "model": r.model,
                "miou": r.miou,
                "f1_macro": r.f1_macro,
                "pixel_acc": r.pixel_acc,
                "returncode": r.returncode,
            }
            for r in rows
        ],
        schema={
            "model": pl.Utf8,
            "miou": pl.Float64,
            "f1_macro": pl.Float64,
            "pixel_acc": pl.Float64,
            "returncode": pl.Int64,
        },
    )


def build_variant_comparison(
    results: Sequence[TrainingResult],
    *,
    baseline_model: str = "tsvit",
    variant_model: str = "tsvit-pheno",
    metrics: Sequence[str] = ("miou", "f1_macro", "pixel_acc"),
) -> pl.DataFrame | None:
    """Tabla comparativa base-vs-variante con delta por metrica.

    Args:
        results: Resultados de ambas variantes.
        baseline_model: Nombre del modelo base.
        variant_model: Nombre del modelo variante.
        metrics: Metricas a comparar.

    Returns:
        DataFrame con columnas ``metrica, <baseline>, <variant>, delta``, o
        ``None`` si falta alguno de los dos modelos o alguna metrica es ``None``.
    """
    by_model = {r.model: r for r in results}
    base = by_model.get(baseline_model)
    variant = by_model.get(variant_model)
    if base is None or variant is None:
        return None
    rows = []
    for metric in metrics:
        b = getattr(base, metric)
        v = getattr(variant, metric)
        if b is None or v is None:
            return None
        rows.append(
            {
                "metrica": metric,
                baseline_model: round(float(b), 4),
                variant_model: round(float(v), 4),
                "delta": round(float(v) - float(b), 4),
            }
        )
    return pl.DataFrame(rows)


def segmentation_eval_table(
    results: dict[str, dict[str, object]],
    *,
    label_col: str = "variante",
) -> pl.DataFrame:
    """Tabla de metricas de evaluacion de uno o varios checkpoints.

    Args:
        results: ``{nombre: metrics_dict}`` (metrics de
            :func:`ml.eval.metrics.dense_metrics_from_cm`).
        label_col: Nombre de la columna de etiqueta.

    Returns:
        DataFrame con ``label_col, mIoU, F1_macro, pixel_acc, balanced_acc,
        cohen_kappa``.
    """
    def _f(m: dict[str, object], key: str) -> float:
        return float(cast("SupportsFloat", m[key]))

    rows = [
        {
            label_col: name,
            "mIoU": round(_f(m, "miou"), 4),
            "F1_macro": round(_f(m, "f1_macro"), 4),
            "pixel_acc": round(_f(m, "pixel_acc"), 4),
            "balanced_acc": round(_f(m, "balanced_acc"), 4),
            "cohen_kappa": round(_f(m, "cohen_kappa"), 4),
        }
        for name, m in results.items()
    ]
    return pl.DataFrame(rows)


def _class_label(c: int, class_names: dict[int, str] | None) -> str:
    if class_names is not None:
        return class_names.get(c, f"clase_{c}")
    return f"grupo_{c}"


def per_class_table(
    metrics: dict[str, object],
    *,
    class_names: dict[int, str] | None = None,
    num_classes: int = 18,
    out_csv: Path | str | None = None,
) -> pl.DataFrame:
    """Tabla IoU/F1 por clase de un solo checkpoint, ordenada por IoU desc.

    Args:
        metrics: metrics_dict con ``per_class_iou`` y ``per_class_f1``.
        class_names: Mapa indice -> nombre (``None`` para ``grupo_c``).
        num_classes: Numero de clases.
        out_csv: Si se da, persiste la tabla.

    Returns:
        DataFrame con columnas ``clase, IoU, F1``.
    """
    rows = [
        {
            "clase": _class_label(c, class_names),
            "IoU": round(float(metrics["per_class_iou"][c]), 4),  # type: ignore[index]
            "F1": round(float(metrics["per_class_f1"][c]), 4),  # type: ignore[index]
        }
        for c in range(num_classes)
    ]
    df = pl.DataFrame(rows).sort("IoU", descending=True)
    if out_csv is not None:
        df.write_csv(out_csv)
    return df


def per_class_comparison_table(
    baseline_metrics: dict[str, object],
    variant_metrics: dict[str, object],
    *,
    class_names: dict[int, str] | None = None,
    num_classes: int = 18,
    out_csv: Path | str | None = None,
) -> pl.DataFrame:
    """Tabla IoU/F1 por clase de dos variantes lado a lado con delta de IoU.

    Args:
        baseline_metrics: metrics_dict de la variante base.
        variant_metrics: metrics_dict de la variante a comparar.
        class_names: Mapa indice -> nombre (``None`` para ``grupo_c``).
        num_classes: Numero de clases.
        out_csv: Si se da, persiste la tabla.

    Returns:
        DataFrame ordenado por ``delta_IoU`` desc con columnas
        ``clase, IoU_base, IoU_pheno, delta_IoU, F1_base, F1_pheno``.
    """
    rows = []
    for c in range(num_classes):
        b_iou = float(baseline_metrics["per_class_iou"][c])  # type: ignore[index]
        v_iou = float(variant_metrics["per_class_iou"][c])  # type: ignore[index]
        rows.append(
            {
                "clase": _class_label(c, class_names),
                "IoU_base": round(b_iou, 4),
                "IoU_pheno": round(v_iou, 4),
                "delta_IoU": round(v_iou - b_iou, 4),
                "F1_base": round(float(baseline_metrics["per_class_f1"][c]), 4),  # type: ignore[index]
                "F1_pheno": round(float(variant_metrics["per_class_f1"][c]), 4),  # type: ignore[index]
            }
        )
    df = pl.DataFrame(rows).sort("delta_IoU", descending=True)
    if out_csv is not None:
        df.write_csv(out_csv)
    return df


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
    title: str = "Matriz de confusion (normalizada)",
    num_classes: int | None = None,
    out_path: Path | str | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Dibuja la matriz de confusion normalizada por fila.

    Args:
        cm: Matriz de confusion ``(C, C)`` sin normalizar.
        class_names: Mapa indice -> nombre para los ticks (``None`` = sin labels).
        title: Titulo de la figura.
        num_classes: Numero de clases (default: ``cm.shape[0]``).
        out_path: Si se da, guarda la figura.
        cmap: Colormap.

    Returns:
        La figura matplotlib (la celda hace ``display(fig)`` + ``plt.close(fig)``).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = num_classes if num_classes is not None else int(cm.shape[0])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdad")
    if class_names is not None:
        labels = [class_names.get(c, str(c)) for c in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def read_segmentation_lineage(
    run_names: str | Sequence[str],
    *,
    experiment_name: str = "agrosat-segmentation",
    tracking_uri: str | None = None,
    max_results: int = 50,
) -> pl.DataFrame | None:
    """Lee el lineage MLflow de las corridas de segmentacion (modo degradado).

    Recupera los runs del experimento, filtra por ``run_names`` client-side
    (evita la fragilidad del operador ``IN`` server-side sobre tags) y devuelve
    un DataFrame con metricas best y tags de version. Devuelve ``None`` (no
    lanza) ante cualquier fallo, para preservar la ejecucion end-to-end del
    notebook en papermill/CI.

    Args:
        run_names: Nombre o lista de nombres de run a recuperar.
        experiment_name: Experimento MLflow.
        tracking_uri: URI del tracking; si ``None`` usa
            :func:`ml.utils.mlflow_utils.resolve_tracking_uri`.
        max_results: Maximo de runs a traer antes de filtrar.

    Returns:
        DataFrame con ``run_name, miou, f1_macro, pixel_acc, code_version,
        data_version`` (columnas presentes), o ``None`` en modo degradado.
    """
    wanted = {run_names} if isinstance(run_names, str) else set(run_names)
    try:
        import mlflow

        from ml.utils.mlflow_utils import resolve_tracking_uri

        mlflow.set_tracking_uri(tracking_uri or resolve_tracking_uri())
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            logger.info("lineage_experiment_absent", experiment=experiment_name)
            return None
        # search_runs con output_format="pandas" (default) devuelve un DataFrame;
        # el stub de mlflow lo tipa como list[Run], de ahi el cast.
        runs_pd = cast(
            "pd.DataFrame",
            mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=max_results,
            ),
        )
        if runs_pd.empty or "tags.mlflow.runName" not in runs_pd.columns:
            return None
        runs_pd = runs_pd[runs_pd["tags.mlflow.runName"].isin(wanted)]
        if runs_pd.empty:
            return None
        rename = {
            "tags.mlflow.runName": "run_name",
            "metrics.best_val_miou": "miou",
            "metrics.best_val_f1_macro": "f1_macro",
            "metrics.best_val_pixel_acc": "pixel_acc",
            "tags.code_version": "code_version",
            "tags.data_version": "data_version",
        }
        keep = [c for c in rename if c in runs_pd.columns]
        return pl.from_pandas(runs_pd[keep]).rename({c: rename[c] for c in keep})
    except Exception as exc:  # noqa: BLE001 - modo degradado en notebook
        logger.warning("lineage_read_failed", error=str(exc))
        return None


def pastis_class_names(num_classes: int = 18) -> dict[int, str]:
    """Mapa indice de entrenamiento ``[0..17]`` -> nombre de cultivo PASTIS.

    El dataset remapea la clase original PASTIS ``cid`` (1..18) al indice de
    entrenamiento ``cid-1`` (0..17); aqui se invierte ese offset para nombrar
    cada indice del modelo. Solo cubre ``semantic18``: para ``hcat6`` el caller
    debe pasar ``class_names=None`` a los helpers (genera ``grupo_c``).

    Args:
        num_classes: Debe ser 18 (semantic18).

    Returns:
        Dict ``{0: 'Meadow', 1: 'Soft winter wheat', ...}``.

    Raises:
        ValueError: si ``num_classes != 18`` (defensa contra mal uso en hcat6).
    """
    if num_classes != 18:
        raise ValueError(
            f"pastis_class_names solo cubre semantic18 (18 clases), recibido "
            f"{num_classes}. Para hcat6 pasa class_names=None a los helpers."
        )
    from ml.features.phenology_class_prototypes import load_class_names

    orig = load_class_names()
    return {c: orig.get(c + 1, f"clase_{c}") for c in range(18)}
