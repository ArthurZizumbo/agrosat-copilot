"""Modelos temporales fenologicos: TempCNN + InceptionTime (US-022b-C).

Wrapper Polars-in / PyTorch para entrenar las arquitecturas oficiales del
benchmark BreizhCrops (Russwurm et al. 2020) sobre la curva NDVI/NDWI/EVI
diaria interpolada de cada parcela PASTIS-R. Reusa :func:`build_spatial_kfold`
y :func:`compute_baseline_metrics` para que los resultados sean comparables
con el baseline tabular cerrado en US-022 (commit ``87b7c57``, F1-macro 0.32).

Decisiones canonicas (plan ``docs/us-planning/us-022b.md`` §6.1 + ADR-006
D-ARQ-2 actualizado):

- **D-ARQ-2 (actualizado 2026-05-22)**: TempCNN e InceptionTime se PORTAN
  nativos al repo en :mod:`ml.models.temporal` (basados en Pelletier 2019 +
  Fawaz 2020, licencia MIT). El wrapper adapta I/O (DataFrame Polars ->
  tensor ``(B, T, C)``), construye los modelos via
  :func:`ml.models.temporal.build_temporal_model`, registra MLflow y
  resuelve el device priorizando CUDA.
- **CV espacial obligatoria** (no random): reusa
  :func:`ml.train.baseline._build_cv_splits` (con cache).
- **Spatial CV 5-fold**, mismas particiones que el baseline (gracias al cache
  por ``n_rows + k + buffer + seed``).
- **MLflow tags**: ``data_version`` (hash DVC) + ``code_version`` (git sha)
  siempre que ``mlflow_uri`` se pase y el run se abra; si la libreria mlflow
  no esta disponible o el URI es ``None``, el wrapper degrada a "sin
  tracking" sin fallar (testabilidad CPU CI).
- **Arquitecturas ligeras**: ADR-006 D3 confirma L4 24 GB como objetivo;
  Wen et al. 2025 entrenaron variantes mas pesadas en RTX 3090. CPU smoke
  para 2 batches funciona para tests.

Referencias agronomicas / arquitectura
--------------------------------------
- Pelletier, Webb & Petitjean 2019 — TempCNN. DOI 10.3390/rs11050523.
- Fawaz et al. 2020 — InceptionTime. DOI 10.1007/s10618-020-00710-y.
- Russwurm et al. 2020 — BreizhCrops dataset + benchmark.
  DOI 10.1109/IGARSS39084.2020.9324249.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

from ml.eval.metrics import compute_baseline_metrics
from ml.train.baseline import _DROP_CLASS_IDS, _build_cv_splits

logger = structlog.get_logger(__name__)

__all__ = [
    "TemporalDataset",
    "TemporalModelKind",
    "TemporalModelResult",
    "build_temporal_tensor",
    "train_temporal_model",
]


#: Supported temporal models (both live in :mod:`ml.models.temporal`).
TemporalModelKind = Literal["tempcnn", "inceptiontime"]

#: Canonical indices used as C channels of the time series (same as
#: :data:`ml.features.temporal_features.DEFAULT_FFT_INDICES`).
DEFAULT_TEMPORAL_INDICES: tuple[str, ...] = ("NDVI", "NDWI", "EVI")

#: Number of FFT harmonics (4 amps + 4 phases per index) present in the
#: US-018 subset. The tensor fed to TempCNN/InceptionTime reconstructs a
#: synthetic daily series from the FFT representation via partial inverse;
#: alternatively it accepts a pre-materialized daily curve.
DEFAULT_FFT_HARMONICS: int = 3

#: Default temporal length of the series (one agronomic year, T=72 ~5d
#: cadence; balance between cost and seasonal resolution).
DEFAULT_SEQUENCE_LENGTH: int = 72


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalModelResult:
    """Resultado del entrenamiento de un modelo temporal con spatial CV.

    Attributes:
        model_kind: ``"tempcnn"`` o ``"inceptiontime"``.
        f1_macro: F1-macro out-of-fold del spatial CV.
        f1_weighted: F1 ponderado por soporte.
        miou: Mean IoU (Jaccard macro) — proxy a nivel parcela.
        cohen_kappa: Indice de acuerdo Cohen.
        train_time_s: Wall-clock total del entrenamiento (suma de folds).
        n_parcels: Numero de parcelas efectivas tras filtrado por clases
            no agronomicas (``_DROP_CLASS_IDS`` heredado del baseline).
        n_classes: Numero de clases efectivas tras el filtrado.
        mlflow_run_id: ID del run MLflow si se registro, ``None`` si no.
    """

    model_kind: TemporalModelKind
    f1_macro: float
    f1_weighted: float
    miou: float
    cohen_kappa: float
    train_time_s: float
    n_parcels: int
    n_classes: int
    mlflow_run_id: str | None
    y_true_oof: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    y_pred_oof: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    checkpoint_path: Path | None = None


# ---------------------------------------------------------------------------
# Polars -> tensor adapter.
# ---------------------------------------------------------------------------


class TemporalDataset:
    """Adaptador minimal Polars -> tensor PyTorch para series temporales.

    Convierte un :class:`polars.DataFrame` de features fenologicas en un
    tensor ``(B, T, C)`` listo para ``forward`` de TempCNN / InceptionTime.

    Estrategia de reconstruccion de la serie:

    - Si el DataFrame contiene columnas pre-materializadas
      ``{idx}_t_{i:02d}`` (i in [0, T)) las usa directamente.
    - Si no, reconstruye una pseudo-curva diaria a partir de los coeficientes
      FFT ``{idx}_fft_amp_k`` y ``{idx}_fft_phase_k`` (decomp inversa
      truncada al numero de armonicos disponibles). Es una representacion
      compacta y agronomicamente fiel: 1 DC + 3 armonicos reconstruyen
      la senal estacional dominante.

    Args:
        df: DataFrame Polars con features temporales (sale de
            :func:`ml.features.temporal_features.extract_temporal_features`
            o del subset US-018).
        indices: Canales C a usar (default ``("NDVI", "NDWI", "EVI")``).
        sequence_length: T (longitud de la serie); default 72.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        indices: tuple[str, ...] = DEFAULT_TEMPORAL_INDICES,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        self.df = df
        self.indices = indices
        self.sequence_length = sequence_length

    def to_tensor(self) -> np.ndarray:
        """Devuelve la matriz ``(B, T, C)`` como ``np.ndarray`` float32.

        Es responsabilidad del caller envolverla en ``torch.from_numpy(...)``.
        """
        n = self.df.height
        out = np.zeros((n, self.sequence_length, len(self.indices)), dtype=np.float32)
        for c_idx, idx in enumerate(self.indices):
            curve = _reconstruct_curve(
                self.df, index_name=idx, sequence_length=self.sequence_length
            )
            out[:, :, c_idx] = curve
        return out


def build_temporal_tensor(
    df: pl.DataFrame,
    *,
    indices: tuple[str, ...] = DEFAULT_TEMPORAL_INDICES,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Atajo funcional sobre :class:`TemporalDataset`.

    Args:
        df: DataFrame Polars con features temporales.
        indices: Canales C (default NDVI/NDWI/EVI).
        sequence_length: T (default 72).

    Returns:
        ``np.ndarray`` shape ``(n_rows, sequence_length, len(indices))``
        en ``float32``.
    """
    return TemporalDataset(df, indices=indices, sequence_length=sequence_length).to_tensor()


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def train_temporal_model(
    features_path: Path | str | None = None,
    *,
    df: pl.DataFrame | None = None,
    model_kind: TemporalModelKind,
    n_epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    mlflow_uri: str | None = None,
    indices: tuple[str, ...] = DEFAULT_TEMPORAL_INDICES,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
    use_class_weights: bool = True,
    use_weighted_sampler: bool = True,
    use_lr_scheduler: bool = True,
    early_stopping_patience: int = 20,
    val_fraction: float = 0.15,
    dropout: float | None = None,
    warmup_epochs: int = 5,
) -> TemporalModelResult:
    """Entrena TempCNN o InceptionTime sobre la FE temporal con spatial CV.

    Args:
        features_path: Ruta al parquet de features fenologicas (US-018 /
            US-015). Si ``df`` se pasa, se ignora.
        df: DataFrame Polars ya cargado (atajo para tests/notebooks).
        model_kind: ``"tempcnn"`` o ``"inceptiontime"``.
        n_epochs: Numero de epocas por fold (default 30).
        batch_size: Tamano de batch (default 256). En CPU dev se baja a 64.
        learning_rate: LR del optimizador Adam (default 1e-3).
        seed: Semilla determinista (``np.random.default_rng``, ``torch.manual_seed``).
        device: ``"cuda"``, ``"cpu"`` o ``None`` (autodetecta). En CI sin GPU
            se fuerza ``"cpu"``.
        mlflow_uri: Si no es ``None``, se intenta abrir un run MLflow. Si la
            libreria mlflow no esta instalada o el URI no responde, degrada
            a "sin tracking" con un warning.
        indices: Canales C (default ``("NDVI", "NDWI", "EVI")``).
        sequence_length: T (default 72).
        k_folds: Numero de folds del CV espacial (default 5).
        buffer_km: Buffer anti-leakage en km (default 1.0).
        max_samples: Subsample uniforme determinista. ``None`` = dataset
            completo.
        checkpoint_dir: Si se pasa, persiste el ``state_dict`` del modelo del
            ultimo fold en disco con metadata embebida.
        use_class_weights: Si ``True`` (default) pondera el loss inversamente
            a la frecuencia de cada clase para abordar el desbalance ~31x del
            subset US-018. Formula: ``w_k = N_total / (N_classes * N_k)``.
        use_weighted_sampler: Si ``True`` (default) usa
            ``WeightedRandomSampler`` para que cada batch vea proporcionalmente
            todas las clases. Crucial para F1-macro con desbalance fuerte.
        use_lr_scheduler: Si ``True`` (default) aplica warmup linear
            (``warmup_epochs`` epocas) + ``CosineAnnealingLR`` para el resto
            del entrenamiento. Estabiliza la convergencia con datasets grandes.
        early_stopping_patience: Epocas sin mejora en val F1-macro antes de
            detener el fold. ``0`` = sin early stopping. Default 20.
        val_fraction: Fraccion del train del fold que se reserva para
            validacion intra-fold (early stopping + best epoch). Default 0.15.
            Stratified por clase para no perder clases minoritarias.
        dropout: Override del dropout del modelo. ``None`` = default del paper
            (0.5 TempCNN, 0.2 InceptionTime). Para series cortas (T=72) bajar
            a 0.2-0.3 ayuda.
        warmup_epochs: Epocas de warmup linear del LR (de 0 a ``learning_rate``)
            antes de activar el cosine decay. Default 5.

    Returns:
        Un :class:`TemporalModelResult` con metricas out-of-fold y metadata.

    Raises:
        ImportError: si ``torch`` no esta instalado.
        ValueError: si ni ``features_path`` ni ``df`` se pasan, o si
            ``model_kind`` no es soportado.
    """
    if model_kind not in ("tempcnn", "inceptiontime"):
        raise ValueError(
            f"`model_kind` debe ser 'tempcnn' o 'inceptiontime'; recibido {model_kind!r}."
        )
    if df is None:
        if features_path is None:
            raise ValueError("Debes pasar `features_path` o `df`.")
        df = pl.read_parquet(Path(features_path))

    clean_df = _prepare_temporal_dataframe(df)
    if max_samples is not None and max_samples > 0 and clean_df.height > max_samples:
        clean_df = clean_df.sample(n=max_samples, seed=seed, with_replacement=False)
        logger.info("temporal_subsampled", max_samples=max_samples, n=clean_df.height)

    label_encoder, y_encoded = _encode_labels(clean_df)
    n_classes = len(label_encoder)

    # Build the tensor (B, T, C) and the spatial splits (shared cache).
    X = build_temporal_tensor(clean_df, indices=indices, sequence_length=sequence_length)
    cv_splits = _build_cv_splits(clean_df, k_folds=k_folds, buffer_km=buffer_km, random_state=seed)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch no esta instalado. Ejecuta `poetry install --with ml,ml-gpu` "
            "o `poetry install --with ml` para entrenar modelos temporales."
        ) from exc

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)

    mlflow_run_id: str | None = None
    mlflow_ctx = _try_mlflow_run(mlflow_uri, model_kind=model_kind)

    per_fold_metrics: list[dict[str, float]] = []
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    with mlflow_ctx as run_ctx:
        mlflow_run_id = run_ctx.run_id if run_ctx is not None else None
        if run_ctx is not None:
            run_ctx.log_params(
                {
                    "model_kind": model_kind,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "k_folds": k_folds,
                    "buffer_km": buffer_km,
                    "sequence_length": sequence_length,
                    "indices": ",".join(indices),
                    "device": resolved_device,
                    "n_parcels": clean_df.height,
                    "n_classes": n_classes,
                    "use_class_weights": use_class_weights,
                    "use_weighted_sampler": use_weighted_sampler,
                    "use_lr_scheduler": use_lr_scheduler,
                    "early_stopping_patience": early_stopping_patience,
                    "val_fraction": val_fraction,
                    "warmup_epochs": warmup_epochs,
                    "dropout_override": dropout if dropout is not None else "default",
                }
            )

        # Global class weights (same for all folds for consistency;
        # computed over the full dataset).
        class_weights_t: Any = None
        if use_class_weights:
            counts = np.bincount(y_encoded, minlength=n_classes).astype(np.float64)
            counts = np.where(counts > 0, counts, 1.0)
            weights = float(y_encoded.size) / (float(n_classes) * counts)
            class_weights_t = torch.from_numpy(weights.astype(np.float32)).to(resolved_device)
            logger.info(
                "temporal_class_weights",
                min_weight=float(weights.min()),
                max_weight=float(weights.max()),
                imbalance_ratio=float(counts.max() / counts.min()),
            )

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            if train_idx.size == 0 or test_idx.size == 0:
                logger.warning("temporal_cv_fold_skipped", fold=fold_idx)
                continue

            # Intra-fold split train -> (train_inner, val_inner) stratified
            # by class for early stopping. If a class has only 1 sample it
            # goes to train_inner (it cannot be used for validation).
            rng = np.random.default_rng(seed + fold_idx)
            train_inner_idx, val_inner_idx = _stratified_inner_split(
                y_encoded[train_idx],
                val_fraction=val_fraction,
                rng=rng,
            )
            train_inner_global = train_idx[train_inner_idx]
            val_inner_global = train_idx[val_inner_idx]

            x_train = X[train_inner_global]
            x_val = X[val_inner_global]
            x_test = X[test_idx]
            y_train = y_encoded[train_inner_global]
            y_val = y_encoded[val_inner_global]
            y_test = y_encoded[test_idx]

            model_kwargs: dict[str, Any] = {}
            if dropout is not None:
                model_kwargs["dropout"] = dropout
            model = _build_temporal_model_native(
                model_kind=model_kind,
                input_dim=len(indices),
                num_classes=n_classes,
                sequence_length=sequence_length,
                device=resolved_device,
                **model_kwargs,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)

            x_train_t = torch.from_numpy(x_train).to(resolved_device)
            y_train_t = torch.from_numpy(y_train).long().to(resolved_device)
            x_val_t = torch.from_numpy(x_val).to(resolved_device)
            x_test_t = torch.from_numpy(x_test).to(resolved_device)

            # WeightedRandomSampler: probability inverse to each class
            # frequency. Each batch sees all classes proportionally.
            n_train = x_train_t.shape[0]
            if use_weighted_sampler:
                fold_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
                fold_counts = np.where(fold_counts > 0, fold_counts, 1.0)
                sample_weights = 1.0 / fold_counts[y_train]
                sample_weights_t = torch.from_numpy(sample_weights.astype(np.float64)).to(
                    resolved_device
                )
            else:
                sample_weights_t = None

            # LR scheduler: linear warmup + cosine annealing for the rest.
            scheduler = None
            if use_lr_scheduler and n_epochs > warmup_epochs:
                cosine_epochs = max(1, n_epochs - warmup_epochs)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs, eta_min=learning_rate * 0.01
                )

            best_val_f1 = -1.0
            best_state_dict: dict[str, Any] | None = None
            epochs_since_improve = 0

            for epoch in range(n_epochs):
                # Manual warmup: LR rises linearly from ~0 to learning_rate.
                if use_lr_scheduler and epoch < warmup_epochs:
                    warm_lr = learning_rate * (epoch + 1) / max(1, warmup_epochs)
                    for pg in optimizer.param_groups:
                        pg["lr"] = warm_lr

                model.train()
                epoch_loss = 0.0
                n_batches = 0
                # Sample selection: either weighted sampler or uniform permutation.
                if use_weighted_sampler and sample_weights_t is not None:
                    # As many samples as n_train (with replacement, by design
                    # of WeightedRandomSampler).
                    indices_iter = torch.multinomial(sample_weights_t, n_train, replacement=True)
                else:
                    indices_iter = torch.randperm(n_train, device=resolved_device)

                for batch_start in range(0, n_train, batch_size):
                    sel = indices_iter[batch_start : batch_start + batch_size]
                    if sel.numel() < 2:
                        continue
                    optimizer.zero_grad()
                    logits = model(x_train_t[sel])
                    loss = criterion(logits, y_train_t[sel])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
                avg_loss = epoch_loss / max(1, n_batches)

                # Cosine annealing after warmup.
                if scheduler is not None and epoch >= warmup_epochs:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

                # Intra-fold validation: F1-macro for early stopping.
                model.eval()
                with torch.no_grad():
                    val_logits = model(x_val_t)
                    val_pred = val_logits.argmax(dim=-1).cpu().numpy()
                val_metrics = compute_baseline_metrics(
                    y_val, val_pred, labels=list(range(n_classes))
                )
                val_f1 = float(val_metrics["f1_macro"])

                if run_ctx is not None:
                    run_ctx.log_metric(f"fold{fold_idx}_train_loss", avg_loss, step=epoch)
                    run_ctx.log_metric(f"fold{fold_idx}_val_f1_macro", val_f1, step=epoch)
                    run_ctx.log_metric(f"fold{fold_idx}_lr", current_lr, step=epoch)

                # Early stopping: keep the best state_dict by val F1-macro.
                if val_f1 > best_val_f1 + 1e-6:
                    best_val_f1 = val_f1
                    best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1

                if early_stopping_patience > 0 and epochs_since_improve >= early_stopping_patience:
                    logger.info(
                        "temporal_early_stop",
                        fold=fold_idx,
                        epoch=epoch + 1,
                        best_val_f1=round(best_val_f1, 4),
                    )
                    break

            # Load the best fold checkpoint before evaluating on test.
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            model.eval()
            with torch.no_grad():
                y_pred = model(x_test_t).argmax(dim=-1).cpu().numpy()
            fold_metrics = compute_baseline_metrics(y_test, y_pred, labels=list(range(n_classes)))
            per_fold_metrics.append(fold_metrics)
            y_true_chunks.append(y_test)
            y_pred_chunks.append(np.asarray(y_pred))
            logger.info(
                "temporal_fold_done",
                model_kind=model_kind,
                fold=f"{fold_idx + 1}/{len(cv_splits)}",
                f1_macro=round(fold_metrics["f1_macro"], 4),
            )

        y_true_oof = (
            np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
        )
        y_pred_oof = (
            np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
        )
        if y_true_oof.size > 0:
            oof_metrics = compute_baseline_metrics(
                y_true_oof, y_pred_oof, labels=list(range(n_classes))
            )
        else:
            oof_metrics = {
                "f1_macro": float("nan"),
                "f1_weighted": float("nan"),
                "miou": float("nan"),
                "accuracy": float("nan"),
                "cohen_kappa": float("nan"),
            }
        train_time_s = time.perf_counter() - t0
        if run_ctx is not None:
            run_ctx.log_metric("oof_f1_macro", oof_metrics["f1_macro"])
            run_ctx.log_metric("oof_f1_weighted", oof_metrics["f1_weighted"])
            run_ctx.log_metric("oof_miou", oof_metrics["miou"])
            run_ctx.log_metric("oof_cohen_kappa", oof_metrics["cohen_kappa"])
            run_ctx.log_metric("train_time_s", train_time_s)
            # Persist the last fold model's state_dict as an artifact.
            # Allows later reload with torch.load(...) for inference.
            if hasattr(run_ctx, "log_state_dict"):
                try:
                    run_ctx.log_state_dict(model, name=f"{model_kind}_last_fold.pt")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("mlflow_state_dict_save_failed", error=str(exc))

    # On-disk persistence of the state_dict (independent of MLflow).
    # Allows reloading the model with torch.load(...) without retraining.
    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        try:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            code_v = _resolve_code_version()[:7]
            f1_str = f"{float(oof_metrics['f1_macro']):.4f}".replace(".", "p")
            checkpoint_path = ckpt_dir / f"{model_kind}_{code_v}_f1_{f1_str}_seed{seed}.pt"
            torch.save(
                {
                    "model_kind": model_kind,
                    "state_dict": model.state_dict(),
                    "input_dim": len(indices),
                    "num_classes": n_classes,
                    "sequence_length": sequence_length,
                    "indices": list(indices),
                    "label_encoder": label_encoder,
                    "f1_macro": float(oof_metrics["f1_macro"]),
                    "miou": float(oof_metrics["miou"]),
                    "n_parcels": int(clean_df.height),
                    "seed": seed,
                    "code_version": _resolve_code_version(),
                    "data_version": _resolve_data_version(),
                    "mlflow_run_id": mlflow_run_id,
                },
                checkpoint_path,
            )
            logger.info(
                "temporal_checkpoint_saved",
                path=str(checkpoint_path),
                f1_macro=round(float(oof_metrics["f1_macro"]), 4),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "temporal_checkpoint_save_failed",
                checkpoint_dir=str(checkpoint_dir),
                error=str(exc),
            )

    result = TemporalModelResult(
        model_kind=model_kind,
        f1_macro=float(oof_metrics["f1_macro"]),
        f1_weighted=float(oof_metrics["f1_weighted"]),
        miou=float(oof_metrics["miou"]),
        cohen_kappa=float(oof_metrics["cohen_kappa"]),
        train_time_s=float(train_time_s),
        n_parcels=int(clean_df.height),
        n_classes=n_classes,
        mlflow_run_id=mlflow_run_id,
        y_true_oof=y_true_oof.astype(np.int64, copy=False),
        y_pred_oof=y_pred_oof.astype(np.int64, copy=False),
        checkpoint_path=checkpoint_path,
    )
    logger.info(
        "temporal_train_done",
        **{
            "model_kind": model_kind,
            "f1_macro": round(result.f1_macro, 4),
            "n_parcels": result.n_parcels,
            "n_classes": result.n_classes,
            "train_time_s": round(result.train_time_s, 2),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _prepare_temporal_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Filtra clases no agronomicas (parche del baseline)."""
    if "class_id" not in df.columns:
        raise ValueError("`df` debe contener la columna `class_id`.")
    clean = df.filter(
        pl.col("class_id").is_not_null() & ~pl.col("class_id").is_in(list(_DROP_CLASS_IDS))
    )
    if clean.height == 0:
        raise ValueError("Tras filtrar clases no agronomicas el DataFrame quedo vacio.")
    return clean


def _encode_labels(df: pl.DataFrame) -> tuple[list[int], np.ndarray]:
    """Re-mapea `class_id` a etiquetas contiguas ``[0, n_classes)``."""
    raw = df.get_column("class_id").to_numpy().astype(np.int64)
    unique_classes = sorted(int(c) for c in np.unique(raw).tolist())
    mapping = {c: i for i, c in enumerate(unique_classes)}
    y_encoded = np.array([mapping[int(v)] for v in raw], dtype=np.int64)
    return unique_classes, y_encoded


def _stratified_inner_split(
    y: np.ndarray,
    *,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split estratificado por clase del train de un fold para early stopping.

    Garantiza que cada clase con >= 2 muestras tenga al menos una en val.
    Las clases con 1 sola muestra van enteras al train.

    Args:
        y: Etiquetas del train (1D, encoded).
        val_fraction: Fraccion objetivo para validacion (0 < x < 1).
        rng: Generador determinista por fold.

    Returns:
        Tupla ``(train_inner_idx, val_inner_idx)``, indices relativos a ``y``.
    """
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = round(len(cls_idx) * val_fraction)
        # Guarantee >= 1 in val if there are >= 2 samples of the class.
        if len(cls_idx) >= 2 and n_val == 0:
            n_val = 1
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
    )


def _build_temporal_model_native(
    *,
    model_kind: TemporalModelKind,
    input_dim: int,
    num_classes: int,
    sequence_length: int,
    device: str,
    **model_overrides: Any,
) -> Any:
    """Construye el modelo TempCNN o InceptionTime desde ``ml.models.temporal``.

    Implementacion propia (no breizhcrops) tras el porteo del ADR-006
    D-ARQ-2 actualizado. Importacion lazy de torch (~3s) y de las
    arquitecturas; los tests pueden monkeypatchear este helper para
    inyectar modelos mock en CI sin tocar la arquitectura real.

    Args:
        model_kind: ``"tempcnn"`` o ``"inceptiontime"``.
        input_dim: Numero de canales C.
        num_classes: Clases efectivas.
        sequence_length: T.
        device: device string.
        **model_overrides: Hiperparametros adicionales (``dropout``,
            ``hidden_dim``, ``depth``, etc.) pasados al constructor.
    """
    import torch

    from ml.models.temporal import build_temporal_model

    model = build_temporal_model(
        model_kind,
        input_dim=input_dim,
        num_classes=num_classes,
        sequence_length=sequence_length,
        **model_overrides,
    )
    return model.to(torch.device(device))


def _resolve_device(requested: str | None) -> str:
    """Resuelve el device deseado priorizando CUDA cuando ``"auto"``.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"`` o ``None``. ``None``
            equivale a ``"auto"``.

    Returns:
        Cadena lista para ``torch.device(...)``: ``"cuda"`` si CUDA esta
        disponible y ``requested`` lo permite, ``"cpu"`` en otro caso.
    """
    import torch

    if requested in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda_requested_but_unavailable_fallback_cpu")
        return "cpu"
    return requested


def _reconstruct_curve(
    df: pl.DataFrame,
    *,
    index_name: str,
    sequence_length: int,
) -> np.ndarray:
    """Reconstruye la curva diaria de un indice como matrix ``(N, T)``.

    Prioridad de origen:
      1. Columnas pre-materializadas ``{idx}_t_{i:02d}``.
      2. Reconstruccion inversa FFT desde ``{idx}_fft_amp_k`` y
         ``{idx}_fft_phase_k``.
      3. Fallback: repite el ``{idx}_mean`` constante en T (modelo
         degenerado, devuelve serie plana).

    Las columnas pueden tener nulls; se imputan a 0.0 por columna T.
    """
    n = df.height
    T = sequence_length

    # 1) pre-materialized curves.
    cols_t = [f"{index_name}_t_{i:02d}" for i in range(T)]
    if all(c in df.columns for c in cols_t):
        matrix = df.select(cols_t).fill_null(0.0).to_numpy().astype(np.float32)
        return matrix

    # 2) FFT inverse reconstruction.
    amp_cols = [f"{index_name}_fft_amp_{k}" for k in range(DEFAULT_FFT_HARMONICS + 1)]
    phase_cols = [f"{index_name}_fft_phase_{k}" for k in range(DEFAULT_FFT_HARMONICS + 1)]
    if all(c in df.columns for c in amp_cols) and all(c in df.columns for c in phase_cols):
        amps = df.select(amp_cols).fill_null(0.0).to_numpy().astype(np.float64)
        phases = df.select(phase_cols).fill_null(0.0).to_numpy().astype(np.float64)
        # Frequencies per harmonic (1 cycle per year for k=1, 2 for k=2, ...).
        t_axis = np.arange(T, dtype=np.float64)
        curve = np.zeros((n, T), dtype=np.float32)
        # k=0 (DC): constant signal = DC amplitude.
        curve += amps[:, 0:1].astype(np.float32)
        for k in range(1, DEFAULT_FFT_HARMONICS + 1):
            freq = 2.0 * np.pi * k * t_axis / T  # shape (T,)
            # amplitude * cos(freq + phase). Single-sided (consistent with FFT).
            phase_k = phases[:, k : k + 1]  # (N, 1)
            amp_k = amps[:, k : k + 1]  # (N, 1)
            curve += (amp_k * np.cos(freq[None, :] + phase_k)).astype(np.float32)
        return curve

    # 3) Flat fallback: repeat the mean.
    mean_col = f"{index_name}_mean"
    if mean_col in df.columns:
        means = df.get_column(mean_col).fill_null(0.0).to_numpy().astype(np.float32)
        return np.broadcast_to(means[:, None], (n, T)).copy()

    # Degenerate case: 0.0 series (tests should avoid it).
    return np.zeros((n, T), dtype=np.float32)


class _NullMlflowRun:
    """Context manager nulo usado cuando MLflow no esta disponible."""

    run_id: str | None = None

    def __enter__(self) -> _NullMlflowRun | None:
        return None

    def __exit__(self, *args: object) -> None:
        return None

    def log_params(self, params: dict[str, object]) -> None:  # pragma: no cover
        return None

    def log_metric(
        self, key: str, value: float, *, step: int | None = None
    ) -> None:  # pragma: no cover
        return None


class _MlflowRun:
    """Context manager fino sobre mlflow.start_run; logueo + tags estandar."""

    def __init__(self, uri: str, model_kind: TemporalModelKind) -> None:
        self.uri = uri
        self.model_kind = model_kind
        self.run_id: str | None = None
        self._mlflow: Any = None
        self._run: Any = None

    def __enter__(self) -> _MlflowRun:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.uri)
        run = mlflow.start_run(run_name=f"phenology_{self.model_kind}")
        self._run = run
        self.run_id = run.info.run_id
        mlflow.set_tags(
            {
                "data_version": _resolve_data_version(),
                "code_version": _resolve_code_version(),
                "module": "ml.train.phenology_models",
                "model_kind": self.model_kind,
            }
        )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()

    def log_params(self, params: dict[str, object]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        if self._mlflow is not None and value == value:  # filter out NaN
            self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Registra un archivo como artifact en el run actual."""
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_state_dict(self, model: Any, name: str = "model_state_dict.pt") -> None:
        """Persiste el ``state_dict`` del modelo como artifact serializado."""
        if self._mlflow is None:
            return
        import tempfile

        import torch as _torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            _torch.save(model.state_dict(), path)
            self._mlflow.log_artifact(str(path), artifact_path="checkpoints")


def _try_mlflow_run(uri: str | None, *, model_kind: TemporalModelKind):  # type: ignore[no-untyped-def]
    """Devuelve un context manager: real si MLflow disponible, nulo si no."""
    if uri is None:
        return _NullMlflowRun()
    try:
        import mlflow  # noqa: F401
    except ImportError:  # pragma: no cover
        logger.warning("mlflow_not_available", uri=uri)
        return _NullMlflowRun()
    return _MlflowRun(uri=uri, model_kind=model_kind)


def _resolve_data_version() -> str:
    """Resuelve el ``data_version`` tag (hash DVC corto si .dvc disponible)."""
    try:
        import subprocess

        repo_root = Path(__file__).resolve().parents[2]
        dvc_file = (
            repo_root / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet.dvc"
        )
        if dvc_file.exists():
            content = dvc_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                if "md5:" in line:
                    return line.split("md5:")[-1].strip()[:12]
        # Fallback: git rev of the data/ folder.
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", "data/"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _resolve_code_version() -> str:
    """Resuelve el ``code_version`` tag (git HEAD sha corto)."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# Sentinel-free helper to override imports in tests.
_TEMPORAL_BUILDER = _build_temporal_model_native


def _set_model_builder(builder):  # type: ignore[no-untyped-def]  # pragma: no cover - test util
    """Inyecta un builder alternativo (uso exclusivo de tests)."""
    global _TEMPORAL_BUILDER
    _TEMPORAL_BUILDER = builder
