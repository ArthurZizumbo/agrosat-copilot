"""ml/ensemble/stacking.py
=========================
Stacking (meta-learning) ensemble for dense semantic segmentation.

**Heterogeneous ensemble**: A pixel-level meta-learner is trained on
the concatenated softmax probabilities of the base models.  Because the
meta-learner sees *which model was confident about what*, it can learn
that TSViT is reliable for cereals but AnySat is better for orchards.

Architecture
------------
::

    Base models (M)  →  (M × C) probability vectors per pixel
                     →  PixelMetaLearner (1×1 Conv or LogisticRegression)
                     →  (C,) final class probabilities per pixel

Two meta-learner flavours are provided:

* ``ConvMetaLearner`` — a lightweight 1×1 Conv head, trained
  end-to-end on the training folds.  Preserves spatial structure.
* ``SklearnMetaLearner`` — wraps scikit-learn (LogisticRegression /
  RandomForest) on flattened pixels; faster to fit but ignores spatial
  context.  Useful as a strong, fast baseline.

Usage::

    from ml.ensemble.stacking import StackingEnsemble

    stack = StackingEnsemble(
        checkpoint_paths=[ckpt_tsvit_pheno, ckpt_utae, ckpt_anysat],
        model_builders=[build_tsvit_pheno, build_utae, build_anysat],
        meta_learner="conv",   # or "logreg"
        num_classes=20,
        device="cuda",
    )

    # Fit meta-learner on training fold predictions
    stack.fit(train_loader, val_loader, epochs=5)

    # Evaluate on test fold
    metrics = stack.evaluate(test_loader)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Conv meta-learner
# ─────────────────────────────────────────────────────────────────────────────

class ConvMetaLearner(nn.Module):
    """1×1 Conv meta-learner that combines M×C probability maps → C classes.

    Input : (B, M*C, H, W) — concatenated softmax probs from M base models.
    Output: (B, C, H, W)   — final class logits.
    """

    def __init__(self, num_models: int, num_classes: int, hidden: int = 64) -> None:
        super().__init__()
        in_ch = num_models * num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, M*C, H, W)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Stacking ensemble
# ─────────────────────────────────────────────────────────────────────────────

class StackingEnsemble:
    """Stacking ensemble: base models → cached probs → meta-learner.

    Args:
        checkpoint_paths: paths to base model checkpoints.
        model_builders:   callables returning base ``nn.Module`` instances.
        meta_learner:     "conv" (ConvMetaLearner) or "logreg"/"rf"
                          (scikit-learn, flattened pixels).
        num_classes:      number of output classes (default 20 for PASTIS-R).
        device:           inference device.
        hidden:           hidden channels in ``ConvMetaLearner``.
        input_key, label_key, positions_key: DataLoader batch keys.
    """

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        model_builders: Sequence[Callable[[], nn.Module]],
        meta_learner: str = "conv",
        num_classes: int = 20,
        device: str = "cuda",
        hidden: int = 64,
        input_key: str = "pixel_values",
        label_key: str = "labels",
        positions_key: str | None = "positions",
        ignore_index: int = 19,
    ) -> None:
        self.num_classes = num_classes
        self.num_models = len(checkpoint_paths)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.meta_type = meta_learner
        self.input_key = input_key
        self.label_key = label_key
        self.positions_key = positions_key
        self.ignore_index = ignore_index

        # Load base models (frozen during meta-learner training)
        self.base_models: list[nn.Module] = []
        for ckpt, builder in zip(checkpoint_paths, model_builders):
            m = builder()
            state = torch.load(str(ckpt), map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            m.load_state_dict(state, strict=False)
            m.eval().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
            self.base_models.append(m)

        # Build meta-learner
        if meta_learner == "conv":
            self.meta: nn.Module = ConvMetaLearner(
                self.num_models, num_classes, hidden
            ).to(self.device)
        elif meta_learner in ("logreg", "rf"):
            self.meta = None  # built lazily in fit()
        else:
            raise ValueError(f"Unknown meta_learner: {meta_learner!r}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_base_probs(self, batch: dict) -> torch.Tensor:
        """Forward all base models → (B, M*C, H, W) softmax probs."""
        x = batch[self.input_key].to(self.device, non_blocking=True)
        target_hw = x.shape[-2:]
        probs_list = []
        for model in self.base_models:
            if self.positions_key and self.positions_key in batch:
                pos = batch[self.positions_key].to(self.device, non_blocking=True)
                out = model(x, pos)
            else:
                out = model(x)
            if hasattr(out, "logits"):
                out = out.logits
            if out.shape[-2:] != target_hw:
                out = F.interpolate(out.float(), size=target_hw,
                                    mode="bilinear", align_corners=False)
            probs_list.append(F.softmax(out.float(), dim=1))  # (B, C, H, W)
        return torch.cat(probs_list, dim=1)  # (B, M*C, H, W)

    # ------------------------------------------------------------------
    def _collect_features(
        self, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect base-model probs and labels for the whole loader.

        Returns:
            features: (N, M*C, H, W)
            labels:   (N, H, W)
        """
        all_feats, all_labels = [], []
        for batch in loader:
            feats = self._get_base_probs(batch).cpu()
            all_feats.append(feats)
            if self.label_key in batch:
                all_labels.append(batch[self.label_key])
        features = torch.cat(all_feats, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
        return features, labels

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> dict[str, list[float]]:
        """Fit the meta-learner on base-model predictions.

        For ``conv`` meta-learner: trains a small CNN head.
        For ``logreg`` / ``rf``: fits a scikit-learn model on flattened pixels
        sampled randomly (max 500 k pixels to keep memory bounded).

        Args:
            train_loader: DataLoader for training folds.
            val_loader:   optional validation loader for per-epoch metrics.
            epochs:       training epochs (conv only).
            lr:           learning rate (conv only).
            weight_decay: L2 regularisation (conv only).

        Returns:
            History dict with ``train_loss`` and optionally ``val_miou``.
        """
        print(f"[Stacking] Collecting base-model features on train set …")
        t0 = time.time()
        train_feats, train_labels = self._collect_features(train_loader)
        print(f"  done in {time.time()-t0:.1f}s  |  shape: {train_feats.shape}")

        history: dict[str, list[float]] = {"train_loss": []}

        # ── Conv meta-learner ──────────────────────────────────────────────
        if self.meta_type == "conv":
            optimizer = torch.optim.AdamW(
                self.meta.parameters(), lr=lr, weight_decay=weight_decay
            )
            criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            ds = TensorDataset(train_feats, train_labels)
            meta_loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

            for epoch in range(1, epochs + 1):
                self.meta.train()
                epoch_loss = 0.0
                for feats_b, labels_b in meta_loader:
                    feats_b = feats_b.to(self.device)
                    labels_b = labels_b.to(self.device)
                    logits = self.meta(feats_b)
                    loss = criterion(logits, labels_b)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg = epoch_loss / len(meta_loader)
                history["train_loss"].append(avg)

                val_str = ""
                if val_loader is not None:
                    m = self._evaluate_conv(val_loader)
                    history.setdefault("val_miou", []).append(m["miou"])
                    val_str = f"  val mIoU={m['miou']:.4f}"
                print(f"  Epoch {epoch}/{epochs}  loss={avg:.4f}{val_str}")

        # ── Sklearn meta-learner ───────────────────────────────────────────
        else:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
            except ImportError as e:
                raise ImportError("Install scikit-learn: pip install scikit-learn") from e

            # Flatten: (N, M*C, H, W) → (N*H*W, M*C)
            N, MC, H, W = train_feats.shape
            X = train_feats.permute(0, 2, 3, 1).reshape(-1, MC).numpy()
            y = train_labels.reshape(-1).numpy()

            # Remove ignored pixels
            valid_mask = y != self.ignore_index
            X, y = X[valid_mask], y[valid_mask]

            # Random subsample to keep RAM bounded (max 500k pixels)
            max_px = 500_000
            if len(y) > max_px:
                idx = np.random.choice(len(y), max_px, replace=False)
                X, y = X[idx], y[idx]

            print(f"[Stacking] Fitting {self.meta_type} on {len(y):,} pixels …")
            if self.meta_type == "logreg":
                clf = LogisticRegression(
                    max_iter=500, C=1.0, solver="saga",
                    multi_class="multinomial", n_jobs=-1
                )
            else:  # rf
                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=8, n_jobs=-1, random_state=42
                )
            clf.fit(X, y)
            self.meta = clf  # type: ignore[assignment]
            history["train_loss"] = [float("nan")]

        self._train_time_s = time.time() - t0
        return history

    # ------------------------------------------------------------------
    def _evaluate_conv(
        self, loader: DataLoader, ignore_index: int | None = None
    ) -> dict[str, float]:
        from torchmetrics import JaccardIndex, F1Score, Accuracy
        ig = ignore_index if ignore_index is not None else self.ignore_index
        miou_m = JaccardIndex(task="multiclass", num_classes=self.num_classes,
                              ignore_index=ig, average="macro").to(self.device)
        f1_m = F1Score(task="multiclass", num_classes=self.num_classes,
                       ignore_index=ig, average="macro").to(self.device)
        acc_m = Accuracy(task="multiclass", num_classes=self.num_classes,
                         ignore_index=ig).to(self.device)
        self.meta.eval()
        with torch.no_grad():
            for batch in loader:
                feats = self._get_base_probs(batch)
                logits = self.meta(feats)
                preds = logits.argmax(dim=1)
                labels = batch[self.label_key].to(self.device)
                miou_m.update(preds, labels)
                f1_m.update(preds, labels)
                acc_m.update(preds, labels)
        return {
            "miou": miou_m.compute().item(),
            "f1_macro": f1_m.compute().item(),
            "pixel_accuracy": acc_m.compute().item(),
        }

    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the full stacking ensemble on ``loader``."""
        if self.meta_type == "conv":
            return self._evaluate_conv(loader)

        # sklearn path
        from torchmetrics import JaccardIndex, F1Score, Accuracy
        miou_m = JaccardIndex(task="multiclass", num_classes=self.num_classes,
                              ignore_index=self.ignore_index, average="macro")
        f1_m = F1Score(task="multiclass", num_classes=self.num_classes,
                       ignore_index=self.ignore_index, average="macro")
        acc_m = Accuracy(task="multiclass", num_classes=self.num_classes,
                         ignore_index=self.ignore_index)

        for batch in loader:
            feats = self._get_base_probs(batch).cpu()
            N, MC, H, W = feats.shape
            X = feats.permute(0, 2, 3, 1).reshape(-1, MC).numpy()
            preds_flat = torch.from_numpy(self.meta.predict(X)).long()
            preds = preds_flat.reshape(N, H, W)
            labels = batch[self.label_key]
            miou_m.update(preds, labels)
            f1_m.update(preds, labels)
            acc_m.update(preds, labels)

        return {
            "miou": miou_m.compute().item(),
            "f1_macro": f1_m.compute().item(),
            "pixel_accuracy": acc_m.compute().item(),
        }

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save meta-learner weights to ``path``."""
        if self.meta_type == "conv":
            torch.save(self.meta.state_dict(), str(path))
        else:
            import pickle
            with open(str(path), "wb") as f:
                pickle.dump(self.meta, f)

    def load(self, path: str | Path) -> None:
        """Load meta-learner weights from ``path``."""
        if self.meta_type == "conv":
            self.meta.load_state_dict(torch.load(str(path), map_location=self.device))
        else:
            import pickle
            with open(str(path), "rb") as f:
                self.meta = pickle.load(f)
