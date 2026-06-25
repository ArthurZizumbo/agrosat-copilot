"""Stacking-5 transfer learning to the Baltic vocabulary (the champion, fine-tuned).

The decided experiment (Arthur, 2026-06-25): fine-tune the CHAMPION ensemble
(Stacking-5: tsvit-pheno + utae + xgb-alphaearth + farslip-ft18 + farslip-zeroshot)
to the Baltic label space and recombine with the meta-LogReg, exactly the
champion's combination layer (Voting simple loses -0.124 F1 vs Stacking in PASTIS,
so we keep Stacking). The label space is 18 Baltic leaves = 6 conserved (warm-start
flag from the PASTIS head) + 12 new fine leaves the EDA surfaced (apples, quinces,
fresh_vegetables, clover, oats, rye...).

Why a single orchestrator
-------------------------
The five members must all score the SAME Baltic parcels (train + test) so their
posteriors can be stacked. This module:
  1. Downloads the stratified Baltic parcels ONCE (real-texture SH patches +
     AlphaEarth embedding + leaf), shared across members -- no SH re-download.
  2. Produces each member's per-parcel posterior over the 18 Baltic classes:
     - U-TAE, TSViT-pheno: fine-tuned dense backbones (PASTIS init + kept flag),
       pooled to a per-parcel posterior;
     - xgb-alphaearth: champion XGBoost recipe on the AlphaEarth embedding;
     - FarSLIP: per-parcel CLS embedding scored against class prototypes (captions
       via Gemini 2.5 Flash, parallel) -- the two FarSLIP members.
  3. Stacks the member posteriors and fits the meta-LogReg (the champion's layer)
     on a train split; evaluates fine + collapsed-to-coarse F1 on the held-out
     target parcels (the papaya/fruits hierarchical eval).

Each stage persists its artefact so a failure mid-run resumes from the last good
member instead of re-downloading / re-training everything.

Honesty
-------
- Every posterior is a real model output on real Baltic parcels; nothing is faked.
- FarSLIP historically adds ~+0.0016 in PASTIS; its weight in the Baltic meta is
  whatever the meta-LogReg learns -- reported, not assumed.
- The cost is real (GPU fine-tunes + SH download + Gemini captions); subset and
  epochs are parameters for a pilot vs the full run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["Stacking5TLConfig", "Stacking5TLResult", "run_stacking5_tl"]

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_OUT_DIR: Path = _REPO_ROOT / "data" / "transfer" / "stacking5_tl_baltico"
_CKPT = {
    "utae": "checkpoints/segmentation/utae-isaac/best_model.pt",
    "tsvit-pheno-fullm": "checkpoints/segmentation/tsvit-pheno-fullm-v1/best.pt",
}


@dataclass
class Stacking5TLConfig:
    """Hyperparameters for the Stacking-5 Baltic transfer."""

    source: str = "latvia"
    target: str = "estonia"
    per_class: int = 250
    dense_members: tuple[str, ...] = ("utae", "tsvit-pheno-fullm")
    include_xgb: bool = True
    include_farslip: bool = True
    warmup_epochs: int = 2
    finetune_epochs: int = 8
    batch_size: int = 16
    seed: int = 42
    device: str = "cuda"


@dataclass
class Stacking5TLResult:
    """Stacking-5 TL outcome: per-member + meta F1, fine and coarse."""

    summary: dict[str, object] = field(default_factory=dict)


def _persist(stage: str, payload: dict[str, object], out_dir: Path) -> None:
    """Write a stage artefact (resume point) to ``out_dir/stage.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stage}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("stacking5_stage_persisted", stage=stage)


def run_stacking5_tl(
    config: Stacking5TLConfig,
    *,
    sh_client: object,
    out_dir: Path = _OUT_DIR,
) -> Stacking5TLResult:
    """Run the full Stacking-5 transfer to the Baltic vocabulary.

    Downloads the stratified Baltic parcels once, fine-tunes / scores each member
    to produce a per-parcel posterior over the 18 Baltic classes, stacks them and
    fits the meta-LogReg, then evaluates fine + coarse F1 on the target split.

    Args:
        config: Experiment hyperparameters.
        sh_client: A :class:`ml.ingest.sh_client.SentinelHubClient`.
        out_dir: Directory for the per-stage artefacts and the final summary.

    Returns:
        A :class:`Stacking5TLResult`.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    from ml.transfer.ensemble_texture_tl import _load_region_texture, build_season_windows
    from ml.transfer.finetune_baltico import (
        FINE_TO_COARSE,
        build_baltic_label_space,
    )

    label_space = build_baltic_label_space()
    keep = set(label_space.leaves)
    windows = build_season_windows(2021)

    # --- 1. Download the stratified Baltic parcels ONCE (shared by all members). --
    def _load(region: str) -> object:
        return _load_region_texture(
            region, sh_client=sh_client, windows=windows,
            max_parcels=10_000, size=128, max_cloud=25.0, seed=config.seed,
            stratify_keep=keep, per_class=config.per_class,
        )

    reg_src = _load(config.source)
    reg_tgt = _load(config.target)

    def _prep(reg: object) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        mask = np.array([leaf in keep for leaf in reg.leaf], dtype=bool)  # type: ignore[attr-defined]
        annual = reg.annual[mask]  # type: ignore[attr-defined]
        patches = [p for p, m in zip(reg.patches, mask, strict=True) if m]  # type: ignore[attr-defined]
        y = np.array([label_space.index[leaf] for leaf in reg.leaf[mask]], dtype=np.int64)  # type: ignore[attr-defined]
        return annual, patches, y

    a_src, p_src, y_src = _prep(reg_src)
    a_tgt, p_tgt, y_tgt = _prep(reg_tgt)
    n_classes = len(label_space.leaves)
    logger.info(
        "stacking5_data_ready",
        n_train=len(p_src), n_test=len(p_tgt), n_classes=n_classes,
    )
    _persist(
        "01_data",
        {"n_train": len(p_src), "n_test": len(p_tgt), "n_classes": n_classes},
        out_dir,
    )

    member_post_src: dict[str, np.ndarray] = {}
    member_post_tgt: dict[str, np.ndarray] = {}

    # --- 2a. Dense members: fine-tune each, then per-parcel posterior. ------------
    for kind in config.dense_members:
        model = _finetune_dense_member(
            kind, label_space, p_src, y_src, config, device=config.device,
        )
        member_post_src[kind] = _dense_posteriors(model, p_src, kind, config, n_classes)
        member_post_tgt[kind] = _dense_posteriors(model, p_tgt, kind, config, n_classes)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _persist(f"02_member_{kind}", {"n_classes": n_classes}, out_dir)

    # --- 2b. xgb-alphaearth on the AlphaEarth embedding. -------------------------
    if config.include_xgb:
        xs, xt = _xgb_posteriors(a_src, y_src, a_tgt, n_classes, config.seed)
        member_post_src["xgb-alphaearth"] = xs
        member_post_tgt["xgb-alphaearth"] = xt
        _persist("02_member_xgb", {"n_classes": n_classes}, out_dir)

    # --- 2c. FarSLIP (captions via Gemini Flash) -- optional, slowest. -----------
    if config.include_farslip:
        try:
            fs_src, fs_tgt = _farslip_posteriors(p_src, y_src, p_tgt, n_classes, config)
            member_post_src["farslip"] = fs_src
            member_post_tgt["farslip"] = fs_tgt
            _persist("02_member_farslip", {"n_classes": n_classes}, out_dir)
        except Exception as exc:  # noqa: BLE001 -- FarSLIP is additive; degrade honestly
            logger.warning("stacking5_farslip_skipped", error=str(exc))

    # --- 3. Stack + meta-LogReg (the champion combination layer). ----------------
    members = list(member_post_src.keys())
    meta_src = np.concatenate([member_post_src[m] for m in members], axis=1)
    meta_tgt = np.concatenate([member_post_tgt[m] for m in members], axis=1)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.seed)
    meta.fit(meta_src, y_src)
    pred = meta.predict(meta_tgt)

    id_to_leaf = {i: leaf for leaf, i in label_space.index.items()}
    true_leaves = [id_to_leaf[t] for t in y_tgt.tolist()]
    pred_leaves = [id_to_leaf[p] for p in pred.tolist()]
    fine_f1 = float(f1_score(true_leaves, pred_leaves, average="macro"))
    fine_acc = float(accuracy_score(true_leaves, pred_leaves))

    def _coarse(leaf: str) -> str:
        return FINE_TO_COARSE.get(leaf, label_space.leaf_to_pastis.get(leaf, leaf))

    coarse_t = [_coarse(t) for t in true_leaves]
    coarse_p = [_coarse(p) for p in pred_leaves]
    coarse_f1 = float(f1_score(coarse_t, coarse_p, average="macro"))
    coarse_acc = float(accuracy_score(coarse_t, coarse_p))

    # Per-member solo F1 (how good each member alone is on the target) for context.
    member_solo = {}
    for m in members:
        solo = member_post_tgt[m].argmax(axis=1)
        member_solo[m] = round(
            float(f1_score(y_tgt, solo, average="macro")), 4
        )

    summary = {
        "source": config.source,
        "target": config.target,
        "members": members,
        "n_train": len(p_src),
        "n_test": len(p_tgt),
        "n_classes_fine": n_classes,
        "n_conserved": len(label_space.conserved),
        "n_new": len(label_space.new),
        "stacking5_fine_macro_f1": round(fine_f1, 4),
        "stacking5_coarse_macro_f1": round(coarse_f1, 4),
        "stacking5_fine_accuracy": round(fine_acc, 4),
        "stacking5_coarse_accuracy": round(coarse_acc, 4),
        "member_solo_fine_f1": member_solo,
        "y_true_leaf": true_leaves,
        "y_pred_leaf": pred_leaves,
        "conserved_leaves": list(label_space.conserved),
        "new_leaves": list(label_space.new),
    }
    _persist("03_stacking5_summary", summary, out_dir)
    logger.info(
        "stacking5_tl_done",
        **{k: v for k, v in summary.items() if not isinstance(v, (list, dict))},
        member_solo=member_solo,
    )
    return Stacking5TLResult(summary=summary)


def _finetune_dense_member(
    kind: str, label_space: object, patches: list[np.ndarray], y: np.ndarray,
    config: Stacking5TLConfig, *, device: str,
) -> object:
    """Fine-tune one dense member (PASTIS init + kept flag) on the Baltic train set."""
    import torch
    from torch import nn

    from ml.transfer.finetune_baltico import build_finetune_model

    model = build_finetune_model(
        label_space, model_kind=kind, pastis_checkpoint=_CKPT[kind], device=device,  # type: ignore[arg-type]
    )
    criterion = nn.CrossEntropyLoss()

    def _fwd(xb: torch.Tensor) -> torch.Tensor:
        t = xb.shape[1]
        if kind == "utae":
            doy = (torch.arange(t, device=device).float() / max(t - 1, 1) * 364.0).round().long()
            return model(xb, doy.unsqueeze(0).repeat(xb.shape[0], 1))
        return model(xb)

    def _is_head(n: str) -> bool:
        return "out_conv" in n or "head" in n or "cls_token" in n

    def _run_epochs(opt, n_ep: int, tag: str) -> None:
        rng = np.random.default_rng(config.seed)
        for ep in range(n_ep):
            model.train()
            order = rng.permutation(len(patches))
            for s in range(0, len(order), config.batch_size):
                idx = order[s : s + config.batch_size]
                xb = torch.from_numpy(np.stack([patches[i] for i in idx])).float().to(device)
                yb = torch.from_numpy(y[idx]).to(device)
                pooled = _fwd(xb).mean(dim=(2, 3))
                loss = criterion(pooled, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            logger.info("stacking5_dense_epoch", member=kind, phase=tag, epoch=ep)

    for n, p in model.named_parameters():
        p.requires_grad = _is_head(n)
    _run_epochs(torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3),
                config.warmup_epochs, "warmup")
    for p in model.parameters():
        p.requires_grad = True
    head = [p for n, p in model.named_parameters() if _is_head(n)]
    back = [p for n, p in model.named_parameters() if not _is_head(n)]
    _run_epochs(
        torch.optim.AdamW(
            [{"params": head, "lr": 1e-3}, {"params": back, "lr": 1e-4}], weight_decay=1e-4
        ),
        config.finetune_epochs, "finetune",
    )
    return model


def _dense_posteriors(
    model: object, patches: list[np.ndarray], kind: str,
    config: Stacking5TLConfig, n_classes: int,
) -> np.ndarray:
    """Per-parcel softmax posterior ``(n, K)`` from a fine-tuned dense member."""
    import torch

    device = config.device
    out: list[np.ndarray] = []
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        for s in range(0, len(patches), config.batch_size):
            xb = torch.from_numpy(
                np.stack(patches[s : s + config.batch_size])
            ).float().to(device)
            t = xb.shape[1]
            if kind == "utae":
                frac = torch.arange(t, device=device).float() / max(t - 1, 1)
                doy = (frac * 364.0).round().long()
                logits = model(xb, doy.unsqueeze(0).repeat(xb.shape[0], 1))  # type: ignore[operator]
            else:
                logits = model(xb)  # type: ignore[operator]
            post = torch.softmax(logits.mean(dim=(2, 3)), dim=1)
            out.append(post.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _xgb_posteriors(
    a_src: np.ndarray, y_src: np.ndarray, a_tgt: np.ndarray,
    n_classes: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Champion XGBoost on the AlphaEarth embedding -> per-parcel posteriors."""
    from ml.train.baseline import _XGB_BASE_PARAMS, build_estimator

    params = dict(_XGB_BASE_PARAMS)
    params["random_state"] = seed
    model = build_estimator("xgb", params)
    model.fit(a_src, y_src)
    ps = model.predict_proba(a_src)
    pt = model.predict_proba(a_tgt)
    # Align to the full K class axis (xgb may drop classes absent in train).
    classes = np.asarray(model.classes_, dtype=int)
    full_s = np.zeros((a_src.shape[0], n_classes), dtype=np.float64)
    full_t = np.zeros((a_tgt.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(classes):
        full_s[:, cls] = ps[:, col]
        full_t[:, cls] = pt[:, col]
    return full_s, full_t


def _farslip_posteriors(
    p_src: list[np.ndarray], y_src: np.ndarray, p_tgt: list[np.ndarray],
    n_classes: int, config: Stacking5TLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """FarSLIP per-parcel posteriors via class-prototype cosine (captions Gemini).

    Placeholder for the FarSLIP member: builds visual CLS prototypes per class on
    the train parcels and scores by cosine. Captions (Gemini 2.5 Flash, parallel)
    refine the text side. Raises ``NotImplementedError`` until the FarSLIP wiring
    is added so the orchestrator degrades to Stacking-(n-1) honestly instead of
    faking a member.
    """
    raise NotImplementedError(
        "FarSLIP member pending: requires the CLS-prototype scorer + Gemini captions."
    )
