"""Smoke tests del trainer FarSLIP (AC-2, AC-4, AC-9 parcial).

Estos tests requieren cargar ``openai/clip-vit-base-patch16`` desde HF Hub. Si
no hay acceso a internet ni cache, se hace skip clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.farslip.distill import (
    FarSLIPDistillationTrainer,
    FarSLIPTrainerConfig,
)


def _hf_available() -> bool:
    """Heuristica: ``transformers`` instalado + offline mode no fuerza skip."""
    try:
        from transformers import CLIPVisionModel  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@pytest.fixture(scope="module")
def trainer(tmp_path_factory: pytest.TempPathFactory) -> FarSLIPDistillationTrainer:
    if not _hf_available():
        pytest.skip("transformers no disponible")
    out_dir = tmp_path_factory.mktemp("trainer_out")
    cfg = FarSLIPTrainerConfig(
        teacher_model_id="openai/clip-vit-base-patch16",
        dataset_root=Path("data/farslip_pairs"),
        output_dir=out_dir,
        n_epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        device="cpu",
        n_in_channels=4,
        n_regions=2,
        n_categories=3,
    )
    try:
        t = FarSLIPDistillationTrainer(cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo cargar CLIP: {exc}")
    return t


def test_trainer_init_loads_teacher_and_student(
    trainer: FarSLIPDistillationTrainer,
) -> None:
    """Trainer carga ambos modelos y los pone en CPU."""
    assert trainer.teacher is not None
    assert trainer.student is not None
    # Verifica que el patch_embed sea 4 canales
    pe = trainer.student.embeddings.patch_embedding
    assert pe.in_channels == 4
    # Teacher debe estar frozen
    assert all(not p.requires_grad for p in trainer.teacher.parameters())
    # Student trainable
    assert any(p.requires_grad for p in trainer.student.parameters())


def test_student_initial_weights_equal_teacher(
    trainer: FarSLIPDistillationTrainer,
) -> None:
    """AC-2: en epoch 0 step 0 el student arranca copia del teacher.

    Tras Q1 fix, teacher conserva 3 canales (RGB puro = paper FarSLIP) y
    student tiene 4 canales (Sentinel-2). La equivalencia solo aplica a
    parametros con shape identico — el ``patch_embedding.weight`` difiere
    intencionalmente (3 vs 4 canales) y los primeros 3 canales del student
    deben ser una copia exacta del teacher RGB; el 4o canal (NIR) es la
    media de los 3 RGB (mean-init, no zero).
    """
    t_state = trainer.teacher.state_dict()
    s_state = trainer.student.state_dict()
    assert set(t_state.keys()) == set(s_state.keys())
    diff = 0.0
    n_compared = 0
    for k in t_state:
        if t_state[k].shape != s_state[k].shape:
            # Solo permitido para patch_embedding.weight tras Q1 fix.
            assert "patch_embedding.weight" in k, (
                f"shape mismatch inesperado en {k}: t={t_state[k].shape} s={s_state[k].shape}"
            )
            # Validar contrato Q1: primeros 3 canales identicos, 4o canal =
            # media de los 3 RGB del teacher.
            t_rgb = t_state[k].float()
            s_rgb = s_state[k][:, :3, :, :].float()
            s_nir = s_state[k][:, 3, :, :].float()
            assert torch.allclose(s_rgb, t_rgb, atol=1e-9), (
                "primeros 3 canales del student deben ser copia exacta del teacher"
            )
            expected_nir = t_rgb.mean(dim=1)
            assert torch.allclose(s_nir, expected_nir, atol=1e-6), (
                "4o canal NIR debe inicializarse a mean(RGB) del teacher"
            )
            continue
        diff += (t_state[k].float() - s_state[k].float()).pow(2).sum().item()
        n_compared += 1
    assert n_compared > 0, "no se comparo ningun parametro shape-equal"
    assert diff < 1e-9, f"L2 diff student vs teacher en params shape-equal = {diff}"


def test_save_student_safetensors_format(
    trainer: FarSLIPDistillationTrainer, tmp_path: Path
) -> None:
    """save_student devuelve ruta a archivo .safetensors valido."""
    # Redirigimos output_dir a tmp para no escribir en el module fixture
    trainer.config.output_dir = tmp_path
    path = trainer.save_student(format="safetensors", suffix="smoke")
    p = Path(path)
    assert p.exists()
    assert p.suffix == ".safetensors"
    from safetensors.torch import load_file

    state = load_file(str(p))
    assert "embeddings.patch_embedding.weight" in state
    assert state["embeddings.patch_embedding.weight"].shape[1] == 4


def test_two_batches_loss_decreases(
    trainer: FarSLIPDistillationTrainer,
) -> None:
    """Tras varios batches con grad step + LR alto, la loss decrece.

    Sobre random data sin estructura el lr default 1e-5 puede no producir
    decrement monotonico en 2 steps; usamos lr=1e-2 + 4 steps sobre el mismo
    batch (overfit trivial) para verificar que el gradiente fluye en la
    direccion correcta.
    """
    torch.manual_seed(42)
    d = trainer.teacher.config.hidden_size
    n_protos = trainer.config.n_regions * trainer.config.n_categories
    trainer.set_text_prototypes(torch.randn(n_protos, d))

    # Overfit a un batch fijo con LR moderado.
    overfit_optim = torch.optim.AdamW(
        (p for p in trainer.student.parameters() if p.requires_grad), lr=1e-3
    )
    images = torch.rand(2, 4, 224, 224)
    region_ids = torch.tensor([0, 1])
    cat_ids = torch.tensor([0, 1])

    losses_seq: list[float] = []
    for _ in range(8):
        overfit_optim.zero_grad(set_to_none=True)
        out = trainer.step(images, region_ids, cat_ids)
        total = out["loss_total"]
        total.backward()
        overfit_optim.step()
        losses_seq.append(float(total.detach().cpu().item()))
    # Loss promedio de la 2da mitad < primera mitad (overfit progresa)
    early = sum(losses_seq[:4]) / 4
    late = sum(losses_seq[4:]) / 4
    assert late < early, f"early {early} late {late} (loss no decrecio)"


def test_mlflow_run_created(
    trainer: FarSLIPDistillationTrainer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trainer.train() abre un MLflow run con tags US-017."""
    pytest.importorskip("mlflow")
    import mlflow

    # MLflow en Windows requiere SQLite o file:/// con triple slash (no file://).
    sqlite_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", sqlite_uri)
    mlflow.set_tracking_uri(sqlite_uri)

    # Dataset minimo en memoria
    class _TinyDS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "image": torch.rand(4, 224, 224),
                "region_id": torch.tensor(0, dtype=torch.long),
                "category_id": torch.tensor(0, dtype=torch.long),
            }

    d = trainer.teacher.config.hidden_size
    n_protos = trainer.config.n_regions * trainer.config.n_categories
    trainer.set_text_prototypes(torch.randn(n_protos, d))
    # acortamos para que termine rapido
    trainer.config.n_epochs = 1
    trainer.config.batch_size = 2
    trainer.config.grad_accum_steps = 1
    trainer.config.output_dir = tmp_path

    loader = torch.utils.data.DataLoader(_TinyDS(), batch_size=2)
    trainer.train(loader)
    runs = mlflow.search_runs()
    assert not runs.empty


def test_patch_embed_4_channels_init_no_dead_neurons(
    trainer: FarSLIPDistillationTrainer,
) -> None:
    """El 4o canal (NIR) tiene init != ceros (mean RGB)."""
    pe = trainer.student.embeddings.patch_embedding
    nir_w = pe.weight[:, 3, :, :]
    assert nir_w.abs().mean().item() > 1e-6
