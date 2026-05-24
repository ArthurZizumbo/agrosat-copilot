"""Tests de las arquitecturas temporales nativas (US-022b-C).

Cubre forma del output, propagacion de gradiente, factory por nombre y
manejo del edge case de ``num_classes=1``.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings(
    "ignore",
    message="Initializing zero-element tensors is a no-op",
    category=UserWarning,
)

torch = pytest.importorskip("torch")

from ml.models.temporal import (  # noqa: E402  (importorskip arriba)
    InceptionTime,
    TempCNN,
    build_temporal_model,
)

# ---------------------------------------------------------------------------
# TempCNN
# ---------------------------------------------------------------------------


def test_tempcnn_forward_shape() -> None:
    """TempCNN devuelve logits ``(B, num_classes)``."""
    model = TempCNN(input_dim=3, num_classes=17, sequencelength=72)
    x = torch.randn(4, 72, 3)
    out = model(x)
    assert out.shape == (4, 17)


def test_tempcnn_backward_propagates_gradient() -> None:
    """Backward pass actualiza pesos sin errores."""
    model = TempCNN(input_dim=3, num_classes=5, sequencelength=24)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(8, 24, 3)
    y = torch.randint(0, 5, (8,))
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    # Al menos uno de los gradientes es no-cero.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad
    optimizer.step()


def test_tempcnn_deterministic_with_seed() -> None:
    """Misma semilla + ``eval()`` produce misma salida (init de pesos)."""
    torch.manual_seed(42)
    m1 = TempCNN(input_dim=3, num_classes=5, sequencelength=24).eval()
    torch.manual_seed(42)
    m2 = TempCNN(input_dim=3, num_classes=5, sequencelength=24).eval()
    x = torch.randn(2, 24, 3)
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x), atol=1e-6)


def test_tempcnn_num_classes_one_edge() -> None:
    """``num_classes=1`` no rompe (output (B, 1))."""
    model = TempCNN(input_dim=3, num_classes=1, sequencelength=24)
    out = model(torch.randn(4, 24, 3))
    assert out.shape == (4, 1)


def test_tempcnn_eval_mode_no_dropout() -> None:
    """En ``eval()`` el dropout es identidad y la salida es determinista."""
    model = TempCNN(input_dim=3, num_classes=5, sequencelength=24, dropout=0.9)
    model.eval()
    x = torch.randn(4, 24, 3)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# InceptionTime
# ---------------------------------------------------------------------------


def test_inceptiontime_forward_shape() -> None:
    """InceptionTime devuelve logits ``(B, num_classes)``."""
    model = InceptionTime(input_dim=3, num_classes=17)
    x = torch.randn(4, 72, 3)
    out = model(x)
    assert out.shape == (4, 17)


def test_inceptiontime_backward() -> None:
    """Backward propaga gradiente sobre depth=6 default."""
    model = InceptionTime(input_dim=3, num_classes=5)
    x = torch.randn(8, 48, 3)
    y = torch.randint(0, 5, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad


def test_inceptiontime_shortcut_count() -> None:
    """Hay un shortcut cada 3 bloques: depth=6 -> 2 shortcuts."""
    model = InceptionTime(input_dim=3, num_classes=5, depth=6)
    assert len(model.shortcuts) == 2


def test_inceptiontime_variable_depth() -> None:
    """Soporta depth=3 (1 shortcut) y depth=9 (3 shortcuts)."""
    m3 = InceptionTime(input_dim=3, num_classes=5, depth=3)
    assert len(m3.shortcuts) == 1
    m9 = InceptionTime(input_dim=3, num_classes=5, depth=9)
    assert len(m9.shortcuts) == 3


def test_inceptiontime_handles_short_sequence() -> None:
    """Series cortas (T < kernel_size) no rompen el forward (padding)."""
    model = InceptionTime(input_dim=3, num_classes=5)
    out = model(torch.randn(4, 16, 3))
    assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# Factory por nombre
# ---------------------------------------------------------------------------


def test_build_temporal_model_tempcnn() -> None:
    """``build_temporal_model('tempcnn', ...)`` devuelve un ``TempCNN``."""
    model = build_temporal_model(
        "tempcnn", input_dim=3, num_classes=5, sequence_length=24
    )
    assert isinstance(model, TempCNN)


def test_build_temporal_model_inceptiontime() -> None:
    """``build_temporal_model('inceptiontime', ...)`` devuelve un ``InceptionTime``."""
    model = build_temporal_model(
        "inceptiontime", input_dim=3, num_classes=5, sequence_length=24
    )
    assert isinstance(model, InceptionTime)


def test_build_temporal_model_invalid_kind() -> None:
    """``build_temporal_model`` rechaza nombres no soportados."""
    with pytest.raises(ValueError, match="model_kind="):
        build_temporal_model(
            "unknown_arch", input_dim=3, num_classes=5, sequence_length=24
        )


def test_build_temporal_model_overrides() -> None:
    """``overrides`` se propagan al constructor (hidden_dim de TempCNN)."""
    model = build_temporal_model(
        "tempcnn",
        input_dim=3,
        num_classes=5,
        sequence_length=24,
        hidden_dim=32,
        dropout=0.1,
    )
    assert isinstance(model, TempCNN)
    assert model.hidden_dim == 32


# ---------------------------------------------------------------------------
# Smoke combinado: training mini-batch one-shot.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_kind", ["tempcnn", "inceptiontime"])
def test_temporal_model_full_minibatch_step(model_kind: str) -> None:
    """One epoch sobre un mini-batch: la loss baja entre paso 0 y paso N."""
    torch.manual_seed(0)
    model = build_temporal_model(
        model_kind, input_dim=3, num_classes=5, sequence_length=24
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(16, 24, 3)
    y = torch.randint(0, 5, (16,))
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # En 10 pasos sobre un batch fijo la loss tiene que bajar.
    assert losses[-1] < losses[0]
