"""Arquitecturas temporales nativas: TempCNN + InceptionTime (US-022b-C).

Implementacion propia (PyTorch) de las dos arquitecturas oficiales del
benchmark BreizhCrops (Russwurm et al. 2020), portadas directamente desde los
papers originales y de la implementacion de referencia (licencia MIT) en
``breizhcrops`` 0.0.4.1. Anteriormente este modulo importaba las clases de
``breizhcrops.models``; el reencuadre (ADR-006 D-ARQ-2 actualizado) las trae
al repo para tener control sobre serializacion MLflow, custom layers,
testing aislado y evolucion independiente de la dependencia externa.

Modelos
-------

- :class:`TempCNN` (Pelletier, Webb & Petitjean 2019). CNN 1D con tres
  bloques convolucionales + un cabezal denso. Pensado para clasificacion
  de cultivos sobre series Sentinel-2.
  DOI: 10.3390/rs11050523. Codigo de referencia: MIT.
- :class:`InceptionTime` (Fawaz et al. 2020). Pila de bloques Inception
  con shortcut residual; arquitectura ganadora del benchmark UCR.
  DOI: 10.1007/s10618-020-00710-y. Codigo de referencia: MIT.

Ambas aceptan input ``(B, T, C)`` (batch, tiempo, canales = indices
espectrales) y producen logits ``(B, num_classes)``.

Decisiones tecnicas
-------------------

- Pesos He uniformemente inicializados (kaiming_uniform_) por convencion
  de PyTorch.
- BatchNorm1d entre conv y activacion (orden Conv-BN-ReLU).
- Global Average Pooling antes del cabezal denso (InceptionTime); para
  TempCNN, flatten despues del ultimo bloque conv segun la implementacion
  original.
- Dropout configurable; defaults heredados de los papers (0.5 para
  TempCNN, 0.2 para InceptionTime).
- Sin dependencia de ``breizhcrops`` en runtime; solo numpy + torch.

Acreditacion
------------

Adaptado de:
- breizhcrops 0.0.4.1, ``breizhcrops/models/TempCNN.py``,
  ``breizhcrops/models/InceptionTime.py`` (MIT License).
- Pelletier, Webb & Petitjean. ``Temporal Convolutional Neural Network for
  the Classification of Satellite Image Time Series``. Remote Sensing
  11(5):523, 2019.
- Fawaz et al. ``InceptionTime: Finding AlexNet for Time Series
  Classification``. Data Mining and Knowledge Discovery 34, 2020.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["InceptionTime", "TempCNN", "build_temporal_model"]


# ---------------------------------------------------------------------------
# TempCNN  (Pelletier, Webb & Petitjean 2019)
# ---------------------------------------------------------------------------


class _TempCNNBlock(nn.Module):
    """Bloque Conv1D + BatchNorm + ReLU + Dropout (kernel_size=5 default)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))


class TempCNN(nn.Module):
    """TempCNN: 3 bloques Conv1D + dense head para clasificacion de series.

    Args:
        input_dim: Numero de canales C (indices espectrales).
        num_classes: Numero de clases en el cabezal final.
        sequencelength: Longitud temporal T de cada serie.
        hidden_dim: Filtros por bloque convolucional (default 64).
        kernel_size: Tamano del kernel temporal (default 5).
        dropout: Dropout despues de cada bloque conv y antes del dense
            (default 0.5, como en el paper).

    Input:
        Tensor ``(B, T, C)``.

    Output:
        Logits ``(B, num_classes)``.

    Referencia:
        Pelletier, Webb & Petitjean (2019), Remote Sensing 11(5):523.
        DOI 10.3390/rs11050523.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequencelength: int,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sequencelength = sequencelength
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dropout_p = dropout

        self.block1 = _TempCNNBlock(input_dim, hidden_dim, kernel_size, dropout)
        self.block2 = _TempCNNBlock(hidden_dim, hidden_dim, kernel_size, dropout)
        self.block3 = _TempCNNBlock(hidden_dim, hidden_dim, kernel_size, dropout)

        flatten_size = hidden_dim * sequencelength
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(flatten_size, hidden_dim * 4)
        self.dense_bn = nn.BatchNorm1d(hidden_dim * 4)
        self.dense_act = nn.ReLU(inplace=True)
        self.dense_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input (B, T, C) -> Conv1d espera (B, C, T)
        x = x.transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.dense_dropout(self.dense_act(self.dense_bn(self.dense(x))))
        return self.classifier(x)


# ---------------------------------------------------------------------------
# InceptionTime  (Fawaz et al. 2020)
# ---------------------------------------------------------------------------


class _InceptionModule(nn.Module):
    """Modulo Inception 1D: bottleneck + 3 convoluciones paralelas + maxpool.

    Args:
        in_channels: Canales de entrada.
        nb_filters: Filtros por rama (4 ramas total, output = 4*nb_filters).
        kernel_sizes: Tamanos de kernel para las 3 convoluciones paralelas.
        bottleneck_channels: Canales del bottleneck inicial (0 = sin
            bottleneck, recomendado si in_channels > 1).
        use_bias: Si las convoluciones llevan bias.
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
        bottleneck_channels: int = 32,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if bottleneck_channels > 0 and in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=use_bias
            )
            conv_in = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in = in_channels

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in,
                    nb_filters,
                    kernel_size=ks,
                    padding=ks // 2,
                    bias=use_bias,
                )
                for ks in kernel_sizes
            ]
        )
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=use_bias),
        )

        total_out = nb_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bottleneck is not None:
            bottlenecked = self.bottleneck(x)
        else:
            bottlenecked = x
        branches = [conv(bottlenecked) for conv in self.conv_branches]
        # Ajusta a la longitud temporal minima (los kernels grandes pueden
        # producir T+1 segun parity).
        target_t = min(b.size(-1) for b in branches)
        branches = [b[..., :target_t] for b in branches]

        maxpool_out = self.maxpool_branch(x)[..., :target_t]
        merged = torch.cat([*branches, maxpool_out], dim=1)
        return self.act(self.bn(merged))


class _ShortcutBlock(nn.Module):
    """Shortcut residual (Conv1d 1x1 + BatchNorm + ReLU)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, residual: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if residual.size(-1) != out.size(-1):
            residual = residual[..., : out.size(-1)]
        shortcut = self.bn(self.conv(residual))
        return self.act(shortcut + out)


class InceptionTime(nn.Module):
    """InceptionTime: 6 modulos Inception con shortcut residual cada 3.

    Args:
        input_dim: Numero de canales C (indices espectrales).
        num_classes: Numero de clases en el cabezal final.
        nb_filters: Filtros por rama del modulo Inception.
        depth: Numero de modulos Inception apilados (default 6).
        kernel_sizes: Tamanos de kernel paralelos en cada modulo.
        bottleneck_channels: Canales del bottleneck (default 32).
        dropout: Dropout antes del classifier final.

    Input:
        Tensor ``(B, T, C)``.

    Output:
        Logits ``(B, num_classes)``.

    Referencia:
        Fawaz, Lucas, Forestier et al. (2020), Data Mining and Knowledge
        Discovery 34. DOI 10.1007/s10618-020-00710-y.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        nb_filters: int = 32,
        depth: int = 6,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
        bottleneck_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.nb_filters = nb_filters
        self.depth = depth

        out_channels_per_module = nb_filters * (len(kernel_sizes) + 1)

        modules: list[nn.Module] = []
        shortcuts: list[nn.Module | None] = []
        current_in = input_dim
        for d in range(depth):
            module = _InceptionModule(
                in_channels=current_in,
                nb_filters=nb_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_bias=False,
            )
            modules.append(module)
            current_in = out_channels_per_module
            # Shortcut cada 3 bloques (segun paper).
            shortcuts.append(None if (d + 1) % 3 != 0 else _ShortcutBlock(0, 0))

        self.inception_modules = nn.ModuleList(modules)
        # Construimos shortcuts sabiendo el input residual exacto en forward.
        # Aqui solo guardamos placeholders; los Conv1d 1x1 se crean lazy.
        self._build_shortcuts(input_dim, out_channels_per_module)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_channels_per_module, num_classes)

        self._init_weights()

    def _build_shortcuts(
        self, input_dim: int, out_channels_per_module: int
    ) -> None:
        """Crea los shortcuts 1x1 sabiendo los canales reales en cada brinco.

        Shortcuts cada 3 bloques: el primero compara input vs salida del bloque
        2 (index 2 con +1=3); el segundo compara salida bloque 2 (out_ch_per)
        vs salida bloque 5 (out_ch_per).
        """
        sc_modules: list[nn.Module] = []
        for d in range(self.depth):
            if (d + 1) % 3 != 0:
                continue
            if d == 2:
                # Shortcut input -> salida bloque 2
                sc_modules.append(_ShortcutBlock(input_dim, out_channels_per_module))
            else:
                # Shortcuts subsiguientes: salida previa -> salida actual
                sc_modules.append(
                    _ShortcutBlock(out_channels_per_module, out_channels_per_module)
                )
        self.shortcuts = nn.ModuleList(sc_modules)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                if module.weight.numel() == 0:
                    continue
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input (B, T, C) -> Conv1d espera (B, C, T).
        x = x.transpose(1, 2)
        residual = x
        shortcut_idx = 0
        out = x
        for d, module in enumerate(self.inception_modules):
            out = module(out)
            if (d + 1) % 3 == 0:
                out = self.shortcuts[shortcut_idx](residual, out)
                residual = out
                shortcut_idx += 1
        pooled = self.global_pool(out).squeeze(-1)  # (B, out_channels)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Factory helper para selector por nombre.
# ---------------------------------------------------------------------------


def build_temporal_model(
    model_kind: str,
    *,
    input_dim: int,
    num_classes: int,
    sequence_length: int,
    **overrides: object,
) -> nn.Module:
    """Construye un modelo temporal por nombre.

    Args:
        model_kind: ``"tempcnn"`` o ``"inceptiontime"``.
        input_dim: Numero de canales C.
        num_classes: Numero de clases efectivas.
        sequence_length: Longitud temporal T.
        **overrides: Hiperparametros adicionales pasados al constructor del
            modelo (``hidden_dim``, ``dropout``, ``depth``, etc.).

    Returns:
        ``nn.Module`` listo para entrenar.

    Raises:
        ValueError: si ``model_kind`` no es uno de los soportados.
    """
    if model_kind == "tempcnn":
        return TempCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            sequencelength=sequence_length,
            **overrides,  # type: ignore[arg-type]
        )
    if model_kind == "inceptiontime":
        return InceptionTime(
            input_dim=input_dim,
            num_classes=num_classes,
            **overrides,  # type: ignore[arg-type]
        )
    raise ValueError(
        f"model_kind={model_kind!r} no soportado. Usa 'tempcnn' o 'inceptiontime'."
    )
