"""U-TAE — U-Net with Lightweight Temporal Attention Encoder (Garnot & Landrieu 2021).

Arquitectura de segmentacion semantica temporal de series Sentinel-2: un U-Net
2D codifica cada timestep de forma independiente, un L-TAE (Lightweight Temporal
Attention Encoder) agrega la dimension temporal en el cuello de botella, y los
pesos de atencion temporal re-ponderan las skip connections antes del decoder.

Entrada ``(B, T, C_in, H, W)`` (parche multi-temporal) + ``batch_positions``
``(B, T)`` (day-of-year para el encoding posicional). Salida
``(B, num_classes, H, W)``.

Esta implementacion es el port verbatim (mismas dimensiones y nombres de modulo)
del modelo entrenado por Isaac en ``notebooks/segmentation/04j_segmentation_utae``,
de modo que su checkpoint (``best_model.pt``, ``model_state_dict`` con claves
``in_conv`` / ``down_convs`` / ``temporal_encoder`` / ...) carga sin renombrar.
Se porta a modulo para poder construir el modelo fuera del notebook (Optuna,
inferencia) respetando separation of concerns (regla CLAUDE.md 8).

Referencia: V. Sainte Fare Garnot, L. Landrieu, "Panoptic Segmentation of
Satellite Image Time Series with Convolutional Temporal Attention Networks",
ICCV 2021.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["UTAE", "build_utae"]


class PositionalEncoding(nn.Module):
    """Encoding posicional sinusoidal sumado a una serie ``(B, T, C, H, W)``."""

    def __init__(self, d: int, T: int = 1000, repeat: int | None = None) -> None:
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Suma el encoding posicional por timestep.

        Args:
            x: Serie ``(B, T, C, H, W)``.
            positions: Posiciones temporales ``(B, T)`` (day-of-year enteros).

        Returns:
            ``x`` con el encoding posicional sumado, misma forma.
        """
        _, _, C, _, _ = x.shape
        pe = self._get_pe(positions, C, x.device)  # (B, T, C)
        pe = pe.unsqueeze(-1).unsqueeze(-1)  # (B, T, C, 1, 1)
        return x + pe

    def _get_pe(self, positions: torch.Tensor, d: int, device: torch.device) -> torch.Tensor:
        B, T = positions.shape
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(self.T) / d)
        )  # (d//2,)
        pos = positions.float().unsqueeze(-1)  # (B, T, 1)
        pe = torch.zeros(B, T, d, device=device)
        pe[:, :, 0::2] = torch.sin(pos * div)
        pe[:, :, 1::2] = torch.cos(pos * div)
        return pe


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder aplicado por posicion espacial.

    Entrada ``(B, T, C, H, W)`` -> salida ``(B, C_out, H, W)`` (mapa agregado en
    el tiempo). Opcionalmente devuelve los pesos de atencion ``(B, n_head, T, H, W)``
    para re-ponderar las skip connections.
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp_in: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        d_model: int = 256,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.return_att = return_att
        self.d_k = d_k

        if positional_encoding:
            self.positional_encoder = PositionalEncoding(d=in_channels, T=T, repeat=None)
        else:
            self.positional_encoder = None

        self.inlayernorm = nn.LayerNorm(in_channels)
        self.outlayernorm = nn.LayerNorm(mlp_in[-1])
        self.key_net = nn.Sequential(nn.Linear(in_channels // n_head, d_k))
        self.query = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU())

        layers: list[nn.Module] = []
        for i in range(len(mlp_in) - 1):
            layers.extend(
                [nn.Linear(mlp_in[i], mlp_in[i + 1]), nn.BatchNorm1d(mlp_in[i + 1]), nn.ReLU()]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: torch.Tensor | None = None,
        return_att: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Agrega la dimension temporal por atencion.

        Args:
            x: Serie ``(B, T, C, H, W)``.
            batch_positions: Posiciones temporales ``(B, T)`` o ``None``.
            return_att: Si ``True`` devuelve tambien los pesos de atencion.

        Returns:
            ``out`` ``(B, C_out, H, W)``, o ``(out, att)`` con
            ``att`` ``(B, n_head, T, H, W)`` si ``return_att``/``self.return_att``.
        """
        B, T, C, H, W = x.shape

        if self.positional_encoder is not None and batch_positions is not None:
            x = self.positional_encoder(x, batch_positions)

        x_flat = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        x_flat = self.inlayernorm(x_flat)

        q = self.query(x_flat.mean(dim=1))  # (B*H*W, C)

        x_heads = x_flat.view(B * H * W, T, self.n_head, C // self.n_head)
        q_heads = q.view(B * H * W, self.n_head, C // self.n_head)

        k = self.key_net(x_heads)  # (B*H*W, T, n_head, d_k)
        q_k = self.key_net(q_heads)  # (B*H*W, n_head, d_k)

        att = torch.einsum("bnd,btnd->bnt", q_k, k) / math.sqrt(self.d_k)
        att = F.softmax(att, dim=-1)  # (B*H*W, n_head, T)
        att = self.dropout(att)

        out = torch.einsum("bnt,btnc->bnc", att, x_heads)  # (B*H*W, n_head, C//n_head)
        out = out.view(B * H * W, C)
        out = self.outlayernorm(self.mlp(out))
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C_out, H, W)

        if self.return_att or return_att:
            att_out = att.view(B, H, W, self.n_head, T).permute(0, 3, 4, 1, 2)
            return out, att_out
        return out


class ConvLayer(nn.Module):
    """Bloque Conv -> GroupNorm/BatchNorm -> ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        p: int = 1,
        norm: str = "group",
        n_groups: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=norm is None)
        ]
        if norm == "group":
            layers.append(nn.GroupNorm(n_groups, out_ch))
        elif norm == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownConv(nn.Module):
    """Conv stride-2 para downsampling."""

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1, norm: str = "group"
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(4, out_ch) if norm == "group" else nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    """ConvTranspose stride-2 para upsampling."""

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1, norm: str = "group"
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(4, out_ch) if norm == "group" else nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UTAE(nn.Module):
    """U-Net con L-TAE en el cuello de botella y skip connections re-ponderadas.

    Entrada ``(B, T, C_in, H, W)`` (serie S2 multi-temporal) + ``batch_positions``
    ``(B, T)``. Salida ``(B, num_classes, H, W)``.
    """

    def __init__(
        self,
        input_dim: int = 10,
        encoder_widths: tuple[int, ...] = (32, 32, 64, 128),
        decoder_widths: tuple[int, ...] = (32, 32, 64, 128),
        out_conv: tuple[int, ...] = (32, 20),
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        encoder_norm: str = "group",
        agg_mode: str = "att_group",
        pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_widths = list(encoder_widths)
        self.decoder_widths = list(decoder_widths)
        self.pad_value = pad_value
        self.agg_mode = agg_mode
        n_levels = len(encoder_widths)

        self.in_conv = ConvLayer(input_dim, encoder_widths[0], norm=encoder_norm)
        self.down_convs = nn.ModuleList(
            [
                DownConv(encoder_widths[i], encoder_widths[i + 1], norm=encoder_norm)
                for i in range(n_levels - 1)
            ]
        )

        ltae_in = encoder_widths[-1]
        self.temporal_encoder = LTAE2d(
            in_channels=ltae_in,
            n_head=n_head,
            d_k=d_k,
            mlp_in=(ltae_in, ltae_in),
            d_model=d_model,
            return_att=True,
        )

        self.skip_agg = nn.ModuleList(
            [
                ConvLayer(encoder_widths[i], encoder_widths[i], norm=encoder_norm)
                for i in range(n_levels - 1)
            ]
        )

        self.up_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(n_levels - 1, 0, -1):
            in_ch = decoder_widths[i]
            skip_ch = encoder_widths[i - 1]
            out_ch = decoder_widths[i - 1]
            self.up_convs.append(UpConv(in_ch, out_ch, norm=encoder_norm))
            self.dec_convs.append(ConvLayer(out_ch + skip_ch, out_ch, norm=encoder_norm))

        head_layers: list[nn.Module] = []
        in_ch = decoder_widths[0]
        for out_ch in out_conv:
            head_layers.append(nn.Conv2d(in_ch, out_ch, 1))
            if out_ch != out_conv[-1]:
                head_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.out_conv = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor, batch_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Segmenta una serie temporal.

        Args:
            x: Serie ``(B, T, C_in, H, W)``.
            batch_positions: Posiciones temporales ``(B, T)`` (day-of-year).

        Returns:
            Logits densos ``(B, num_classes, H, W)``.
        """
        B, T, C, H, W = x.shape

        skips: list[torch.Tensor] = []
        xt = x.view(B * T, C, H, W)
        xt = self.in_conv(xt)
        _, w0, h0, w0s = xt.shape
        skips.append(xt.view(B, T, w0, h0, w0s))

        for down in self.down_convs:
            xt = down(xt)
            _, wi, hi, wis = xt.shape
            skips.append(xt.view(B, T, wi, hi, wis))

        bottleneck = skips[-1]
        feat, att = self.temporal_encoder(bottleneck, batch_positions)

        agg_skips: list[torch.Tensor] = []
        for i, skip in enumerate(skips[:-1]):
            _, _, _Wi, Hi, Wis = skip.shape
            att_up = F.interpolate(
                att.reshape(B * att.shape[1], T, att.shape[3], att.shape[4]),
                size=(Hi, Wis),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, att.shape[1], T, Hi, Wis)
            att_mean = att_up.mean(dim=1)  # (B, T, Hi, Wis)
            att_mean = F.softmax(att_mean, dim=1).unsqueeze(2)  # (B, T, 1, Hi, Wis)
            agg = (skip * att_mean).sum(dim=1)  # (B, Wi, Hi, Wis)
            agg = self.skip_agg[i](agg)
            agg_skips.append(agg)

        d = feat
        for up, dec, skip in zip(self.up_convs, self.dec_convs, reversed(agg_skips), strict=False):
            d = up(d)
            d = torch.cat([d, skip], dim=1)
            d = dec(d)

        return self.out_conv(d)


def build_utae(num_classes: int = 20, input_dim: int = 10) -> UTAE:
    """Construye el U-TAE con la config del checkpoint de Isaac (04j).

    Args:
        num_classes: Numero de clases de salida. El checkpoint de Isaac se
            entreno con 20 (18 cultivos PASTIS + background + void), por eso es
            el default; para 18 clases hay que reentrenar la cabeza.
        input_dim: Bandas de entrada (10 bandas S2).

    Returns:
        Modelo :class:`UTAE` listo para cargar ``model_state_dict``.
    """
    return UTAE(
        input_dim=input_dim,
        encoder_widths=(32, 32, 64, 128),
        decoder_widths=(32, 32, 64, 128),
        out_conv=(32, num_classes),
        n_head=16,
        d_model=256,
        d_k=4,
        encoder_norm="group",
        agg_mode="att_group",
    )
