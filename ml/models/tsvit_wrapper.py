"""TSViT factorizado para segmentacion densa de series Sentinel-2.

Reimplementacion limpia (no clon del repo externo) del Temporal-Spatial
Vision Transformer (TSViT) de Tarasiou, Chavez & Zafeiriou (2023),
"ViTs for SITS: Vision Transformers for Satellite Image Time Series"
(arXiv:2301.04944, CVPR 2023). La idea central de TSViT es **invertir el
orden tipico de los video-ViT**: en lugar de atender primero al espacio y
luego al tiempo, TSViT factoriza el self-attention aplicando **primero el
encoder temporal** (a lo largo del eje de adquisiciones) y **despues el
encoder espacial** (a lo largo de los tokens de parche). Esto explota la
estructura agronomica de las series: el patron fenologico temporal es la
senal mas discriminante para el tipo de cultivo.

Componentes (Secciones 3.1-3.3 del paper):

1. **Tokenizacion 3D por parches** ``(t=1, h, w)``: una ``Conv2d`` con
   ``kernel=stride=patch_size`` se aplica de forma independiente a cada
   imagen temporal, produciendo ``N = (H/p) * (W/p)`` tokens espaciales por
   timestep con dimension ``dim``.
2. **Positional encoding temporal por fecha real**: tabla aprendida indexada
   por dia-del-anio (DOY, 1..365) en lugar de por posicion ordinal. Acepta
   el ``doy`` del batch, lo que hace al modelo invariante al numero de
   adquisiciones y consciente de la fecha real (Seccion 3.2). Si no se pasa
   ``doy`` se cae a un PE temporal ordinal aprendido.
3. **Encoder TEMPORAL**: ``K`` cls-tokens separables (uno por clase) se
   anteponen a la secuencia temporal de cada posicion espacial; el
   self-attention recorre el eje temporal. Cada cls-token aprende a resumir
   la evidencia temporal para SU clase (Seccion 3.3, "multiple cls tokens").
4. **Encoder ESPACIAL**: tras el temporal se conserva un token por
   ``(clase, posicion-espacial)``; un PE espacial aprendido se suma y el
   self-attention recorre el eje espacial para cada clase.
5. **Head de segmentacion densa**: reconstruye logits ``(B, K, H, W)``
   proyectando cada token de parche a ``p*p`` y reordenando a resolucion
   plena.
6. **Rama visual contrastiva (US-025 Seccion A)**: ademas del head de
   segmentacion, expone una proyeccion opcional de las **features visuales
   por pixel** al espacio semantico de dimension ``semantic_dim`` (384, el
   de los prototipos fenologicos de
   :mod:`ml.features.phenology_class_prototypes`). Permite la alineacion
   contrastiva pixel-visual <-> prototipo-de-clase del metodo de Wen et al.
   (2025) sin concatenar texto al vector (ver Seccion A.0 del plan US-025).

Recorte para L4 (de-risk, no se dispone de H100 hoy): con ``T=10``, 128px,
``patch_size=8`` (16x16 = 256 tokens espaciales) y ``dim=128``,
``depth_temporal=depth_spatial=4`` el modelo entra holgado en una L4 24GB
con ``batch=4``. No se exageran ``dim``/``depth`` para mantener el
entrenamiento viable en la ventana de computo asignada.

Atribucion: arquitectura de Tarasiou et al. (2023), arXiv:2301.04944
(repo de referencia ``michaeltrs/DeepSatModels``, licencia Apache-2.0).
Esta es una reimplementacion propia; documentada en
``docs/licenses/DATA_LICENSE.md``.
"""

from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn

__all__ = ["TSViT", "build_tsvit"]


# ---------------------------------------------------------------------------
# Bloques Transformer base (pre-norm, estilo ViT)
# ---------------------------------------------------------------------------


class _FeedForward(nn.Module):
    """MLP de dos capas con GELU usado dentro de cada bloque Transformer.

    Args:
        dim: Dimension de entrada y salida.
        hidden_dim: Dimension oculta (expansion intermedia).
        dropout: Probabilidad de dropout aplicada tras cada lineal.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    """Multi-head self-attention escalada (scaled dot-product).

    Args:
        dim: Dimension del token.
        heads: Numero de cabezas de atencion.
        dim_head: Dimension por cabeza.
        dropout: Dropout sobre la salida de la proyeccion.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_aplanado, n_tokens, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv
        )
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _TransformerBlock(nn.Module):
    """Bloque Transformer pre-norm: LN -> Attn -> res; LN -> MLP -> res.

    Args:
        dim: Dimension del token.
        heads: Cabezas de atencion.
        dim_head: Dimension por cabeza.
        mlp_dim: Dimension oculta del feed-forward.
        dropout: Dropout en atencion y MLP.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class _Transformer(nn.Module):
    """Pila de ``depth`` bloques Transformer con LayerNorm final.

    Args:
        dim: Dimension del token.
        depth: Numero de bloques.
        heads: Cabezas de atencion.
        dim_head: Dimension por cabeza.
        mlp_dim: Dimension oculta del feed-forward.
        dropout: Dropout interno.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# TSViT
# ---------------------------------------------------------------------------


class TSViT(nn.Module):
    """Temporal-Spatial ViT factorizado para segmentacion densa de SITS.

    Implementa la arquitectura de Tarasiou et al. (2023): tokenizacion 3D por
    parches, encoder **temporal primero** con ``K`` cls-tokens separables
    (uno por clase), encoder **espacial despues**, positional encoding
    temporal por fecha real (DOY) y un head de segmentacion densa que
    reconstruye ``(B, K, H, W)``.

    Ademas del head de segmentacion expone una **proyeccion visual por pixel**
    al espacio semantico (``semantic_dim``) para la alineacion contrastiva con
    los prototipos fenologicos por clase (US-025 Seccion A, Wen et al. 2025).

    Args:
        num_classes: Numero ``K`` de clases; tambien el numero de cls-tokens
            temporales separables y los canales de salida del head de
            segmentacion.
        n_timesteps: Longitud temporal ``T`` esperada (define el PE temporal
            ordinal de respaldo cuando no se pasa ``doy``).
        img_size: Lado ``H = W`` del patch de entrada en pixeles.
        in_channels: Numero de bandas de entrada por timestep (10 para
            Sentinel-2 PASTIS-R).
        patch_size: Lado del parche espacial ``p``. Produce
            ``(img_size/p)^2`` tokens espaciales por timestep.
        dim: Dimension del token Transformer.
        depth_temporal: Numero de bloques del encoder temporal.
        depth_spatial: Numero de bloques del encoder espacial.
        heads: Cabezas de atencion en ambos encoders.
        dim_head: Dimension por cabeza de atencion.
        mlp_ratio: Factor de expansion del feed-forward (``mlp_dim = dim *
            mlp_ratio``).
        semantic_dim: Dimension del espacio semantico de la rama contrastiva
            (384 para coincidir con los embeddings de
            ``all-MiniLM-L6-v2`` de los prototipos por clase).
        dropout: Dropout aplicado en los Transformers.
        max_doy: Maximo dia-del-anio admitido por la tabla de PE temporal
            (366 para cubrir anios bisiestos; el indice 0 queda sin uso).

    Raises:
        ValueError: Si ``img_size`` no es divisible por ``patch_size``.
    """

    def __init__(
        self,
        num_classes: int = 18,
        n_timesteps: int = 10,
        img_size: int = 128,
        in_channels: int = 10,
        patch_size: int = 8,
        dim: int = 128,
        depth_temporal: int = 4,
        depth_spatial: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        mlp_ratio: int = 4,
        semantic_dim: int = 384,
        dropout: float = 0.0,
        max_doy: int = 366,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) debe ser divisible por patch_size "
                f"({patch_size})."
            )

        self.num_classes = num_classes
        self.n_timesteps = n_timesteps
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.semantic_dim = semantic_dim
        self.max_doy = max_doy

        self.grid = img_size // patch_size  # tokens por lado
        self.num_patches = self.grid * self.grid  # N tokens espaciales

        mlp_dim = dim * mlp_ratio

        # --- Tokenizacion 3D (t=1, p, p): Conv2d por imagen temporal --------
        self.to_patch_embedding = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # --- Positional encoding temporal por DOY (tabla aprendida) ---------
        # Indexada por dia-del-anio real (1..max_doy). Fila 0 reservada.
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(max_doy + 1, dim) * 0.02
        )
        # PE temporal ordinal de respaldo cuando no se pasa doy.
        self.temporal_pos_ordinal = nn.Parameter(
            torch.randn(1, n_timesteps, dim) * 0.02
        )

        # --- K cls-tokens temporales separables (uno por clase) -------------
        self.temporal_cls_tokens = nn.Parameter(
            torch.randn(1, num_classes, dim) * 0.02
        )

        # --- Encoder temporal ----------------------------------------------
        self.temporal_transformer = _Transformer(
            dim, depth_temporal, heads, dim_head, mlp_dim, dropout
        )

        # --- Positional encoding espacial aprendido -------------------------
        self.spatial_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, dim) * 0.02
        )

        # --- Encoder espacial ----------------------------------------------
        self.spatial_transformer = _Transformer(
            dim, depth_spatial, heads, dim_head, mlp_dim, dropout
        )

        # --- Head de segmentacion densa ------------------------------------
        # Cada token de parche se proyecta a p*p valores; el reorder reconstruye
        # la resolucion plena. Una proyeccion por clase mantiene la separacion
        # de los K cls-tokens.
        self.to_seg = nn.Linear(dim, patch_size * patch_size)

        # --- Rama visual contrastiva (proyeccion al espacio semantico) ------
        # Proyecta la feature por (clase, parche) a semantic_dim; el reorder a
        # pixel produce (B, semantic_dim, H, W).
        self.to_visual_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, semantic_dim * patch_size * patch_size),
        )

    def _tokenize(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Tokeniza la entrada ``(B, T, C, H, W)`` a tokens de parche.

        Args:
            x: Tensor de entrada ``(B, T, C, H, W)``.

        Returns:
            Tupla ``(tokens, batch)`` donde ``tokens`` tiene forma
            ``(B, T, N, dim)`` con ``N`` tokens espaciales por timestep.
        """
        b, t = x.shape[0], x.shape[1]
        # Conv2d se aplica a cada imagen temporal de forma independiente.
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.to_patch_embedding(x)  # (b*t, dim, grid, grid)
        x = rearrange(x, "(b t) d gh gw -> b t (gh gw) d", b=b, t=t)
        return x, b

    def _temporal_pos(
        self, doy: torch.Tensor | None, batch: int, t: int, device: torch.device
    ) -> torch.Tensor:
        """Devuelve el PE temporal ``(B, T, dim)`` por DOY o ordinal.

        Args:
            doy: Dia-del-anio por timestep ``(B, T)`` int, o ``None``.
            batch: Tamano de batch ``B``.
            t: Numero de timesteps ``T``.
            device: Dispositivo destino.

        Returns:
            Tensor ``(B, T, dim)`` con el positional encoding temporal.
        """
        if doy is None:
            return self.temporal_pos_ordinal[:, :t, :].expand(batch, -1, -1)
        doy_idx = doy.long().clamp(0, self.max_doy).to(device)
        return self.temporal_pos_embedding[doy_idx]  # (B, T, dim)

    def forward(
        self,
        x: torch.Tensor,
        doy: torch.Tensor | None = None,
        *,
        return_visual_proj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pasa la serie por los encoders factorizados y reconstruye logits.

        Args:
            x: Serie de entrada ``(B, T, C, H, W)`` float (B4 idx2, B8 idx6,
                10 bandas Sentinel-2 PASTIS-R, escala /10000).
            doy: Dia-del-anio de adquisicion por timestep ``(B, T)`` int. Si
                ``None`` se usa el PE temporal ordinal de respaldo.
            return_visual_proj: Si ``True``, devuelve ademas la proyeccion
                visual por pixel al espacio semantico ``semantic_dim``.

        Returns:
            - ``return_visual_proj=False``: logits de segmentacion
              ``(B, num_classes, H, W)``.
            - ``return_visual_proj=True``: tupla ``(logits, visual_proj)`` con
              ``visual_proj`` de forma ``(B, semantic_dim, H, W)``.
        """
        tokens, b = self._tokenize(x)  # (B, T, N, dim)
        t = tokens.shape[1]
        n = tokens.shape[2]
        device = tokens.device

        # --- PE temporal por DOY (sumado a cada token de cada posicion) -----
        temp_pos = self._temporal_pos(doy, b, t, device)  # (B, T, dim)
        tokens = tokens + temp_pos.unsqueeze(2)  # broadcast sobre N posiciones

        # --- Encoder temporal por posicion espacial -------------------------
        # Aplana (B, N) al eje batch para atender solo el eje temporal.
        seq = rearrange(tokens, "b t n d -> (b n) t d")
        cls = repeat(
            self.temporal_cls_tokens, "1 k d -> bn k d", bn=b * n
        )  # (B*N, K, dim)
        seq = torch.cat([cls, seq], dim=1)  # (B*N, K + T, dim)
        seq = self.temporal_transformer(seq)
        # Conservar solo los K cls-tokens (resumen temporal por clase).
        cls_out = seq[:, : self.num_classes, :]  # (B*N, K, dim)

        # --- Encoder espacial por clase -------------------------------------
        # Reordena a (B*K, N, dim): para cada clase, atiende el eje espacial.
        spatial = rearrange(cls_out, "(b n) k d -> (b k) n d", b=b, n=n)
        spatial = spatial + self.spatial_pos_embedding  # (B*K, N, dim)
        spatial = self.spatial_transformer(spatial)  # (B*K, N, dim)

        # --- Head de segmentacion: token de parche -> p*p pixeles -----------
        seg = self.to_seg(spatial)  # (B*K, N, p*p)
        logits = rearrange(
            seg,
            "(b k) (gh gw) (ph pw) -> b k (gh ph) (gw pw)",
            b=b,
            k=self.num_classes,
            gh=self.grid,
            gw=self.grid,
            ph=self.patch_size,
            pw=self.patch_size,
        )  # (B, K, H, W)

        if not return_visual_proj:
            return logits

        # --- Rama visual: feature por (clase, parche) -> pixel semantico ----
        # Se promedian las K ramas de clase para obtener una feature visual por
        # posicion espacial, y se proyecta a semantic_dim por pixel.
        per_class = rearrange(spatial, "(b k) n d -> b k n d", b=b)
        pooled = per_class.mean(dim=1)  # (B, N, dim) feature visual por parche
        proj = self.to_visual_proj(pooled)  # (B, N, semantic_dim*p*p)
        visual_proj = rearrange(
            proj,
            "b (gh gw) (s ph pw) -> b s (gh ph) (gw pw)",
            gh=self.grid,
            gw=self.grid,
            s=self.semantic_dim,
            ph=self.patch_size,
            pw=self.patch_size,
        )  # (B, semantic_dim, H, W)
        return logits, visual_proj


def build_tsvit(
    num_classes: int = 18,
    n_timesteps: int = 10,
    img_size: int = 128,
    in_channels: int = 10,
    patch_size: int = 8,
    dim: int = 128,
    depth_temporal: int = 4,
    depth_spatial: int = 4,
    semantic_dim: int = 384,
) -> nn.Module:
    """Construye un :class:`TSViT` con los defaults recortados para L4.

    Factory publica del wrapper TSViT (US-025 Tarea 3). Los defaults
    (``patch_size=8`` -> 16x16 tokens, ``dim=128``, profundidad 4+4) mantienen
    el modelo entrenable en una L4 24GB con ``T=10``, 128px y ``batch=4``.

    Args:
        num_classes: Numero ``K`` de clases / cls-tokens separables.
        n_timesteps: Longitud temporal ``T`` esperada.
        img_size: Lado del patch de entrada en pixeles.
        in_channels: Bandas de entrada por timestep (10 para Sentinel-2).
        patch_size: Lado del parche espacial.
        dim: Dimension del token Transformer.
        depth_temporal: Bloques del encoder temporal.
        depth_spatial: Bloques del encoder espacial.
        semantic_dim: Dimension del espacio semantico de la rama contrastiva
            (384 para los prototipos fenologicos por clase).

    Returns:
        Modulo :class:`TSViT` listo para entrenar/inferir.
    """
    return TSViT(
        num_classes=num_classes,
        n_timesteps=n_timesteps,
        img_size=img_size,
        in_channels=in_channels,
        patch_size=patch_size,
        dim=dim,
        depth_temporal=depth_temporal,
        depth_spatial=depth_spatial,
        semantic_dim=semantic_dim,
    )
