"""Extractor de embeddings FarSLIP para inferencia (US-017 / US-016b).

Clase consumida por US-016 (fusion multisensor) y US-025 (SegFormer-B2 cabezal
open-vocab). Carga pesos student desde GCS con cache local + checksum opcional;
graceful fallback al cache si GCS esta offline.

API:
    extract_embeddings(crops) -> (B, 512) float32 L2-norm  (CLIP projection)
    extract_patch_features(crops) -> (B, 196, 768) float32 (vision last hidden)
    encode_text(texts) -> (N, 512) float32 L2-norm

Notas:
    - El student fue entrenado con 4 bandas Sentinel-2 (B02/B03/B04/B08).
    - El text encoder es el del teacher (frozen) -- por eso ``encode_text``
      carga ``openai/clip-vit-base-patch16`` por defecto.
"""

from __future__ import annotations

import hashlib
from contextlib import suppress
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F

try:
    from transformers import CLIPModel, CLIPTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("transformers requerido para FarSLIPExtractor") from exc

from ml.farslip.distill import adapt_patch_embed_to_n_channels
from ml.utils.gcs_errors import is_gcs_auth_error

_log = structlog.get_logger(__name__)


DEFAULT_TEACHER_ID = "openai/clip-vit-base-patch16"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "agrosat" / "farslip"


class FarSLIPExtractor:
    """Lazy loader de pesos student FarSLIP + text encoder teacher.

    Args:
        weights_uri: ``gs://...`` o ruta local al archivo ``student.safetensors``
            (o un directorio que lo contenga). Si es ``None``, usa los pesos
            teacher CLIP sin destilar (modo placeholder degradado).
        device: ``"cuda"``, ``"cpu"`` o ``"auto"``.
        cache_dir: carpeta local de cache (default ``~/.cache/agrosat/farslip/``).
        n_in_channels: bandas Sentinel-2 (default 4).
        teacher_model_id: HF id usado para text encoder + arquitectura base.
        expected_sha1: checksum SHA1 opcional del student.safetensors.
    """

    def __init__(
        self,
        weights_uri: str | None = "gs://agrosat-models/farslip/farslip-clip-italy-v1/",
        device: str = "auto",
        cache_dir: Path | None = None,
        n_in_channels: int = 4,
        teacher_model_id: str = DEFAULT_TEACHER_ID,
        expected_sha1: str | None = None,
    ) -> None:
        self.weights_uri = weights_uri
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_in_channels = n_in_channels
        self.teacher_model_id = teacher_model_id
        self.expected_sha1 = expected_sha1

        # Modelo base (vision + text projection). Cargamos CLIPModel completo
        # para tener acceso a visual_projection y text_projection (512-dim).
        base = CLIPModel.from_pretrained(teacher_model_id)
        base.eval()
        # Adaptar vision_model.embeddings.patch_embedding a n_in_channels.
        adapt_patch_embed_to_n_channels(base.vision_model, n_in_channels)
        self.model = base.to(self.device)  # type: ignore[arg-type]

        self.tokenizer = CLIPTokenizer.from_pretrained(teacher_model_id)

        # Cargar pesos student si disponibles.
        weights_local = self._resolve_weights_local()
        if weights_local is not None:
            self._load_student_weights(weights_local)
        else:
            _log.warning(
                "FarSLIP weights no disponibles; corriendo en modo teacher (degradado)",
                weights_uri=weights_uri,
            )

        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_weights_local(self) -> Path | None:
        """Resuelve a ruta local. Descarga de GCS si necesario; cache local fallback."""
        if self.weights_uri is None:
            return None
        uri = str(self.weights_uri)

        # Si es path local directo (file://, ruta absoluta o relativa existente)
        local_candidate = Path(uri)
        if local_candidate.is_file():
            return local_candidate
        if local_candidate.is_dir():
            cand = local_candidate / "student.safetensors"
            if cand.is_file():
                return cand
            return None

        if uri.startswith("gs://"):
            cached = self._maybe_use_cache(uri)
            if cached is not None:
                # Antes de descargar, intentar usar cache valido.
                if self._validate_checksum(cached):
                    _log.info("usando cache local valido (sin GCS)", path=str(cached))
                    return cached

            try:
                return self._download_from_gcs(uri)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
                _log.warning(
                    "GCS download fallido; intentando cache local",
                    error=str(exc),
                    uri=uri,
                )
                if cached is not None and cached.exists():
                    return cached
                return None
            except Exception as exc:
                # google.api_core / google.auth: 403, 401, NotFound, creds
                # ausentes. Degradar a cache local o modo teacher en lugar
                # de propagar y romper el constructor. Cualquier otro error
                # (AttributeError, KeyError, ValueError reales) burbujea.
                if is_gcs_auth_error(exc):
                    _log.warning(
                        "GCS auth/permiso denegado; intentando cache local",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        uri=uri,
                    )
                    if cached is not None and cached.exists():
                        return cached
                    return None
                raise
        return None

    def _cache_path_for(self, uri: str) -> Path:
        # sha1 sin usebsecurity flag: solo deduplicacion de cache, no cripto.
        h = hashlib.sha1(uri.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        return self.cache_dir / f"{h}_student.safetensors"

    def _maybe_use_cache(self, uri: str) -> Path | None:
        cached = self._cache_path_for(uri)
        return cached if cached.exists() else None

    def _validate_checksum(self, path: Path) -> bool:
        if self.expected_sha1 is None:
            return True
        try:
            h = hashlib.sha1(usedforsecurity=False)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 16), b""):
                    h.update(chunk)
            return h.hexdigest() == self.expected_sha1
        except OSError:  # pragma: no cover
            return False

    #: Retries para auto-download en caso de checksum invalido tras wait.
    _DOWNLOAD_MAX_ATTEMPTS = 3
    #: Segundos de espera por iteracion del polling cuando otro proceso descarga.
    _DOWNLOAD_WAIT_INTERVAL_S = 0.5
    #: Iteraciones maximas de polling (= 30 s con interval 0.5 s).
    _DOWNLOAD_WAIT_MAX_ITERATIONS = 60

    def _is_complete_download(self, path: Path) -> bool:
        """Valida que ``path`` exista, tenga ``size > 0`` y matchee checksum.

        Esto cubre el caso en que un proceso paralelo murio mid-download
        dejando un archivo parcial (size=0 o checksum invalido). El
        checksum solo se evalua si ``self.expected_sha1`` esta seteado;
        en ausencia de hash conocido nos quedamos con el chequeo de size.
        """
        if not path.exists():
            return False
        try:
            if path.stat().st_size == 0:
                return False
        except OSError:  # pragma: no cover
            return False
        return self._validate_checksum(path)

    def _download_from_gcs(self, uri: str) -> Path:
        from google.cloud import storage  # type: ignore[import-untyped]

        if not uri.startswith("gs://"):
            raise ValueError(f"URI no GCS: {uri}")
        without_scheme = uri[len("gs://") :]
        parts = without_scheme.split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""
        # Si apunta a una carpeta, agrega student.safetensors.
        if blob_path.endswith("/") or not blob_path.endswith(".safetensors"):
            blob_path = blob_path.rstrip("/") + "/student.safetensors"

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        dest = self._cache_path_for(uri)

        import time as _time

        for attempt in range(1, self._DOWNLOAD_MAX_ATTEMPTS + 1):
            lock_path = dest.with_suffix(".safetensors.lock")
            try:
                with lock_path.open("x"):
                    try:
                        blob.download_to_filename(str(dest))
                    except BaseException:
                        # MT-3 fix: si download_to_filename revienta (403,
                        # NotFound, network drop) puede haber creado el archivo
                        # destino vacio o parcial. Limpiar antes de propagar
                        # para que la proxima invocacion no vea ese basura como
                        # cache valido (sin expected_sha1 _validate_checksum
                        # devuelve True y load_file revienta con
                        # SafetensorError: header too small).
                        with suppress(OSError):
                            dest.unlink()
                        raise
            except FileExistsError:
                # Otra ejecucion ya esta descargando. Polling al archivo final.
                for _ in range(self._DOWNLOAD_WAIT_MAX_ITERATIONS):
                    if dest.exists():
                        # Esperar a que el otro proceso libere el lock antes
                        # de validar (evita leer durante el flush final).
                        if not lock_path.exists():
                            break
                    _time.sleep(self._DOWNLOAD_WAIT_INTERVAL_S)
            finally:
                with suppress(OSError):
                    lock_path.unlink()

            # Validacion post-wait/post-download (Q7): size>0 + checksum.
            if self._is_complete_download(dest):
                return dest

            _log.warning(
                "download incompleto/checksum invalido; reintentando",
                attempt=attempt,
                dest=str(dest),
            )
            with suppress(OSError):
                dest.unlink()

        raise RuntimeError(
            f"download GCS fallo {self._DOWNLOAD_MAX_ATTEMPTS} intentos: {uri}"
        )

    def _load_student_weights(self, path: Path) -> None:
        from safetensors.torch import load_file

        state = load_file(str(path), device=str(self.device))
        # Los pesos vienen de CLIPVisionModel (state del student). Cargamos al
        # vision_model interno; ignoramos missing/unexpected keys (text encoder).
        missing, unexpected = self.model.vision_model.load_state_dict(
            state, strict=False
        )
        if missing:
            _log.warning("missing keys al cargar student", n=len(missing))
        if unexpected:
            _log.warning("unexpected keys al cargar student", n=len(unexpected))
        _log.info("FarSLIP student weights cargados", path=str(path))

    # ------------------------------------------------------------------ API

    @torch.inference_mode()
    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        """Extrae embeddings CLS proyectados a 512-dim, L2-norm.

        Args:
            crops: ``(B, 4, H, W)`` float [0,1] o uint16 raw (se normaliza).

        Returns:
            ``(B, 512)`` float32 L2-normalizados.
        """
        crops = self._prep_crops(crops)
        vision_out = self.model.vision_model(pixel_values=crops)
        pooled = vision_out.pooler_output  # (B, 768)
        embeds = self.model.visual_projection(pooled)  # (B, 512)
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds.float()

    @torch.inference_mode()
    def extract_patch_features(self, crops: torch.Tensor) -> torch.Tensor:
        """Extrae patch features (sin CLS) ``(B, 196, 768)`` para SegFormer US-025."""
        crops = self._prep_crops(crops)
        vision_out = self.model.vision_model(pixel_values=crops)
        # last_hidden_state: (B, 1+P, 768). Quitamos CLS.
        hidden: torch.Tensor = vision_out.last_hidden_state
        return hidden[:, 1:, :].float()

    def load_crops_batch(self, paths: list[str | Path]) -> torch.Tensor:
        """Lee crops Sentinel-2 desde rutas TIFF y devuelve tensor batch.

        Helper publico consumido por el asset Dagster ``farslip_embeddings_italy``
        para abstraer el I/O TIFF (4 bandas B02/B03/B04/B08 a 10 m, uint16
        reflectancia escalada). El preprocessing fino (uint16 -> [0,1], resize a
        224x224) se aplica dentro de ``extract_embeddings`` via ``_prep_crops``.

        Args:
            paths: lista de rutas a ``.tif`` (4 bandas, mismo shape esperado).

        Returns:
            ``(B, 4, H, W)`` ``torch.int32`` con valores raw uint16 (la
            normalizacion ocurre downstream).

        Raises:
            ImportError: si ``rasterio`` no esta instalado.
            FileNotFoundError: si alguna ruta no existe.
        """
        try:
            import rasterio
        except ImportError as exc:  # pragma: no cover
            raise ImportError("rasterio requerido para load_crops_batch") from exc

        if not paths:
            raise ValueError("paths esta vacio")

        arrays = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"crop TIFF no existe: {path}")
            with rasterio.open(path) as src:
                arr = src.read()  # (C, H, W)
            if arr.shape[0] != self.n_in_channels:
                raise ValueError(
                    f"crop {path} tiene {arr.shape[0]} bandas; "
                    f"esperado {self.n_in_channels}"
                )
            arrays.append(arr)

        # Stack (B, C, H, W). Cast a int32 para preservar uint16 sin overflow
        # en torch (torch no tiene uint16 nativo).
        import numpy as np

        batch = np.stack(arrays, axis=0).astype(np.int32)
        return torch.from_numpy(batch)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Codifica textos via text encoder + text_projection. L2-norm."""
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        text_out = self.model.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled = text_out.pooler_output
        embeds = self.model.text_projection(pooled)
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds.float()

    # ------------------------------------------------------------------ utils

    def _prep_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """Mueve crops al device, normaliza uint16 -> [0,1], asegura 4 canales y 224x224."""
        if crops.dtype in (torch.int16, torch.int32, torch.uint8):
            crops = crops.to(torch.float32) / 10000.0
        elif crops.dtype == torch.float64:
            crops = crops.to(torch.float32)
        crops = crops.to(self.device)
        if crops.dim() != 4:
            raise ValueError(f"crops debe ser (B,C,H,W); got {crops.shape}")
        if crops.shape[1] != self.n_in_channels:
            raise ValueError(
                f"esperado C={self.n_in_channels}; got C={crops.shape[1]}"
            )
        target = 224
        if crops.shape[-1] != target or crops.shape[-2] != target:
            crops = F.interpolate(
                crops, size=(target, target), mode="bilinear", align_corners=False
            )
        return crops


__all__ = ["FarSLIPExtractor"]
