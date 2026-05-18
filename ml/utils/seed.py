"""Helper unico de propagacion de seed para training reproducible (US-017+).

Reemplaza las copias duplicadas en `ml/farslip/distill.py` y
`ml/farslip/train.py`. Activa ``torch.use_deterministic_algorithms`` con
``warn_only=True`` para no romper kernels que carecen de implementacion
deterministica, y setea ``CUBLAS_WORKSPACE_CONFIG`` para CUDA >= 10.2.
"""

from __future__ import annotations

import os
import random

import numpy as np
import structlog
import torch

_log = structlog.get_logger(__name__)


def propagate_seed(seed: int, *, deterministic: bool = True) -> None:
    """Propaga ``seed`` a ``random``, ``numpy``, ``torch`` (CPU + CUDA).

    Args:
        seed: entero usado como semilla en los 4 RNGs.
        deterministic: si ``True``, activa
            ``torch.use_deterministic_algorithms(True, warn_only=True)`` y
            setea ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` (requerido por
            CUDA >= 10.2 para algoritmos cuBLAS deterministicos).

    Notas:
        - ``warn_only=True`` permite que operaciones sin implementacion
          deterministica caigan a la version no-determinista emitiendo solo
          warning, en lugar de lanzar excepcion. Necesario para suites de
          tests que ejercitan modelos completos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError) as exc:  # pragma: no cover
            _log.debug("deterministic algorithms no soportadas", error=str(exc))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


__all__ = ["propagate_seed"]
