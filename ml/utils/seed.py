"""Single seed-propagation helper for reproducible training (US-017+).

Replaces the duplicated copies in `ml/farslip/distill.py` and
`ml/farslip/train.py`. Enables ``torch.use_deterministic_algorithms`` with
``warn_only=True`` so as not to break kernels that lack a deterministic
implementation, and sets ``CUBLAS_WORKSPACE_CONFIG`` for CUDA >= 10.2.
"""

from __future__ import annotations

import os
import random

import numpy as np
import structlog
import torch

_log = structlog.get_logger(__name__)


def propagate_seed(seed: int, *, deterministic: bool = True) -> None:
    """Propagate ``seed`` to ``random``, ``numpy``, ``torch`` (CPU + CUDA).

    Args:
        seed: integer used as the seed for the 4 RNGs.
        deterministic: if ``True``, enables
            ``torch.use_deterministic_algorithms(True, warn_only=True)`` and
            sets ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` (required by
            CUDA >= 10.2 for deterministic cuBLAS algorithms).

    Notes:
        - ``warn_only=True`` lets operations without a deterministic
          implementation fall back to the non-deterministic version emitting
          only a warning, instead of raising an exception. Needed for test
          suites that exercise full models.
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
