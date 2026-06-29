"""Class-separability diagnostics for the Italian TL dataset (US-082 scoping).

Answers a single scoping question with zero training cost: how many Italian crop
classes are *physically* separable in the AlphaEarth embedding space, and at what
class support, BEFORE committing GPU to a fine-tune or a UDA loop. The diagnostic
reads the per-parcel AlphaEarth features (``data/features/alphaearth_italia_2018``
materialised by US-078) and reports, per class:

- the parcel support (the real ceiling for a fold-5 macro-F1, distinct from the
  pixel count that the dense report inflates),
- the mean pairwise Jeffries-Matusita (JM) distance to the other classes, a
  bounded [0, 2] separability score derived from the Bhattacharyya distance under
  a Gaussian assumption (JM > 1.9 = well separated, < 1.0 = heavy overlap),
- the nearest confusable class (smallest JM), which names the structural
  collision the crosswalk cannot resolve.

The verdict aggregates how many classes clear support and separability gates, so a
realistic per-class target (e.g. "rescue N classes at F1 >= 0.6") can be set on
evidence instead of hope. No model, no GPU, no synthetic data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

#: Default US-078 homologue dataset root (where ``class_mapping.json`` lives).
DEFAULT_ITALIA_ROOT = Path("data/pastis_italia_2018")

#: Default per-parcel AlphaEarth feature table (US-078 output, DVC-tracked).
DEFAULT_FEATURES_PATH = Path("data/features/alphaearth_italia_2018.parquet")

#: AlphaEarth Satellite Embedding V1 Annual dimensionality.
_EMBED_DIM = 64

#: Embedding column names in the feature table (``dim_00`` .. ``dim_63``).
_EMBED_COLS = tuple(f"dim_{i:02d}" for i in range(_EMBED_DIM))

#: Minimum parcels for a class to be a credible fold-5 target. Below this the
#: held-out fold carries too few parcels for a stable F1 (a single parcel swings
#: it by tens of points), so the class is reported as support-starved.
_MIN_PARCELS_FOR_TARGET = 30

#: JM gate (of 2.0) above which two class centroids are considered separable in
#: the embedding space. 1.0 is the classic "fair separability" threshold.
_JM_SEPARABLE = 1.0


@dataclass(frozen=True)
class ClassSeparability:
    """Per-class separability summary in the AlphaEarth embedding space.

    Attributes:
        class_id: Dense Italian class id (``[1, K]``).
        name: HCAT4 leaf name.
        n_parcels: Parcel support in the feature table (the fold-5 F1 ceiling).
        mean_jm: Mean Jeffries-Matusita distance to all other classes (``[0, 2]``).
        nearest_name: Name of the most confusable class (smallest JM).
        nearest_jm: JM distance to that nearest class.
    """

    class_id: int
    name: str
    n_parcels: int
    mean_jm: float
    nearest_name: str
    nearest_jm: float

    @property
    def has_support(self) -> bool:
        """Whether the class clears the minimum-parcel gate for a fold-5 target."""
        return self.n_parcels >= _MIN_PARCELS_FOR_TARGET

    @property
    def is_separable(self) -> bool:
        """Whether the class is separable from its nearest neighbour (JM gate)."""
        return self.nearest_jm >= _JM_SEPARABLE

    @property
    def is_rescuable(self) -> bool:
        """A class is a realistic target only with both support AND separability."""
        return self.has_support and self.is_separable


@dataclass(frozen=True)
class SeparabilityReport:
    """Dataset-level separability verdict for US-082 scoping.

    Attributes:
        classes: Per-class summaries, sorted by descending mean JM.
        n_total: Number of classes with at least one parcel.
        n_with_support: Classes clearing the parcel-support gate.
        n_separable: Classes clearing the JM-separability gate.
        n_rescuable: Classes clearing BOTH gates (the realistic target count).
    """

    classes: tuple[ClassSeparability, ...]
    n_total: int
    n_with_support: int
    n_separable: int
    n_rescuable: int


def _bhattacharyya_gaussian(
    mu_a: np.ndarray, cov_a: np.ndarray, mu_b: np.ndarray, cov_b: np.ndarray
) -> float:
    """Bhattacharyya distance between two multivariate Gaussians.

    Uses the closed form ``1/8 (mu_a - mu_b)^T Sigma^-1 (mu_a - mu_b) +
    1/2 ln(det Sigma / sqrt(det Sigma_a det Sigma_b))`` with ``Sigma`` the average
    covariance. Covariances are diagonally regularised so the inverse and log-det
    stay finite when a class has few parcels.

    Args:
        mu_a: Mean vector of class A (``(D,)``).
        cov_a: Covariance of class A (``(D, D)``).
        mu_b: Mean vector of class B (``(D,)``).
        cov_b: Covariance of class B (``(D, D)``).

    Returns:
        The non-negative Bhattacharyya distance.
    """
    dim = mu_a.shape[0]
    eps = 1e-6
    reg = eps * np.eye(dim)
    cov_a = cov_a + reg
    cov_b = cov_b + reg
    sigma = 0.5 * (cov_a + cov_b)
    diff = (mu_a - mu_b).reshape(-1, 1)
    sigma_inv = np.linalg.pinv(sigma)
    term_mahalanobis = float(0.125 * (diff.T @ sigma_inv @ diff).item())
    sign, logdet_sigma = np.linalg.slogdet(sigma)
    _, logdet_a = np.linalg.slogdet(cov_a)
    _, logdet_b = np.linalg.slogdet(cov_b)
    term_cov = float(0.5 * (logdet_sigma - 0.5 * (logdet_a + logdet_b)))
    if sign <= 0:
        return term_mahalanobis
    return term_mahalanobis + term_cov


def _jeffries_matusita(bhattacharyya: float) -> float:
    """Convert a Bhattacharyya distance to a Jeffries-Matusita distance in ``[0, 2]``.

    Args:
        bhattacharyya: A non-negative Bhattacharyya distance.

    Returns:
        ``2 (1 - exp(-B))``, the bounded JM separability score.
    """
    return float(2.0 * (1.0 - np.exp(-max(bhattacharyya, 0.0))))


def compute_separability(
    *,
    features_path: Path = DEFAULT_FEATURES_PATH,
    italia_root: Path = DEFAULT_ITALIA_ROOT,
) -> SeparabilityReport:
    """Compute per-class separability of the Italian dataset in AlphaEarth space.

    Args:
        features_path: Per-parcel AlphaEarth feature parquet (``dim_00``..``dim_63``
            plus ``class_id``).
        italia_root: Dataset root holding ``class_mapping.json`` for the id->name map.

    Returns:
        A :class:`SeparabilityReport` with per-class JM, support and gate counts.

    Raises:
        FileNotFoundError: if the feature table or class mapping is missing.
    """
    if not features_path.is_file():
        raise FileNotFoundError(
            f"missing {features_path}; run `dvc pull {features_path}` (US-078 feature)"
        )
    mapping_path = italia_root / "class_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"missing {mapping_path}; run the US-078 builder first")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    id_to_name = {int(c["class_id"]): str(c["hcat4_name"]) for c in mapping["classes"]}

    frame = pl.read_parquet(features_path)
    embed = frame.select(_EMBED_COLS).to_numpy()
    class_ids = frame["class_id"].to_numpy()

    present_ids = sorted(int(c) for c in np.unique(class_ids))
    stats: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for cid in present_ids:
        rows = embed[class_ids == cid]
        cov = np.cov(rows, rowvar=False) if rows.shape[0] > 1 else np.eye(_EMBED_DIM)
        stats[cid] = (rows.mean(axis=0), np.atleast_2d(cov), rows.shape[0])

    summaries: list[ClassSeparability] = []
    for cid in present_ids:
        mu_a, cov_a, n_a = stats[cid]
        jms: list[tuple[float, int]] = []
        for other in present_ids:
            if other == cid:
                continue
            mu_b, cov_b, _ = stats[other]
            jm = _jeffries_matusita(_bhattacharyya_gaussian(mu_a, cov_a, mu_b, cov_b))
            jms.append((jm, other))
        if not jms:
            continue
        mean_jm = float(np.mean([j for j, _ in jms]))
        nearest_jm, nearest_id = min(jms, key=lambda t: t[0])
        summaries.append(
            ClassSeparability(
                class_id=cid,
                name=id_to_name.get(cid, f"class_{cid}"),
                n_parcels=n_a,
                mean_jm=mean_jm,
                nearest_name=id_to_name.get(nearest_id, f"class_{nearest_id}"),
                nearest_jm=nearest_jm,
            )
        )

    summaries.sort(key=lambda s: s.mean_jm, reverse=True)
    return SeparabilityReport(
        classes=tuple(summaries),
        n_total=len(summaries),
        n_with_support=sum(1 for s in summaries if s.has_support),
        n_separable=sum(1 for s in summaries if s.is_separable),
        n_rescuable=sum(1 for s in summaries if s.is_rescuable),
    )
