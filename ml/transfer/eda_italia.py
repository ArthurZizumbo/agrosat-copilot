"""Formal EDA of the complete Italian TL dataset (US-082 Fase 3).

Quantifies the real shape of the 1,438-patch Italian homologue (US-078) BEFORE
and AFTER the AlphaEarth re-extraction, so the honest per-class target of the
transfer (10-15 classes at F1 >= 0.6-0.8) rests on the dataset's own statistics
instead of the 1 %-pilot artefact that produced the misleading F1 0.13 (US-079).

Everything is read with Polars from the patch ``metadata.parquet`` (one row per
patch, materialised by US-078) plus the per-patch ``TARGET_<id>.npy`` masks when a
pixel-level class distribution is requested. No model, no GPU, no synthetic data:
the rows are the real patches, the dates are the real Sentinel-2 acquisition
counts, and the class presences are the real HCAT leaves.

The report answers four scoping questions:

- **Volume**: patches, parcels, mean parcels/patch -- the dataset is NOT small
  (1,438 patches / ~107k parcels, ~58 % / ~87 % of PASTIS-France).
- **Class support**: per-class patch presence and parcel count, so a class with
  hundreds of patches (a credible fold-5 target) is told apart from the ~23
  minority tail leaves that stay hard even with the full extraction.
- **Temporal ceiling**: the per-patch Sentinel-2 date count distribution (mean
  ~24.3 vs PASTIS 43), the real structural limit on inter-cereal discrimination;
  the champion resamples to ``n_timesteps=32`` so the count of patches below 16
  dates is reported (they pad heavily and carry weak phenology).
- **Inter-class co-occurrence**: how often two leaves share a patch, a proxy for
  the spatial-context confusion the dense head must resolve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "ItaliaEdaReport",
    "compute_italia_eda",
    "load_patch_metadata",
]

#: Default US-078 homologue dataset root (where ``metadata.parquet`` and
#: ``class_mapping.json`` live). The complete dataset (1,438 patches) lives here.
DEFAULT_ITALIA_ROOT = Path("data/pastis_italia_2018")

#: The champion ``tsvit-pheno-fullm-v2`` resamples every series to this many
#: timesteps; a patch with fewer real dates pads heavily and carries weak
#: phenology, so the count below half of this is reported as the temporal-ceiling
#: cohort.
_CHAMPION_TIMESTEPS = 32
_WEAK_PHENOLOGY_DATES = _CHAMPION_TIMESTEPS // 2  # 16

#: Patch-presence floor above which a class is a credible fold-5 target. Mirrors
#: the parcel floor of :mod:`ml.transfer.separability_italia` at the patch grain.
_MIN_PATCHES_FOR_TARGET = 200


def load_patch_metadata(root: Path = DEFAULT_ITALIA_ROOT) -> pl.DataFrame:
    """Load the per-patch metadata table of the Italian homologue dataset.

    Args:
        root: Dataset root holding ``metadata.parquet`` (US-078 output).

    Returns:
        The metadata :class:`polars.DataFrame`, one row per patch. Expected
        columns (US-078 schema): ``patch_id``, ``n_parcelas``, ``n_fechas``,
        ``clases_presentes`` (list of HCAT leaf ids present in the patch),
        ``pct_cubierto``, ``fold_espacial``, and the four ``bbox_*`` corners.

    Raises:
        FileNotFoundError: If ``metadata.parquet`` is absent under ``root``.
    """
    meta_path = root / "metadata.parquet"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.parquet not found under {root}")
    return pl.read_parquet(meta_path)


@dataclass
class ItaliaEdaReport:
    """The formal EDA of the Italian TL dataset (JSON-serialisable).

    Attributes:
        n_patches: Number of patches (rows in the metadata).
        n_parcels: Total parcels across all patches.
        mean_parcels_per_patch: ``n_parcels / n_patches``.
        date_stats: ``{"mean", "min", "max", "median", "p25", "p75"}`` of the
            per-patch Sentinel-2 acquisition count.
        n_patches_weak_phenology: Patches with fewer than 16 real dates (the
            temporal-ceiling cohort that pads heavily to 32 timesteps).
        per_class: One row per HCAT leaf present, sorted by patch presence:
            ``{"class_id", "class_name", "n_patches", "n_parcels_est",
            "pct_patches", "is_target_candidate"}``.
        n_target_candidate_classes: Classes with >= 200 patch presence.
        top_cooccurrence: The most frequent inter-class patch co-occurrences:
            ``{"class_a", "class_b", "n_patches"}`` (top 25).
        fold_distribution: ``{fold_id: n_patches}`` of the spatial-CV folds.
    """

    n_patches: int
    n_parcels: int
    mean_parcels_per_patch: float
    date_stats: dict[str, float]
    n_patches_weak_phenology: int
    per_class: list[dict[str, object]] = field(default_factory=list)
    n_target_candidate_classes: int = 0
    top_cooccurrence: list[dict[str, object]] = field(default_factory=list)
    fold_distribution: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        """Return a flat JSON-friendly summary (no per-class / co-occurrence lists)."""
        return {
            "n_patches": self.n_patches,
            "n_parcels": self.n_parcels,
            "mean_parcels_per_patch": round(self.mean_parcels_per_patch, 2),
            "date_mean": round(self.date_stats.get("mean", 0.0), 2),
            "date_min": self.date_stats.get("min", 0.0),
            "date_max": self.date_stats.get("max", 0.0),
            "n_patches_weak_phenology": self.n_patches_weak_phenology,
            "n_target_candidate_classes": self.n_target_candidate_classes,
        }

    def to_json(self, path: Path) -> None:
        """Write the full report (including the lists) to ``path`` as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.summary(),
            "date_stats": self.date_stats,
            "per_class": self.per_class,
            "top_cooccurrence": self.top_cooccurrence,
            "fold_distribution": self.fold_distribution,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_class_names(root: Path) -> dict[int, str]:
    """Load the ``class_id -> HCAT leaf name`` map from ``class_mapping.json``.

    The US-078 ``class_mapping.json`` may store either ``{id: name}`` or
    ``{name: id}``; both are normalised to ``{int_id: name}``. Returns an empty
    map (callers fall back to the stringified id) if the file is absent.
    """
    path = root / "class_mapping.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = raw.get("class_mapping", raw) if isinstance(raw, dict) else {}
    out: dict[int, str] = {}
    for key, value in mapping.items():
        if isinstance(value, str) and str(key).lstrip("-").isdigit():
            out[int(key)] = value  # {id: name}
        elif isinstance(value, int):
            out[value] = str(key)  # {name: id}
    return out


def _date_stats(dates: pl.Series) -> dict[str, float]:
    """Compute the descriptive stats of the per-patch date counts."""
    arr = dates.to_numpy().astype(np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _explode_classes(meta: pl.DataFrame, presence_col: str) -> pl.DataFrame:
    """Explode the per-patch ``clases_presentes`` list into ``(patch_id, class_id)``.

    Accepts the list either as a Polars ``List`` column or as JSON-encoded
    strings (US-078 has written both); normalises to a long frame with one row
    per (patch, class) pair.
    """
    col = meta.get_column(presence_col)
    if col.dtype == pl.List:
        long = meta.select("patch_id", presence_col).explode(presence_col)
    else:
        # JSON-encoded list of ids per patch (string); parse then explode.
        parsed = [
            (row["patch_id"], int(cid))
            for row in meta.select("patch_id", presence_col).iter_rows(named=True)
            for cid in json.loads(row[presence_col])
        ]
        long = pl.DataFrame(
            parsed, schema={"patch_id": meta["patch_id"].dtype, presence_col: pl.Int64}
        )
    return long.rename({presence_col: "class_id"}).with_columns(pl.col("class_id").cast(pl.Int64))


def _cooccurrence(long: pl.DataFrame, top_n: int = 25) -> list[dict[str, object]]:
    """Count how often two class ids share a patch (top ``top_n`` unordered pairs)."""
    joined = long.join(long, on="patch_id", suffix="_b")
    pairs = (
        joined.filter(pl.col("class_id") < pl.col("class_id_b"))
        .group_by("class_id", "class_id_b")
        .len()
        .sort("len", descending=True)
        .head(top_n)
    )
    return [
        {"class_a": int(r["class_id"]), "class_b": int(r["class_id_b"]), "n_patches": int(r["len"])}
        for r in pairs.iter_rows(named=True)
    ]


def compute_italia_eda(
    root: Path = DEFAULT_ITALIA_ROOT,
    *,
    presence_col: str = "clases_presentes",
    parcels_col: str = "n_parcelas",
    dates_col: str = "n_fechas",
    fold_col: str = "fold_espacial",
) -> ItaliaEdaReport:
    """Compute the formal EDA of the Italian TL dataset from its patch metadata.

    Args:
        root: Dataset root holding ``metadata.parquet`` + ``class_mapping.json``.
        presence_col: Metadata column with the per-patch list of present class ids.
        parcels_col: Metadata column with the per-patch parcel count.
        dates_col: Metadata column with the per-patch Sentinel-2 date count.
        fold_col: Metadata column with the spatial-CV fold id.

    Returns:
        An :class:`ItaliaEdaReport` with volume, per-class support, the temporal
        ceiling and inter-class co-occurrence, all from the real metadata.
    """
    meta = load_patch_metadata(root)
    names = _load_class_names(root)

    n_patches = meta.height
    n_parcels = int(meta.get_column(parcels_col).sum()) if parcels_col in meta.columns else 0
    mean_ppp = n_parcels / n_patches if n_patches else 0.0

    date_stats = (
        _date_stats(meta.get_column(dates_col))
        if dates_col in meta.columns
        else _date_stats(pl.Series([]))
    )
    n_weak = (
        int((meta.get_column(dates_col) < _WEAK_PHENOLOGY_DATES).sum())
        if dates_col in meta.columns
        else 0
    )

    long = _explode_classes(meta, presence_col)
    # Distribute the patch parcel count uniformly across its present classes as a
    # parcel-support estimate (the exact per-class parcel count needs the masks; the
    # estimate is the documented proxy at the metadata grain).
    parcels_by_patch = (
        meta.select("patch_id", parcels_col).rename({parcels_col: "n_parcels_patch"})
        if parcels_col in meta.columns
        else meta.select("patch_id").with_columns(pl.lit(0).alias("n_parcels_patch"))
    )
    classes_per_patch = long.group_by("patch_id").len().rename({"len": "n_classes_patch"})
    est = (
        long.join(parcels_by_patch, on="patch_id")
        .join(classes_per_patch, on="patch_id")
        .with_columns((pl.col("n_parcels_patch") / pl.col("n_classes_patch")).alias("parcel_share"))
    )
    per_class_df = (
        est.group_by("class_id")
        .agg(
            pl.len().alias("n_patches"),
            pl.col("parcel_share").sum().round(0).cast(pl.Int64).alias("n_parcels_est"),
        )
        .sort("n_patches", descending=True)
    )
    per_class = [
        {
            "class_id": int(r["class_id"]),
            "class_name": names.get(int(r["class_id"]), str(r["class_id"])),
            "n_patches": int(r["n_patches"]),
            "n_parcels_est": int(r["n_parcels_est"]),
            "pct_patches": round(100.0 * int(r["n_patches"]) / n_patches, 2) if n_patches else 0.0,
            "is_target_candidate": int(r["n_patches"]) >= _MIN_PATCHES_FOR_TARGET,
        }
        for r in per_class_df.iter_rows(named=True)
    ]
    n_target = sum(1 for row in per_class if row["is_target_candidate"])

    fold_dist: dict[str, int] = {}
    if fold_col in meta.columns:
        fold_counts = meta.group_by(fold_col).len().sort(fold_col)
        fold_dist = {str(r[fold_col]): int(r["len"]) for r in fold_counts.iter_rows(named=True)}

    report = ItaliaEdaReport(
        n_patches=n_patches,
        n_parcels=n_parcels,
        mean_parcels_per_patch=mean_ppp,
        date_stats=date_stats,
        n_patches_weak_phenology=n_weak,
        per_class=per_class,
        n_target_candidate_classes=n_target,
        top_cooccurrence=_cooccurrence(long),
        fold_distribution=fold_dist,
    )
    logger.info("italia_eda_computed", **report.summary())
    return report
