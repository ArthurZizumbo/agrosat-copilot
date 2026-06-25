"""EDA across transfer datasets: class distribution + ensemble-enrichment map.

Arthur's question (2026-06-25): before launching more transfer runs, do an EDA to
see the class distribution per dataset, which classes are well-supported, and
which NEW fine classes could ENRICH the champion ensemble (the papaya/fruits
hypothesis: a dataset that resolves a fine class PASTIS lumps -- e.g. PASTIS has a
single "Fruits, vegetables, flowers" bucket -- could teach the ensemble that
granularity).

What it computes
----------------
For every transfer dataset with labels (EuroCropsML EE/LV via HCAT, WorldCereal
BR/IN via class_name):
  1. class distribution (support per leaf, sorted),
  2. overlap with the PASTIS-18 label space the ensemble already predicts,
  3. ENRICHMENT CANDIDATES: well-supported classes that are NOT in PASTIS-18 (new
     granularity the ensemble cannot currently express), ranked by support.

The output is a per-dataset table + a consolidated enrichment shortlist: the fine
classes that (a) have enough samples to learn and (b) PASTIS-18 does not resolve,
which are the concrete candidates to add to the ensemble's label space.

Honesty
-------
- Support counts are the real parquet row counts (one row per parcel pixel).
- "Maps to PASTIS" is decided by an explicit, auditable name-mapping table, not a
  fuzzy match; an unmapped class is reported as a genuine NEW class, not silently
  bucketed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["DatasetEDA", "run_transfer_eda"]

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_TRANSFER_DIR: Path = _REPO_ROOT / "data" / "transfer"
_OUT_DIR: Path = _TRANSFER_DIR / "transfer_eda"
_HCAT3_CSV: Path = _REPO_ROOT / "data" / "reference" / "eurocrops_hcat3.csv"

#: Minimum support for a class to be a learnable enrichment candidate.
_MIN_SUPPORT_CANDIDATE: int = 50

#: Mapping from a normalised dataset leaf name to the PASTIS-18 class it already
#: covers. A leaf absent here is treated as a NEW class (potential enrichment).
#: Built on agronomy, auditable -- not fuzzy matched. PASTIS-18:
#: Meadow, Soft winter wheat, Corn, Winter barley, Winter rapeseed, Spring barley,
#: Sunflower, Grapevine, Beet, Winter triticale, Winter durum wheat,
#: Fruits/vegetables/flowers, Potatoes, Leguminous fodder, Soybeans, Orchard,
#: Mixed cereal, Sorghum.
_LEAF_TO_PASTIS: dict[str, str] = {
    "pasture_meadow_grassland_grass": "Meadow",
    "winter_common_soft_wheat": "Soft winter wheat",
    "maize": "Corn",
    "winter_barley": "Winter barley",
    "winter_rapeseed_rape": "Winter rapeseed",
    "spring_barley": "Spring barley",
    "sunflower": "Sunflower",
    "grapes": "Grapevine",
    "sugar_beet": "Beet",
    "winter_triticale": "Winter triticale",
    "potatoes": "Potatoes",
    "soya_soybean": "Soybeans",
    "wintercereals": "Soft winter wheat",  # WorldCereal coarse winter-cereal bucket
}


@dataclass
class DatasetEDA:
    """EDA result for one dataset."""

    name: str
    n_parcels: int
    n_classes: int
    distribution: pl.DataFrame  # leaf, support, share, maps_to_pastis, is_new
    enrichment_candidates: pl.DataFrame  # NEW classes with support >= floor


def _load_hcat_name_map() -> dict[int, str]:
    """Return ``{hcat_code: hcat_leaf_name}`` from the HCAT v3 reference CSV."""
    h = pl.read_csv(
        _HCAT3_CSV, schema_overrides={"HCAT3_code": pl.Utf8, "HCAT3_name": pl.Utf8}
    )
    return {int(c): str(n) for c, n in zip(h["HCAT3_code"], h["HCAT3_name"], strict=True)}


def _eda_one(name: str, leaves: list[str], total: int) -> DatasetEDA:
    """Build the EDA for one dataset from its per-parcel leaf list."""
    df = pl.DataFrame({"leaf": leaves})
    dist = (
        df.group_by("leaf")
        .len()
        .rename({"len": "support"})
        .with_columns(
            (pl.col("support") / total).round(4).alias("share"),
            pl.col("leaf").is_in(list(_LEAF_TO_PASTIS.keys())).alias("maps_to_pastis"),
        )
        .with_columns((~pl.col("maps_to_pastis")).alias("is_new"))
        .sort("support", descending=True)
    )
    candidates = (
        dist.filter(pl.col("is_new") & (pl.col("support") >= _MIN_SUPPORT_CANDIDATE))
        .select(["leaf", "support", "share"])
        .sort("support", descending=True)
    )
    return DatasetEDA(
        name=name,
        n_parcels=total,
        n_classes=dist.height,
        distribution=dist,
        enrichment_candidates=candidates,
    )


def _eurocrops_leaves(region: str) -> list[str]:
    """Resolve EuroCropsML HCAT codes to leaf names for one region."""
    parquet = _TRANSFER_DIR / f"eurocropsml_alphaearth_{region}.parquet"
    name_map = _load_hcat_name_map()
    df = pl.read_parquet(parquet, columns=["hcat_code"]).with_columns(
        pl.col("hcat_code").cast(pl.Int64)
        .replace_strict(name_map, default="unknown_hcat", return_dtype=pl.Utf8)
        .alias("leaf")
    )
    return df["leaf"].to_list()


def _worldcereal_leaves(region: str) -> list[str]:
    """Read WorldCereal class_name for one region."""
    parquet = _TRANSFER_DIR / f"worldcereal_{region}.parquet"
    return pl.read_parquet(parquet, columns=["class_name"])["class_name"].to_list()


def run_transfer_eda() -> dict[str, DatasetEDA]:
    """Run the EDA over all labelled transfer datasets.

    Returns:
        Mapping ``dataset_name -> DatasetEDA``.
    """
    edas: dict[str, DatasetEDA] = {}
    for region in ("estonia", "latvia"):
        leaves = _eurocrops_leaves(region)
        edas[f"eurocrops_{region}"] = _eda_one(f"eurocrops_{region}", leaves, len(leaves))
    for region in ("brazil_cerrado", "india_karnataka"):
        leaves = _worldcereal_leaves(region)
        key = f"worldcereal_{region.split('_')[0]}"
        edas[key] = _eda_one(key, leaves, len(leaves))
    for name, eda in edas.items():
        logger.info(
            "transfer_eda_dataset",
            dataset=name,
            n_parcels=eda.n_parcels,
            n_classes=eda.n_classes,
            n_enrichment_candidates=eda.enrichment_candidates.height,
        )
    return edas


def consolidate_enrichment(edas: dict[str, DatasetEDA]) -> pl.DataFrame:
    """Pool the enrichment candidates across datasets, summing support.

    A class that is NEW (not in PASTIS-18) and well-supported in one or more
    datasets is a concrete candidate to add to the ensemble's label space. The
    pooled view shows total support and which datasets contribute it.

    Args:
        edas: Per-dataset EDA results.

    Returns:
        A frame ``leaf, total_support, n_datasets, datasets`` sorted by support.
    """
    rows: list[dict[str, object]] = []
    for name, eda in edas.items():
        for leaf, support, _share in eda.enrichment_candidates.iter_rows():
            rows.append({"leaf": leaf, "dataset": name, "support": support})
    if not rows:
        return pl.DataFrame(
            schema={
                "leaf": pl.Utf8,
                "total_support": pl.Int64,
                "n_datasets": pl.Int64,
                "datasets": pl.Utf8,
            }
        )
    pooled = pl.DataFrame(rows)
    return (
        pooled.group_by("leaf")
        .agg(
            pl.col("support").sum().alias("total_support"),
            pl.col("dataset").n_unique().alias("n_datasets"),
            pl.col("dataset").str.concat(", ").alias("datasets"),
        )
        .sort("total_support", descending=True)
    )


def save_outputs(edas: dict[str, DatasetEDA], out_dir: Path = _OUT_DIR) -> None:
    """Persist per-dataset distributions + the consolidated enrichment shortlist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for name, eda in edas.items():
        eda.distribution.write_parquet(out_dir / f"dist_{name}.parquet")
        summary[name] = {
            "n_parcels": eda.n_parcels,
            "n_classes": eda.n_classes,
            "n_enrichment_candidates": eda.enrichment_candidates.height,
            "top_candidates": eda.enrichment_candidates.head(8).to_dicts(),
        }
    enrichment = consolidate_enrichment(edas)
    enrichment.write_parquet(out_dir / "enrichment_shortlist.parquet")
    summary["enrichment_shortlist"] = enrichment.head(20).to_dicts()
    (out_dir / "transfer_eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("transfer_eda_saved", out_dir=str(out_dir))
