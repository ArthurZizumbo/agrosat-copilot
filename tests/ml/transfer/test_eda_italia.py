"""Mechanics tests for the Italian dataset EDA (US-082).

A tiny synthetic metadata.parquet drives these tests; they verify the EDA wiring
(volume aggregation, per-class support, the weak-phenology cohort, co-occurrence),
NOT any headline statistic. The real EDA numbers come from the actual 1,438-patch
metadata.parquet on disk (1,438 / 107,493 verified), read by the same code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ml.transfer.eda_italia import (
    ItaliaEdaReport,
    compute_italia_eda,
    load_patch_metadata,
)


@pytest.fixture
def toy_dataset(tmp_path: Path) -> Path:
    """Write a 4-patch toy dataset (metadata.parquet + class_mapping.json)."""
    meta = pl.DataFrame(
        {
            "patch_id": [0, 1, 2, 3],
            "n_parcelas": [100, 50, 80, 20],
            "n_fechas": [30, 12, 24, 9],  # 2 below the weak-phenology floor (16)
            "clases_presentes": [[1, 2], [1], [2, 3], [1, 2, 3]],
            "fold_espacial": [0, 1, 0, 4],
        }
    )
    meta.write_parquet(tmp_path / "metadata.parquet")
    (tmp_path / "class_mapping.json").write_text(
        json.dumps({"class_mapping": {"1": "soft_wheat", "2": "durum_wheat", "3": "olive"}}),
        encoding="utf-8",
    )
    return tmp_path


def test_load_metadata_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_patch_metadata(tmp_path)


def test_volume_aggregation(toy_dataset: Path) -> None:
    report = compute_italia_eda(toy_dataset)
    assert isinstance(report, ItaliaEdaReport)
    assert report.n_patches == 4
    assert report.n_parcels == 250  # 100+50+80+20
    assert report.mean_parcels_per_patch == 62.5


def test_weak_phenology_cohort(toy_dataset: Path) -> None:
    # n_fechas 12 and 9 are below 16 -> 2 patches in the weak-phenology cohort.
    report = compute_italia_eda(toy_dataset)
    assert report.n_patches_weak_phenology == 2
    assert report.date_stats["min"] == 9.0
    assert report.date_stats["max"] == 30.0


def test_per_class_support_sorted_by_presence(toy_dataset: Path) -> None:
    report = compute_italia_eda(toy_dataset)
    # class 1 in patches {0,1,3} -> 3; class 2 in {0,2,3} -> 3; class 3 in {2,3} -> 2.
    by_id = {row["class_id"]: row for row in report.per_class}
    assert by_id[1]["n_patches"] == 3
    assert by_id[2]["n_patches"] == 3
    assert by_id[3]["n_patches"] == 2
    assert by_id[1]["class_name"] == "soft_wheat"
    # Sorted descending by presence.
    presences = [row["n_patches"] for row in report.per_class]
    assert presences == sorted(presences, reverse=True)


def test_cooccurrence_counts_shared_patches(toy_dataset: Path) -> None:
    report = compute_italia_eda(toy_dataset)
    pairs = {(r["class_a"], r["class_b"]): r["n_patches"] for r in report.top_cooccurrence}
    # classes 1 & 2 share patches {0, 3} -> 2; 2 & 3 share {2, 3} -> 2; 1 & 3 share {3} -> 1.
    assert pairs[(1, 2)] == 2
    assert pairs[(2, 3)] == 2
    assert pairs[(1, 3)] == 1


def test_fold_distribution(toy_dataset: Path) -> None:
    report = compute_italia_eda(toy_dataset)
    assert report.fold_distribution == {"0": 2, "1": 1, "4": 1}


def test_to_json_roundtrip(toy_dataset: Path, tmp_path: Path) -> None:
    report = compute_italia_eda(toy_dataset)
    out = tmp_path / "eda.json"
    report.to_json(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_patches"] == 4
    assert "per_class" in loaded and "top_cooccurrence" in loaded
