"""Unit tests for the data-driven paper tables (US-070).

The core contract is **anti-hardcode**: every cell of a ``.tex`` table must come
from its source artifact. We verify this by feeding a synthetic CSV/JSON fixture
with values that do not exist anywhere in the repo, then asserting the generated
LaTeX contains exactly those fixture numbers (and the booktabs structure).

We also smoke-test the real-artifact path: tables built from the actual
``reports/**`` files emit valid LaTeX when the source exists (skip otherwise).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.report import paper_tables as pt

REPORTS_DIR = Path("reports")


def _booktabs_ok(tex: str) -> bool:
    """Return whether the LaTeX string has the expected booktabs skeleton."""
    return all(
        token in tex
        for token in (
            r"\begin{table}",
            r"\toprule",
            r"\midrule",
            r"\bottomrule",
            r"\end{table}",
        )
    )


def test_render_latex_table_structure() -> None:
    tex = pt.render_latex_table(
        ["A", "B"],
        [["x", "1.0000"], ["y", "2.0000"]],
        caption="cap",
        label="tab:t",
    )
    assert _booktabs_ok(tex)
    assert r"\label{tab:t}" in tex
    assert "x & 1.0000" in tex


def test_fmt_preserves_nan_not_zero() -> None:
    assert pt._fmt(float("nan")) == "NaN"
    assert pt._fmt(None) == "NaN"
    assert pt._fmt(0.123456) == "0.1235"
    assert pt._fmt(3) == "3"


def test_escape_latex_underscore() -> None:
    assert pt._escape_latex("a_b") == r"a\_b"


def test_segmentation_table_anti_hardcode(tmp_path: Path) -> None:
    """The fold-5 segmentation table must echo the fixture, not repo numbers."""
    csv = tmp_path / "fixture_fold5.csv"
    # Sentinel values that exist nowhere in reports/.
    csv.write_text(
        "model,miou,f1_macro,pixel_accuracy,fold,n_patches,status,model_kind,"
        "needs_resize,in_channels,cohen_kappa,balanced_acc\n"
        "fakemodel,0.913700,0.824600,0.700100,5,10,ok,fakemodel,true,7,"
        "0.555500,0.444400\n",
        encoding="utf-8",
    )
    tex = pt.build_segmentation_table(csv)
    assert _booktabs_ok(tex)
    assert "fakemodel" in tex
    assert "0.9137" in tex  # mIoU read from fixture
    assert "0.8246" in tex  # F1 read from fixture
    assert "7" in tex  # in_channels read from fixture
    assert "si" in tex  # needs_resize true -> "si"


def test_fm_comparison_anti_hardcode(tmp_path: Path) -> None:
    s2 = tmp_path / "s2.csv"
    s2.write_text(
        "scenario,model,n_features,f1_macro,f1_weighted,miou,train_time_s\n"
        "FakeRep,XGB,42,0.111100,0.222200,0.333300,9.9\n",
        encoding="utf-8",
    )
    farslip = tmp_path / "farslip.csv"
    farslip.write_text(
        "space,silhouette,f1_macro_mean,f1_macro_std,n_dims,n_samples,"
        "delta_vs_0163,delta_vs_alphaearth_here\n"
        "fakespace,0.01,0.555500,0.02,768,567,-0.1,0.0\n",
        encoding="utf-8",
    )
    tex = pt.build_fm_comparison_table(s2_csv=s2, farslip_csv=farslip)
    assert "FakeRep" in tex
    assert "0.1111" in tex  # from fixture, not repo
    assert "fakespace" in tex
    assert "0.5555" in tex


def test_llm_benchmark_anti_hardcode(tmp_path: Path) -> None:
    payload = {
        "gemini": {
            "tool_calling": {
                "tool_selection_accuracy": {"mean": 0.1234, "std": 0.0},
                "arg_match_accuracy": {"mean": 0.5678, "std": 0.0},
            },
            "grounded_crop": {
                "crop_match_accuracy": {"mean": 0.9012, "std": 0.0},
                "routing_accuracy": {"mean": 0.3456, "std": 0.0},
            },
            "rag_ab": {"hallucination_rate_grounded": {"mean": 0.0789, "std": 0.0}},
        }
    }
    js = tmp_path / "eval.json"
    js.write_text(json.dumps(payload), encoding="utf-8")
    tex = pt.build_llm_benchmark_table(js, models=("gemini",))
    assert "0.1234" in tex
    assert "0.5678" in tex
    assert "0.9012" in tex
    assert pt.PENDING in tex  # AgroMind-IT/ES column blocked, never fabricated


def test_tool_ablation_anti_hardcode(tmp_path: Path) -> None:
    payload = {
        "gemini": {
            "tool_calling": {
                "tool_calling_native": {"mean": 0.8001, "std": 0.0},
                "parse_failure_rate": {"mean": 0.0202, "std": 0.0},
                "tool_selection_accuracy": {"mean": 0.6003, "std": 0.0},
                "arg_match_accuracy": {"mean": 0.7004, "std": 0.0},
                "no_call_rate": {"mean": 0.0505, "std": 0.0},
            },
            "grounded_crop": {
                "crop_match_accuracy": {"mean": 0.9006, "std": 0.0},
                "routing_accuracy": {"mean": 0.8507, "std": 0.0},
                "faithfulness_crop": {"mean": 0.9508, "std": 0.0},
            },
        }
    }
    js = tmp_path / "eval.json"
    js.write_text(json.dumps(payload), encoding="utf-8")
    tex = pt.build_tool_ablation_table(js, model="gemini")
    assert "0.6003" in tex  # tool_selection_accuracy from fixture
    assert "0.9006" in tex  # crop_match from fixture
    assert pt.PENDING in tex  # on-prem Qwen column blocked


def test_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pt.build_segmentation_table(tmp_path / "does_not_exist.csv")


# --------------------------------------------------------------------------- #
# Real-artifact smoke tests (skip if the artifact is not present).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("builder", "source"),
    [
        (
            pt.build_segmentation_table,
            REPORTS_DIR / "segmentation" / "metrics" / "model_comparison_fold5.csv",
        ),
        (
            pt.build_llm_benchmark_table,
            REPORTS_DIR / "agent_bench" / "us049_system_eval.json",
        ),
    ],
)
def test_real_artifacts_emit_valid_latex(builder, source: Path) -> None:
    if not source.exists():
        pytest.skip(f"real artifact missing: {source}")
    tex = builder()
    assert _booktabs_ok(tex)


def test_segmentation_real_value_matches_csv() -> None:
    """The number written to the .tex equals the number read from the CSV."""
    import polars as pl

    source = REPORTS_DIR / "segmentation" / "metrics" / "model_comparison_fold5.csv"
    if not source.exists():
        pytest.skip("real fold-5 metrics missing")
    df = pl.read_csv(source)
    top = df.sort("miou", descending=True).row(0, named=True)
    expected = f"{top['miou']:.4f}"
    tex = pt.build_segmentation_table(source)
    assert expected in tex


def test_write_all_tables_real(tmp_path: Path) -> None:
    written = pt.write_all_tables(out_dir=tmp_path)
    # At least the autonomous tables backed by present artifacts must exist.
    assert "segmentation_individual_fold5" in written
    for path in written.values():
        assert path.exists()
        assert _booktabs_ok(path.read_text(encoding="utf-8"))
