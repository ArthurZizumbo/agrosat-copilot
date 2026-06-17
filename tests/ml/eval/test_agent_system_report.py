"""Tests for the system-eval HTML report (US-049).

Drives :func:`ml.eval.agent_system_report.build_system_report_html` on a small
NaN-bearing fixture dict and asserts that the written file contains the three
Spanish eval section headers and is NaN-safe (NaN cells render as ``n/a``,
never as a Python ``nan`` literal nor a crash). ZERO network: the renderer only
touches matplotlib (Agg backend) and the filesystem.

Conventions: identifiers and docstrings in English; visible prose in Spanish;
no emojis; full type hints.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from ml.eval.agent_system_report import build_system_report_html


def _fixture() -> dict[str, Any]:
    """Return a tiny two-variant results dict with a NaN cell.

    Returns:
        A ``{variant: {eval: {metric: {"mean", "std"}}}}`` mapping mirroring the
        real US-049 shape, with the ungrounded-hallucination cell of one variant
        set to NaN to exercise the ``n/a`` rendering path.
    """
    return {
        "gemini": {
            "tool_calling": {
                "tool_selection_accuracy": {"mean": 0.55, "std": 0.0},
                "arg_match_accuracy": {"mean": 0.95, "std": 0.0},
                "tool_calling_native": {"mean": 1.0, "std": 0.0},
                "no_call_rate": {"mean": 0.0, "std": 0.0},
            },
            "grounded_crop": {
                "routing_accuracy": {"mean": 1.0, "std": 0.0},
                "crop_match_accuracy": {"mean": 0.923, "std": 0.0},
                "faithfulness_crop": {"mean": 1.0, "std": 0.0},
            },
            "rag_ab": {
                "hallucination_rate_ungrounded": {
                    "mean": math.nan,
                    "std": math.nan,
                },
                "hallucination_rate_grounded": {"mean": 0.1, "std": 0.0},
                "hallucination_reduction_delta": {
                    "mean": math.nan,
                    "std": math.nan,
                },
            },
        },
        "qwen36-vl": {
            "tool_calling": {
                "tool_selection_accuracy": {"mean": 0.95, "std": 0.0},
                "arg_match_accuracy": {"mean": 0.579, "std": 0.0},
                "tool_calling_native": {"mean": 0.0, "std": 0.0},
                "no_call_rate": {"mean": 0.0, "std": 0.0},
            },
            "grounded_crop": {
                "routing_accuracy": {"mean": 0.923, "std": 0.0},
                "crop_match_accuracy": {"mean": 0.923, "std": 0.0},
                "faithfulness_crop": {"mean": 1.0, "std": 0.0},
            },
            "rag_ab": {
                "hallucination_rate_ungrounded": {"mean": 0.9, "std": 0.0},
                "hallucination_rate_grounded": {"mean": 0.1, "std": 0.0},
                "hallucination_reduction_delta": {"mean": 0.8, "std": 0.0},
            },
        },
    }


def test_build_system_report_html_sections_and_nan_safe(tmp_path: Path) -> None:
    """The report writes a file with the three Spanish eval headers, NaN-safe."""
    out_path = tmp_path / "system_report.html"

    returned = build_system_report_html(_fixture(), out_path)

    assert returned == out_path
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")

    # The three eval section headers (Spanish prose).
    assert "Tool-calling sobre las 10 herramientas reales" in text
    assert "Orquestacion grounded-crop" in text
    assert "RAG A/B (reduccion de alucinacion)" in text

    # The embedded base64 chart is present and self-contained.
    assert "data:image/png;base64," in text

    # NaN-safe: rendered as the n/a marker, never as a raw nan literal in the
    # visible markup. The base64 PNG blob can contain the substring "nan" by
    # chance, so strip the data URI before checking the textual content.
    visible = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", text)
    assert "n/a" in visible
    assert "nan" not in visible.lower()
