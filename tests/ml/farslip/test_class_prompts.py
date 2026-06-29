"""Tests for the FarSLIP per-class prompts (US-080, :mod:`ml.farslip.class_prompts`)."""

from __future__ import annotations

from ml.farslip.class_prompts import (
    build_class_prompts,
    load_cached_prompts,
    save_prompts,
)


def test_build_uses_template_and_hint_for_known_class() -> None:
    prompts = build_class_prompts(["Soft winter wheat"])
    text = prompts["Soft winter wheat"]
    assert "wheat" in text.lower()
    assert "winter cereal" in text  # the deterministic phenology hint


def test_build_prefers_provided_description() -> None:
    prompts = build_class_prompts(
        ["Corn"], descriptions={"Corn": "an LLM-written phenology description"}
    )
    assert prompts["Corn"] == "an LLM-written phenology description"


def test_build_blank_description_falls_back_to_template() -> None:
    prompts = build_class_prompts(["Beet"], descriptions={"Beet": "   "})
    assert "beet" in prompts["Beet"].lower()
    assert "root crop" in prompts["Beet"]


def test_build_unknown_class_uses_generic_fallback() -> None:
    prompts = build_class_prompts(["Olive"])  # not a france-9 class
    assert "olive" in prompts["Olive"].lower()


def test_cache_roundtrip(tmp_path) -> None:
    prompts = {"Corn": "a corn field", "Beet": "a beet field"}
    save_prompts(prompts, tmp_path / "prompts.parquet")
    assert load_cached_prompts(tmp_path / "prompts.parquet") == prompts


def test_load_missing_cache_returns_empty(tmp_path) -> None:
    assert load_cached_prompts(tmp_path / "absent.parquet") == {}
