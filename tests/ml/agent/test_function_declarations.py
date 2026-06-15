"""Registry and FunctionDeclaration tests (US-045 AC-1, AC-2, AC-4, AC-5).

Covers the orchestration seam in ``ml.agent.tools``:

- ``import ml.agent.tools`` exposes a registry of exactly nine tools.
- ``build_function_declarations`` yields nine ``google-genai``
  ``FunctionDeclaration`` with unique names and a valid object parameter schema.
- The 5 sync / 4 deferred split is internally consistent across
  ``TOOL_SPECS``, ``get_sync_tools``, ``get_deferred_tools`` and the declaration
  ``behavior`` (BLOCKING vs NON_BLOCKING).
- The lazy registry resolves every per-tool module's ``run`` coroutine.
"""

from __future__ import annotations

import inspect

import pytest
from google.genai import types

import ml.agent.tools as tools_mod
from ml.agent.tools import (
    TOOL_REGISTRY,
    TOOL_SPECS,
    build_function_declarations,
    build_registry,
    get_deferred_tools,
    get_sync_tools,
    get_tool,
)

_EXPECTED_TOOLS = {
    "list_parcels",
    "get_parcel_timeseries",
    "get_aoi_stats",
    "classify_new_parcel",
    "explain_prediction",
    "search_stac",
    "get_tiles",
    "add_aoi",
    "compare_models",
}
_SYNC = {
    "list_parcels",
    "get_parcel_timeseries",
    "get_aoi_stats",
    "classify_new_parcel",
    "explain_prediction",
}
_DEFERRED = {"search_stac", "get_tiles", "add_aoi", "compare_models"}


def test_registry_has_nine_known_tools() -> None:
    """AC-1: the registry exposes exactly the nine expected tools."""
    assert set(TOOL_SPECS) == _EXPECTED_TOOLS
    assert len(TOOL_REGISTRY) == 9
    assert set(TOOL_REGISTRY) == _EXPECTED_TOOLS


def test_module_importable_as_registry() -> None:
    """AC-1: ``import ml.agent.tools`` resolves to the registry module.

    The package/module name collision was resolved by hosting the registry in
    ``ml/agent/tools/__init__.py``; this guards that ``TOOL_REGISTRY`` lives there.
    """
    assert hasattr(tools_mod, "TOOL_REGISTRY")
    assert hasattr(tools_mod, "build_function_declarations")


def test_build_function_declarations_count_and_unique_names() -> None:
    """AC-2: nine declarations with unique snake_case names."""
    decls = build_function_declarations()
    assert len(decls) == 9
    names = [d.name for d in decls]
    assert len(set(names)) == 9
    assert set(names) == _EXPECTED_TOOLS


def test_function_declarations_have_valid_object_schema() -> None:
    """AC-2: every declaration carries an OBJECT parameter schema with props."""
    for decl in build_function_declarations():
        params = decl.parameters
        assert params is not None, f"{decl.name} has no parameters schema"
        assert params.type == types.Type.OBJECT
        assert params.properties, f"{decl.name} has no properties"
        assert decl.description, f"{decl.name} has no description"


def test_input_models_emit_valid_json_schema() -> None:
    """AC-2: each tool input model produces a JSON schema (object root)."""
    for _name, (_mod, input_model, _out, _deferred, _desc) in TOOL_SPECS.items():
        schema = input_model.model_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema


def test_behavior_matches_deferred_flag() -> None:
    """AC-5: deferred tools are NON_BLOCKING, sync tools BLOCKING."""
    decls = {d.name: d for d in build_function_declarations()}
    for name in _SYNC:
        assert decls[name].behavior == types.Behavior.BLOCKING
    for name in _DEFERRED:
        assert decls[name].behavior == types.Behavior.NON_BLOCKING


def test_sync_and_deferred_split() -> None:
    """AC-4/AC-5: five synchronous demo tools, four deferred ones."""
    sync_names = {t.name for t in get_sync_tools()}
    deferred_names = {t.name for t in get_deferred_tools()}
    assert sync_names == _SYNC
    assert deferred_names == _DEFERRED
    assert len(sync_names) == 5
    assert len(deferred_names) == 4
    assert sync_names.isdisjoint(deferred_names)


def test_deferred_flag_consistent_in_specs() -> None:
    """The ``deferred`` flag in each resolved ToolSpec matches the split."""
    registry = build_registry()
    assert len(registry) == 9
    for name, spec in registry.items():
        assert spec.deferred is (name in _DEFERRED)


def test_get_tool_binds_async_run_coroutine() -> None:
    """The lazy loader binds a real ``async def run(inp, ctx)`` per tool."""
    for name in _EXPECTED_TOOLS:
        spec = get_tool(name)
        assert spec.name == name
        assert inspect.iscoroutinefunction(spec.fn)
        sig_params = list(inspect.signature(spec.fn).parameters)
        assert sig_params == ["inp", "ctx"]


def test_get_tool_unknown_raises_keyerror() -> None:
    """An unknown tool name raises ``KeyError`` (no silent fallback)."""
    with pytest.raises(KeyError):
        get_tool("teleport")
