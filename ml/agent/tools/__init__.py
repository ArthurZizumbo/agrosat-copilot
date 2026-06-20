"""Central registry of the nine geospatial agent tools.

This module is the orchestration seam between the per-tool implementation
modules (``ml/agent/tools/<name>.py``, each exposing
``async def run(inp, ctx) -> Output``) and the ``google-genai`` function-calling
loop (US-047).

Import policy: this module must import cleanly even before the per-tool modules
exist, so it does **not** import their ``run`` coroutines at top level. Instead a
static descriptor table :data:`TOOL_SPECS` maps each tool name to its module
path, Pydantic input/output models, deferred flag and description. The bound
``run`` callable is resolved lazily via :func:`importlib.import_module` on first
access (see :func:`get_tool` / :func:`build_registry`). Each per-tool module is
expected to expose exactly::

    async def run(inp: <ToolInput>, ctx: ToolContext) -> <ToolOutput>: ...

The 5 synchronous demo tools run inline in the agent loop; the 4 deferred tools
are flagged ``deferred=True`` and registered with ``Behavior.NON_BLOCKING`` so
the SDK knows they may complete out of band.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from google.genai import types
from pydantic import BaseModel

from ml.agent.context import ToolContext
from ml.agent.schemas import (
    AddAoiInput,
    AoiRef,
    AoiStats,
    AoiStatsInput,
    ClassificationResult,
    ClassifyParcelInput,
    CompareModelsInput,
    ExplainPredictionInput,
    Explanation,
    GetTilesInput,
    ListParcelsInput,
    ModelComparison,
    ParcelList,
    ParcelTimeseriesInput,
    SceneList,
    SearchStacInput,
    TileUrl,
    TimeSeries,
)

# US-046: the deferred Spatial-RAG tool defines its own input/output contracts in
# its tool module (``ml/agent/schemas.py`` belongs to US-045 and is off-limits to
# this work-stream). Importing them here is safe: the module imports cleanly.
from ml.agent.tools.retrieve_context import RetrieveContextInput, RetrievedContext

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_SPECS",
    "ToolSpec",
    "build_function_declarations",
    "build_registry",
    "get_deferred_tools",
    "get_sync_tools",
    "get_tool",
]

# Coroutine signature every per-tool ``run`` must implement.
ToolFn = Callable[[BaseModel, ToolContext], Awaitable[BaseModel]]


@dataclass(frozen=True)
class ToolSpec:
    """Immutable descriptor of a registered tool.

    Attributes:
        name: Stable tool name exposed to the LLM (snake_case).
        fn: Bound ``run`` coroutine ``(input_model, ctx) -> output_model``.
        input_model: Pydantic model validating the tool arguments.
        output_model: Pydantic model of the tool result.
        deferred: ``True`` for background/non-blocking tools.
        description: One-line natural-language description for the LLM.
    """

    name: str
    fn: ToolFn
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    deferred: bool
    description: str


# Static descriptor table. ``module`` is imported lazily so this file has no
# import-time dependency on the per-tool modules created by sub-tasks B/C/D.
# Tuple shape: (module_path, input_model, output_model, deferred, description).
_ToolDescriptor = tuple[str, type[BaseModel], type[BaseModel], bool, str]

TOOL_SPECS: dict[str, _ToolDescriptor] = {
    "list_parcels": (
        "ml.agent.tools.parcels",
        ListParcelsInput,
        ParcelList,
        False,
        "List the parcels of the current session, optionally restricted to an AOI polygon.",
    ),
    "get_parcel_timeseries": (
        "ml.agent.tools.timeseries",
        ParcelTimeseriesInput,
        TimeSeries,
        False,
        "Return the NDVI/NDWI/EVI time series of a parcel over a date window.",
    ),
    "get_aoi_stats": (
        "ml.agent.tools.aoi_stats",
        AoiStatsInput,
        AoiStats,
        False,
        "Aggregate crop statistics (area, dominant crop, class fractions) over an AOI for a year.",
    ),
    "search_stac": (
        "ml.agent.tools.stac",
        SearchStacInput,
        SceneList,
        True,
        "Search Sentinel-2 scenes in a STAC catalogue by bbox, datetime range and cloud cover.",
    ),
    "get_tiles": (
        "ml.agent.tools.tiles",
        GetTilesInput,
        TileUrl,
        True,
        "Build a TiTiler XYZ tile-template URL for a scene rendered as an index or RGB.",
    ),
    "classify_new_parcel": (
        "ml.agent.tools.classify",
        ClassifyParcelInput,
        ClassificationResult,
        False,
        "Classify a parcel's crop. By default serves the xgb-alphaearth tabular "
        "member restricted to the active label-space's well-resolved classes "
        "(france-9); with use_stacking=True and cached fold-5 OOF it serves the "
        "Stacking-5 champion posterior. Not a generic 'stacking ensemble' by default.",
    ),
    "add_aoi": (
        "ml.agent.tools.add_aoi",
        AddAoiInput,
        AoiRef,
        True,
        "Persist a named Area Of Interest polygon for the current session.",
    ),
    "compare_models": (
        "ml.agent.tools.compare",
        CompareModelsInput,
        ModelComparison,
        True,
        "Compare the crop predictions of several ensemble members for one parcel.",
    ),
    "explain_prediction": (
        "ml.agent.tools.explain",
        ExplainPredictionInput,
        Explanation,
        False,
        "Explain a parcel prediction with phenology, vigor and a natural-language description.",
    ),
    "retrieve_context": (
        "ml.agent.tools.retrieve_context",
        RetrieveContextInput,
        RetrievedContext,
        True,
        "Retrieve real neighbouring-parcel grounding (Spatial-RAG lite) for an AOI; "
        "gated by the rag_enabled flag (no-op when off).",
    ),
}

# Names of the 5 synchronous demo tools (must match TOOL_SPECS deferred=False).
_SYNC_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_parcels",
        "get_parcel_timeseries",
        "get_aoi_stats",
        "classify_new_parcel",
        "explain_prediction",
    }
)


def _resolve_run(module_path: str, tool_name: str) -> ToolFn:
    """Import a per-tool module and return its ``run`` coroutine.

    Args:
        module_path: Dotted path of the per-tool module, e.g.
            ``ml.agent.tools.parcels``.
        tool_name: Tool name, used only for clearer error messages.

    Returns:
        The module-level ``run`` coroutine.

    Raises:
        AttributeError: If the module does not expose a ``run`` attribute.
    """
    module = importlib.import_module(module_path)
    run_fn = getattr(module, "run", None)
    if run_fn is None:
        raise AttributeError(
            f"tool module {module_path!r} for tool {tool_name!r} must expose "
            "'async def run(inp, ctx)'"
        )
    return run_fn


def get_tool(name: str) -> ToolSpec:
    """Resolve a single :class:`ToolSpec` by name, importing its module lazily.

    Args:
        name: Registered tool name.

    Returns:
        The fully-bound :class:`ToolSpec` (with its ``run`` coroutine resolved).

    Raises:
        KeyError: If ``name`` is not a registered tool.
    """
    if name not in TOOL_SPECS:
        raise KeyError(f"unknown tool {name!r}; registered: {sorted(TOOL_SPECS)}")
    module_path, input_model, output_model, deferred, description = TOOL_SPECS[name]
    return ToolSpec(
        name=name,
        fn=_resolve_run(module_path, name),
        input_model=input_model,
        output_model=output_model,
        deferred=deferred,
        description=description,
    )


def build_registry() -> dict[str, ToolSpec]:
    """Resolve every tool into a name -> :class:`ToolSpec` mapping.

    Imports all per-tool modules. Call once the per-tool modules exist (after
    sub-tasks B/C/D). The agent loop typically calls this at startup.

    Returns:
        Mapping of tool name to its bound :class:`ToolSpec`.
    """
    return {name: get_tool(name) for name in TOOL_SPECS}


class _LazyRegistry(dict):
    """Dict that resolves :class:`ToolSpec` values on first key access.

    Lets ``TOOL_REGISTRY`` be importable before the per-tool modules exist while
    still behaving like ``dict[str, ToolSpec]`` once they do. Keys are present
    immediately; values are materialised lazily and cached.
    """

    def __missing__(self, key: str) -> ToolSpec:
        if key not in TOOL_SPECS:
            raise KeyError(key)
        spec = get_tool(key)
        self[key] = spec
        return spec

    def __iter__(self):
        return iter(TOOL_SPECS)

    def keys(self):
        return TOOL_SPECS.keys()

    def values(self):
        return [self[name] for name in TOOL_SPECS]

    def items(self):
        return [(name, self[name]) for name in TOOL_SPECS]

    def __len__(self) -> int:
        return len(TOOL_SPECS)

    def __contains__(self, key: object) -> bool:
        return key in TOOL_SPECS


# Public registry. Behaves like ``dict[str, ToolSpec]`` with 9 entries; each
# ``ToolSpec`` is resolved (its module imported) on first access to that key.
TOOL_REGISTRY: dict[str, ToolSpec] = _LazyRegistry()


# ---------------------------------------------------------------------------
# google-genai FunctionDeclaration derivation
# ---------------------------------------------------------------------------
# Map JSON-schema primitive type names to the google-genai Type enum.
_JSON_TYPE_TO_GENAI: dict[str, types.Type] = {
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
    "array": types.Type.ARRAY,
    "object": types.Type.OBJECT,
    "null": types.Type.NULL,
}


def _resolve_ref(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Inline a local ``$ref`` (``#/$defs/Name``) against the schema's ``$defs``.

    Args:
        node: JSON-schema node, possibly containing ``$ref``.
        defs: The ``$defs`` section of the root schema.

    Returns:
        The dereferenced node (the referenced definition), or ``node`` itself if
        it carries no ``$ref``.
    """
    ref = node.get("$ref")
    if not ref:
        return node
    # Refs are of the form "#/$defs/Name".
    name = ref.rsplit("/", 1)[-1]
    return defs.get(name, {})


def _pydantic_to_genai_schema(json_schema: dict[str, Any], defs: dict[str, Any]) -> types.Schema:
    """Convert a Pydantic JSON schema node into a ``google-genai`` ``Schema``.

    Handles the subset of JSON-schema that Pydantic v2 emits for the tool input
    models: objects with properties, arrays, primitives, ``$ref`` to ``$defs``,
    ``anyOf`` (used for ``X | None`` optionals) and ``enum`` (``Literal``).

    Args:
        json_schema: A JSON-schema node (root or nested).
        defs: The ``$defs`` map from the root schema, used to resolve ``$ref``.

    Returns:
        The equivalent :class:`google.genai.types.Schema`.
    """
    node = _resolve_ref(json_schema, defs)

    # ``X | None`` and other unions render as ``anyOf``. Pick the first
    # non-null branch as the representative schema and mark it nullable when a
    # ``null`` branch is present (google-genai has no native union type).
    any_of = node.get("anyOf")
    if any_of:
        non_null = [b for b in any_of if b.get("type") != "null"]
        nullable = len(non_null) != len(any_of)
        chosen = non_null[0] if non_null else any_of[0]
        schema = _pydantic_to_genai_schema(chosen, defs)
        if nullable:
            schema.nullable = True
        if node.get("description") and not schema.description:
            schema.description = node["description"]
        return schema

    kwargs: dict[str, Any] = {}
    if node.get("description"):
        kwargs["description"] = node["description"]

    # Enums (from ``Literal[...]``) carry their own type plus the value list.
    enum_values = node.get("enum")
    if enum_values is not None:
        kwargs["enum"] = [str(v) for v in enum_values]

    json_type = node.get("type")
    genai_type = _JSON_TYPE_TO_GENAI.get(json_type, types.Type.STRING)
    kwargs["type"] = genai_type

    if genai_type is types.Type.OBJECT:
        properties = node.get("properties", {})
        kwargs["properties"] = {
            prop_name: _pydantic_to_genai_schema(prop_schema, defs)
            for prop_name, prop_schema in properties.items()
        }
        required = node.get("required")
        if required:
            kwargs["required"] = list(required)

    if genai_type is types.Type.ARRAY:
        items = node.get("items")
        if items is not None:
            kwargs["items"] = _pydantic_to_genai_schema(items, defs)

    return types.Schema(**kwargs)


def _input_model_to_parameters(input_model: type[BaseModel]) -> types.Schema:
    """Derive the ``parameters`` ``Schema`` of a tool from its input model.

    Args:
        input_model: The tool's Pydantic ``*Input`` model.

    Returns:
        A ``google-genai`` object ``Schema`` describing the tool parameters.
    """
    json_schema = input_model.model_json_schema(ref_template="#/$defs/{model}")
    defs = json_schema.get("$defs", {})
    return _pydantic_to_genai_schema(json_schema, defs)


def build_function_declarations() -> list[types.FunctionDeclaration]:
    """Build one ``google-genai`` ``FunctionDeclaration`` per registered tool.

    The parameter schema of each declaration is derived from the tool's Pydantic
    input model. Deferred tools are tagged ``Behavior.NON_BLOCKING`` so the SDK
    knows they may resolve out of band; synchronous tools use ``BLOCKING``.

    Returns:
        Nine :class:`google.genai.types.FunctionDeclaration`, in registry order.
    """
    declarations: list[types.FunctionDeclaration] = []
    for name, (_module, input_model, _output, deferred, description) in TOOL_SPECS.items():
        parameters = _input_model_to_parameters(input_model)
        behavior = types.Behavior.NON_BLOCKING if deferred else types.Behavior.BLOCKING
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=parameters,
                behavior=behavior,
            )
        )
    return declarations


def get_sync_tools() -> list[ToolSpec]:
    """Return the 5 synchronous demo tools (``deferred=False``).

    Returns:
        The synchronous :class:`ToolSpec` list, in registry order.
    """
    return [get_tool(name) for name in TOOL_SPECS if name in _SYNC_TOOL_NAMES]


def get_deferred_tools() -> list[ToolSpec]:
    """Return the 4 deferred/background tools (``deferred=True``).

    Returns:
        The deferred :class:`ToolSpec` list, in registry order.
    """
    return [get_tool(name) for name in TOOL_SPECS if name not in _SYNC_TOOL_NAMES]
