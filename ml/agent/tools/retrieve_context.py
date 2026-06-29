"""``retrieve_context`` tool: Spatial-RAG *lite* grounding (synchronous, US-046).

This tool injects real corpus grounding into the reasoner. It is a SYNCHRONOUS
in-loop tool (a fast pgvector query, not background work), but it sits behind the
``rag_enabled`` feature flag (default ``False``): :func:`ml.agent.agent.create_agent`
only adds it to the agent's tool set when the flag is on, so the default copilot is
ungrounded exactly as before and the reasoner can ground itself on request when RAG
is enabled (anti-hallucination).

Graceful degradation (AC-10) is the central behaviour:

- ``rag_enabled = False`` (default): the tool returns an empty
  :class:`RetrievedContext` with ``rag_enabled=False`` and ``grounding_text=""``
  WITHOUT touching the database. The reasoner then runs ungrounded, exactly as
  before US-046.
- ``rag_enabled = True``: the tool runs the hybrid lite pipeline
  (:func:`ml.agent.rag.spatial_rag`) over the session-scoped corpus and packs the
  retrieved documents into a single ``grounding_text`` block the reasoner can read.

The new Pydantic contracts (:class:`RetrieveContextInput`, :class:`RetrievedContext`)
live here rather than in ``ml/agent/schemas.py`` because that module is owned by a
sibling US (US-045) and must not be edited from this work-stream. The shared
:class:`~ml.agent.schemas.GeoJSONGeometry` value object IS reused.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ml.agent.context import ToolContext
from ml.agent.rag import RAGDocument, spatial_rag
from ml.agent.schemas import GeoJSONGeometry

logger = structlog.get_logger(__name__)

__all__ = ["RetrieveContextInput", "RetrievedContext", "run"]

#: ``strict`` rejects implicit coercion; ``extra="forbid"`` rejects hallucinated
#: keys -- the same hardening every US-045 tool input uses.
_STRICT_CONFIG = ConfigDict(strict=True, extra="forbid")

#: Default number of grounding documents to retrieve.
_DEFAULT_TOP_K: int = 5

#: Name of the Settings flag gating the whole RAG path (default off).
_RAG_FLAG: str = "rag_enabled"


class RetrieveContextInput(BaseModel):
    """Arguments for ``retrieve_context``.

    Attributes:
        session_id: Tenant session; every corpus query filters by it.
        query: Natural-language query the grounding should support.
        aoi: GeoJSON geometry of the area of interest (EPSG:4326).
        top_k: Number of grounding documents to retrieve.
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    query: str
    aoi: GeoJSONGeometry
    top_k: int = _DEFAULT_TOP_K


class RetrievedContext(BaseModel):
    """Result of ``retrieve_context``.

    Attributes:
        documents: Retrieved corpus documents with their fused scores (empty when
            RAG is disabled or nothing lies near the AOI).
        grounding_text: Concatenated, citation-tagged block the reasoner consumes
            (empty string when RAG is disabled).
        rag_enabled: Echo of the effective ``rag_enabled`` flag, so the caller can
            assert the isolation contract (AC-10) from the tool output alone.
    """

    model_config = ConfigDict(strict=True)

    documents: list[RAGDocument] = Field(default_factory=list)
    grounding_text: str = ""
    rag_enabled: bool = False


def _build_grounding_text(documents: list[RAGDocument]) -> str:
    """Concatenate retrieved documents into one citation-tagged grounding block.

    Each line is prefixed with its source and parcel id so the reasoner can cite
    the origin of every grounded statement (the agent's citation contract).

    Args:
        documents: Retrieved documents, already ranked by fused score.

    Returns:
        A Spanish-prefixed multi-line grounding block, or ``""`` when empty.
    """
    if not documents:
        return ""
    lines: list[str] = ["Contexto recuperado de parcelas vecinas (corpus PASTIS-R):"]
    for doc in documents:
        ref = doc.parcel_id if doc.parcel_id else doc.source
        lines.append(f"[{doc.source}:{ref}] {doc.content}")
    return "\n".join(lines)


async def run(inp: RetrieveContextInput, ctx: ToolContext) -> RetrievedContext:
    """Retrieve spatial-semantic grounding for the AOI, gated by ``rag_enabled``.

    Args:
        inp: Validated arguments (session id, query, AOI polygon, top-k).
        ctx: Tool execution context (asyncpg pool, settings, session id).

    Returns:
        A :class:`RetrievedContext`. When ``rag_enabled`` is off, an empty result
        with ``rag_enabled=False`` and no DB access; when on, the retrieved corpus
        documents plus their concatenated grounding text.
    """
    rag_enabled = bool(getattr(ctx.settings, _RAG_FLAG, False))
    if not rag_enabled:
        # AC-10: with the flag off the loop never touches the corpus -- return a
        # controlled empty grounding without opening a connection.
        logger.info(
            "retrieve_context_disabled",
            session_id=str(inp.session_id),
        )
        return RetrievedContext(documents=[], grounding_text="", rag_enabled=False)

    logger.info(
        "retrieve_context_started",
        session_id=str(inp.session_id),
        query_len=len(inp.query),
        top_k=inp.top_k,
        geometry_type=inp.aoi.type,
    )

    documents = await spatial_rag(
        ctx,
        query=inp.query,
        aoi=inp.aoi.model_dump(),
        top_k=inp.top_k,
    )
    grounding_text = _build_grounding_text(documents)

    logger.info(
        "retrieve_context_finished",
        session_id=str(inp.session_id),
        n_documents=len(documents),
        grounding_chars=len(grounding_text),
    )
    return RetrievedContext(
        documents=documents,
        grounding_text=grounding_text,
        rag_enabled=True,
    )
