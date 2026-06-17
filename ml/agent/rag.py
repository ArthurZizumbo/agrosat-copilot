"""Spatial-RAG *lite* retrieval layer for the conversational agent (US-046).

This is the "lite" variant of the hybrid Spatial-RAG: it grounds the reasoner on
the real PASTIS-R phenology corpus (parcel captions) plus scene metadata, ranked
by a two-signal fusion of spatial proximity and semantic similarity. The full
Spatial-RAG (``intfloat/e5-mistral-7b-instruct`` 4096-dim embeddings + HNSW index
+ cross-encoder reranking, per the ``agrosat-spatial-rag`` skill) is FUTURE and is
deliberately NOT implemented here.

Two intentional departures from the full design, both mandated by US-046:

1. **Vector**: the lite layer reuses the AlphaEarth 64-dim annual embedding that is
   already persisted in ``rag_documents.embedding`` (and in ``features_parcels``);
   it never loads e5-mistral, never touches a GPU and never embeds text at query
   time. The "semantic" axis is therefore embedding-space similarity in the same
   64-dim AlphaEarth space the classifier consumes, not text similarity.
2. **No ANN index**: there is no HNSW/IVFFlat over ``embedding``. The pipeline runs
   in series -- a PostGIS ``ST_DWithin`` pre-filter first narrows the corpus to a
   small spatial candidate set, and only then does pgvector run a flat cosine scan
   over that set. For the demo corpus (fold-5 subset) this is cheap.

Pipeline (in series, matching AC-8):

    1. ``ST_DWithin(geom::geography, aoi::geography, radius_m)`` -> candidate set.
    2. pgvector cosine ``embedding <=> query_embedding`` over the candidates.
    3. weighted fusion ``spatial_weight * prox + (1 - spatial_weight) * sem``.

The query embedding is the AlphaEarth 64-dim vector of the AOI. The agent does not
carry an AlphaEarth embedding for an arbitrary drawn polygon, so it is resolved
from the corpus itself: the embedding of the spatially-nearest document to the AOI
centroid. When the AOI has no spatial neighbour with an embedding, the layer
degrades gracefully to a spatial-only ranking (documented below).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict

from ml.agent.db import session_scoped_conn

if TYPE_CHECKING:
    import asyncpg

    from ml.agent.context import ToolContext

logger = structlog.get_logger(__name__)

__all__ = ["RAGDocument", "ingest_rag_documents", "spatial_rag"]

#: AlphaEarth annual Satellite Embedding V1 dimensionality (matches VECTOR(64)).
_EMBED_DIM: int = 64

#: Source tag for the per-parcel phenology caption corpus.
SOURCE_PHENOLOGY_CAPTION: str = "phenology_caption"

#: Source tag for scene-level metadata documents.
SOURCE_SCENE_META: str = "scene_meta"


class RAGDocument(BaseModel):
    """A retrieved corpus document with its fused relevance score.

    Attributes:
        id: Primary key of the document in ``rag_documents``.
        content: Document text (phenology description or scene metadata).
        source: Document kind (``"phenology_caption"`` | ``"scene_meta"``).
        parcel_id: Composite corpus parcel id, or ``None`` for scene-level docs.
        distance_m: Geodesic distance in metres from the AOI to the document
            geometry, or ``None`` when the document carries no geometry.
        score: Fused relevance score in ``[0, 1]`` (spatial proximity + semantic
            similarity); higher is more relevant.
    """

    model_config = ConfigDict(strict=True)

    id: int
    content: str
    source: str
    parcel_id: str | None = None
    distance_m: float | None = None
    score: float


# Strict cap on how many spatial candidates pgvector scans per call. The lite
# layer relies on ST_DWithin to bound this; the LIMIT is a hard backstop so a
# pathologically large radius cannot trigger a full-corpus cosine scan.
_MAX_CANDIDATES: int = 256


async def spatial_rag(
    ctx: ToolContext,
    query: str,
    aoi: dict,
    *,
    top_k: int = 5,
    spatial_weight: float = 0.4,
    radius_m: float = 5000.0,
) -> list[RAGDocument]:
    """Retrieve the most relevant corpus documents near an AOI (hybrid, in series).

    Runs the three-stage lite pipeline (spatial pre-filter -> pgvector cosine ->
    weighted fusion) and returns the ``top_k`` documents by fused score. Every
    query is session-scoped through :func:`ml.agent.db.session_scoped_conn`, so a
    future RLS policy on ``rag_documents`` applies transparently.

    Args:
        ctx: Tool execution context (asyncpg pool, settings, session id).
        query: Natural-language query (kept for tracing/parity with the full RAG;
            the lite layer ranks on the AlphaEarth embedding axis, not text).
        aoi: GeoJSON geometry of the area of interest (EPSG:4326).
        top_k: Number of documents to return after fusion.
        spatial_weight: Weight of the spatial-proximity signal in ``[0, 1]``; the
            semantic signal gets ``1 - spatial_weight``.
        radius_m: ``ST_DWithin`` search radius in metres (geography).

    Returns:
        Up to ``top_k`` :class:`RAGDocument` ordered by descending fused score.
        Empty when no document lies within ``radius_m`` of the AOI.
    """
    aoi_json = json.dumps(aoi)
    logger.info(
        "spatial_rag_started",
        session_id=str(ctx.session_id),
        query_len=len(query),
        top_k=top_k,
        spatial_weight=spatial_weight,
        radius_m=radius_m,
    )

    async with session_scoped_conn(ctx.session_id) as conn:
        # Stage 1: ST_DWithin spatial pre-filter (geodesic, on geography). The
        # candidate set is bounded by both the radius and a hard LIMIT so the
        # downstream cosine scan stays cheap (no ANN index by design).
        candidates = await conn.fetch(
            """
            SELECT
                d.id,
                d.content,
                d.source,
                d.parcel_id,
                d.embedding,
                ST_Distance(
                    d.geom::geography,
                    ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)::geography
                ) AS distance_m
            FROM rag_documents d
            WHERE d.geom IS NOT NULL
              AND ST_DWithin(
                    d.geom::geography,
                    ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)::geography,
                    $2
                )
            ORDER BY distance_m ASC
            LIMIT $3
            """,
            aoi_json,
            radius_m,
            _MAX_CANDIDATES,
        )

        if not candidates:
            logger.info(
                "spatial_rag_no_candidates",
                session_id=str(ctx.session_id),
                radius_m=radius_m,
            )
            return []

        # Resolve the AOI query embedding from the corpus: the AlphaEarth vector
        # of the spatially-nearest document that actually has one. The agent has
        # no AlphaEarth embedding for an arbitrary drawn polygon, and re-sampling
        # GEE here is out of scope (no GPU / no text embedder in the lite layer).
        query_embedding = _nearest_embedding(candidates)

        cosine_by_id: dict[int, float] = {}
        if query_embedding is not None:
            # Stage 2: pgvector cosine over the (already small) candidate set.
            # ``<=>`` is the cosine-distance operator; 1 - distance is similarity.
            candidate_ids = [int(row["id"]) for row in candidates]
            semantic_rows = await conn.fetch(
                """
                SELECT
                    d.id,
                    (d.embedding <=> $1::vector) AS cosine_distance
                FROM rag_documents d
                WHERE d.id = ANY($2::bigint[])
                  AND d.embedding IS NOT NULL
                ORDER BY cosine_distance ASC
                """,
                _to_pgvector_literal(query_embedding),
                candidate_ids,
            )
            cosine_by_id = {int(row["id"]): float(row["cosine_distance"]) for row in semantic_rows}

    # Stage 3: weighted fusion of spatial proximity and semantic similarity.
    # When no embedding could be resolved, the semantic term is absent for every
    # candidate, so the ranking degrades cleanly to spatial-only (documented).
    documents = _fuse_and_rank(
        candidates,
        cosine_by_id=cosine_by_id,
        spatial_weight=spatial_weight,
        radius_m=radius_m,
        top_k=top_k,
    )
    logger.info(
        "spatial_rag_finished",
        session_id=str(ctx.session_id),
        n_candidates=len(candidates),
        n_with_embedding=len(cosine_by_id),
        n_returned=len(documents),
        semantic_used=query_embedding is not None,
    )
    return documents


def _nearest_embedding(candidates: list[asyncpg.Record]) -> list[float] | None:
    """Pick the AlphaEarth embedding of the spatially-nearest candidate.

    The candidate rows arrive ordered by ascending distance, so the first row that
    carries a parseable 64-dim embedding is the nearest usable query embedding.

    Args:
        candidates: Spatially pre-filtered rows (distance-ordered) with an
            ``embedding`` column.

    Returns:
        A 64-element ``float`` list, or ``None`` if no candidate has a usable
        embedding (the pipeline then degrades to spatial-only ranking).
    """
    for row in candidates:
        embedding = _parse_pgvector(row["embedding"])
        if embedding is not None:
            return embedding
    return None


def _parse_pgvector(raw: object) -> list[float] | None:
    """Parse a pgvector value returned by asyncpg into a ``list[float]``.

    Without a registered codec, asyncpg returns a ``vector`` column as the text
    literal ``"[0.1,0.2,...]"``; native sequences are also accepted.

    Args:
        raw: The raw ``embedding`` cell from a query row.

    Returns:
        A 64-element ``float`` list, or ``None`` when the value is missing or has
        an unexpected dimensionality.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        body = raw.strip().strip("[]")
        if not body:
            return None
        values = [float(v) for v in body.split(",") if v.strip()]
    elif isinstance(raw, Iterable):
        values = [float(v) for v in raw]
    else:
        logger.warning("rag_embedding_unexpected_type", got=type(raw).__name__)
        return None
    if len(values) != _EMBED_DIM:
        logger.warning("rag_embedding_unexpected_dim", expected=_EMBED_DIM, got=len(values))
        return None
    return values


def _to_pgvector_literal(embedding: list[float]) -> str:
    """Render a float vector as a pgvector text literal (``"[v0,v1,...]"``).

    Args:
        embedding: The query embedding (64-dim).

    Returns:
        The pgvector literal accepted by an ``$1::vector`` bind.
    """
    # pgvector rejects NaN/Infinity in a ``vector`` literal; coerce non-finite
    # components to 0.0 (mirrors the AlphaEarth feature sanitisation in
    # ``tools/classify.py``) so a single bad ``dim_k`` does not abort the whole
    # ingest batch on the ``::vector`` cast.
    return "[" + ",".join(repr(f if math.isfinite(f := float(v)) else 0.0) for v in embedding) + "]"


def _fuse_and_rank(
    candidates: list[asyncpg.Record],
    *,
    cosine_by_id: dict[int, float],
    spatial_weight: float,
    radius_m: float,
    top_k: int,
) -> list[RAGDocument]:
    """Fuse spatial proximity and semantic similarity into a ranked document list.

    The fused score for each candidate is::

        score = spatial_weight * (1 / (1 + dist / radius_m))
              + (1 - spatial_weight) * (1 - cosine_distance)

    The spatial term decays smoothly from 1 (at the AOI) toward 0 (at the radius);
    the semantic term is the cosine similarity (``1 - cosine_distance``), defaulting
    to 0 for candidates without an embedding so they rank purely on proximity.

    Args:
        candidates: Distance-ordered spatial candidates.
        cosine_by_id: Map of document id -> cosine distance from the query
            embedding (empty when the semantic stage was skipped).
        spatial_weight: Weight of the spatial term in ``[0, 1]``.
        radius_m: Search radius used to normalise the spatial decay.
        top_k: Number of documents to keep after ranking.

    Returns:
        The ``top_k`` :class:`RAGDocument` by descending fused score.
    """
    radius = radius_m if radius_m > 0.0 else 1.0
    scored: list[RAGDocument] = []
    for row in candidates:
        doc_id = int(row["id"])
        distance_m = row["distance_m"]
        dist = float(distance_m) if distance_m is not None else radius
        spatial_term = 1.0 / (1.0 + dist / radius)

        cosine_distance = cosine_by_id.get(doc_id)
        # pgvector cosine distance is in [0, 2] (2 for opposite-hemisphere
        # vectors); clamp the semantic term to [0, 1] so the fused score honours
        # its documented range and the ranking stays monotone.
        semantic_term = max(0.0, 1.0 - cosine_distance) if cosine_distance is not None else 0.0

        score = spatial_weight * spatial_term + (1.0 - spatial_weight) * semantic_term
        scored.append(
            RAGDocument(
                id=doc_id,
                content=row["content"],
                source=row["source"],
                parcel_id=row["parcel_id"],
                distance_m=float(distance_m) if distance_m is not None else None,
                score=float(score),
            )
        )

    scored.sort(key=lambda doc: doc.score, reverse=True)
    return scored[:top_k]


async def ingest_rag_documents(conn: asyncpg.Connection, documents: list[dict]) -> int:
    """Insert corpus documents into ``rag_documents`` in a single batch.

    Each document dict accepts the keys:

    - ``content`` (str, required): the document text.
    - ``source`` (str, required): ``"phenology_caption"`` | ``"scene_meta"``.
    - ``parcel_id`` (str | None): composite corpus parcel id.
    - ``embedding`` (Sequence[float] | None): AlphaEarth 64-dim vector.
    - ``geom_geojson`` (str | None): GeoJSON geometry string of the source patch.
    - ``geom_srid`` (int): SRID of ``geom_geojson`` (default 4326). The geometry is
      reprojected to 4326 and reduced to its centroid by PostGIS, so the stored
      ``geom`` is always a 4326 POINT regardless of the input projection.
    - ``geom_wkt`` (str | None): alternative geometry as WKT in EPSG:4326 (used
      only when ``geom_geojson`` is absent).

    The reprojection + centroid happen server-side via
    ``ST_Centroid(ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(...), srid), 4326))``,
    so callers never need shapely/pyproj. Documents without a geometry are stored
    with ``geom = NULL`` (they simply never pass the ``ST_DWithin`` pre-filter).

    Args:
        conn: An asyncpg connection (the caller owns the transaction).
        documents: Document dicts as described above.

    Returns:
        The number of rows inserted.
    """
    if not documents:
        return 0

    insert_sql = """
        INSERT INTO rag_documents (parcel_id, content, source, embedding, geom)
        VALUES (
            $1,
            $2,
            $3,
            $4::vector,
            CASE
                WHEN $5::text IS NOT NULL THEN
                    ST_Centroid(
                        ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON($5), $6), 4326)
                    )
                WHEN $7::text IS NOT NULL THEN
                    ST_SetSRID(ST_GeomFromText($7), 4326)
                ELSE NULL
            END
        )
    """

    rows: list[tuple] = []
    for doc in documents:
        embedding = doc.get("embedding")
        embedding_literal = (
            _to_pgvector_literal([float(v) for v in embedding]) if embedding is not None else None
        )
        rows.append(
            (
                doc.get("parcel_id"),
                doc["content"],
                doc["source"],
                embedding_literal,
                doc.get("geom_geojson"),
                int(doc.get("geom_srid", 4326)),
                doc.get("geom_wkt"),
            )
        )

    await conn.executemany(insert_sql, rows)
    logger.info("rag_documents_ingested", n=len(rows))
    return len(rows)
