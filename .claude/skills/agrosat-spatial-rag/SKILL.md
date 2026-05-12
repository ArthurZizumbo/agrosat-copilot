---
name: agrosat-spatial-rag
description: Implement hybrid Spatial-RAG combining PostGIS ST_DWithin spatial filter with pgvector semantic similarity (e5-mistral-7b embeddings) for AgroSatCopilot agent. Use to reduce hallucinations ~30% per GeoAnalystBench 2025 pattern.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Spatial-RAG Skill

## Rules — NON-NEGOTIABLE

- Pipeline en serie: (1) filtrado espacial `ST_DWithin` (2) similitud semántica pgvector (3) fusión weighted score
- Embeddings con `intfloat/e5-mistral-7b-instruct` (4096-dim, licencia MIT)
- HNSW index sobre `embedding VECTOR(4096)` con `m=16, ef_construction=64`
- Top-k semantic = 4× top-k final para diversidad
- Documentos versionados en `rag_documents` table con geom + embedding
- Reranking opcional con cross-encoder

## Embedding Function

```python
# ml/agent/rag.py
from sentence_transformers import SentenceTransformer
import torch

_model: SentenceTransformer | None = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "intfloat/e5-mistral-7b-instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _model

def embed_query(query: str) -> list[float]:
    """Devuelve embedding 4096-dim normalizado."""
    return get_embedder().encode(
        [f"Instruct: Retrieve agronomic information.\nQuery: {query}"],
        normalize_embeddings=True,
    )[0].tolist()
```

## Spatial-RAG Híbrido

```python
from sqlmodel import select
from geoalchemy2.functions import ST_DWithin, ST_GeomFromGeoJSON

async def spatial_rag(
    session,
    query: str,
    aoi: dict,
    top_k: int = 5,
    spatial_weight: float = 0.4,
    radius_m: float = 5000,
) -> list[RAGDocument]:
    """Hybrid retrieval: spatial proximity + semantic similarity.

    Steps:
    1. Spatial filter: parcels/docs within radius_m of AOI centroid.
    2. Semantic similarity: pgvector cosine over top_k*4 candidates.
    3. Weighted fusion: spatial_weight * (1/distance) + (1-spatial_weight) * cosine_score.
    """
    aoi_geom = ST_GeomFromGeoJSON(json.dumps(aoi))

    # 1. Spatial neighbors
    spatial_stmt = select(RAGDocument, ST_Distance(RAGDocument.geom, aoi_geom).label("dist")) \
        .where(ST_DWithin(RAGDocument.geom, aoi_geom, radius_m)) \
        .limit(top_k * 4)
    spatial_rows = (await session.execute(spatial_stmt)).all()

    # 2. Semantic candidates
    query_emb = embed_query(query)
    semantic_stmt = select(RAGDocument, RAGDocument.embedding.cosine_distance(query_emb).label("cos_dist")) \
        .order_by("cos_dist") \
        .limit(top_k * 4)
    semantic_rows = (await session.execute(semantic_stmt)).all()

    # 3. Weighted fusion
    scored = {}
    for doc, dist in spatial_rows:
        scored.setdefault(doc.id, {"doc": doc, "spatial": 0, "semantic": 0})
        scored[doc.id]["spatial"] = 1.0 / (1.0 + dist / radius_m)
    for doc, cos_dist in semantic_rows:
        scored.setdefault(doc.id, {"doc": doc, "spatial": 0, "semantic": 0})
        scored[doc.id]["semantic"] = 1.0 - cos_dist

    ranked = sorted(
        scored.values(),
        key=lambda r: spatial_weight * r["spatial"] + (1 - spatial_weight) * r["semantic"],
        reverse=True,
    )
    return [r["doc"] for r in ranked[:top_k]]
```

## Ingestion Pipeline

```python
async def ingest_rag_documents(session, documents: list[dict]):
    """Carga documentos agronómicos + scene metadata al RAG."""
    embedder = get_embedder()
    texts = [d["content"] for d in documents]
    embeddings = embedder.encode(texts, normalize_embeddings=True, batch_size=8)
    for doc, emb in zip(documents, embeddings):
        rag_doc = RAGDocument(
            content=doc["content"],
            embedding=emb.tolist(),
            source=doc["source"],
            geom=doc.get("geom"),
        )
        session.add(rag_doc)
    await session.commit()
```

## Integración con Agente

```python
# ml/agent/agent.py
from ml.agent.rag import spatial_rag

async def with_rag_context(query: str, aoi: dict, session) -> str:
    docs = await spatial_rag(session, query, aoi, top_k=5)
    context = "\n\n".join(f"[{d.source}] {d.content}" for d in docs)
    return f"Contexto recuperado:\n{context}\n\nPregunta: {query}"
```

## QA Checklist RAG

- [ ] HNSW index sobre embedding column
- [ ] GIST index sobre geom column
- [ ] Embeddings normalizados (cosine)
- [ ] Fusion weighted con pesos configurables
- [ ] Eval reducción hallucinations vs no-RAG (esperado ~30%)
- [ ] Ingest pipeline batch
