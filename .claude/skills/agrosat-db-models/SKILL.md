---
name: agrosat-db-models
description: Define SQLModel + GeoAlchemy2 ORM models for AgroSatCopilot PostgreSQL schema. Use when creating/modifying models with geometry/geography columns, vector columns (pgvector), relationships, enums, or Pydantic response types.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot DB Models Skill

## Rules — NON-NEGOTIABLE

- Three-class pattern: `XBase` (shared, Pydantic) → `X` (table=True ORM) → `XResponse` (API output)
- Geometry: `Column(Geometry("POLYGON", srid=4326))`
- Vector: `Column(Vector(64))` AlphaEarth, `Vector(4096)` e5-mistral
- Enums via `enum.Enum` mapped to PostgreSQL enum
- Async session with `sqlalchemy.ext.asyncio`
- Never `SQLModel.metadata.create_all()` — dbmate only

## Model Pattern

```python
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from geoalchemy2 import Geometry

class AOIBase(SQLModel):
    session_id: str = Field(index=True, max_length=64)
    label: str | None = None
    area_ha: float | None = None

class AOI(AOIBase, table=True):
    __tablename__ = "aois"
    id: int | None = Field(default=None, primary_key=True)
    geom: dict = Field(sa_column=Column(Geometry("POLYGON", srid=4326), nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

class AOIResponse(AOIBase):
    id: int
    geom_geojson: dict
    created_at: datetime
```

## Vector Embeddings

```python
from pgvector.sqlalchemy import Vector

class RAGDocument(SQLModel, table=True):
    __tablename__ = "rag_documents"
    id: int | None = Field(default=None, primary_key=True)
    content: str
    embedding: list[float] = Field(sa_column=Column(Vector(4096), nullable=False))
    source: str
    geom: dict | None = Field(sa_column=Column(Geometry("POLYGON", srid=4326)))
```

## Enums

```python
import enum
from sqlalchemy.types import Enum as SAEnum

class LLMVariant(str, enum.Enum):
    gemini = "gemini"
    qwen35 = "qwen35"

class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"
    id: str | None = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    llm_variant: LLMVariant = Field(sa_column=Column(SAEnum(LLMVariant), nullable=False))
```

## Spatial Query

```python
from sqlmodel import select
from geoalchemy2.functions import ST_DWithin, ST_GeomFromGeoJSON

async def find_neighbors(session, aoi_geojson, radius_m):
    aoi = ST_GeomFromGeoJSON(json.dumps(aoi_geojson))
    stmt = select(Parcel).where(ST_DWithin(Parcel.geom, aoi, radius_m))
    return (await session.execute(stmt)).scalars().all()
```

## Vector Similarity

```python
async def search_similar(session, query_emb, top_k):
    stmt = select(RAGDocument).order_by(RAGDocument.embedding.cosine_distance(query_emb)).limit(top_k)
    return (await session.execute(stmt)).scalars().all()
```
