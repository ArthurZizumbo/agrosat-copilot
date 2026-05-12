---
name: agrosat-backend-api
description: Create or modify FastAPI endpoints, routers, and REST/SSE handlers for AgroSatCopilot. Use when adding endpoints (/chat SSE, /aois, /timeseries, /stac/search, /tiles, /llm/switch, /jobs), implementing handlers, integrating the Google ADK agent into HTTP layer, or modifying FastAPI routers.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Backend API Skill

Current branch: `! git branch --show-current`
Existing routers: `! ls backend/app/api/ 2>/dev/null || echo "no api dir found"`

## Rules — NON-NEGOTIABLE

- Every protected endpoint MUST use `Depends(get_current_user)` (Clerk JWT)
- Every endpoint touching session data MUST call `_check_session_owner(session_id, current_user)`
- Router receives request → delegates to Service → never contains business logic
- GeoJSON inputs validated with Pydantic `Feature`/`FeatureCollection` schema BEFORE service call
- ML inference >2 s → publish to Pub/Sub `inference-jobs`, return `job_id`
- `/chat` endpoint returns `StreamingResponse(media_type="text/event-stream")` with ADK events
- Return Pydantic response models, never raw SQLModel or GeoAlchemy2 objects
- Rate limit `/chat` 10 req/min, `/llm/switch` 5 req/min, all by user_id

## Router Structure (chat SSE)

```python
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.auth import get_current_user
from app.core.database import get_session
from app.core.rate_limit import limiter
from app.models.chat import ChatRequest, ChatEvent
from app.models.user import UserPublic
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_class=StreamingResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    session: AsyncSession = Depends(get_session),
    current_user: UserPublic = Depends(get_current_user),
) -> StreamingResponse:
    """Stream ADK agent events: plan_created, tool_call, tool_result, final_answer."""
    _check_session_owner(body.session_id, current_user)

    async def event_stream():
        async for event in ChatService.run(
            session=session,
            query=body.query,
            aoi=body.aoi_geojson,
            llm_variant=body.llm_variant,
            session_id=body.session_id,
            user_id=current_user.id,
        ):
            yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

## Session Owner Check

```python
def _check_session_owner(session_id: str, current_user: UserPublic) -> None:
    if current_user.role != "super_admin":
        if not current_user.session_ids or session_id not in current_user.session_ids:
            raise HTTPException(status_code=403, detail="Access denied")
```

## TiTiler Tile Endpoint

```python
from titiler.core.factory import TilerFactory

cog_tiler = TilerFactory(router_prefix="/tiles/cog")
router.include_router(cog_tiler.router)
# GET /tiles/cog/{z}/{x}/{y}.png?url=gs://agrosat-data/alphaearth/toscana/2025.tif&rescale=0,1&colormap_name=viridis
```

## Pub/Sub Job Dispatch

```python
@router.post("/aois/{aoi_id}/classify")
async def classify(aoi_id: int, current_user=Depends(get_current_user)):
    job_id = await JobService.enqueue(
        topic="inference-jobs",
        payload={"aoi_id": aoi_id, "user_id": current_user.id, "action": "classify"},
    )
    return {"job_id": job_id, "status": "queued"}
```

## SSE Event Schema

```python
from typing import Literal
from pydantic import BaseModel

class ChatEvent(BaseModel):
    type: Literal["plan_created", "tool_call", "tool_result", "final_answer", "error"]
    data: dict
    timestamp: str
```

## Endpoints Canónicos

| Method | Path | EPIC | Skill |
|--------|------|------|-------|
| GET    | `/healthz` | E0 | — |
| POST   | `/chat` SSE | E7-E8 | this + `agrosat-google-adk-agent` |
| GET    | `/aois/{id}` | E8 | this |
| POST   | `/aois` | E8 | this |
| GET    | `/aois/{id}/segmentation` | E8 | this |
| GET    | `/timeseries` | E8 | this + `agrosat-ml-features` |
| GET    | `/stac/search` | E1 | this + `agrosat-db-models` |
| GET    | `/tiles/{z}/{x}/{y}.png` | E8 | `agrosat-titiler-cog` |
| POST   | `/llm/switch` | E7 | this |
| GET    | `/llm/health` | E7 | this |
| GET    | `/jobs/{id}` | E8 | this |

## Structured Logging

```python
import structlog
logger = structlog.get_logger()
logger.info(
    "chat_request",
    session_id=body.session_id,
    user_id=current_user.id,
    llm_variant=body.llm_variant,
)
```

## Register Router

```python
# backend/app/main.py
from app.api.chat import router as chat_router
app.include_router(chat_router, prefix="/api/v1")
```
