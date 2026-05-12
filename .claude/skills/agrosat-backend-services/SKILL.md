---
name: agrosat-backend-services
description: Create or modify FastAPI service classes, dependency injection, and Pub/Sub workers for AgroSatCopilot. Use when implementing business logic in the service layer, integrating the Google ADK agent client, building async workers for inference jobs, or wiring DI containers.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Backend Services Skill

## Rules — NON-NEGOTIABLE

- Service classes hold business logic; routers only orchestrate request → service → response
- Vertex AI / vLLM / GEE / DINOv3 calls live ONLY in services (never in routers)
- Heavy ML inference (>2 s) goes through Pub/Sub → Cloud Run GPU L4 worker
- Services use async DB sessions and async HTTP clients
- Structured logging with `structlog` at every external boundary
- Errors raise typed exceptions (`AgentTimeoutError`, `STACNotFoundError`)

## Service Pattern (Chat)

```python
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from app.core.adk_client import get_agrosat_agent
from app.models.chat import ChatEvent

logger = structlog.get_logger()

class ChatService:
    @staticmethod
    async def run(session, query, aoi, llm_variant, session_id, user_id):
        agent = get_agrosat_agent(variant=llm_variant)
        logger.info("agent_run_started", session_id=session_id, llm=llm_variant)
        async for event in agent.run_async(
            new_message=query,
            session=session_id,
            context={"aoi": aoi, "user_id": user_id},
        ):
            yield ChatEvent(type=event.type, data=event.data, timestamp=event.ts)
```

## Pub/Sub Worker (Cloud Run GPU L4)

```python
# backend/app/workers/inference_worker.py
from cloudevents.http import from_http
from fastapi import FastAPI, Request
from app.services.inference_service import InferenceService

app = FastAPI()

@app.post("/", status_code=204)
async def handle_pubsub(request: Request):
    event = from_http(request.headers, await request.body())
    await InferenceService.run(event.data["message"])
```

## Dependency Injection

```python
# backend/app/core/adk_client.py
from functools import lru_cache
from ml.agent.agent import build_agent

@lru_cache(maxsize=2)
def get_agrosat_agent(variant: str):
    return build_agent(llm_variant=variant)
```

## Job Service

```python
from google.cloud import pubsub_v1
import json, uuid

class JobService:
    @staticmethod
    async def enqueue(topic: str, payload: dict) -> str:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(settings.GCP_PROJECT, topic)
        job_id = str(uuid.uuid4())
        publisher.publish(topic_path, json.dumps({"job_id": job_id, **payload}).encode()).result(timeout=10)
        return job_id
```

## Error Hierarchy

```python
class AgrosatError(Exception): ...
class AgentTimeoutError(AgrosatError): ...
class STACNotFoundError(AgrosatError): ...
class GEEQuotaExceededError(AgrosatError): ...
class VLLMUnavailableError(AgrosatError): ...
```

## QA Checklist

- [ ] Business logic en service
- [ ] Clientes externos via DI
- [ ] Inferencia pesada → Pub/Sub
- [ ] Errores tipados
- [ ] Logging estructurado
- [ ] Tests con mocks ADK / GEE / Vertex
