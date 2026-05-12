---
name: agrosat-testing
description: Write pytest + pytest-asyncio unit/integration/e2e tests and Playwright browser tests for AgroSatCopilot. Use when testing backend services, ADK tools with mocks (Vertex AI, vLLM, GEE), frontend components, or end-to-end /chat flow.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Testing Skill

## Rules — NON-NEGOTIABLE

- Cobertura backend ≥70%, frontend ≥50%
- Mocks para Vertex AI, vLLM, Google Earth Engine, HuggingFace
- Tests E2E con Playwright sobre flujo crítico: login → AOI → chat → resultado
- Fixtures deterministas (no llamadas reales a APIs externas)
- pytest-asyncio para tests async
- Testcontainers para PostgreSQL + Redis integration tests

## Backend Unit Test Pattern

```python
# backend/tests/unit/services/test_chat_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.chat_service import ChatService

@pytest.fixture
def mock_adk_agent():
    agent = AsyncMock()
    async def fake_run(**kwargs):
        yield MagicMock(type="plan_created", data={"steps": ["alphaearth_query"]}, ts="2026-05-15T10:00:00")
        yield MagicMock(type="tool_call", data={"tool": "alphaearth_query"}, ts="...")
        yield MagicMock(type="tool_result", data={"area_ha": 42.3}, ts="...")
        yield MagicMock(type="final_answer", data={"text": "42.3 ha"}, ts="...")
    agent.run_async = fake_run
    return agent

@pytest.mark.asyncio
async def test_chat_service_streams_events(mock_adk_agent, monkeypatch):
    monkeypatch.setattr("app.services.chat_service.get_agrosat_agent", lambda v: mock_adk_agent)
    events = []
    async for ev in ChatService.run(
        session=AsyncMock(), query="ciao", aoi={}, llm_variant="gemini",
        session_id="s1", user_id="u1",
    ):
        events.append(ev)
    assert [e.type for e in events] == ["plan_created", "tool_call", "tool_result", "final_answer"]
```

## Mocking GEE

```python
@pytest.fixture
def mock_ee(monkeypatch):
    monkeypatch.setattr("ml.ingest.alphaearth.ee.Initialize", lambda *a, **kw: None)
    mock_task = MagicMock()
    mock_task.id = "TASK_TEST_123"
    mock_task.start = MagicMock()
    monkeypatch.setattr("ml.ingest.alphaearth.ee.batch.Export.image.toCloudStorage", lambda **kw: mock_task)
```

## Mocking vLLM / Vertex AI

```python
@pytest.fixture
def mock_vllm(httpx_mock):
    httpx_mock.add_response(
        url="http://vllm-qwen35.internal:8000/v1/chat/completions",
        json={"choices": [{"message": {"content": "mocked response"}}]},
    )

@pytest.fixture
def mock_vertex():
    with patch("vertexai.preview.generative_models.GenerativeModel") as m:
        m.return_value.generate_content_async.return_value.text = "mocked"
        yield m
```

## Integration con Testcontainers

```python
# backend/tests/integration/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def postgres():
    with PostgresContainer("postgis/postgis:15-3.4") as pg:
        # Apply migrations
        subprocess.run(["dbmate", "--url", pg.get_connection_url(), "up"], check=True)
        yield pg

@pytest.fixture(scope="session")
def redis():
    with RedisContainer("redis:7-alpine") as r:
        yield r
```

## Playwright E2E

```typescript
// frontend/tests/e2e/chat_flow.spec.ts
import { test, expect } from '@playwright/test'

test('user can draw AOI and chat about it', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Sign in' }).click()
  await page.fill('[name="email"]', 'test@agrosat.io')
  await page.fill('[name="password"]', process.env.TEST_PWD!)
  await page.getByRole('button', { name: 'Continue' }).click()
  await page.waitForURL('/dashboard')

  // Draw polygon on map
  await page.getByTitle('Draw polygon').click()
  await page.click('#map-container', { position: { x: 100, y: 100 } })
  await page.click('#map-container', { position: { x: 200, y: 100 } })
  await page.click('#map-container', { position: { x: 200, y: 200 } })
  await page.dblclick('#map-container', { position: { x: 100, y: 200 } })

  // Send query
  await page.fill('[data-testid="chat-input"]', 'Di che coltura si tratta?')
  await page.getByRole('button', { name: 'Send' }).click()

  // Verify streaming response
  await expect(page.getByTestId('chat-message-assistant')).toBeVisible({ timeout: 15000 })
  await expect(page.getByTestId('tool-call-trace')).toBeVisible()
})
```

## Comandos

```bash
make test                    # pytest backend con coverage
make test-unit               # unit only
make test-integration        # con testcontainers
make test-e2e                # Playwright
make test-frontend           # vitest
pytest -k "test_session_isolation" -v
```

## QA Checklist Tests

- [ ] Cobertura ≥70% backend, ≥50% frontend
- [ ] Mocks para APIs externas (no llamadas reales)
- [ ] Testcontainers para Postgres + Redis
- [ ] Playwright E2E para flujo crítico
- [ ] Tests pasan en CI < 5 min
