"""API tests for sessions / aois / llm with the service layer faked.

The DB session dependency yields ``None`` and the service classes used by the
routers are monkeypatched, so the routers (validation, status codes, response
models, ownership guard) are exercised without Postgres.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app.core.db import get_session
from backend.app.main import create_app
from backend.app.services.aoi_service import AoiView

SESSION_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
DEMO_USER = "demo@agrosat.dev"


class _FakeChatSession:
    def __init__(self, variant: str = "gemini") -> None:
        self.id = SESSION_ID
        self.user_id = DEMO_USER
        self.llm_variant = variant


@pytest.fixture
def app_with_db():  # type: ignore[no-untyped-def]
    app = create_app()

    async def fake_get_session():  # type: ignore[no-untyped-def]
        yield None

    app.dependency_overrides[get_session] = fake_get_session
    return app


def test_create_session_returns_201(app_with_db, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_or_create(self, *, user_id, llm_variant):  # type: ignore[no-untyped-def]
        return _FakeChatSession(variant=llm_variant)

    monkeypatch.setattr(
        "backend.app.services.session_service.SessionService.get_or_create_latest",
        fake_get_or_create,
    )
    client = TestClient(app_with_db)
    resp = client.post("/sessions", json={"llm_variant": "qwen35"})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["session_id"] == str(SESSION_ID)
    assert body["user_id"] == DEMO_USER
    assert body["llm_variant"] == "qwen35"


def test_create_session_invalid_variant_422(app_with_db) -> None:
    client = TestClient(app_with_db)
    resp = client.post("/sessions", json={"llm_variant": "gpt4"})
    assert resp.status_code == 422


def _patch_owner(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    async def fake_owner(*, session_id, user_id, db):  # type: ignore[no-untyped-def]
        if session_id != SESSION_ID:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="session not found")
        return _FakeChatSession()

    monkeypatch.setattr(f"{module}._check_session_owner", fake_owner)


def test_llm_switch_ok(app_with_db, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_owner(monkeypatch, "backend.app.api.llm")

    async def fake_switch(self, *, session_id, llm_variant):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "backend.app.services.session_service.SessionService.switch_llm_variant",
        fake_switch,
    )
    client = TestClient(app_with_db)
    resp = client.post(
        "/llm/switch",
        json={"session_id": str(SESSION_ID), "llm_variant": "qwen35"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["llm_variant"] == "qwen35"


def test_llm_switch_not_owner_404(app_with_db, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_owner(monkeypatch, "backend.app.api.llm")
    client = TestClient(app_with_db)
    other = "44444444-4444-4444-4444-444444444444"
    resp = client.post("/llm/switch", json={"session_id": other, "llm_variant": "gemini"})
    assert resp.status_code == 404


def test_create_aoi_computes_area(app_with_db, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_owner(monkeypatch, "backend.app.api.aois")

    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[11.10, 43.30], [11.11, 43.30], [11.11, 43.31], [11.10, 43.31], [11.10, 43.30]]
        ],
    }

    async def fake_create(self, *, session_id, geometry, label):  # type: ignore[no-untyped-def]
        return AoiView(
            id=1,
            session_id=session_id,
            label=label,
            area_ha=88.0,
            geometry=geometry,
        )

    monkeypatch.setattr("backend.app.services.aoi_service.AoiService.create", fake_create)
    client = TestClient(app_with_db)
    resp = client.post(
        f"/aois?session_id={SESSION_ID}",
        json={"geometry": geometry, "label": "field"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["area_ha"] == 88.0
    assert body["geometry"]["type"] == "Polygon"


def test_create_aoi_rejects_bad_ring(app_with_db, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_owner(monkeypatch, "backend.app.api.aois")
    client = TestClient(app_with_db)
    resp = client.post(
        f"/aois?session_id={SESSION_ID}",
        json={
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]},
            "label": None,
        },
    )
    assert resp.status_code == 422


def test_list_aois_ok(app_with_db, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_owner(monkeypatch, "backend.app.api.aois")

    async def fake_list(self, *, session_id):  # type: ignore[no-untyped-def]
        return [
            AoiView(
                id=1,
                session_id=session_id,
                label="a",
                area_ha=10.0,
                geometry={"type": "Polygon", "coordinates": [[[0, 0]]]},
            )
        ]

    monkeypatch.setattr("backend.app.services.aoi_service.AoiService.list_for_session", fake_list)
    client = TestClient(app_with_db)
    resp = client.get(f"/aois?session_id={SESSION_ID}")
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["items"]) == 1
