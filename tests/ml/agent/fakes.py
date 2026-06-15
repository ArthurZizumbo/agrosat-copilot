"""Deterministic in-memory fakes for the agent unit tests.

No network, no DB, no Vertex/vLLM. The fakes satisfy the ``ParcelReader`` /
``ChatMemory`` Protocols and an ``LLMBackend``-shaped object so the orchestrator
and tools can be exercised in isolation.
"""

from __future__ import annotations

from ml.agent.backends import ChatMessage, LLMResult
from ml.agent.ports import ChatTurn, FeatureRecord, ParcelRecord


class FakeParcelReader:
    """In-memory ``ParcelReader`` backed by plain dicts."""

    def __init__(
        self,
        parcels: list[ParcelRecord],
        features: dict[int, FeatureRecord] | None = None,
    ) -> None:
        self._parcels = parcels
        self._features = features or {}

    async def list_parcels_in_aoi(
        self, *, session_id: str, aoi_id: int | None = None, year: int | None = None
    ) -> list[ParcelRecord]:
        out = self._parcels
        if aoi_id is not None:
            out = [p for p in out if p.aoi_id == aoi_id]
        if year is not None:
            out = [p for p in out if p.year == year]
        return list(out)

    async def get_features(
        self, *, session_id: str, parcel_id: int, year: int
    ) -> FeatureRecord | None:
        return self._features.get(parcel_id)


class FakeChatMemory:
    """In-memory ``ChatMemory`` recording appended turns."""

    def __init__(self, history: list[ChatTurn] | None = None) -> None:
        self.turns: list[ChatTurn] = list(history or [])

    async def load_history(self, *, session_id: str, limit: int = 20) -> list[ChatTurn]:
        return list(self.turns[-limit:])

    async def append_turn(self, *, session_id: str, turn: ChatTurn) -> None:
        self.turns.append(turn)


class FakeLLMBackend:
    """An ``LLMBackend`` returning a canned, deterministic completion."""

    name = "fake"

    def __init__(self, text: str = "Respuesta sintetizada del LLM.") -> None:
        self.text = text
        self.calls: list[list[ChatMessage]] = []

    async def generate(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(text=self.text, model="fake-model", finish_reason="stop")


class FakeBundle:
    """Stand-in for a ``BaselineResult`` exposing the classify_parcel contract."""

    def __init__(self, model: object, label_classes: tuple[int, ...], n_features: int) -> None:
        self.model = model
        self.label_classes = label_classes
        self.feature_cols = tuple(f"ae_{i:02d}" for i in range(n_features))


class FakeXGBModel:
    """Minimal sklearn-like classifier with deterministic ``predict_proba``."""

    def __init__(self, classes: tuple[int, ...], forced_index: int = 0) -> None:
        self.classes_ = classes
        self._forced_index = forced_index

    def predict_proba(self, x):  # type: ignore[no-untyped-def]
        import numpy as np

        n = len(self.classes_)
        row = np.full(n, 0.1 / max(n - 1, 1))
        row[self._forced_index] = 0.9
        return np.asarray([row for _ in range(len(x))])
