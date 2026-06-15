"""Agent-local configuration (hexagonal: no dependency on ``backend.app``).

The conversational agent must stay independent of the FastAPI backend so it can
be tested with fakes and later ported to Vertex AI Agent Engine. Instead of
importing ``backend.app.core.config`` we declare a small pydantic-settings model
that reads the same environment variables documented in ``.env.example`` (lines
68-88: LLM backends + google-genai SDK).

Always read configuration through :func:`get_settings` (cached) — never read
``os.environ`` directly from the tools or the orchestrator.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Typed configuration the agent needs from the environment.

    ``extra="ignore"`` (not ``"forbid"``) so the shared ``.env.local`` — which
    carries many backend-only keys — does not abort the agent at import time.
    """

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM backend selection ---------------------------------------------
    llm_variant_default: str = "gemini"
    """Default backend variant when the caller does not pin one ('gemini'|'qwen35')."""

    # --- Gemini / Vertex AI (variant A) ------------------------------------
    gemini_model: str = "gemini-3.1-pro"
    google_genai_use_vertexai: bool = True
    google_cloud_project: str = "agrosat-copilot"
    google_cloud_location: str = "us-central1"
    vertex_ai_location: str = "us-central1"
    gemini_api_key: str = ""

    # --- Qwen3.5-35B-A3B over vLLM (variant B) -----------------------------
    vllm_qwen35_url: str = "http://vllm-qwen35.internal:8000/v1"
    vllm_api_key: str = "not-needed"
    vllm_qwen35_model: str = "Qwen/Qwen3.5-35B-A3B"

    # --- Crop classifier (MLflow Model Registry) ---------------------------
    crop_classifier_model_uri: str = "models:/baseline-xgb-hcat6@champion"
    """MLflow registry URI for the trained XGBoost+AlphaEarth baseline.

    Default points at the Model Registry alias; override with the env var
    ``CROP_CLASSIFIER_MODEL_URI`` (e.g. ``models:/baseline-xgb-hcat6/Production``
    or a local ``runs:/<run_id>/model`` URI) when the registry is unavailable.
    """

    # --- Generation knobs --------------------------------------------------
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 1024
    llm_http_timeout_ms: int = 60_000


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    """Return the cached :class:`AgentSettings` instance."""
    return AgentSettings()


__all__ = ["AgentSettings", "get_settings"]
