"""Configuración central de la aplicación.

Carga variables de entorno vía Pydantic Settings. Nunca leer ``os.environ``
directamente desde routers o services — siempre vía ``get_settings()``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración tipada del backend AgroSatCopilot."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    database_url: str = Field(default="postgresql+asyncpg://agrosat:agrosat@localhost:5432/agrosat")
    redis_url: str = Field(default="redis://localhost:6379/0")

    gcp_project_id: str = "agrosat-prod"
    gcp_region: str = "europe-west1"

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la instancia singleton de configuración."""
    return Settings()
