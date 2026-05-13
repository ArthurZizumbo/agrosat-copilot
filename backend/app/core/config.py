"""Configuración central de la aplicación.

Carga variables de entorno vía Pydantic Settings. Nunca leer ``os.environ``
directamente desde routers o services — siempre vía ``get_settings()``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_DATABASE_URL = "postgresql+asyncpg://agrosat:agrosat@localhost:5432/agrosat"
_DEV_REDIS_URL = "redis://localhost:6379/0"
# Placeholder rechazado por validator si env != dev.
_JWT_PLACEHOLDER = "change-me-in-prod"


class Settings(BaseSettings):
    """Configuración tipada del backend AgroSatCopilot.

    ``extra="forbid"`` detecta typos en ``.env.local`` (variable definida pero
    no declarada aqui) y aborta el arranque. Agregar cualquier variable nueva
    del ``.env.example`` que el backend deba leer.
    """

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    app_name: str = "agrosatcopilot"
    debug: bool = False

    # Conexiones — defaults locales del docker-compose. En staging/prod son
    # obligatorios y se validan en ``_require_real_urls_in_cloud``.
    database_url: str = Field(default=_DEV_DATABASE_URL)
    dbmate_database_url: str = Field(default="")
    redis_url: str = Field(default=_DEV_REDIS_URL)
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    # Cloud
    gcp_project_id: str = "agrosat-prod"
    gcp_region: str = "europe-west1"
    google_application_credentials: str = ""
    gcs_data_bucket: str = ""
    gcs_artifacts_bucket: str = ""
    gcs_dvc_bucket: str = ""
    pubsub_inference_topic: str = "inference-jobs"
    azure_subscription_id: str = ""
    azure_resource_group: str = "agrosat-rg"
    azure_h100_vm_name: str = "agrosat-h100-prod"
    azure_storage_connection_string: str = ""
    azure_blob_checkpoints_container: str = "agrosat-checkpoints"

    # Earth Engine / CDSE
    gee_service_account_path: str = ""
    gee_project_id: str = ""
    cdse_username: str = ""
    cdse_password: str = ""

    # HuggingFace
    huggingface_token: str = ""
    hf_home: str = ""

    # LLM backends
    llm_variant_default: Literal["gemini", "qwen35"] = "gemini"
    vertex_ai_location: str = "us-central1"
    gemini_model: str = "gemini-3.1-pro"
    vllm_qwen35_url: str = ""
    vllm_api_key: str = ""

    # MLflow / Dagster
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_store: str = ""
    dagster_home: str = ""

    # Auth (Clerk)
    clerk_secret_key: str = ""
    clerk_publishable_key: str = ""

    # Frontend
    frontend_url: str = "http://localhost:3000"
    nuxt_public_api_url: str = "http://localhost:8000"

    # Observability
    prometheus_pushgateway: str = ""
    sentry_dsn: str = ""

    # JWT / security
    jwt_secret: str = _JWT_PLACEHOLDER
    jwt_algorithm: str = "HS256"
    cors_allowed_origins: str = "http://localhost:3000"
    rate_limit_chat_per_min: int = 10
    rate_limit_llm_switch_per_min: int = 5

    # Terraform passthrough (no usadas en backend, declaradas para extra=forbid)
    tf_var_project_id: str = ""
    tf_var_gcp_region: str = ""
    tf_var_azure_location: str = ""
    tf_var_allowed_ssh_cidrs: str = ""

    # DVC
    dvc_remote_name: str = ""
    dvc_remote_url: str = ""

    # Host ports (docker-compose, no usados por backend pero declarados)
    postgres_host_port: int = 5432
    redis_host_port: int = 6379
    api_host_port: int = 8000
    frontend_host_port: int = 3000
    titiler_host_port: int = 8001
    mlflow_host_port: int = 5000
    dagster_host_port: int = 3001
    ollama_host_port: int = 11434

    @property
    def cors_allow_origins(self) -> list[str]:
        """Lista parseada de origenes CORS desde la string CSV de .env.local."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _require_real_urls_in_cloud(self) -> "Settings":
        """Rechaza arrancar con defaults locales si ``env`` es staging/prod.

        Defiende contra el caso en que un deploy real arranque sin ``.env.local``
        valido y termine conectandose a Postgres con credenciales por defecto.
        """
        if self.env != "dev":
            if self.database_url == _DEV_DATABASE_URL:
                raise ValueError(
                    f"DATABASE_URL es obligatorio cuando env={self.env!r} "
                    "(no usar el default de desarrollo en cloud)."
                )
            if self.redis_url == _DEV_REDIS_URL and not self.upstash_redis_rest_url:
                raise ValueError(
                    f"REDIS_URL o UPSTASH_REDIS_REST_URL es obligatorio cuando "
                    f"env={self.env!r} (no usar el default de desarrollo en cloud)."
                )
            if self.jwt_secret == _JWT_PLACEHOLDER:
                raise ValueError(f"JWT_SECRET no puede ser el placeholder en env={self.env!r}.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la instancia singleton de configuración."""
    return Settings()
