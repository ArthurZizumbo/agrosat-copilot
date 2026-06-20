"""Central application configuration.

Loads environment variables via Pydantic Settings. Never read ``os.environ``
directly from routers or services — always via ``get_settings()``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_DATABASE_URL = "postgresql+asyncpg://agrosat:agrosat@localhost:5432/agrosat"
# Application-role DSN (role ``agrosat_app``, NOBYPASSRLS) used by the backend
# pool so RLS policies actually enforce (US-051). Separate from the superuser
# ``agrosat`` migration role, which bypasses RLS.
_DEV_APP_DATABASE_URL = "postgresql+asyncpg://agrosat_app:agrosat_app@localhost:55432/agrosat"
_DEV_REDIS_URL = "redis://localhost:6379/0"
# Placeholder rejected by the validator if env != dev.
_JWT_PLACEHOLDER = "change-me-in-prod"


class Settings(BaseSettings):
    """Typed configuration of the AgroSatCopilot backend.

    ``extra="forbid"`` detects typos in ``.env.local`` (variable defined but not
    declared here) and aborts startup. Add any new variable from
    ``.env.example`` that the backend must read.
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

    # Connections — local docker-compose defaults. In staging/prod they are
    # mandatory and validated in ``_require_real_urls_in_cloud``.
    database_url: str = Field(default=_DEV_DATABASE_URL)
    # DSN of the non-superuser application role ``agrosat_app`` (NOBYPASSRLS):
    # the backend pool connects with this so the multi-tenant RLS policies
    # enforce (US-051). The superuser ``agrosat`` (``database_url`` /
    # ``dbmate_database_url``) is the migration role and bypasses RLS.
    app_database_url: str = Field(default=_DEV_APP_DATABASE_URL)
    dbmate_database_url: str = Field(default="")
    redis_url: str = Field(default=_DEV_REDIS_URL)
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    db_pass: str = ""

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
    # H100 VM access via Cloudflare tunnel (declared for extra=forbid; the tunnel
    # is operated outside the backend).
    azure_tunnel_cloudflare_token: str = ""
    cf_access_client_id: str = ""
    cf_access_client_secret: str = ""

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
    # Ollama OpenAI-compatible endpoint for the local Gemma variant (US-049).
    ollama_base_url: str = ""
    # Gemini / google-genai credentials (read by the SDK via the environment;
    # declared here so ``extra="forbid"`` accepts them in ``.env.local``).
    gemini_api_key: str = ""
    google_genai_use_vertexai: str = ""
    google_cloud_project: str = ""
    google_cloud_location: str = ""
    agrosat_llm_provider: str = ""

    # Spatial-RAG lite (US-046). Feature flag gating the deferred
    # ``retrieve_context`` tool. Default off: with it disabled the reasoner runs
    # ungrounded and the agent loop never touches the ``rag_documents`` corpus
    # (AC-5, AC-10). Set ``RAG_ENABLED=true`` in ``.env.local`` to opt in.
    rag_enabled: bool = False

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

    # Terraform passthrough (not used in backend, declared for extra=forbid)
    tf_var_project_id: str = ""
    tf_var_gcp_region: str = ""
    tf_var_azure_location: str = ""
    tf_var_allowed_ssh_cidrs: str = ""

    # DVC
    dvc_remote_name: str = ""
    dvc_remote_url: str = ""

    # Host ports (docker-compose, not used by backend but declared)
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
        """Parsed list of CORS origins from the CSV string in .env.local."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _require_real_urls_in_cloud(self) -> "Settings":
        """Refuse to start with local defaults if ``env`` is staging/prod.

        Defends against the case where a real deploy starts without a valid
        ``.env.local`` and ends up connecting to Postgres with default
        credentials.
        """
        if self.env != "dev":
            if self.database_url == _DEV_DATABASE_URL:
                raise ValueError(
                    f"DATABASE_URL is required when env={self.env!r} "
                    "(do not use the development default in cloud)."
                )
            if self.app_database_url == _DEV_APP_DATABASE_URL:
                raise ValueError(
                    f"APP_DATABASE_URL is required when env={self.env!r} "
                    "(do not ship the dev agrosat_app password to cloud)."
                )
            if self.redis_url == _DEV_REDIS_URL and not self.upstash_redis_rest_url:
                raise ValueError(
                    f"REDIS_URL or UPSTASH_REDIS_REST_URL is required when "
                    f"env={self.env!r} (do not use the development default in cloud)."
                )
            if self.jwt_secret == _JWT_PLACEHOLDER:
                raise ValueError(f"JWT_SECRET cannot be the placeholder in env={self.env!r}.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton configuration instance."""
    return Settings()
