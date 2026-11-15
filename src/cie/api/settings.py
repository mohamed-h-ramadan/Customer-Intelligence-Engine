"""API configuration settings and environment variable bindings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """FastAPI service settings configurable via environment variables."""

    api_prefix: str = "/v1"
    api_key: str | None = None
    max_batch_size: int = 1000
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    model_config = SettingsConfigDict(
        env_prefix="CIE_",
        env_file=".env",
        extra="ignore",
    )
