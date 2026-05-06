from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VERSION: str = "1.0.0"

    # Database
    MYSQL_SERVER: str = "localhost"
    MYSQL_USER: str = "smartrent"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DB: str = "smartrent_ai"
    MYSQL_PORT: int = 3306

    @computed_field  # type: ignore[misc]
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@"
            f"{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
        )

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Google Cloud / Vertex AI Configuration
    GCP_CREDENTIALS_BASE64: str = ""  # Base64-encoded service account JSON
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = "global"
    GEMINI_CHAT_MODEL: str = "gemini-2.5-flash"  # Model used for chat responses
    GEMINI_VISION_MODEL: str = (
        "gemini-2.5-flash"  # Model used for listing verification (vision + text)
    )
    GEMINI_PRICE_MODEL: str = (
        "gemini-2.5-flash"  # Model used for price prediction (function calling)
    )

    # SmartRent Backend Configuration
    SMARTRENT_BACKEND_URL: str = "http://localhost:8080"
    SMARTRENT_AI_URL: str = "http://localhost:8000"
    INTERNAL_AI_API_KEY: str = (
        "your_secret_internal_key_here"  # Override in production via env var
    )

    # Chat Configuration
    MAX_LISTINGS_RETURN: int = 5  # Maximum number of listings to return to user

    # Langfuse Observability
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # CORS — comma-separated list of allowed origins.
    # Dev default covers Next.js (3000), Vite (5173), and the Spring Boot
    # backend (8080) which proxies the chat endpoint. In production, set
    # this env var explicitly to your FE origin(s), e.g.
    #   CORS_ALLOWED_ORIGINS=https://app.smartrent.com,https://smartrent.com
    # Use "*" for wildcard (will force allow_credentials=False per spec).
    CORS_ALLOWED_ORIGINS: str = (
        "http://localhost:3000,http://localhost:5173,http://localhost:8080"
    )

    @computed_field  # type: ignore[misc]
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ALLOWED_ORIGINS env var into a list of origins."""
        return [o.strip() for o in self.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }


settings = Settings()
