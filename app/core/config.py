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

    # LLM Provider Configuration
    # All agents are built via the OpenAI Agents SDK. The provider is selected
    # at startup; switching providers is a single env-var change.
    #   "gemini"  → LitellmModel; auth auto-detected:
    #               - GCP_CREDENTIALS_BASE64 set → Vertex AI (service account)
    #               - else GEMINI_API_KEY set    → Google AI Studio (api key)
    #   "openai"  → LitellmModel via OPENAI_API_KEY
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Google Cloud / Vertex AI credentials — only needed when using Vertex AI
    # for Gemini (service-account auth). Leave blank to use GEMINI_API_KEY.
    GCP_CREDENTIALS_BASE64: str = ""  # Base64-encoded service account JSON
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = "us-central1"

    # Per-task model identifiers — bare model names (e.g. "gemini-2.5-flash",
    # "gpt-4o-mini"). The factory prepends the provider prefix when needed.
    LLM_CHAT_MODEL: str = "gemini-2.5-flash"  # Chat / agent loop
    LLM_VISION_MODEL: str = "gemini-2.5-flash"  # Listing verification (text+image)
    LLM_PRICE_MODEL: str = "gemini-2.5-flash"  # Price prediction
    # Low temperature for the chat/agent loop → deterministic tool-calling and
    # fewer hallucinated params. Configurable so it can be tuned without a deploy.
    LLM_CHAT_TEMPERATURE: float = 0.2

    # Google Maps — for get_nearby_places (POI search / distance). When unset,
    # the tool falls back to haversine distance between two given coordinates.
    GOOGLE_MAPS_API_KEY: str = ""

    # SmartRent Backend Configuration
    SMARTRENT_BACKEND_URL: str = "http://localhost:8080"
    SMARTRENT_AI_URL: str = "http://localhost:8000"
    # Public frontend site URL — used to build canonical listing links the LLM
    # can share (e.g. https://thuenhatro.net/listing-detail/{id}). The model must
    # echo the `url` field from tool payloads, never fabricate a domain/path.
    FRONTEND_URL: str = "https://thuenhatro.net"
    # Shared secret the Spring backend sends as X-Internal-Api-Key when proxying
    # chat. Empty = open (local dev); set in deployed envs to reject public callers.
    INTERNAL_AI_API_KEY: str = ""

    # Chat Configuration
    MAX_LISTINGS_RETURN: int = 5  # Maximum number of listings to return to user

    # Concurrency cap for CPU-bound work (TF-IDF, pHash decode) on the single
    # uvicorn worker. Keeps a request burst from pinning the shared VM's CPU.
    CPU_BOUND_CONCURRENCY: int = 2

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
