"""
Agent factory — single place that knows how to build an Agents-SDK Model
for the configured provider.

Adding a new provider is a one-function change here; the rest of the codebase
(orchestrator, services) builds Agents using the Model returned by `make_model`.

For the `gemini` provider we support both auth schemes:
  - Vertex AI (service-account JSON via GCP_CREDENTIALS_BASE64) — preferred
    in production, matches the project's pre-existing setup.
  - Google AI Studio (GEMINI_API_KEY) — convenient for local development.
The choice is auto-detected from which env var is set.
"""

import base64
import logging
import os
import tempfile
from typing import Optional

from agents import Model, ModelSettings  # type: ignore[import]
from agents.extensions.models.litellm_model import LitellmModel  # type: ignore[import]

from app.core.config import settings

logger = logging.getLogger(__name__)

# Set once per process — guards against re-decoding the base64 credentials
# blob and re-configuring litellm globals on every request.
_vertex_initialised = False


def _ensure_vertex_credentials() -> None:
    """
    Decode GCP_CREDENTIALS_BASE64 to a temp file, point
    GOOGLE_APPLICATION_CREDENTIALS at it, and set LiteLLM's vertex globals.

    Idempotent — the first call wins; later calls are no-ops.
    """
    global _vertex_initialised
    if _vertex_initialised:
        return

    if not settings.GCP_PROJECT_ID:
        raise ValueError(
            "GCP_PROJECT_ID is not configured. "
            "Set GCP_PROJECT_ID (and GCP_CREDENTIALS_BASE64) to use Vertex AI auth, "
            "or set GEMINI_API_KEY to use the Google AI Studio API key."
        )

    if settings.GCP_CREDENTIALS_BASE64:
        creds_bytes = base64.b64decode(settings.GCP_CREDENTIALS_BASE64)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, prefix="gcp_creds_"
        )
        tmp.write(creds_bytes)
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        logger.info("GCP credentials decoded from GCP_CREDENTIALS_BASE64.")
    else:
        logger.info(
            "GCP_CREDENTIALS_BASE64 not set — relying on existing "
            "GOOGLE_APPLICATION_CREDENTIALS or application-default credentials."
        )

    # Tell LiteLLM where to send Vertex requests. These globals are read by
    # litellm.completion() when the model name is prefixed with `vertex_ai/`.
    import litellm  # type: ignore[import]

    litellm.vertex_project = settings.GCP_PROJECT_ID
    litellm.vertex_location = settings.GCP_LOCATION or "us-central1"

    _vertex_initialised = True
    logger.info(
        "Vertex AI configured for LiteLLM (project=%s, location=%s)",
        settings.GCP_PROJECT_ID,
        litellm.vertex_location,
    )


def _gemini_model(model_name: str) -> Model:
    """
    Build a Gemini-backed model, auto-selecting Vertex AI vs Google AI Studio
    based on which credential is configured.
    """
    use_vertex = bool(settings.GCP_CREDENTIALS_BASE64 or settings.GCP_PROJECT_ID)

    if use_vertex:
        _ensure_vertex_credentials()
        # LitellmModel.api_key is unused for vertex_ai — auth flows through
        # GOOGLE_APPLICATION_CREDENTIALS / litellm.vertex_project.
        return LitellmModel(model=f"vertex_ai/{model_name}")

    if settings.GEMINI_API_KEY:
        return LitellmModel(
            model=f"gemini/{model_name}",
            api_key=settings.GEMINI_API_KEY,
        )

    raise ValueError(
        "Gemini provider selected but no credentials configured. "
        "Set GCP_CREDENTIALS_BASE64 + GCP_PROJECT_ID (Vertex AI) "
        "or GEMINI_API_KEY (Google AI Studio)."
    )


def make_model(model_name: Optional[str] = None) -> Model:
    """
    Return an Agents-SDK `Model` instance for the configured LLM_PROVIDER.

    Args:
        model_name: Bare model name (e.g. "gemini-2.5-flash"). When None,
                    falls back to settings.LLM_CHAT_MODEL.
    """
    name = model_name or settings.LLM_CHAT_MODEL
    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        return _gemini_model(name)

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured.")
        return LitellmModel(
            model=name,
            api_key=settings.OPENAI_API_KEY,
        )

    if provider == "litellm":
        # Catch-all: caller passes the fully-qualified LiteLLM model id
        # (e.g. "anthropic/claude-3-5-sonnet"). API key is read from the
        # appropriate provider env var by LiteLLM itself.
        return LitellmModel(model=name)

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}'. "
        "Expected one of: gemini, openai, litellm."
    )


def default_model_settings(temperature: float = 0.7) -> ModelSettings:
    """Centralised place to tweak temperature/top_p across all agents."""
    return ModelSettings(temperature=temperature)
