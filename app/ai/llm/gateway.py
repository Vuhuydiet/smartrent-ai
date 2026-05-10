"""
LLM Gateway — Langfuse trace and prompt-management helpers.

After the migration to the OpenAI Agents SDK, all LLM calls go through
`agents.Runner` and the `LitellmModel` built by `agent_factory`. This module
no longer wraps any provider SDK directly; it only exposes the observability
surface (tracing + Langfuse prompt fetch) that orchestrator and services
need.

Used by:
- AgentOrchestrator (chat with function calling)
- ListingVerificationService (vision + text)
- PricePredictionService (one-shot generate with tools)
- /api/v1/completion (raw generate)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# No-op stubs — used when Langfuse is not configured so callers never need
# to branch on `if trace is not None`.
# ---------------------------------------------------------------------------


class _NoOpSpan:
    def end(self, **kwargs: Any) -> None:
        pass


class _NoOpTrace:
    def generation(self, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def span(self, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class LLMGateway:
    """
    Thin observability layer between application services and Langfuse.

    Usage:

        gateway = get_gateway()
        trace   = gateway.create_trace("chat-request", session_id=..., input=msg)
        span    = trace.generation(name="agent-run", model=..., input=...)
        ...
        span.end(output=...)
    """

    def __init__(self) -> None:
        from app.core.config import settings

        self._langfuse: Optional[Any] = None
        self._langfuse_enabled = False

        if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
            try:
                from langfuse import Langfuse  # type: ignore[import]

                self._langfuse = Langfuse(
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                self._langfuse_enabled = True
                logger.info(
                    "Langfuse tracing enabled (host: %s)", settings.LANGFUSE_HOST
                )
            except ImportError:
                logger.warning("langfuse package not installed — tracing disabled.")
        else:
            logger.info(
                "Langfuse keys not set — tracing disabled. "
                "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to enable."
            )

    # ------------------------------------------------------------------
    # Trace / span factory
    # ------------------------------------------------------------------

    def create_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a top-level Langfuse trace for a single request.

        Returns a real Langfuse trace when configured, or a _NoOpTrace so
        callers never need to guard against None.
        """
        if self._langfuse_enabled and self._langfuse is not None:
            return self._langfuse.trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                input=input,
                metadata=metadata or {},
            )
        return _NoOpTrace()

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def get_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        fallback: Optional[str] = None,
        cache_ttl_seconds: int = 300,
    ) -> tuple[Optional[str], Optional[Any]]:
        """
        Fetch a text prompt from Langfuse Prompt Management.

        Returns:
            Tuple of (prompt_text, prompt_object). On failure or when Langfuse
            is disabled, returns (fallback, None).
        """
        if not self._langfuse_enabled or self._langfuse is None:
            logger.debug("Langfuse disabled — using fallback prompt for '%s'", name)
            return fallback, None

        try:
            kwargs: Dict[str, Any] = {"cache_ttl_seconds": cache_ttl_seconds}
            if label:
                kwargs["label"] = label
            prompt_obj = self._langfuse.get_prompt(name, **kwargs)
            return prompt_obj.prompt, prompt_obj  # type: ignore[union-attr]
        except Exception as e:
            logger.warning(
                "Failed to fetch prompt '%s' from Langfuse: %s — using fallback",
                name,
                e,
            )
            return fallback, None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush any buffered Langfuse events. Call on application shutdown."""
        if self._langfuse_enabled and self._langfuse is not None:
            self._langfuse.flush()
            logger.info("Langfuse events flushed.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_gateway_instance: Optional[LLMGateway] = None


def get_gateway() -> LLMGateway:
    """Return the module-level LLMGateway singleton."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = LLMGateway()
        logger.info("LLMGateway singleton created.")
    return _gateway_instance
