"""
LLM Gateway — central point for ALL Gemini model interactions via Vertex AI.

Every LLM call in the application goes through this class so that:
- Langfuse observability spans are created automatically
- Model construction is consistent (system_instruction, tools)
- Token usage is captured in one place
- Failures surface with structured logging

Used by:
- AgentOrchestrator (chat with function calling)
- ListingVerificationService (vision + text)
- PricePredictionService (one-shot generate)
"""

import logging
from typing import Any, Dict, List, Optional

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
    Thin orchestration layer between application services and the Vertex AI SDK.

    Usage patterns:

        gateway = get_gateway()

        # --- Chat with function calling (AgentOrchestrator) ---
        trace    = gateway.create_trace("chat-request", session_id=..., input=msg)
        model    = gateway.build_model(settings.GEMINI_CHAT_MODEL, system_prompt, tools)
        chat     = gateway.start_chat(model, history)
        response = await gateway.send_message(chat, user_message, trace)

        # --- One-shot text generation (PricePrediction, etc.) ---
        trace    = gateway.create_trace("price-prediction", input=prompt)
        response = await gateway.generate(prompt, model_name=..., trace=trace)

        # --- Vision + text (ListingVerification) ---
        trace    = gateway.create_trace("listing-verify", input=text)
        response = await gateway.generate_with_images(prompt, images, trace=trace)
    """

    def __init__(self) -> None:
        import base64
        import os
        import tempfile

        import vertexai  # type: ignore[import]

        from app.core.config import settings

        if not settings.GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID is not configured")

        # Decode base64 service account JSON → temp file → set env var
        self._credentials_tmp_path: Optional[str] = None
        if settings.GCP_CREDENTIALS_BASE64:
            credentials_json = base64.b64decode(settings.GCP_CREDENTIALS_BASE64)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, prefix="gcp_creds_"
            )
            tmp.write(credentials_json)
            tmp.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
            self._credentials_tmp_path = tmp.name
            logger.info("GCP credentials loaded from base64 env var.")
        else:
            logger.warning(
                "GCP_CREDENTIALS_BASE64 not set — falling back to "
                "GOOGLE_APPLICATION_CREDENTIALS or application default credentials."
            )

        vertexai.init(
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
        logger.info(
            "Vertex AI initialised (project=%s, location=%s)",
            settings.GCP_PROJECT_ID,
            settings.GCP_LOCATION,
        )

        # Langfuse is optional — gracefully disabled when keys are absent
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
                logger.warning(
                    "langfuse package not installed — tracing disabled. "
                    "Run: uv add langfuse"
                )
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
    # Model / chat construction (used by AgentOrchestrator)
    # ------------------------------------------------------------------

    def build_model(
        self,
        model_name: str,
        system_instruction: str,
        tools: Optional[Any] = None,
    ) -> Any:
        """
        Build a Vertex AI GenerativeModel with a proper system_instruction.
        """
        from vertexai.generative_models import GenerativeModel  # type: ignore[import]

        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "system_instruction": system_instruction,
        }
        if tools is not None:
            kwargs["tools"] = [tools]

        logger.debug("Building model '%s' (tools=%s)", model_name, tools is not None)
        return GenerativeModel(**kwargs)

    def start_chat(
        self,
        model: Any,
        history: Optional[List[Any]] = None,
    ) -> Any:
        """
        Start a stateful chat session, optionally seeding it with prior history.
        """
        return model.start_chat(history=history or [])

    # ------------------------------------------------------------------
    # Instrumented send (chat mode — AgentOrchestrator)
    # ------------------------------------------------------------------

    async def send_message(
        self,
        chat: Any,
        message: Any,
        trace: Any,
        span_name: str = "llm-call",
    ) -> Any:
        """
        Send a message through an active ChatSession, wrapped in a Langfuse span.
        Uses Vertex AI native async (send_message_async).
        """
        generation = trace.generation(
            name=span_name,
            model=getattr(
                getattr(chat, "_model", None),
                "model_name",
                getattr(getattr(chat, "_model", None), "_model_name", "unknown"),
            ),
            input=str(message)[:2000],
        )
        try:
            response = await chat.send_message_async(message)

            usage = self._extract_usage(response)
            output_text = self._extract_text_safe(response)

            generation.end(output=output_text or "(function call)", usage=usage)
            return response

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            logger.error("LLM call failed [%s]: %s", span_name, e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # One-shot generate (text only — PricePrediction, future services)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Any] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        span_name: str = "generate",
    ) -> Any:
        """
        One-shot text generation (no chat session).

        Uses Vertex AI native async (generate_content_async).
        """
        from vertexai.generative_models import (  # type: ignore[import]
            GenerationConfig,
            GenerativeModel,
        )

        from app.core.config import settings

        trace = trace or _NoOpTrace()
        model_name = model_name or settings.GEMINI_CHAT_MODEL

        model_kwargs: Dict[str, Any] = {"model_name": model_name}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        if tools is not None:
            model_kwargs["tools"] = [tools] if not isinstance(tools, list) else tools

        model = GenerativeModel(**model_kwargs)

        gen_kwargs: Dict[str, Any] = {}
        if generation_config:
            gen_kwargs["generation_config"] = GenerationConfig(**generation_config)

        generation = trace.generation(
            name=span_name,
            model=model_name,
            input=prompt[:2000],
        )

        try:
            response = await model.generate_content_async(prompt, **gen_kwargs)

            usage = self._extract_usage(response)
            output_text = self._extract_text_safe(response)

            generation.end(
                output=output_text[:2000] if output_text else "", usage=usage
            )
            return response

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            logger.error("generate failed [%s]: %s", span_name, e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Vision generate (images + text — ListingVerification)
    # ------------------------------------------------------------------

    async def generate_with_images(
        self,
        prompt: str,
        images: List[Any],
        *,
        model_name: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        span_name: str = "vision-generate",
    ) -> Any:
        """
        Multimodal generation: text prompt + PIL Image objects.

        Uses Vertex AI native async (generate_content_async).
        """
        from vertexai.generative_models import (  # type: ignore[import]
            GenerationConfig,
            GenerativeModel,
        )

        from app.core.config import settings

        trace = trace or _NoOpTrace()
        model_name = model_name or settings.GEMINI_VISION_MODEL
        model = GenerativeModel(model_name)

        gen_kwargs: Dict[str, Any] = {}
        if generation_config:
            gen_kwargs["generation_config"] = GenerationConfig(**generation_config)

        content_parts: List[Any] = [prompt] + images

        generation = trace.generation(
            name=span_name,
            model=model_name,
            input=prompt[:2000],
            metadata={"image_count": len(images)},
        )

        try:
            response = await model.generate_content_async(content_parts, **gen_kwargs)

            usage = self._extract_usage(response)
            output_text = self._extract_text_safe(response)

            generation.end(
                output=output_text[:2000] if output_text else "", usage=usage
            )
            return response

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            logger.error("vision generate failed [%s]: %s", span_name, e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
        """Extract token usage from a Vertex AI response."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            return {
                "input": getattr(meta, "prompt_token_count", 0),
                "output": getattr(meta, "candidates_token_count", 0),
                "total": getattr(meta, "total_token_count", 0),
            }
        return None

    @staticmethod
    def _extract_text_safe(response: Any) -> str:
        """Safely extract text from a Vertex AI response (won't raise on function calls)."""
        try:
            return response.text
        except (ValueError, AttributeError):
            return ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Flush any buffered Langfuse events and clean up temp credentials.
        Call this on application shutdown (e.g. FastAPI `lifespan` teardown).
        """
        if self._langfuse_enabled and self._langfuse is not None:
            self._langfuse.flush()
            logger.info("Langfuse events flushed.")

        if self._credentials_tmp_path:
            import os

            try:
                os.unlink(self._credentials_tmp_path)
                logger.info("Temp credentials file cleaned up.")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_gateway_instance: Optional[LLMGateway] = None


def get_gateway() -> LLMGateway:
    """
    Return the module-level LLMGateway singleton.
    All services should use this instead of creating their own instance.
    """
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = LLMGateway()
        logger.info("LLMGateway singleton created.")
    return _gateway_instance
