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

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from google.api_core.exceptions import ResourceExhausted  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helper for 429 / RESOURCE_EXHAUSTED
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BASE_DELAY = 15  # seconds — generous because free-tier quota is ~5 RPM


async def _retry_on_quota(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Call an async function with exponential backoff on RESOURCE_EXHAUSTED (429).
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await fn(*args, **kwargs)
        except ResourceExhausted:
            if attempt == _MAX_RETRIES:
                logger.error(
                    "Quota exhausted after %d retries — giving up.", _MAX_RETRIES
                )
                raise
            delay = _BASE_DELAY * (2**attempt)
            logger.warning(
                "429 RESOURCE_EXHAUSTED — retry %d/%d in %ds",
                attempt + 1,
                _MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)


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

        location = settings.GCP_LOCATION or "us-central1"

        # Force the API endpoint to match the configured location
        # This prevents requests from being routed to a wrong region
        api_endpoint = f"{location}-aiplatform.googleapis.com"

        vertexai.init(
            project=settings.GCP_PROJECT_ID,
            location=location,
            api_endpoint=api_endpoint,
        )
        logger.info(
            "Vertex AI initialised (project=%s, location=%s, endpoint=%s)",
            settings.GCP_PROJECT_ID,
            location,
            api_endpoint,
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

        Args:
            name: Prompt name (e.g. "smartrent-chat-system").
            label: Optional label like "production" or "latest".
            fallback: Returned when Langfuse is disabled or the fetch fails.
            cache_ttl_seconds: Client-side cache TTL (default 5 min).

        Returns:
            Tuple of (prompt_text, prompt_object).
            prompt_object can be linked to a Langfuse generation for version tracking.
            On failure, returns (fallback, None).
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

        model = GenerativeModel(**kwargs)
        logger.debug(
            "Building model '%s' (tools=%s, location=%s)",
            model_name,
            tools is not None,
            getattr(model, "_location", "unknown"),
        )
        return model

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
        prompt: Optional[Any] = None,
    ) -> Any:
        """
        Send a message through an active ChatSession, wrapped in a Langfuse span.
        Uses Vertex AI native async (send_message_async).

        Args:
            prompt: Optional Langfuse prompt object for version tracking.
        """
        gen_kwargs: Dict[str, Any] = {
            "name": span_name,
            "model": getattr(
                getattr(chat, "_model", None),
                "model_name",
                getattr(getattr(chat, "_model", None), "_model_name", "unknown"),
            ),
            "input": str(message)[:2000],
        }
        if prompt is not None:
            gen_kwargs["prompt"] = prompt
        generation = trace.generation(**gen_kwargs)
        try:
            response = await _retry_on_quota(chat.send_message_async, message)

            usage = self._extract_usage(response)
            output_text = self._extract_text_safe(response)

            generation.end(output=output_text or "(function call)", usage=usage)
            return response

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            logger.error("LLM call failed [%s]: %s", span_name, e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Instrumented streaming send (chat mode — AgentOrchestrator)
    # ------------------------------------------------------------------

    async def send_message_stream(
        self,
        chat: Any,
        message: Any,
        trace: Any,
        span_name: str = "llm-stream",
        prompt: Optional[Any] = None,
    ):
        """
        Streaming variant of send_message — yields incremental events as Vertex AI
        produces them.

        Event shape (dicts yielded):
            {"type": "text_delta", "delta": str}       — incremental text
            {"type": "final", "text": str,             — emitted once at end
                              "function_calls": list,
                              "response": GenerationResponse | None}

        The Langfuse generation span is closed once the stream completes with
        the accumulated text and token usage from the last chunk.
        """
        gen_kwargs: Dict[str, Any] = {
            "name": span_name,
            "model": getattr(
                getattr(chat, "_model", None),
                "model_name",
                getattr(getattr(chat, "_model", None), "_model_name", "unknown"),
            ),
            "input": str(message)[:2000],
        }
        if prompt is not None:
            gen_kwargs["prompt"] = prompt
        generation = trace.generation(**gen_kwargs)

        accumulated: List[str] = []
        function_calls: List[Any] = []
        last_chunk: Any = None

        try:
            stream = await _retry_on_quota(
                chat.send_message_async, message, stream=True
            )
            async for chunk in stream:
                last_chunk = chunk
                try:
                    parts = chunk.candidates[0].content.parts
                except (AttributeError, IndexError):
                    continue

                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        accumulated.append(text)
                        yield {"type": "text_delta", "delta": text}

                    fc = getattr(part, "function_call", None)
                    if fc is not None and getattr(fc, "name", None):
                        function_calls.append(fc)

            full_text = "".join(accumulated)
            yield {
                "type": "final",
                "text": full_text,
                "function_calls": function_calls,
                "response": last_chunk,
            }

            usage = self._extract_usage(last_chunk) if last_chunk is not None else None
            generation.end(output=full_text or "(function call)", usage=usage)

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            logger.error("LLM stream failed [%s]: %s", span_name, e, exc_info=True)
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

        active_trace: Any = trace if trace is not None else _NoOpTrace()
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

        generation = active_trace.generation(
            name=span_name,
            model=model_name,
            input=prompt[:2000],
        )

        try:
            response = await _retry_on_quota(
                model.generate_content_async, prompt, **gen_kwargs
            )

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
        from io import BytesIO

        from vertexai.generative_models import GenerationConfig, GenerativeModel
        from vertexai.generative_models import (
            Image as VertexImage,  # type: ignore[import]
        )
        from vertexai.generative_models import Part

        from app.core.config import settings

        active_trace: Any = trace if trace is not None else _NoOpTrace()
        model_name = model_name or settings.GEMINI_VISION_MODEL
        model = GenerativeModel(model_name)

        gen_kwargs: Dict[str, Any] = {}
        if generation_config:
            gen_kwargs["generation_config"] = GenerationConfig(**generation_config)

        # Convert PIL Image objects → Vertex AI Part objects
        def _pil_to_part(img: Any) -> Any:
            try:
                from PIL import Image as PILImage

                if isinstance(img, PILImage.Image):
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buf = BytesIO()
                    img.save(buf, format="JPEG")
                    return Part.from_image(VertexImage.from_bytes(buf.getvalue()))
            except Exception as conv_err:
                logger.warning("Could not convert image to Vertex Part: %s", conv_err)
            return img  # fallback: pass as-is and let Vertex SDK handle it

        image_parts = [_pil_to_part(img) for img in images]
        content_parts: List[Any] = [prompt] + image_parts

        generation = active_trace.generation(
            name=span_name,
            model=model_name,
            input=prompt[:2000],
            metadata={"image_count": len(images)},
        )

        try:
            response = await _retry_on_quota(
                model.generate_content_async, content_parts, **gen_kwargs
            )

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
