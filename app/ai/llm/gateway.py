"""
LLM Gateway — central point for ALL Gemini model interactions.

Migrated from the deprecated `vertexai.generative_models` SDK to the unified
`google-genai` SDK (https://googleapis.github.io/python-genai/). The deprecated
SDK is removed on 2026-06-24; this migration is required to keep the service
running past that date.

Every LLM call in the application goes through this class so that:
- Langfuse observability spans are created automatically
- Client construction is consistent (project, location, credentials)
- Token usage is captured in one place
- Failures surface with structured logging

Used by:
- AgentOrchestrator (chat with function calling)
- ListingVerificationService (vision + text)
- PricePredictionService (one-shot generate with tools)
- /api/v1/completion (raw generate)
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from google.api_core.exceptions import ResourceExhausted  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helper for 429 / RESOURCE_EXHAUSTED
# ---------------------------------------------------------------------------

_MAX_RETRIES = 1  # quota retries — fail fast for chat UX
_BASE_DELAY = 3  # seconds

# Per-LLM-call timeout for the streaming path. The agent's outer timeout
# (REQUEST_TIMEOUT_SECONDS in orchestrator) only covers the non-streaming
# path; without this, a stuck Gemini call hangs the SSE stream forever.
# 90s is generous enough for legitimate slowness on round 2 with large
# tool results, short enough to surface real hangs to the user.
_STREAM_TIMEOUT_SECONDS = 90

# Vertex AI explicit context cache (CachedContent) — registers the stable
# system_instruction + tool schemas once, then references the cache from
# every chat request. Cached tokens bill at ~25% of normal input rate and
# eliminate prefix re-encoding latency. The cache is invisible to callers
# beyond the optional `cached_content` argument on start_chat.
_CACHE_TTL_SECONDS = 3600  # refresh hourly
_CACHE_REFRESH_MARGIN = 60  # treat cache as expired this many seconds early
# Vertex requires cached content to exceed a per-model minimum (~1024
# tokens for Flash, ~2048 for Pro). Skip caching below this character
# threshold rather than waste an API call we know will fail.
_MIN_CACHEABLE_CHARS = 3000


async def _retry_on_quota(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Call an async function with bounded retry on RESOURCE_EXHAUSTED (429).

    Conservative retry policy: 1 retry with 3-second backoff. Long retries
    (15-105s) hurt user-perceived latency more than they help; for chat
    workloads it is better to surface a friendly error fast.
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
            delay = _BASE_DELAY
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
    Thin orchestration layer between application services and the google-genai SDK.

    Usage patterns:

        gateway = get_gateway()

        # --- Chat with function calling (AgentOrchestrator) ---
        trace    = gateway.create_trace("chat-request", session_id=..., input=msg)
        chat     = gateway.start_chat(
            model_name=settings.GEMINI_CHAT_MODEL,
            system_instruction=system_prompt,
            tools=tools,
            history=history,
        )
        response = await gateway.send_message(chat, user_message, trace)

        # --- One-shot text generation (PricePrediction, completion) ---
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

        from google import genai  # type: ignore[import]

        from app.core.config import settings

        if not settings.GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID is not configured")

        # Decode base64 service account JSON → temp file → set env var
        # google-genai picks up GOOGLE_APPLICATION_CREDENTIALS automatically.
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

        # Single shared client for both sync and async calls (use .aio.* for async).
        self._client = genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT_ID,
            location=location,
        )
        logger.info(
            "google-genai client initialised (project=%s, location=%s, vertexai=True)",
            settings.GCP_PROJECT_ID,
            location,
        )

        # Explicit CachedContent state. The cache is created lazily on the
        # first request that needs it, then reused across subsequent requests
        # and refreshed transparently when the TTL approaches.
        self._cache_handle: Optional[str] = None
        self._cache_key: Optional[int] = None
        self._cache_expires_at: float = 0.0

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

        Returns (prompt_text, prompt_object). On failure, returns (fallback, None).
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
    # Explicit context cache (Vertex AI CachedContent)
    # ------------------------------------------------------------------

    async def get_or_create_cache(
        self,
        model_name: str,
        system_instruction: str,
        tools: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Return a CachedContent resource name for the given (model, system, tools)
        triple. Creates and caches the resource lazily on first call, refreshes
        before TTL expiry, and recreates on key change (e.g. prompt edit).

        Returns None when caching is unavailable (content below per-model
        minimum, region not supported, API rejection). Callers should fall
        back to the non-cached path.
        """
        # Hash the inputs to detect drift (e.g. Langfuse prompt edited mid-process).
        # `str(tools)` is stable because google-genai Tool objects have deterministic repr.
        cache_key = hash((model_name, system_instruction, str(tools)))
        now = time.time()

        cache_valid = (
            self._cache_handle is not None
            and self._cache_key == cache_key
            and now < self._cache_expires_at - _CACHE_REFRESH_MARGIN
        )
        if cache_valid:
            return self._cache_handle

        # Skip when content is too small to be cacheable on Vertex.
        if len(system_instruction) < _MIN_CACHEABLE_CHARS:
            logger.info(
                "Cache skipped: system_instruction too small (%d chars < %d threshold)",
                len(system_instruction),
                _MIN_CACHEABLE_CHARS,
            )
            return None

        # Create or recreate the cache.
        from google.genai import types  # type: ignore[import]

        try:
            config_kwargs: Dict[str, Any] = {
                "system_instruction": system_instruction,
                "ttl": f"{_CACHE_TTL_SECONDS}s",
            }
            if tools is not None:
                config_kwargs["tools"] = (
                    tools if isinstance(tools, list) else [tools]
                )
            cache = await self._client.aio.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(**config_kwargs),
            )
            self._cache_handle = cache.name
            self._cache_key = cache_key
            self._cache_expires_at = now + _CACHE_TTL_SECONDS
            logger.info(
                "CachedContent created: name=%s ttl=%ds (system=%d chars)",
                cache.name,
                _CACHE_TTL_SECONDS,
                len(system_instruction),
            )
            return cache.name

        except Exception as e:
            # Fail soft: caller falls back to non-cached path.
            logger.warning(
                "CachedContent creation failed (%s) — falling back to "
                "uncached path. This is non-fatal but raises per-request "
                "input token cost.",
                e,
            )
            # Avoid retry storm: pretend we have a temporary cache to hold
            # off retrying for a short window.
            self._cache_expires_at = now + 60
            return None

    # ------------------------------------------------------------------
    # Chat construction (used by AgentOrchestrator)
    # ------------------------------------------------------------------

    def start_chat(
        self,
        model_name: str,
        system_instruction: str,
        tools: Optional[Any] = None,
        history: Optional[List[Any]] = None,
        cached_content: Optional[str] = None,
    ) -> Any:
        """
        Create a stateful AsyncChat session pre-configured with the given
        system instruction, tools, and seeded history.

        When `cached_content` is provided (a CachedContent resource name), the
        system_instruction and tools are loaded from the cache by Vertex; the
        local arguments must still match what was cached.

        Returns an async chat object that supports
        `send_message(message=...)` and `send_message_stream(message=...)`.
        """
        from google.genai import types  # type: ignore[import]

        config_kwargs: Dict[str, Any] = {}
        if cached_content:
            # Cache supplies system + tools; do NOT duplicate them in config.
            config_kwargs["cached_content"] = cached_content
        else:
            config_kwargs["system_instruction"] = system_instruction
            if tools is not None:
                # `tools` may be either a single Tool or list — normalise to list.
                config_kwargs["tools"] = (
                    tools if isinstance(tools, list) else [tools]
                )

        config = types.GenerateContentConfig(**config_kwargs)
        return self._client.aio.chats.create(
            model=model_name,
            history=history or [],
            config=config,
        )

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
        Send a message through an active AsyncChat, wrapped in a Langfuse span.
        """
        gen_kwargs: Dict[str, Any] = {
            "name": span_name,
            "model": getattr(chat, "_model", "unknown"),
            "input": str(message)[:2000],
        }
        if prompt is not None:
            gen_kwargs["prompt"] = prompt
        generation = trace.generation(**gen_kwargs)
        try:
            response = await _retry_on_quota(self._send_message_call, chat, message)

            usage = self._extract_usage(response)
            output_text = self._extract_text_safe(response)

            generation.end(output=output_text or "(function call)", usage=usage)
            return response

        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            logger.error("LLM call failed [%s]: %s", span_name, e, exc_info=True)
            raise

    @staticmethod
    async def _send_message_call(chat: Any, message: Any) -> Any:
        """Bridge for _retry_on_quota — google-genai uses keyword arg `message`."""
        return await chat.send_message(message=message)

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
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming variant of send_message — yields incremental events.

        Event shape (dicts yielded):
            {"type": "text_delta", "delta": str}       — incremental text
            {"type": "final", "text": str,             — emitted once at end
                              "function_calls": list,
                              "response": GenerateContentResponse | None}
        """
        gen_kwargs: Dict[str, Any] = {
            "name": span_name,
            "model": getattr(chat, "_model", "unknown"),
            "input": str(message)[:2000],
        }
        if prompt is not None:
            gen_kwargs["prompt"] = prompt
        generation = trace.generation(**gen_kwargs)

        accumulated: List[str] = []
        function_calls: List[Any] = []
        last_chunk: Any = None

        # Diagnostic counters — to distinguish "Gemini sent 1 big chunk"
        # (model behavior, not a bug) from "pipeline is buffering".
        sdk_chunks_received = 0
        text_deltas_yielded = 0
        stream_start = time.perf_counter()

        try:
            logger.info(
                "Stream [%s] issuing SDK request (timeout=%ds)",
                span_name,
                _STREAM_TIMEOUT_SECONDS,
            )
            # Per-call timeout — without this a stuck Gemini call hangs
            # the SSE stream forever and the FE shows an indefinite spinner.
            async with asyncio.timeout(_STREAM_TIMEOUT_SECONDS):
                stream = await chat.send_message_stream(message=message)
                async for chunk in stream:
                    sdk_chunks_received += 1
                    # Timestamp each chunk arrival. Spread over time → SDK is
                    # streaming. Bunched at end → SDK or transport is buffering.
                    logger.info(
                        "Stream [%s] SDK chunk #%d arrived at +%.0fms",
                        span_name,
                        sdk_chunks_received,
                        (time.perf_counter() - stream_start) * 1000,
                    )
                    last_chunk = chunk
                    try:
                        parts = chunk.candidates[0].content.parts
                    except (AttributeError, IndexError, TypeError):
                        continue

                    if not parts:
                        continue

                    for part in parts:
                        text = getattr(part, "text", None)
                        if text:
                            accumulated.append(text)
                            text_deltas_yielded += 1
                            yield {"type": "text_delta", "delta": text}

                        fc = getattr(part, "function_call", None)
                        if fc is not None and getattr(fc, "name", None):
                            function_calls.append(fc)

            logger.info(
                "Stream [%s]: SDK chunks=%d, text deltas yielded=%d, total chars=%d",
                span_name,
                sdk_chunks_received,
                text_deltas_yielded,
                sum(len(s) for s in accumulated),
            )

            full_text = "".join(accumulated)
            yield {
                "type": "final",
                "text": full_text,
                "function_calls": function_calls,
                "response": last_chunk,
            }

            usage = self._extract_usage(last_chunk) if last_chunk is not None else None
            generation.end(output=full_text or "(function call)", usage=usage)

        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - stream_start) * 1000
            logger.error(
                "LLM stream [%s] timed out after %dms (cap=%ds, chunks received=%d)",
                span_name,
                elapsed,
                _STREAM_TIMEOUT_SECONDS,
                sdk_chunks_received,
            )
            generation.end(
                level="ERROR",
                status_message=f"LLM stream timeout after {_STREAM_TIMEOUT_SECONDS}s",
            )
            raise
        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            logger.error("LLM stream failed [%s]: %s", span_name, e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # One-shot generate (text only — PricePrediction, completion)
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
        """
        from google.genai import types  # type: ignore[import]

        from app.core.config import settings

        active_trace: Any = trace if trace is not None else _NoOpTrace()
        model_name = model_name or settings.GEMINI_CHAT_MODEL

        config_kwargs: Dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools is not None:
            config_kwargs["tools"] = tools if isinstance(tools, list) else [tools]
        if generation_config:
            # Map old keys → new SDK names where they differ.
            mapped = dict(generation_config)
            if "max_output_tokens" in mapped:
                config_kwargs["max_output_tokens"] = mapped.pop("max_output_tokens")
            if "temperature" in mapped:
                config_kwargs["temperature"] = mapped.pop("temperature")
            if "top_p" in mapped:
                config_kwargs["top_p"] = mapped.pop("top_p")
            if "top_k" in mapped:
                config_kwargs["top_k"] = mapped.pop("top_k")
            # Pass through any remaining keys; new SDK will reject unknowns
            config_kwargs.update(mapped)

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        generation = active_trace.generation(
            name=span_name,
            model=model_name,
            input=prompt[:2000],
        )

        try:
            response = await _retry_on_quota(
                self._generate_content_call, model_name, prompt, config
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

    async def _generate_content_call(
        self, model_name: str, contents: Any, config: Any
    ) -> Any:
        """Bridge for _retry_on_quota."""
        return await self._client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

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
        Multimodal generation: text prompt + PIL Image objects (or already-Part inputs).

        PIL images are encoded as PNG bytes and wrapped in `types.Part.from_bytes`.
        Items already shaped as Part objects pass through unchanged.
        """
        import io

        from google.genai import types  # type: ignore[import]

        from app.core.config import settings

        active_trace: Any = trace if trace is not None else _NoOpTrace()
        model_name = model_name or settings.GEMINI_VISION_MODEL

        # Build content list: prompt as plain string, images converted to Parts.
        contents: List[Any] = [prompt]
        for img in images:
            # Already a genai Part — pass through.
            if hasattr(img, "inline_data") or isinstance(img, types.Part):
                contents.append(img)
                continue
            # PIL image — encode to PNG bytes.
            if hasattr(img, "save"):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                contents.append(
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
                )
                continue
            # Raw bytes — assume PNG.
            if isinstance(img, (bytes, bytearray)):
                contents.append(
                    types.Part.from_bytes(data=bytes(img), mime_type="image/png")
                )
                continue
            logger.warning("generate_with_images: unsupported image type %r", type(img))

        config_kwargs: Dict[str, Any] = {}
        if generation_config:
            mapped = dict(generation_config)
            for k in ("max_output_tokens", "temperature", "top_p", "top_k"):
                if k in mapped:
                    config_kwargs[k] = mapped.pop(k)
            config_kwargs.update(mapped)

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        generation = active_trace.generation(
            name=span_name,
            model=model_name,
            input=prompt[:2000],
            metadata={"image_count": len(images)},
        )

        try:
            response = await _retry_on_quota(
                self._generate_content_call, model_name, contents, config
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
        """Extract token usage from a google-genai response."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            return {
                "input": getattr(meta, "prompt_token_count", 0) or 0,
                "output": getattr(meta, "candidates_token_count", 0) or 0,
                "total": getattr(meta, "total_token_count", 0) or 0,
            }
        return None

    @staticmethod
    def _extract_text_safe(response: Any) -> str:
        """
        Safely extract text from a google-genai response.

        `.text` may raise or return None when the response contains only
        function calls; iterate parts as a fallback.
        """
        try:
            text = response.text
            if text:
                return text
        except (ValueError, AttributeError):
            pass

        try:
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    return part.text
        except (AttributeError, IndexError, TypeError):
            pass

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
