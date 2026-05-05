# LLM Integration

All Gemini calls in SmartRent AI flow through a single component: `LLMGateway`.
Every service (chat, price prediction, listing verification, raw completion) is
just a thin caller on top of it.

## Why a single gateway

- **One place** to configure Vertex AI (`vertexai.init(project, location, api_endpoint)`)
- **One place** to instrument calls — Langfuse traces, spans, token usage
- **One place** to handle quota failures — `_retry_on_quota` wraps every call
  with exponential backoff on `ResourceExhausted` (429)
- **One place** to fetch prompts from Langfuse Prompt Management

## Module layout

```
app/
├── ai/
│   └── llm/
│       ├── gateway.py                 # LLMGateway — the only LLM entry point
│       └── gemini_listing_helper.py   # Vision + text wrapper for verification
├── agent/
│   ├── orchestrator.py                # Chat agent loop (uses gateway)
│   ├── tools/                         # Tool implementations for function calling
│   └── rag/                           # Static + dynamic RAG context
├── service/
│   ├── chat_service.py                # Thin HTTP layer → AgentOrchestrator
│   ├── price_prediction_service.py    # Function-calling price estimator
│   └── listing_verification_service.py
└── api/v1/
    ├── chat.py                        # POST /api/v1/chat
    ├── completion.py                  # POST /api/v1/completion/
    ├── price_suggestion.py            # POST /api/v1/price-suggestion/...
    └── listing_verification.py        # POST /ai/listing-verification
```

## LLMGateway — public surface

Import via `from app.ai.llm.gateway import get_gateway`. The singleton is
eagerly constructed in `app.main:lifespan` at startup so Vertex AI is
initialised before any request is served.

### Trace / span factory

```python
trace = gateway.create_trace(
    name="chat-request",
    session_id=session_id,
    input={"message": user_message},
    metadata={"model": settings.GEMINI_CHAT_MODEL},
)
```

When Langfuse keys are not configured this returns a `_NoOpTrace` so callers
never need to guard against `None`.

### Prompt management

```python
base_prompt, prompt_obj = gateway.get_prompt(
    "smartrent-chat-system",
    label="production",
    fallback=_SYSTEM_BASE,           # local string used when Langfuse is down
    cache_ttl_seconds=300,
)
```

Pass `prompt_obj` to `send_message(..., prompt=prompt_obj)` so the generation
is linked to that prompt version in Langfuse.

### Chat mode (function calling)

```python
model = gateway.build_model(
    model_name=settings.GEMINI_CHAT_MODEL,
    system_instruction=system_instruction,
    tools=tool_registry.get_tool(),
)
chat = gateway.start_chat(model, history)
response = await gateway.send_message(chat, user_message, trace, span_name="llm-initial")
```

### One-shot text generation

```python
response = await gateway.generate(
    prompt="...",
    model_name=settings.GEMINI_CHAT_MODEL,
    system_instruction=optional_system_prompt,
    generation_config={"temperature": 0.1, "max_output_tokens": 4096},
    trace=trace,
    span_name="completion-generate",
)
```

### Vision (images + text)

```python
response = await gateway.generate_with_images(
    prompt=full_prompt,
    images=pil_image_list,
    generation_config={"temperature": 0.1, "max_output_tokens": 8192},
    trace=trace,
)
```

### Safe text extraction

`response.text` raises `ValueError` when the response contains function calls
or no text parts (safety filter, empty candidate). Always guard:

```python
try:
    text = response.text
except (ValueError, AttributeError):
    text = ""
    for part in response.candidates[0].content.parts:
        if getattr(part, "text", None):
            text = part.text
            break
```

## Request flow per endpoint

### `POST /api/v1/chat` — agentic chat

`ChatService` → `AgentOrchestrator.run()`:

1. Create Langfuse trace (`chat-request`, seeded with `session_id`)
2. RAG retrieval — static province/amenity prefix + dynamic context
3. Fetch `smartrent-chat-system` prompt from Langfuse (with local fallback)
4. Build model with `system_instruction` + registered tools
5. Agent loop (max 5 rounds): send message → inspect function calls →
   execute via `ToolRegistry` → feed tool responses back
6. Extract final text, build listings payload from raw search results
7. Return `AgentResult`

### `POST /api/v1/price-suggestion/...`

`PricePredictionService` uses `gateway.build_model()` with a dedicated
`search_listings` `FunctionDeclaration`. Handles up to 5 function-call rounds,
then parses the final JSON. Falls back to a rule-based market estimate if the
LLM call fails.

### `POST /ai/listing-verification`

`ListingVerificationService` → `GeminiListingVerificationHelper` →
`gateway.generate_with_images()` (or `generate()` for text-only).
Images are downloaded **concurrently** via `httpx.AsyncClient` and
PIL decode/resize is offloaded to `asyncio.to_thread`.

### `POST /api/v1/completion/`

Thin endpoint for backend-side description generation. Calls
`gateway.generate()` directly — no tools, no chat history.

## Configuration

All settings live in `app/core/config.py` and are loaded from `.env`:

| Variable | Purpose |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID — **required**, boot fails without it |
| `GCP_CREDENTIALS_BASE64` | Base64-encoded service account JSON (written to a temp file at startup) |
| `GCP_LOCATION` | Vertex AI region (default `us-central1`) |
| `GEMINI_CHAT_MODEL` | Model for chat + completion (default `gemini-2.5-flash`) |
| `GEMINI_VISION_MODEL` | Model for listing verification (default `gemini-2.5-flash`) |
| `GEMINI_PRICE_MODEL` | Model for price prediction (default `gemini-2.5-flash`) |
| `LANGFUSE_SECRET_KEY` | Optional — enables tracing when set with public key |
| `LANGFUSE_PUBLIC_KEY` | Optional — enables tracing when set with secret key |
| `LANGFUSE_HOST` | Default `https://cloud.langfuse.com` |

`GOOGLE_APPLICATION_CREDENTIALS` is set automatically from
`GCP_CREDENTIALS_BASE64` during gateway initialisation — you should **not**
set it manually in production.

## Lifecycle

`app.main:lifespan` handles both ends:

- **Startup:** `get_gateway()` is called eagerly so `vertexai.init()` runs
  before the first request. Missing `GCP_PROJECT_ID` or bad credentials
  surface here, not on the first HTTP call.
- **Shutdown:** `_gateway_instance.flush()` drains buffered Langfuse events
  so no traces are lost on container stop.

## Adding a new LLM-powered feature

1. Add (or reuse) a tool in `app/agent/tools/` if function calling is needed.
2. In your service, call `get_gateway()` and pick the method that matches:
   - function-calling conversation → `build_model` + `start_chat` + `send_message`
   - single prompt → `generate`
   - prompt + images → `generate_with_images`
3. Always pass a `trace` so the call shows up in Langfuse.
4. Use the safe text-extraction snippet above — never touch `response.text`
   unguarded.
