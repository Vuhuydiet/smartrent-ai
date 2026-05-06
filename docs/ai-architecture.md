# SmartRent AI — Full AI System Architecture

This document gives a complete picture of every AI surface in the
`smartrent-ai` Python service: what each feature does, what AI paradigm
it uses, what models it calls, and how the surfaces share infrastructure.

For a focused look at the LLM Provider abstraction (current vs target
state), see [architecture.md](architecture.md).
For chat-specific details, see [chatbot-integration.md](chatbot-integration.md).
For frontend chat integration, see [chat-streaming-frontend.md](chat-streaming-frontend.md).

---

## 1. The six AI surfaces, at a glance

| # | Feature | Paradigm | Model(s) | HTTP endpoint | Primary files |
|---|---|---|---|---|---|
| 1 | **Chatbot / Agent** | LLM agent + tools + RAG | Gemini 2.5 Flash | `POST /api/v1/chat`, `POST /api/v1/chat/stream` | [`app/agent/orchestrator.py`](../app/agent/orchestrator.py), [`app/agent/tools/`](../app/agent/tools/), [`app/agent/rag/`](../app/agent/rag/) |
| 2 | **Listing Verification** | Multimodal LLM (vision + text) | Gemini 2.5 Flash | `POST /ai/verify-listing` | [`app/service/listing_verification_service.py`](../app/service/listing_verification_service.py), [`app/ai/llm/gemini_listing_helper.py`](../app/ai/llm/gemini_listing_helper.py) |
| 3 | **Price Prediction** | **Hybrid**: classical ML + LLM with tools | XGBoost (`TwoStageUncertaintyModel`) + Gemini 2.5 Flash | `POST /api/v1/price-suggestion/get-price-suggestion` | [`app/service/price_prediction_service.py`](../app/service/price_prediction_service.py), [`app/ai/house_pricing/price_predictor.py`](../app/ai/house_pricing/price_predictor.py) |
| 4 | **Recommendation** | Classical ML (CBF + CF + boosts) | scikit-learn (cosine similarity, MinMaxScaler) | `POST /api/v1/recommendations/similar`, `POST /api/v1/recommendations/personalized` | [`app/service/recommendation_service.py`](../app/service/recommendation_service.py) |
| 5 | **Completion (raw)** | Single-shot LLM | Gemini 2.5 Flash | `POST /api/v1/completion/` | [`app/api/v1/completion.py`](../app/api/v1/completion.py) |
| 6 | **MCP Server** | External tool exposure (Model Context Protocol) | n/a (transport layer) | stdio (separate process) | [`app/ai/mcp/backend_server.py`](../app/ai/mcp/backend_server.py) |

Three of the six are LLM-driven (1, 2, 5). One is classical ML (4).
One is a hybrid of both (3). The last (6) is an integration surface for
external AI clients.

---

## 2. System map

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          External Consumers                                      │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────┐   │
│  │ Web frontend │  │ Spring Boot      │  │ Spring Boot     │  │ MCP clients  │   │
│  │  (Next.js)   │  │  (proxy + auth)  │  │  (description   │  │ (Claude      │   │
│  │              │  │                  │  │   gen, etc.)    │  │  Desktop)    │   │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬────────┘  └──────┬───────┘   │
└─────────┼───────────────────┼─────────────────────┼──────────────────┼───────────┘
          │ /api/v1/chat*     │ /api/v1/...         │ /api/v1/         │ stdio
          │                   │                     │  completion/     │ (MCP)
          ▼                   ▼                     ▼                  ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       FastAPI service (smartrent-ai, :8000)                      │
│                                                                                  │
│   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐   │
│   │ Chatbot              │  │ Listing Verification │  │ Price Prediction     │   │
│   │ /api/v1/chat[/stream]│  │ /ai/verify-listing   │  │ /api/v1/price-       │   │
│   │ ─ orchestrator       │  │ ─ vision + text      │  │   suggestion         │   │
│   │ ─ 7 tools            │  │ ─ multimodal Gemini  │  │ ─ XGBoost (primary)  │   │
│   │ ─ RAG retriever      │  │                      │  │ ─ Gemini agent (fb)  │   │
│   └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘   │
│              │                         │                         │               │
│   ┌──────────────────────┐  ┌──────────────────────┐                             │
│   │ Recommendation       │  │ Completion           │                             │
│   │ /api/v1/             │  │ /api/v1/completion/  │                             │
│   │  recommendations/*   │  │ ─ raw LLM            │                             │
│   │ ─ scikit-learn       │  │   pass-through       │                             │
│   │   (CBF + CF)         │  │                      │                             │
│   └──────────────────────┘  └──────────┬───────────┘                             │
│                                        │                                         │
│                       ┌────────────────┴───────────┬────────────────────┐        │
│                       │                            │                    │        │
│                       ▼                            ▼                    ▼        │
│   ┌──────────────────────────────┐  ┌──────────────────────────┐                 │
│   │ LLMGateway (singleton)       │  │ XGBoost predictor        │                 │
│   │ app/ai/llm/gateway.py        │  │ TwoStageUncertaintyModel │                 │
│   │ ─ create_trace (Langfuse)    │  │ pickled .pkl artefact    │                 │
│   │ ─ build_model                │  │                          │                 │
│   │ ─ send_message[_stream/_async]                              │                 │
│   │ ─ generate / generate_with_images                           │                 │
│   │ ─ get_prompt (Langfuse mgmt)                                │                 │
│   │ ─ flush                                                     │                 │
│   └────────────┬─────────────────┘  └──────────────────────────┘                 │
└────────────────┼─────────────────────────────────────────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│ Vertex AI    │    │ Langfuse (cloud)     │    │ Spring Boot backend    │
│ Gemini 2.5   │    │ traces + prompts     │    │ (listings, users, etc.)│
│ Flash        │    │                      │    │ called via httpx       │
└──────────────┘    └──────────────────────┘    └────────────────────────┘

Separate process (not part of the FastAPI app):
┌──────────────────────────────────────────┐
│ MCP Server (app/ai/mcp/backend_server.py)│
│ ─ FastMCP, exposes search_listings tool  │
│ ─ stdio transport for Claude Desktop etc.│
│ ─ Calls Spring Boot via httpx            │
└──────────────────────────────────────────┘
```

---

## 3. Feature deep-dives

### 3.1 Chatbot / Agent

**Paradigm**: ReAct-style agent loop (Yao et al., 2022) with parallel
tool calling, augmented by retrieval (RAG).

**Endpoints**:
- `POST /api/v1/chat` — blocking JSON response.
- `POST /api/v1/chat/stream` — Server-Sent Events.

**Flow** ([`orchestrator.py`](../app/agent/orchestrator.py)):

```
1. Create Langfuse trace
2. RAG: retrieve relevant context (locations, amenities, FAQ, guides)
3. Build system instruction (stable prefix — cached implicitly by Gemini)
4. Convert chat history to provider format
5. Agentic loop (max 5 rounds):
   a. Send message to LLM
   b. Extract function calls from response
   c. If none → break (final text response)
   d. Execute every tool concurrently
   e. Feed tool results back to LLM
6. Build listings payload from collected raw listing objects
7. Return AgentResult
```

**Tools registered** ([`app/agent/tools/`](../app/agent/tools/)):

| Tool | Purpose |
|---|---|
| `search_listings` | Search backend listings by location, price, amenities |
| `get_listing_detail` | Fetch full detail (incl. contact info) for one listing |
| `get_price_estimate` | Call price-prediction service for market estimate |
| `get_price_history` | Price history & changes for a listing |
| `get_recommendations` | Personalized recommendations |
| `get_user_info` | Profile, membership, saved listings |
| `save_listing` | Bookmark / unbookmark |

**RAG** ([`app/agent/rag/retriever.py`](../app/agent/rag/retriever.py)) is
keyword-based with diacritic normalization and synonym expansion over
four JSON knowledge bases:
- `area_codes.json` — provinces + districts with aliases
- `amenities.json` — amenity name → ID
- `faq.json` — Q&A entries with keywords
- `platform_guide.json` — step-by-step usage guides

Retrieval is per-request and pure CPU (no I/O after startup).

**Model**: `gemini-2.5-flash` via Vertex AI (configurable via
`GEMINI_CHAT_MODEL`).

**Caching**: System instruction was recently restructured to be
byte-stable so Gemini's implicit prefix cache hits across requests.
Explicit `CachedContent` is on the roadmap.

---

### 3.2 Listing Verification (multimodal)

**Paradigm**: Single-turn multimodal generation. Images + structured
text prompt → JSON validation report.

**Endpoint**: `POST /ai/verify-listing` (note: prefix is `/ai`, not
`/api/v1`).

**Flow** ([`listing_verification_service.py`](../app/service/listing_verification_service.py)):

```
1. Prepare text content from listing fields (title, desc, price, etc.)
2. Download images from URLs concurrently (httpx + Pillow)
3. Build verification prompt
4. Call gateway.generate_with_images(prompt, images, ...)
5. Parse JSON response into ListingVerificationResponse
6. Return structured validation:
   - image_validation: relevance, quality, count
   - content_validation: clarity, category match
   - completeness_validation: required fields present
   - violations: list of detected issues
   - suggestions: improvement recommendations
   - score: aggregate 0..1
   - is_valid: boolean
```

**Model**: `gemini-2.5-flash` via Vertex AI (configurable via
`GEMINI_VISION_MODEL`).

**Why multimodal**: A pure text model would only see the title and
description. Vision lets the AI catch mismatches (e.g. "luxury
apartment" with low-quality bathroom photos), screenshot-only listings,
inappropriate content, and missing key views.

**Helper class**: [`GeminiListingVerificationHelper`](../app/ai/llm/gemini_listing_helper.py)
encapsulates image download, prompt assembly, and JSON parsing. It
delegates the actual model call to `LLMGateway.generate_with_images()`.

---

### 3.3 Price Prediction (hybrid ML + LLM)

**Paradigm**: This is the **most complex AI surface** — it combines
classical machine learning with an LLM agent as a fallback.

**Endpoint**: `POST /api/v1/price-suggestion/get-price-suggestion`

**Two layers**:

#### Layer A — XGBoost (primary path)

- File: [`app/ai/house_pricing/price_predictor.py`](../app/ai/house_pricing/price_predictor.py)
- Class: `TwoStageUncertaintyModel`
  - **Stage 0**: predicts mean price.
  - **Stage 1**: predicts residual variance, gives confidence intervals.
- Features: location (city/district/ward), property type, area,
  geographic coordinates.
- Output: predicted price range with statistical confidence.
- Loaded once at startup from a pickled `.pkl` artefact trained
  offline.

#### Layer B — Gemini agent (fallback / cross-check path)

- File: [`app/service/price_prediction_service.py`](../app/service/price_prediction_service.py)
- When XGBoost lacks data (uncommon location, rare property type),
  the service builds a Vertex AI `GenerativeModel` with a single
  function-calling tool (`search_listings`) and a strict system
  instruction.
- The LLM:
  1. Calls `search_listings` for nearby comparable properties.
  2. Filters by area (±30%) and product type.
  3. Returns a JSON `{min_price, max_price, listings_found, confidence}`.
- Output: market-anchored price range derived from real listings.
- Hardcoded VND/m² baselines for HN/HCM/DN as last-resort defaults.

This is a textbook **hybrid AI** pattern — fast classical ML for the
common case, LLM-with-tools for the long tail. Defense-friendly framing:
*"Each path's weakness is the other's strength. ML is data-efficient
and deterministic where training data exists; the LLM provides
reasoning over fresh market data where the model's training set is
sparse."*

**Models**:
- XGBoost: trained offline on Vietnamese rental market data.
- Gemini: `gemini-2.5-flash` (configurable via `GEMINI_PRICE_MODEL`).

---

### 3.4 Recommendation (pure classical ML)

**Paradigm**: Hybrid recommender — Content-Based Filtering (CBF) +
Collaborative Filtering (CF) + business-rule boosts. **No LLM
involvement.**

**Endpoints**:
- `POST /api/v1/recommendations/similar` — given target listing +
  candidates, rank by similarity.
- `POST /api/v1/recommendations/personalized` — given user
  interaction history + candidates, rank by personalized fit.

**Algorithm** ([`recommendation_service.py`](../app/service/recommendation_service.py)):

```
Feature matrix (per listing):
  ─ price (MinMax-normalised)
  ─ area (MinMax-normalised)
  ─ bedrooms (MinMax-normalised)
  ─ product_type (one-hot: ROOM/APARTMENT/HOUSE/STUDIO/OFFICE)
  ─ listing_type (one-hot: RENT/SALE/SHARE)
  ─ province_code (one-hot)

Similar listings:
  similarity = cosine(target_vec, candidate_vec)
  apply geographic penalty (×0.1 cross-province, ×0.8 cross-district)
  blend with personalization if user_interactions provided
  apply VIP boost (NORMAL=1.0, SILVER=1.05, GOLD=1.10, DIAMOND=1.15)
  apply freshness boost (decays with post age)

Personalized feed:
  build user profile vector = weighted mean of interacted listings
  CBF score = cosine(profile, candidate)
  CF score  = item-item co-occurrence over all_interactions
  base = 0.4 * CF + 0.6 * CBF if CF > 0 else 0.9 * CBF
  apply same VIP and freshness boosts
```

**Why no LLM**: Recommendation is a real-time ranking problem with
hundreds-to-thousands of candidates per request. LLMs are too slow and
too expensive for this. Classical hybrid recommenders are the industry
standard (Netflix, Spotify, etc.).

**Stack**: pure NumPy + scikit-learn (`MinMaxScaler`, `cosine_similarity`).

---

### 3.5 Completion (raw text generation)

**Paradigm**: Stateless single-prompt generation, no tools, no agent loop.

**Endpoint**: `POST /api/v1/completion/`

**Flow** ([`completion.py`](../app/api/v1/completion.py)):

```
1. Validate prompt non-empty
2. Get LLMGateway singleton
3. Create Langfuse trace
4. gateway.generate(prompt, model_name, generation_config, ...)
5. Extract text safely from response
6. Return CompletionResponse(text, model_used)
```

**Caller**: Spring Boot backend uses this endpoint for tasks like
listing description generation. Not exposed to the frontend.

**Why a separate endpoint**: gives Spring Boot a stable, observable
hook into the same Vertex AI quota and tracing infrastructure used by
the agent — without re-implementing the gateway in Java.

---

### 3.6 MCP Server (external integration surface)

**Paradigm**: Model Context Protocol — a protocol for exposing tools to
external AI assistants (Claude Desktop, Cursor, etc.).

**Not** part of the FastAPI HTTP service. Runs as a separate process
([`app/ai/mcp/backend_server.py`](../app/ai/mcp/backend_server.py))
using stdio transport.

**Exposes**: a single tool `search_listings` with extensive filter
parameters (location, price, area, amenities, status, ownership).

**Implementation**: built on `FastMCP` (`mcp` Python package). Each
tool call results in an `httpx` call to the Spring Boot backend.

**Why it exists**: lets a developer with Claude Desktop search the
listings database directly from chat. Independent of the chatbot agent.

---

## 4. Shared infrastructure

### 4.1 LLMGateway

[`app/ai/llm/gateway.py`](../app/ai/llm/gateway.py) — every LLM call in
the application goes through this class. Initialized as a module-level
singleton at FastAPI startup ([`app/main.py:34`](../app/main.py#L34)
via `lifespan` so Vertex AI is ready before the first request).

**Responsibilities**:
- One-time `vertexai.init()` with project/location.
- GCP credentials handling (base64 env var → temp file).
- Optional Langfuse client construction (no-op stubs when disabled).
- Trace and span factories (`create_trace`, `_NoOpTrace`).
- Prompt management (`get_prompt`) — fetches versioned prompts from
  Langfuse with client-side caching.
- Model construction (`build_model`) — wraps `GenerativeModel(...)`.
- Chat lifecycle (`start_chat`, `send_message`, `send_message_stream`).
- Single-shot generation (`generate`, `generate_with_images`).
- Quota retry with exponential backoff (`_retry_on_quota`).
- Lifecycle hook (`flush`) — flushes Langfuse buffer, cleans temp creds.

**Used by**:
- Chatbot agent (chat + streaming + function calling)
- Listing verification (vision + text)
- Price prediction Layer B (function calling for fallback)
- Completion (raw generate)

### 4.2 Langfuse

Optional but recommended — enabled when `LANGFUSE_SECRET_KEY` and
`LANGFUSE_PUBLIC_KEY` are present in the env.

What gets traced:
- Every chat turn (`chat-request` or `chat-stream` trace).
- Every LLM call (`llm-initial`, `llm-round-N`, `completion-generate`,
  `vision-generate`).
- RAG retrieval (`rag-retrieve` span — input query, output context length).
- Each tool call (`tool-{name}` span — input args, output status).
- Token usage extracted from `usage_metadata` and attached to
  generation spans.

Prompts are also managed in Langfuse (`smartrent-chat-system`,
production label) with a 5-minute client-side TTL — non-blocking and
falls back to a hardcoded base prompt if Langfuse is down.

### 4.3 Backend client

[`app/core/backend_client.py`](../app/core/backend_client.py) — thin
`httpx`-based client used by every tool that needs to read/write
listing data on Spring Boot. Centralizes:
- Base URL (`SMARTRENT_BACKEND_URL`)
- 30s timeout
- Auth header injection when `auth_token` is provided
- Error normalization

This is **not** the `LLMGateway`, but is the parallel infrastructure
layer for the *backend* side of every AI feature that needs real data.

---

## 5. AI paradigms used (academic framing)

| Paradigm | Where | Defining trait |
|---|---|---|
| **LLM agent** (ReAct, Yao 2022) | Chatbot | Multi-turn loop with tool use; LLM decides which tool to invoke |
| **Multimodal LLM** | Listing Verification | Single-turn; image + text in, structured output |
| **Hybrid AI** | Price Prediction | ML primary + LLM fallback; covers both data-rich and data-sparse cases |
| **Classical ML** | Recommendation, Price Predictor (XGBoost layer) | No LLM; deterministic, fast, scales to thousands of items |
| **RAG** (Retrieval-Augmented Generation) | Chatbot RAG layer | Per-query keyword retrieval injected as context |
| **Function Calling / Tool Use** | Chatbot tools, Price Prediction Layer B | Structured JSON Schema → model selects tool → orchestrator executes |
| **MCP** (Model Context Protocol) | MCP Server | Standardized protocol for external AI clients to invoke tools |

For thesis defense, this gives you a clean talking sequence: *"The
SmartRent AI service combines six AI surfaces spanning four paradigms
— classical ML, LLM agents, multimodal generation, and hybrid ML+LLM —
unified by a single observability and gateway layer."*

---

## 6. Data flows

### 6.1 Chat (typical search flow)

```
User → /api/v1/chat/stream
     → AgentOrchestrator.run_stream
       → RAG.retrieve (in-memory, ~ms)
       → LLMGateway.send_message_stream
         → Vertex AI Gemini (round 1)
       ← function_call: search_listings
       → ToolRegistry.execute(search_listings)
         → backend_client.post(/v1/listings/search)
           → Spring Boot (DB)
       ← results
       → LLMGateway.send_message_stream
         → Vertex AI Gemini (round 2 — composes answer)
       ← text deltas + listings payload
     ← SSE events to frontend
```

### 6.2 Listing verification

```
Spring Boot → /ai/verify-listing
            → ListingVerificationService.verify_listing
              → image download (concurrent httpx)
              → prompt assembly
              → LLMGateway.generate_with_images
                → Vertex AI Gemini (multimodal)
              ← JSON validation result
            ← parsed ListingVerificationResponse
```

### 6.3 Price prediction (hybrid)

```
Spring Boot → /api/v1/price-suggestion/get-price-suggestion
            → PricePredictionService.predict_price
              ┌─────────────────────────────────────────────┐
              │ Try: TwoStageUncertaintyModel.predict       │
              │      (XGBoost, deterministic, fast)         │
              └─────────────────────────────────────────────┘
              ┌─────────────────────────────────────────────┐
              │ If no/weak signal:                          │
              │   Build Gemini model with search_listings   │
              │   tool                                      │
              │   Loop: LLM → tool call → results → LLM     │
              │   Parse final JSON                          │
              └─────────────────────────────────────────────┘
            ← min_price, max_price, confidence
```

### 6.4 Recommendation

```
Spring Boot → /api/v1/recommendations/personalized
            → RecommendationService.get_personalized_feed
              → build feature matrix (NumPy)
              → build user profile vector (weighted mean)
              → CBF score (cosine similarity)
              → CF score (item-item co-occurrence)
              → blend, apply VIP + freshness boosts
              → sort, slice top_n
            ← List[RecommendationItem]
```

---

## 7. Configuration & deployment

All settings via `pydantic-settings` in [`app/core/config.py`](../app/core/config.py):

| Setting | Used by |
|---|---|
| `GCP_PROJECT_ID`, `GCP_LOCATION`, `GCP_CREDENTIALS_BASE64` | LLMGateway → Vertex AI |
| `GEMINI_CHAT_MODEL`, `GEMINI_VISION_MODEL`, `GEMINI_PRICE_MODEL` | Per-feature model selection |
| `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` | Tracing + prompt management |
| `SMARTRENT_BACKEND_URL` | All tools' `backend_client` calls |
| `MAX_LISTINGS_RETURN` | Chat — cap on listings shown |
| `MYSQL_*` | User service (not AI-related, separate concern) |

**Lifecycle** ([`app/main.py`](../app/main.py)):
- Startup: `LLMGateway` singleton created eagerly to ensure
  `vertexai.init()` runs before any request.
- Shutdown: `LLMGateway.flush()` flushes Langfuse buffer and cleans up
  the temp credentials file.

---

## 8. Provider abstraction status

The current code is coupled to **Vertex AI Gemini** at five points:

| File | Coupling |
|---|---|
| [`gateway.py`](../app/ai/llm/gateway.py) | Imports Vertex SDK directly |
| [`orchestrator.py`](../app/agent/orchestrator.py) | Constructs `Content`/`Part` |
| [`gemini_listing_helper.py`](../app/ai/llm/gemini_listing_helper.py) | Uses Gateway, but accepts response shape |
| [`price_prediction_service.py`](../app/service/price_prediction_service.py) | Imports `Tool`, `FunctionDeclaration`, `Content`, `Part` directly |
| 7 × [`agent/tools/*.py`](../app/agent/tools/) | Each imports `FunctionDeclaration` |

**Designed extension** (see [architecture.md](architecture.md)):
introduce `LLMProvider` ABC with provider-agnostic types
(`LLMMessage`, `ToolCall`, `ToolDefinition`, `LLMResponse`,
`LLMStreamChunk`, `CacheHandle`). Refactor `LLMGateway` body into a
`GeminiProvider`. Adding `OpenAIProvider` or `AnthropicProvider`
becomes a single ~150-LOC file with no orchestrator/tool changes.

The XGBoost predictor and the recommender are **already** model/vendor
neutral — they depend only on NumPy and scikit-learn.

---

## 9. Adding a new AI feature — extension recipe

Each AI surface follows roughly the same five-step pattern. To add a
new one:

1. **Define DTOs** in `app/dto/<feature>.py` — `Request` and
   `Response` with `BaseModel`.
2. **Implement service** in `app/service/<feature>_service.py` — encapsulate
   the AI logic. If LLM-based, **always go through `LLMGateway`** so
   Langfuse traces and quota retries work uniformly.
3. **Add HTTP layer** in `app/api/v1/<feature>.py` — thin router with
   `Depends(...)` for the service.
4. **Register the router** in [`app/api/v1/api.py`](../app/api/v1/api.py).
5. **(Optional) Tool exposure** — if the chat agent should be able to
   invoke this feature, add a tool in `app/agent/tools/<name>.py`,
   register it in [`app/agent/tools/__init__.py`](../app/agent/tools/__init__.py).

For pure-ML features (no LLM), skip the gateway and use NumPy /
scikit-learn / etc. directly inside the service.

---

## 10. Defense-ready Q&A

### Q: How many distinct AI features does the system have?

> Six: a multi-tool LLM chatbot, multimodal listing verification, a
> hybrid ML/LLM price predictor, a classical-ML hybrid recommender, a
> raw completion endpoint, and an MCP server for external AI clients.
> Three are LLM-driven, one is pure classical ML, one is hybrid, and
> one is an integration surface.

### Q: Is the chatbot the only place using Gemini?

> No. Four features call Gemini through a shared `LLMGateway`
> singleton: the chatbot agent, the listing-verification multimodal
> analyzer, the price predictor's fallback path, and the raw
> completion endpoint. Centralising in `LLMGateway` means observability
> (Langfuse traces, token usage, prompt versioning) and quota retries
> are consistent across all four.

### Q: Why is price prediction hybrid and not pure ML or pure LLM?

> XGBoost, our primary path, is fast, deterministic, and accurate
> where training data exists — most provinces and property types in
> Vietnam. But for sparse cases (rare property types, new districts),
> XGBoost predictions become unreliable. The LLM fallback uses
> function calling to query the live listings database for nearby
> comparables and reasons over them in real time. This is a textbook
> hybrid AI pattern: fast classical ML for the common case, LLM-with-tools
> for the long tail.

### Q: Why does the recommender not use an LLM?

> Recommendation is a real-time ranking problem with hundreds to
> thousands of candidates per request. The latency and cost profile
> of LLM inference per candidate is incompatible with that workload.
> Classical hybrid recommenders — Content-Based Filtering plus
> Collaborative Filtering with business-rule boosts — are the industry
> standard for this exact problem (Netflix Prize literature, Su &
> Khoshgoftaar 2009).

### Q: How is observability handled across all these AI surfaces?

> Every LLM call goes through `LLMGateway`, which wraps each call in a
> Langfuse generation span. Tool calls in the chat agent are wrapped
> in spans as well. Token usage, prompt versions, errors, and
> latencies all flow into Langfuse where they can be filtered by trace
> name, user, session, or model. Pure-ML features are observed via
> standard FastAPI logging and request timing.

### Q: What if you need to swap Gemini for OpenAI or Anthropic?

> The shared `LLMGateway` already isolates Vertex SDK construction.
> The full provider-abstraction design is documented in
> [architecture.md](architecture.md): introduce an `LLMProvider`
> interface with provider-agnostic data types, refactor the existing
> gateway body into a `GeminiProvider` class, and add new providers
> as ~150-LOC files. The chat tools' parameter schemas are already in
> universal JSON Schema, so the tool layer requires only ~5 lines of
> change per file.

---

## 11. References

- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in
  Language Models.* arXiv:2210.03629.
- Su, X., Khoshgoftaar, T. (2009). *A Survey of Collaborative Filtering
  Techniques.* Advances in Artificial Intelligence.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.* NeurIPS.
- JSON Schema Specification, Draft 2020-12.
- Anthropic, *Model Context Protocol (MCP) Specification.* 2024.
- Google Cloud — *Vertex AI Generative Models documentation.*
- Liu, N. et al. (2023). *Lost in the Middle: How Language Models Use
  Long Contexts.* NAACL 2024.

---

## 12. Implementation status snapshot

| Surface | Implemented | Notes |
|---|---|---|
| Chatbot — blocking `/api/v1/chat` | ✓ | Production |
| Chatbot — streaming `/api/v1/chat/stream` | ✓ | Stable-prefix system instruction shipped; `CachedContent` pending |
| Listing Verification | ✓ | Production |
| Price Prediction — XGBoost | ✓ | Production |
| Price Prediction — Gemini fallback | ✓ | Production |
| Recommendation — similar | ✓ | Production |
| Recommendation — personalized | ✓ | Production |
| Completion | ✓ | Used by Spring Boot |
| MCP Server | ✓ | Standalone process |
| Provider abstraction | designed | See [architecture.md](architecture.md) |
| Long-term user memory | future | Not yet implemented |
| Sliding window for chat | ✓ | `MAX_HISTORY_MESSAGES = 12` in [`orchestrator.py`](../app/agent/orchestrator.py); applied at both `run()` and `run_stream()` entry points |
| Vector RAG | future | Currently keyword-based |
