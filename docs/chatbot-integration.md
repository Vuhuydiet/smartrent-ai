# SmartRent Chatbot Integration

SmartRent's conversational AI is an agent that helps users search and discuss
real estate listings. It runs on Vertex AI Gemini through the shared
`LLMGateway` (see [llm-integration.md](llm-integration.md)) and uses function
calling to invoke backend tools.

## What the agent can do

- Search for listings by location, price, area, amenities, etc.
- Fetch full details (including contact info) for a specific listing
- Estimate fair market rent for a property
- Compare multiple listings
- Answer general questions about the SmartRent platform

It is **scope-constrained**: it refuses off-topic requests (cooking, sports,
programming, etc.) and always replies in Vietnamese.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in:

```env
# Vertex AI (required)
GCP_PROJECT_ID=your-gcp-project-id
GCP_CREDENTIALS_BASE64=<base64 of your service account JSON>
GCP_LOCATION=us-central1
GEMINI_CHAT_MODEL=gemini-2.5-flash

# Langfuse observability (optional but recommended)
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Backend
SMARTRENT_BACKEND_URL=http://localhost:8080
```

There is **no `GEMINI_API_KEY`** — the project uses Vertex AI with GCP
service account authentication.

### 3. Run the service

```bash
uvicorn app.main:app --reload
```

At startup `LLMGateway` is eagerly initialised, so any misconfiguration in
GCP credentials surfaces immediately in the logs (not on the first request).

## API

### `POST /api/v1/chat`

Sends the full conversation history for one user turn. The client is
responsible for carrying the history forward; the server is stateless.

**Request body:**

```json
{
  "messages": [
    { "role": "user", "content": "Tôi muốn tìm phòng trọ ở Quận 1, TP.HCM dưới 5 triệu" }
  ]
}
```

Constraints: `messages` must be non-empty and the last item must have
`role: "user"`.

**Response body:**

```json
{
  "message": {
    "role": "assistant",
    "content": "Tìm thấy 12 kết quả ở Quận 1. Đây là 5 BĐS phù hợp nhất cho bạn."
  },
  "metadata": {
    "model": "gemini-2.5-flash",
    "tools_used": ["search_listings"],
    "rag_context_injected": true
  },
  "listings": {
    "listings": [/* raw backend listing objects */],
    "totalCount": 5,
    "selectedFromTotal": 12,
    "currentPage": 1,
    "pageSize": 5,
    "totalPages": 1
  }
}
```

When the user is not searching, `listings` is `null`.

### `GET /api/v1/health`

Chat service health check. Returns `200` when the agent can be initialised.

## Architecture

```
app/
├── api/v1/chat.py                     # FastAPI router
├── service/chat_service.py            # Thin wrapper → AgentOrchestrator
├── agent/
│   ├── orchestrator.py                # Agent loop (function calling)
│   ├── tools/
│   │   ├── registry.py                # ToolRegistry (builds Gemini Tool)
│   │   ├── search_listings.py         # SmartRent backend search
│   │   ├── get_listing_detail.py      # Fetch full listing
│   │   └── get_price_estimate.py      # XGBoost + rule-based estimator
│   └── rag/
│       ├── retriever.py               # Dynamic per-query RAG context
│       └── knowledge_base/            # Province codes, amenities, FAQ
└── ai/llm/gateway.py                  # Vertex AI + Langfuse entry point
```

### Per-request flow

1. `ChatService` delegates to the `AgentOrchestrator` singleton
2. `AgentOrchestrator.run()`:
   - Creates a Langfuse `chat-request` trace
   - Runs RAG retrieval (`retriever.retrieve(user_message)`)
   - Fetches the `smartrent-chat-system` prompt from Langfuse (with local
     fallback) and assembles the system instruction
   - Builds the Gemini model with `system_instruction` + registered tools
   - Seeds a chat session with the prior turns from the request
   - Runs up to **5 tool-call rounds**: call the LLM → extract any
     `function_call` parts → dispatch via `ToolRegistry` → feed results back
   - Extracts the final assistant text and builds the listings payload from
     any raw listings collected during tool execution
3. Returns `AgentResult` → `ChatService` maps to `ChatResponse`

### Stateless session handling

The server does **not** store conversations. Every request must include the
full `messages` array. To continue a conversation, the client appends the
assistant's response and the next user turn and posts the full list again.

## Features

- **Function calling** — the agent decides when to call `search_listings`,
  `get_listing_detail`, or `get_price_estimate` instead of hallucinating.
- **RAG context** — static (province codes, amenity IDs) + dynamic
  (district lookup, FAQ matches for the current query) context is injected
  into every `system_instruction`.
- **Langfuse tracing** — every chat call produces a full trace with
  per-round spans, tool spans, token usage, and linked prompt version.
- **Prompt management** — the system prompt is fetched from Langfuse
  (`smartrent-chat-system` / label `production`) and is editable without a
  redeploy. Falls back to a local constant if Langfuse is unreachable.
- **Quota retry** — 429 `ResourceExhausted` errors are retried with
  exponential backoff (`_retry_on_quota` in the gateway).
- **Request timeout** — 120 s overall per request; the agent loop is capped
  at 5 tool rounds to prevent runaway function calling.

## Troubleshooting

1. **`GoogleAuthError: Unable to find your project`**
   - `GCP_PROJECT_ID` is not set, or `GCP_CREDENTIALS_BASE64` is missing /
     invalid. Check startup logs for `Vertex AI initialised (project=...)`.

2. **Agent ignores tools and answers from memory**
   - Usually a prompt regression. Check the active version of
     `smartrent-chat-system` in Langfuse.

3. **429 `RESOURCE_EXHAUSTED` after retries**
   - You've hit the Vertex quota for the configured project/region. Either
     raise the quota or reduce traffic. The retry logic waits up to
     ~1 minute before giving up.

4. **Langfuse traces missing**
   - `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` not set, or network
     blocked. The gateway silently uses no-op traces when Langfuse is
     unavailable — check startup logs for `Langfuse tracing enabled`.
