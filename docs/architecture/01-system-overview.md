# 01 — System Overview

End-to-end stack at production deployment time. FE talks to the
Spring Boot backend for everything traditional; the AI service is a
peer that the FE calls directly for chat (so SSE doesn't traverse two
hops) and the backend calls internally for batch jobs (listing
verification + price prediction).

```mermaid
graph TB
  subgraph Client["Client (Browser)"]
    FE["Next.js Frontend<br/>(smartrent-fe)"]
  end

  subgraph Edge["Edge — Droplet (Caddy reverse proxy)"]
    Caddy["Caddy 2<br/>:80 / :443"]
  end

  subgraph Services["App services — Docker Compose"]
    BE["Spring Boot Backend<br/>(smartrent-backend)<br/>:8080"]
    AI["FastAPI AI Service<br/>(smartrent-ai)<br/>:8000"]
    Redis[("Redis<br/>cache / sessions")]
  end

  subgraph Data["Data plane"]
    MySQL[("MySQL<br/>DigitalOcean Managed DB")]
    R2[("Cloudflare R2<br/>media / images")]
  end

  subgraph External["External AI providers"]
    Vertex[/"Vertex AI<br/>(Gemini 2.5 Flash)"/]
    Langfuse[/"Langfuse<br/>traces + prompts"/]
  end

  FE -->|HTTPS| Caddy
  Caddy -->|/api/*| BE
  Caddy -->|/api/v1/chat/stream<br/>SSE| AI

  BE <-->|JDBC| MySQL
  BE <-->|S3 SDK| R2
  BE <-->|RESP| Redis

  AI -->|HTTP| BE
  AI -->|LiteLLM| Vertex
  AI -->|trace events| Langfuse
  AI -.->|GET prompt "smartrent-chat-system"<br/>(cached LRU)| Langfuse

  classDef external fill:#fffae5,stroke:#d49100,stroke-width:1px
  classDef data fill:#e5f3ff,stroke:#1a6fdb,stroke-width:1px
  class Vertex,Langfuse external
  class MySQL,R2,Redis data
```

## Notes

- **AI is a peer, not behind the backend.** This matters for SSE — if
  the backend proxied chat traffic we'd lose chunked transfer at the
  reverse hop. Caddy routes `/api/v1/chat/stream` straight to the AI
  service with `X-Accel-Buffering: no` honoured.
- **Provider pluggability.** `make_model()` in `agent_factory.py`
  switches on `LLM_PROVIDER` (`gemini` / `openai` / `litellm`); only
  the box labelled "Vertex AI" changes when we swap providers. Same
  diagram, different external node.
- **Langfuse is observability-only.** It does not sit in the request
  path. The dotted line indicates the prompt fetch happens once per
  process and is then cached by `lru_cache` in the orchestrator.
- **Backend ↔ AI internal calls** (post-listing verify, price predict)
  go directly over the Docker bridge network, not through Caddy.
