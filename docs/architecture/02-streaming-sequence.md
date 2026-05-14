# 02 — SSE Streaming Sequence

Lifecycle of a single chat request from user keystroke to listings
rendered in the chat bubble. The headline contribution is that
**text streams continuously** even though the underlying agent loop
makes multiple round-trips to the LLM and the backend.

```mermaid
sequenceDiagram
  autonumber
  participant U as User (browser)
  participant FE as Next.js FE<br/>(useChatLogic + fetchEventSource)
  participant API as FastAPI route<br/>POST /api/v1/chat/stream
  participant ORCH as AgentOrchestrator<br/>.run_stream()
  participant RAG as RAGRetriever
  participant LLM as Vertex Gemini 2.5<br/>(via LiteLLM)
  participant BE as Spring Boot Backend

  U->>FE: "Tìm trọ ở Bình Thạnh 5-10tr"
  FE->>API: POST /chat/stream<br/>{messages, auth_token}
  API-->>FE: HTTP 200 (text/event-stream)
  API->>FE: ": " + 16KB padding<br/>(force kernel flush)

  API->>ORCH: run_stream(messages, ctx)
  ORCH->>RAG: retrieve(query)
  RAG-->>ORCH: dynamic_context

  ORCH->>ORCH: build Agent(instructions, tools=14)
  ORCH-->>FE: event: status<br/>{phase:"thinking", round:1}

  Note over ORCH,LLM: Round 1 — decide tool to call
  ORCH->>LLM: Runner.run_streamed(input)
  loop SDK stream chunks
    LLM-->>ORCH: text_delta "Để mình tìm thử..."
    ORCH-->>FE: event: text {delta}
  end
  LLM-->>ORCH: tool_call_item<br/>(search_listings, args)

  ORCH-->>FE: event: status<br/>{phase:"tool_call", tool, summary, args}

  Note over ORCH,BE: Tool fan-out (productTypes=[ROOM, APARTMENT])
  par parallel
    ORCH->>BE: POST /v1/listings/search<br/>productType=ROOM
    BE-->>ORCH: [] (0 results)
  and
    ORCH->>BE: POST /v1/listings/search<br/>productType=APARTMENT
    BE-->>ORCH: [5 listings]
  end

  ORCH->>ORCH: dedupe + merge → ctx.collected_listings
  ORCH-->>FE: event: status<br/>{phase:"tool_result", status:"success"}

  Note over ORCH,LLM: Round 2 — write response from tool output
  ORCH->>LLM: feed tool result, continue
  loop SDK stream chunks
    LLM-->>ORCH: text_delta "Đây là 5 căn hộ..."
    ORCH-->>FE: event: text {delta}
  end
  LLM-->>ORCH: stop (no more tool calls)

  ORCH->>ORCH: build listings payload<br/>from collected_listings
  ORCH-->>FE: event: listings {listings, totalCount}
  ORCH-->>FE: event: done {metadata, tools_used}

  FE->>U: stream text + render listing cards
```

## Notes

- **Padding event #0 is load-bearing.** The 16KB SSE comment at the
  start defeats Windows uvicorn's kernel send-buffer accumulation;
  without it the FE's `onopen` doesn't fire until ~5s into the
  request.
- **`status` events carry rich `summary` + `args`** (sprint v2). FE
  renders "Đang tìm BĐS: quận 765 phòng/căn hộ giá 5-10tr..." instead
  of a generic spinner.
- **Tool fan-out is the productTypes contribution.** A single user
  utterance maps to N parallel HTTP calls when the Vietnamese term is
  ambiguous (`nhà trọ` → `[ROOM, APARTMENT]`). Dedupe is by
  listingId; totalCount aggregates conservatively.
- **Round 2 starts immediately after the tool result event** — the
  text gap there is just the LLM TTFT (2-3s with thinking disabled,
  see [03-agent-dispatch.md](./03-agent-dispatch.md)).
- **`listings` event is currently emitted once at the end.**
  Progressive per-card emit is on the Phase 2 polish list.
- **Cancel path** (`useChatLogic.cancelStream`): FE aborts the SSE
  connection; AI orchestrator catches `asyncio.CancelledError` and
  cleanly tears down `Runner.run_streamed`.
