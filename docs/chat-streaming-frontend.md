# Chat Streaming — Frontend Integration Guide

Practical reference for integrating with `POST /api/v1/chat/stream` (Server-Sent Events).

For agent architecture & tool behavior, see [chatbot-integration.md](chatbot-integration.md).

---

## TL;DR

- **Service**: This is the **FastAPI AI service** (default port `8000`, env `SMARTRENT_AI_URL`) — a different host from the Spring Boot backend. The Spring Boot backend's chat path `/v1/ai/chat` (if any) is a separate proxy and does NOT serve streaming today.
- **Endpoint**: `POST /api/v1/chat/stream` on the FastAPI service.
- **Content-Type**: `text/event-stream` (SSE)
- **Request body**: identical to `POST /api/v1/chat`
- **Auth**: pass JWT as `auth_token` field **inside the JSON body** (NOT as `Authorization: Bearer …` header — the FastAPI endpoint reads only the body).
- **5 event types**: `status`, `text`, `listings`, `done`, `error`
- **Cannot use native `EventSource`** — it's GET-only. Use `fetch()` + ReadableStream, or `@microsoft/fetch-event-source`.
- Both `/api/v1/chat` (blocking JSON) and `/api/v1/chat/stream` (SSE) are live — pick one per call site.

---

## Why streaming?

The blocking `/api/v1/chat` endpoint waits until the full agent loop finishes (LLM thinking + tool calls + final answer) before returning a single JSON response. For chat UX this means:

- Users stare at a spinner for 3-30 seconds with no feedback.
- Browser/proxy timeouts can fire before the response arrives.
- No way to show progress (e.g. "searching listings…").

The streaming endpoint emits incremental events as the agent works:

```
[~100ms]  status: thinking      (round 1 starts)
[~800ms]  status: tool_call     (calling search_listings)
[~2.1s]   status: tool_result   (got 5 listings)
[~2.3s]   status: thinking      (round 2 starts — composing answer)
[~2.5s]   text: "Tôi tìm "
[~2.6s]   text: "thấy 5 căn "
[~2.7s]   text: "hộ phù hợp..."
[~2.9s]   listings: { listings: [...], totalCount: 5 }
[~3.0s]   done: { metadata, tools_used }
```

---

## Request

Same shape as `/api/v1/chat`. Pydantic schema in [`app/dto/chat.py`](../app/dto/chat.py).

```ts
type ChatRequest = {
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
  user_id?: string | null;       // authenticated user ID (optional)
  auth_token?: string | null;    // JWT for backend tool calls (optional)
  last_listings?: Array<{        // listings shown in previous turn (for "the 2nd one" follow-ups)
    position: number;
    listingId: string;
    title?: string;
  }> | null;
};
```

### Required headers

```
Content-Type: application/json
Accept: text/event-stream
```

### Validation rules (server-side)

- `messages` cannot be empty → 400
- Last message must have `role: "user"` → 400

---

## Response: SSE event format

Each event is a block separated by `\n\n`:

```
event: <event_name>
data: <json string>

```

### Event types

| Event | Payload shape | When | UI action |
|---|---|---|---|
| `status` | `{ phase: 'thinking', round: number }` | Start of each LLM round | Show typing indicator |
| `status` | `{ phase: 'tool_call', tool: string }` | Before invoking a tool | Show "Đang tìm…" with tool name |
| `status` | `{ phase: 'tool_result', tool: string, status: 'success' \| 'error', error?: string }` | After tool returns | Optional — fade tool indicator. When `status === 'error'`, `error` carries the failure message (truncated to 500 chars). |
| `text` | `{ delta: string }` | Each text chunk from the model | **Append `delta` to the assistant bubble** |
| `listings` | `{ listings: [...], totalCount, selectedFromTotal, currentPage, pageSize, totalPages }` | After search/detail tools collect listings | Render listing cards under the bubble |
| `done` | `{ metadata: { model, tools_used, rag_context_injected }, tools_used: string[] }` | Stream complete | Mark message complete, stop spinner |
| `error` | `{ message: string }` | Unrecoverable failure | Show error toast, re-enable input. Stream ends. |

### Notes on event ordering

- `status: thinking` → `text` deltas → `done` is the **happy path** (no tool needed, e.g. an FAQ question).
- `status: thinking` → `status: tool_call` → `status: tool_result` → `status: thinking` → `text` deltas → `listings` → `done` is a **typical search flow**.
- `text` events may be interleaved with `status` events when the model emits prose between tool calls.
- Multiple tool rounds are possible (max 5). Always render based on event type, not order.
- `listings` arrives at most **once** per response, near the end.

### Tool name → friendly label mapping (suggested)

| Tool name | Vietnamese label | English label |
|---|---|---|
| `search_listings` | Đang tìm BĐS | Searching listings |
| `get_listing_detail` | Đang lấy chi tiết | Loading details |
| `get_price_estimate` | Đang ước tính giá | Estimating price |
| `get_price_history` | Đang xem lịch sử giá | Loading price history |
| `get_recommendations` | Đang gợi ý | Building recommendations |
| `get_user_info` | Đang lấy thông tin tài khoản | Loading account info |
| `save_listing` | Đang lưu tin | Saving listing |

---

## Code examples

### Option A — `@microsoft/fetch-event-source` (recommended)

Handles parsing, reconnection, and POST + custom headers out of the box.

```bash
npm install @microsoft/fetch-event-source
```

```ts
import { fetchEventSource } from '@microsoft/fetch-event-source';

type StreamHandlers = {
  onStatus: (data: { phase: string; round?: number; tool?: string; status?: string }) => void;
  onTextDelta: (delta: string) => void;
  onListings: (payload: ListingsPayload) => void;
  onDone: (data: { metadata: Record<string, unknown>; tools_used: string[] }) => void;
  onError: (message: string) => void;
};

export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await fetchEventSource('/api/v1/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify(request),
    signal,
    openWhenHidden: true, // don't pause when tab is backgrounded

    onmessage(ev) {
      const data = JSON.parse(ev.data);
      switch (ev.event) {
        case 'status':   handlers.onStatus(data); break;
        case 'text':     handlers.onTextDelta(data.delta); break;
        case 'listings': handlers.onListings(data); break;
        case 'done':     handlers.onDone(data); break;
        case 'error':    handlers.onError(data.message); break;
      }
    },

    onerror(err) {
      handlers.onError(err.message ?? 'Stream connection error');
      throw err; // stop auto-reconnect
    },
  });
}
```

### Option B — Vanilla `fetch` + ReadableStream

No dependencies, ~50 lines.

```ts
export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!res.ok || !res.body) {
    handlers.onError(`HTTP ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE blocks are separated by a blank line (\n\n)
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() ?? ''; // keep incomplete trailing block

    for (const block of blocks) {
      if (!block.trim()) continue;

      let event = 'message';
      let data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7).trim();
        else if (line.startsWith('data: ')) data += line.slice(6);
      }
      if (!data) continue;

      let parsed: any;
      try { parsed = JSON.parse(data); }
      catch { continue; }

      switch (event) {
        case 'status':   handlers.onStatus(parsed); break;
        case 'text':     handlers.onTextDelta(parsed.delta); break;
        case 'listings': handlers.onListings(parsed); break;
        case 'done':     handlers.onDone(parsed); break;
        case 'error':    handlers.onError(parsed.message); return;
      }
    }
  }
}
```

### React hook example

```tsx
import { useCallback, useRef, useState } from 'react';

type Status = { phase: string; tool?: string; round?: number };

export function useChatStream() {
  const [text, setText] = useState('');
  const [status, setStatus] = useState<Status | null>(null);
  const [listings, setListings] = useState<ListingsPayload | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (request: ChatRequest) => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setText('');
    setStatus(null);
    setListings(null);
    setError(null);
    setIsStreaming(true);

    try {
      await streamChat(request, {
        onStatus: setStatus,
        onTextDelta: (delta) => setText((prev) => prev + delta),
        onListings: setListings,
        onDone: () => { setIsStreaming(false); setStatus(null); },
        onError: (msg) => { setError(msg); setIsStreaming(false); },
      }, abortRef.current.signal);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError((err as Error).message);
      }
      setIsStreaming(false);
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  return { text, status, listings, isStreaming, error, send, cancel };
}
```

---

## UX patterns

### Showing the assistant message

Render the bubble as soon as the first `text` delta arrives. Append each delta — never replace. The full message string is `accumulated_deltas.join('')`.

```tsx
<MessageBubble>
  {text}
  {isStreaming && <TypingCursor />}
</MessageBubble>
```

### Showing tool progress

Map `status.tool` to a friendly label (see table above). Show ONE indicator at a time — replace, don't stack.

```tsx
{status?.phase === 'tool_call' && (
  <StatusBar>{toolLabel(status.tool)}…</StatusBar>
)}
```

When a `text` delta arrives, hide the status indicator — the model has switched to writing.

### Rendering listings

The `listings` event arrives once near the end. Treat it as the source of truth for listing cards. Don't try to parse listing IDs from the text — use the `listings` payload.

```tsx
{listings && <ListingGrid listings={listings.listings} />}
```

### Cancellation

Always wire an `AbortController` to the request. When the user navigates away, sends a new message, or hits a "stop generating" button — call `controller.abort()`. The backend handles client disconnect cleanly via `asyncio.CancelledError` ([`orchestrator.py`](../app/agent/orchestrator.py)).

### Multi-turn conversations

After `done` fires, append the assistant's full text + listings to your local conversation state. Send the **full history** in the next request's `messages` array (the backend is stateless — it reconstructs context from `messages`).

```ts
// On done
setHistory((prev) => [
  ...prev,
  { role: 'assistant', content: text },
]);

// On next user message
const request: ChatRequest = {
  messages: [...history, { role: 'user', content: userInput }],
  user_id,
  auth_token,
  last_listings: listings?.listings.map((l, i) => ({
    position: i + 1,
    listingId: l.listingId,
    title: l.title,
  })),
};
```

`last_listings` is optional but **strongly recommended** — without it, follow-ups like *"chi tiết cái thứ 2"* won't work because the model can't see prior listing IDs.

---

## Error handling

### Per-event error

If you receive `event: error`, the stream is terminated by the server. Show the message to the user and stop reading.

```ts
case 'error':
  showToast(parsed.message);
  return; // exit reader loop
```

### HTTP errors before the stream starts

400/500 errors arrive as a normal JSON response (not SSE). Check `res.ok` and `res.headers.get('content-type')` before reading the stream.

```ts
if (!res.ok) {
  const body = await res.json().catch(() => ({}));
  throw new Error(body.detail ?? `HTTP ${res.status}`);
}
```

### Network drop mid-stream

If the connection drops mid-stream, `reader.read()` rejects. Treat as an error — show "kết nối bị gián đoạn" and let the user retry.

`@microsoft/fetch-event-source` will auto-reconnect by default; pass `openWhenHidden: true` and **`throw err` from `onerror`** to disable auto-reconnect for chat (you don't want it to silently re-fire the request).

### Server timeout

The agent has a 120s hard timeout. If exceeded, you'll receive a final `text` event with `"Xin lỗi, yêu cầu đã mất quá nhiều thời gian xử lý..."` followed by `done`. Render normally — it's a graceful degradation, not a stream error.

---

## Backend gotchas — verify in your environment

1. **CORS**: confirm your CORS middleware allows `text/event-stream` and that `Accept` header passes through. If you see streams cut off after the first event, CORS is likely the cause.
2. **Reverse proxy buffering**: nginx/CloudFront often buffer SSE. The endpoint sets `X-Accel-Buffering: no` ([`app/api/v1/chat.py`](../app/api/v1/chat.py)) which fixes nginx, but check whatever proxy fronts FastAPI in your deploy.
3. **Browser DevTools**: SSE streams appear in the Network tab as a long-running request with no preview until done. To debug events in real time, use the EventStream tab (Chrome/Edge: Network → click request → EventStream).
4. **HTTP/2**: works fine with SSE. Some proxies behave differently — test in staging.

---

## Migration from `/api/v1/chat` → `/api/v1/chat/stream`

Both endpoints stay live indefinitely. Recommended rollout:

1. Implement streaming behind a feature flag.
2. Enable for internal/QA users first.
3. Roll out to all users; monitor error rates and TTFT in Langfuse.
4. Keep `/api/v1/chat` as a fallback for any non-browser client (mobile native, server-to-server).

### Behavioral differences

| Aspect | `/api/v1/chat` | `/api/v1/chat/stream` |
|---|---|---|
| Time to first byte | Full latency (3-30s) | ~100-500ms |
| Cancellation | Not possible mid-request | `AbortController` works cleanly |
| Error reporting | HTTP status + `detail` | `event: error` mid-stream OR HTTP status pre-stream |
| Listings field | In final JSON `response.listings` | `event: listings` near end of stream |
| Metadata | In final JSON `response.metadata` | In `event: done` payload |
| Final text | `response.message.content` | Concatenation of all `text.delta` events |

---

## Quick reference card

```
POST /api/v1/chat/stream
  Headers:  Content-Type: application/json
            Accept: text/event-stream
  Body:     ChatRequest (same as /api/v1/chat)
  Returns:  text/event-stream

Events (in arrival order):
  status    {phase: 'thinking', round}                — show spinner
  status    {phase: 'tool_call', tool}                — show tool label
  status    {phase: 'tool_result', tool, status}      — fade tool label
  text      {delta: '...'}                            — append to bubble
  listings  {listings: [...], totalCount, ...}        — render cards
  done      {metadata, tools_used}                    — finalize
  error     {message: '...'}                          — show toast, stop
```
