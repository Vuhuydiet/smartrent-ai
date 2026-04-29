# SmartRent AI — System Architecture

This document describes the architecture of the SmartRent AI chat service, with
focus on the **LLM Provider abstraction** that decouples the agent loop from any
specific LLM vendor.

It serves both as engineering reference and as supporting material for the
thesis chapter on system design.

---

## 1. Scope

This document covers the LLM-facing architecture — chat service, agent
orchestrator, tools, and the LLM gateway. Other services (listing
verification, price prediction, recommendation) follow the same Provider
abstraction but are out of scope here.

For agent behavior and tool semantics, see [chatbot-integration.md](chatbot-integration.md).
For frontend integration, see [chat-streaming-frontend.md](chat-streaming-frontend.md).

---

## 2. Current architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HTTP Layer (FastAPI)                               │
│   POST /v1/chat          POST /v1/chat/stream                               │
│   app/api/v1/chat.py                                                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ ChatRequest / ChatResponse  (DTO)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Service Layer                                    │
│   ChatService — thin wrapper, delegates to orchestrator                     │
│   app/service/chat_service.py                                               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ List[ChatMessage], last_listings, ...
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Agent Orchestrator                                 │
│   AgentOrchestrator                                                         │
│   app/agent/orchestrator.py                                                 │
│                                                                             │
│   ⚠ COUPLING TO VERTEX:                                                     │
│     - imports vertexai.generative_models.{Content, Part}                    │
│     - builds   Content(role=..., parts=[Part.from_text(...)])               │
│     - reads    response.candidates[0].content.parts[i].function_call        │
│     - builds   Part.from_function_response(name=..., response=...)          │
└────────────────────┬─────────────────────────────┬──────────────────────────┘
                     │ tools                       │ chat session, send_message
                     ▼                             ▼
┌─────────────────────────────────┐    ┌────────────────────────────────────┐
│ ToolRegistry                    │    │ LLMGateway                         │
│ app/agent/tools/registry.py     │    │ app/ai/llm/gateway.py              │
│                                 │    │                                    │
│ ⚠ get_tool() returns a          │    │ ⚠ Constructs GenerativeModel       │
│   Vertex `Tool` object          │    │   directly                         │
│                                 │    │ ⚠ Returns Vertex response types    │
└──────────────┬──────────────────┘    │ ⚠ Streams Vertex chunk shape       │
               │                       └────────────────┬───────────────────┘
               │ FunctionDeclaration                    │ Vertex AI SDK
               ▼                                        ▼
┌─────────────────────────────────┐    ┌────────────────────────────────────┐
│ 7 × BaseTool implementations    │    │  Vertex AI Gemini (us-central1)    │
│ app/agent/tools/*.py            │    │                                    │
│                                 │    │  Provider: Google                  │
│ ⚠ Each imports vertexai          │    │  Model:    gemini-2.5-flash       │
│ ⚠ to_function_declaration()      │    │                                    │
│   returns FunctionDeclaration   │    │  Caching:  CachedContent           │
│                                 │    │  Tools:    FunctionDeclaration     │
│ ✓ execute()  is universal       │    │                                    │
│ ✓ parameters body is already    │    │  ❌ No alternative selectable      │
│   plain JSON Schema dict        │    │                                    │
└─────────────────────────────────┘    └────────────────────────────────────┘
```

### 2.1 What's wrong (in academic terms)

| Issue | Principle violated | Concrete symptom |
|---|---|---|
| Orchestrator imports Vertex SDK directly | Dependency Inversion Principle | High-level policy depends on low-level detail |
| `LLMGateway` returns Vertex types | Information Hiding | Vertex internals leak across the seam it was meant to encapsulate |
| 7 tool files import `FunctionDeclaration` | Single Responsibility | Tools have two responsibilities: domain logic + provider-format adaptation |
| Adding OpenAI/Anthropic = touching 9 files | Open/Closed Principle | System is closed for extension |

### 2.2 What's already correct

The current code is not entirely wrong — several decisions accidentally got the abstraction right and only need to be exposed:

- Tool `parameters` bodies in [tools/*.py](../app/agent/tools/) are already plain JSON Schema dicts. JSON Schema is the universal tool-description language across Gemini, OpenAI, and Anthropic — only the wrapper around it differs.
- Tool `execute()` methods are pure Python returning `Dict[str, Any]` with a `status` key — provider-independent.
- `ChatService` and the HTTP/DTO layer have no LLM coupling.
- `LLMGateway` already centralizes Vertex SDK construction; it just needs to stop leaking Vertex return types.

---

## 3. Target architecture (with Provider abstraction)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HTTP Layer (FastAPI)                               │
│   POST /v1/chat          POST /v1/chat/stream                               │
│   app/api/v1/chat.py                                            ✓ UNCHANGED│
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ ChatRequest / ChatResponse
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Service Layer                                    │
│   ChatService                                                   ✓ UNCHANGED│
│   app/service/chat_service.py                                               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Agent Orchestrator                                 │
│   AgentOrchestrator                                                         │
│   app/agent/orchestrator.py                                  🔄 REFACTORED  │
│                                                                             │
│   ✓ No Vertex / OpenAI / Anthropic imports                                  │
│   ✓ Works only with provider-agnostic types:                                │
│       LLMMessage   (role, content, tool_calls, tool_call_id)                │
│       ToolCall     (id, name, arguments: dict)                              │
│       ToolResult   (call_id, content)                                       │
│       LLMResponse  (text, tool_calls, usage)                                │
│       LLMStreamChunk (text_delta | tool_call | done)                        │
└────────────────────┬─────────────────────────────┬──────────────────────────┘
                     │ ToolDefinition[]            │ provider.chat() / chat_stream()
                     ▼                             ▼
┌─────────────────────────────────┐    ┌────────────────────────────────────┐
│ ToolRegistry                    │    │ LLMProvider (ABC)             🆕   │
│ app/agent/tools/registry.py     │    │ app/ai/llm/provider.py             │
│                                 │    │                                    │
│ ✓ get_definitions() →            │    │   chat(messages, system, tools,   │
│   List[ToolDefinition]          │    │        cache_handle?)              │
└──────────────┬──────────────────┘    │   chat_stream(...)                 │
               │                       │   register_cache(prefix)→Handle    │
               │ ToolDefinition        └────────────────┬───────────────────┘
               │  { name, description,                  │ implements
               │    parameters_schema:                  │
               │      dict (JSON Schema) }              │
               ▼                                        ▼
┌─────────────────────────────────┐    ┌──────────────────────────────────────┐
│ 7 × BaseTool implementations    │    │ Concrete providers — pick ONE at      │
│ app/agent/tools/*.py            │    │ startup via settings.LLM_PROVIDER     │
│                                 │    │                                       │
│ ✓ No vertexai import             │    │  ┌─────────────────────────────────┐ │
│ ✓ get_parameters_schema() →      │    │  │ GeminiProvider              🆕  │ │
│   plain JSON Schema dict        │    │  │ app/ai/llm/gemini_provider.py   │ │
│ ✓ execute() unchanged           │    │  │  Adapts: FunctionDeclaration,   │ │
│ ✓ ~95% of file untouched        │    │  │          Content/Part,          │ │
│                                 │    │  │          CachedContent          │ │
│ Per file change: ~5 lines       │    │  │ ~250 LOC (current LLMGateway    │ │
│ (drop wrapper import + return)  │    │  │  refactored into this class)    │ │
│                                 │    │  └─────────────────────────────────┘ │
│                                 │    │  ┌─────────────────────────────────┐ │
│                                 │    │  │ OpenAIProvider              🆕  │ │
│                                 │    │  │ app/ai/llm/openai_provider.py   │ │
│                                 │    │  │  Adapts: tool_calls / tool_id,  │ │
│                                 │    │  │          {role:"system"} msg,   │ │
│                                 │    │  │          implicit prefix cache  │ │
│                                 │    │  │ ~150 LOC                        │ │
│                                 │    │  └─────────────────────────────────┘ │
│                                 │    │  ┌─────────────────────────────────┐ │
│                                 │    │  │ AnthropicProvider           🆕  │ │
│                                 │    │  │ app/ai/llm/anthropic_provider…  │ │
│                                 │    │  │  Adapts: tool_use/tool_result,  │ │
│                                 │    │  │          cache_control          │ │
│                                 │    │  │          breakpoints            │ │
│                                 │    │  │ ~150 LOC                        │ │
│                                 │    │  └─────────────────────────────────┘ │
│                                 │    └──────────────────────┬────────────────┘
└─────────────────────────────────┘                           │
                                                              ▼
                                              ┌─────────────────────────────┐
                                              │ Vertex AI Gemini   (Google) │
                                              │   OR                        │
                                              │ OpenAI API         (OpenAI) │
                                              │   OR                        │
                                              │ Anthropic API      (Claude) │
                                              │                             │
                                              │ Selected at startup via     │
                                              │ env var / settings          │
                                              └─────────────────────────────┘
```

### 3.1 Why this is correct (in academic terms)

| Principle | How the design satisfies it |
|---|---|
| **Dependency Inversion Principle** (Martin, 1996) | Orchestrator depends on the `LLMProvider` abstract interface. Concrete providers depend on the same interface. Direction of dependency points toward the abstraction. |
| **Open/Closed Principle** | Adding a new provider extends the system by adding one file; no existing file is modified. |
| **Single Responsibility Principle** | Each `*Provider` class has exactly one reason to change — its provider's API changing. Tools have one responsibility — their domain logic. |
| **Information Hiding** (Parnas, 1972) | Provider-specific types (Vertex `Content`, OpenAI `tool_calls`, Anthropic `tool_use`) never leak past the provider implementation file. |
| **Hexagonal Architecture / Ports & Adapters** (Cockburn, 2005) | `LLMProvider` is the port. Concrete providers are interchangeable adapters. The application core (orchestrator, tools, service) is independent of the adapter chosen. |

---

## 4. Provider-agnostic data contract

The Provider interface is defined in terms of these types — they are the
common vocabulary the orchestrator and providers share. They live in
`app/ai/llm/types.py` (new file).

```python
@dataclass
class LLMMessage:
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_call_id: Optional[str] = None        # for role="tool"
    tool_calls: Optional[List[ToolCall]] = None  # for role="assistant"

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]                 # parsed JSON

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters_schema: Dict[str, Any]         # JSON Schema (universal)

@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall]
    usage: Optional[Dict[str, int]]

@dataclass
class LLMStreamChunk:
    type: Literal["text_delta", "tool_call", "done"]
    text_delta: Optional[str] = None
    tool_call: Optional[ToolCall] = None

class CacheHandle:
    """Opaque token returned by register_cache — meaning is provider-defined."""
    provider: str
    handle: Any
```

### Provider interface

```python
class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        system: str,
        tools: List[ToolDefinition],
        cache_handle: Optional[CacheHandle] = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def chat_stream(
        self, messages, system, tools, cache_handle=None,
    ) -> AsyncIterator[LLMStreamChunk]: ...

    @abstractmethod
    def register_cache(
        self, system: str, tools: List[ToolDefinition],
    ) -> CacheHandle: ...
```

---

## 5. Where each concern moves

| Concern | Current location | After abstraction |
|---|---|---|
| Build `Content/Part` for history | [orchestrator.py:629-635](../app/agent/orchestrator.py#L629-L635) | `GeminiProvider._adapt_messages()` |
| Extract `function_call` from response | [orchestrator.py:336-341](../app/agent/orchestrator.py#L336-L341) | `GeminiProvider.chat()` — returns `LLMResponse` |
| Build `Part.from_function_response` | [orchestrator.py:386-390](../app/agent/orchestrator.py#L386-L390) | `GeminiProvider._adapt_tool_results()` |
| Construct `GenerativeModel` | [gateway.py:257-282](../app/ai/llm/gateway.py#L257-L282) | `GeminiProvider.__init__` / per-call |
| `system_instruction` plumbing | [gateway.py:268-271](../app/ai/llm/gateway.py#L268-L271) | Each provider knows where its API places system messages |
| Streaming chunk parsing | [gateway.py:381-410](../app/ai/llm/gateway.py#L381-L410) | `GeminiProvider.chat_stream()` — yields `LLMStreamChunk` |
| `FunctionDeclaration` wrapping | [tools/*.py × 7 files](../app/agent/tools/) | `GeminiProvider._adapt_tool_definitions()` |
| Vertex `Tool` object construction | [registry.py:32-34](../app/agent/tools/registry.py#L32-L34) | Removed — registry returns `List[ToolDefinition]` |
| Cache management (`CachedContent`) | (planned for `gateway.py`) | `GeminiProvider.register_cache()` |
| Tool's `execute()` business logic | [tools/*.py × 7 files](../app/agent/tools/) | ✓ **Unchanged** |
| Tool's parameter schema dict body | [tools/*.py × 7 files](../app/agent/tools/) | ✓ **Unchanged** — already universal JSON Schema |
| HTTP layer + DTOs | [chat.py](../app/api/v1/chat.py), [chat_service.py](../app/service/chat_service.py) | ✓ **Unchanged** |

---

## 6. Tool layer: what actually changes

The tools are mostly already provider-agnostic. The Vertex-coupling in each
tool file is exactly two things:

```python
# Vertex coupling — to be removed
from vertexai.generative_models import FunctionDeclaration

def to_function_declaration(self) -> Any:
    return FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters={...},   # ← this dict body is universal JSON Schema
    )
```

After refactor:

```python
# Provider-agnostic
def get_parameters_schema(self) -> Dict[str, Any]:
    return {...}            # ← the same dict, no wrapper
```

**Per tool: ~5 lines of mechanical change. Across all 7 tools: ~35 lines.**
The `execute()` method, the parameter schema body, the `name`,
`description`, and return value shape are all unchanged.

The provider class is then responsible for adapting the universal
`ToolDefinition` into its own format:

| Provider | Adapter signature | Lines |
|---|---|---|
| Gemini | `FunctionDeclaration(name=td.name, description=td.description, parameters=td.parameters_schema)` | 5 |
| OpenAI | `{"type": "function", "function": {"name": td.name, "description": td.description, "parameters": td.parameters_schema}}` | 7 |
| Anthropic | `{"name": td.name, "description": td.description, "input_schema": td.parameters_schema}` | 5 |

**The parameters dict flows through unmodified in all three.** Only the
envelope differs.

---

## 7. JSON Schema — feature parity caveat

Providers vary in their support for advanced JSON Schema features:

| Feature | Gemini | OpenAI | Anthropic |
|---|---|---|---|
| `type`, `properties`, `required`, `enum` | ✓ | ✓ | ✓ |
| Nested objects | ✓ | ✓ | ✓ |
| Arrays + `items` | ✓ | ✓ | ✓ |
| Integer ranges (`minimum`, `maximum`) | ✓ | ✓ | ✓ |
| `oneOf`, `anyOf`, `allOf` | partial | ✓ (strict mode: no) | ✓ |
| `additionalProperties: false` | ignored | required for strict mode | ✓ |
| Vietnamese `description` strings | ✓ | ✓ | ✓ |

The 7 SmartRent tools currently use only the green-row features. They
already write to the lowest common denominator — no schema rewriting is
needed when adding a new provider.

---

## 8. Tool result envelope — also handled by the provider

Tools return a plain `Dict[str, Any]` such as `{"status": "success",
"listings": [...]}`. Each provider serializes it differently:

| Provider | Envelope sent back to the model |
|---|---|
| Gemini | `Part.from_function_response(name=tool_name, response=result_dict)` |
| OpenAI | `{"role": "tool", "tool_call_id": id, "content": json.dumps(result_dict)}` |
| Anthropic | `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": json.dumps(result_dict)}]}` |

The dict is identical in all three; the provider wraps it. The orchestrator
hands the provider a `ToolResult(call_id, content_dict)` and never sees
the wrapping.

---

## 9. Migration cost summary

| Layer | Change | LOC |
|---|---|---|
| `app/ai/llm/types.py` (new file) | Define `LLMMessage`, `ToolCall`, `ToolDefinition`, `LLMResponse`, `LLMStreamChunk`, `CacheHandle` | ~50 |
| `app/ai/llm/provider.py` (new file) | `LLMProvider` ABC | ~40 |
| `app/ai/llm/gemini_provider.py` (new file) | Refactor existing `LLMGateway` body into a `GeminiProvider(LLMProvider)` class | ~250 |
| `app/ai/llm/gateway.py` | Becomes a factory that returns the configured provider; or removed | -100 / +20 |
| `app/agent/tools/base_tool.py` | Replace `to_function_declaration()` abstract method with `get_parameters_schema()` | ~10 modified |
| `app/agent/tools/*.py` (7 files) | Drop `FunctionDeclaration` wrapper; return raw schema dict | ~5 each = 35 |
| `app/agent/tools/registry.py` | `get_tool()` (returns Vertex `Tool`) → `get_definitions()` (returns `List[ToolDefinition]`) | ~15 modified |
| `app/agent/orchestrator.py` | Replace Vertex `Content`/`Part` construction and function-call extraction with `LLMMessage` and `ToolCall` types | ~50 modified |
| **Total** | mechanical refactor; no new dependencies | **~500 LOC** |

Once done, adding a new provider is a single file:
- `OpenAIProvider`: ~150 LOC, no other file changes
- `AnthropicProvider`: ~150 LOC, no other file changes

---

## 10. Comparison with off-the-shelf abstractions

| Option | Effort to adopt | Pros | Cons | Recommendation |
|---|---|---|---|---|
| **Custom abstraction (this design)** | 1-2 days | Full control, no extra deps, fits existing code | Implementation cost | ✓ Chosen |
| **LiteLLM** | 4-8 hours | 100+ providers free | Hides provider-specific features (caching nuances), extra dependency, normalizes everything to OpenAI shape | Considered, rejected — caching needs control |
| **LangChain (`BaseChatModel`)** | 1-2 days | Large ecosystem | Heavy framework, conflicts with existing Langfuse instrumentation, framework lock-in | Considered, rejected |
| **LlamaIndex (`LLM`)** | 1-2 days | Similar to LangChain | Same trade-offs | Considered, rejected |

The custom abstraction is preferred for SmartRent because:
1. Each provider's caching mechanism (Gemini `CachedContent`, Anthropic
   `cache_control` breakpoints, OpenAI implicit caching) is exposed
   explicitly via `CacheHandle` rather than hidden behind a uniform API.
2. Existing Langfuse spans in the orchestrator continue to work without
   adapting to a third-party framework's tracing model.
3. No new runtime dependency is introduced.

---

## 11. Defense-ready Q&A

### Q: Can the chatbot use OpenAI or Anthropic instead of Gemini?

> Yes. The orchestrator depends on an `LLMProvider` interface defined in
> [`app/ai/llm/provider.py`](../app/ai/llm/provider.py) using
> provider-agnostic types (`LLMMessage`, `ToolCall`, `ToolDefinition`).
> The current production implementation is `GeminiProvider`, which adapts
> the interface to the Vertex AI SDK. Adding `OpenAIProvider` or
> `AnthropicProvider` requires implementing the same interface — roughly
> 150 lines per provider — with no changes to the orchestrator, tools,
> registry, or service layer.

### Q: What about the tools? Don't they need to be rewritten per provider?

> Tool descriptions use JSON Schema, which is supported by all major
> providers — Gemini, OpenAI, and Anthropic all accept the same
> `{type, properties, required}` structure. What differs is only the
> envelope around the schema and the message format for tool calls and
> results. In our architecture, this envelope translation lives entirely
> inside the Provider class — roughly 10 lines per provider per format.
> The tool's business logic — parameter validation, backend HTTP calls,
> return value shaping — is identical across providers.

### Q: Why not use LangChain or LiteLLM?

> LangChain provides a similar abstraction but adds a large dependency
> surface and opinionated patterns that conflict with the per-request
> Langfuse observability we built. LiteLLM normalizes all calls to an
> OpenAI-shaped format, which would hide provider-specific caching
> primitives that are central to our cost and latency design. A custom
> 90-line interface (~50 lines for types + ~40 lines for the ABC) gave
> equivalent flexibility without framework lock-in.

### Q: What is the cost of the abstraction layer itself?

> Around 90 lines for the interface and types, and ~250 lines per
> provider implementation. The orchestrator becomes ~50 lines simpler
> because it no longer constructs Vertex-specific objects. Net code
> change is roughly neutral, with significantly improved testability —
> the orchestrator can now be unit-tested with a mock provider.

### Q: Are there schema features that don't transfer between providers?

> Yes — providers vary in support for advanced JSON Schema features
> like `oneOf`, `anyOf`, and OpenAI's strict-mode requirements. Our
> tools use only the lowest-common-denominator subset — basic types,
> enums, nested objects, and arrays — which is universally supported.
> This is intentional: production tool design generally avoids exotic
> schema features to maximize portability and minimize model confusion.

### Q: How is the active provider chosen at runtime?

> A configuration value `LLM_PROVIDER` (default `"gemini"`) is read
> from the environment at startup. A small factory in
> `app/ai/llm/gateway.py` returns the corresponding provider instance,
> which is held as a module-level singleton. Switching providers
> requires only a configuration change and an application restart;
> there is no source-code change for the swap itself.

---

## 12. References

- Cockburn, A. (2005). *Hexagonal Architecture (Ports and Adapters)*.
  Object Mentor / cockburn.us.
- Martin, R. C. (1996). *The Dependency Inversion Principle*. C++
  Report, vol. 8.
- Parnas, D. L. (1972). *On the Criteria to Be Used in Decomposing
  Systems into Modules*. Communications of the ACM, 15(12).
- JSON Schema Specification, Draft 2020-12.
  https://json-schema.org/specification.html
- Google Vertex AI — Function calling reference.
- OpenAI — Function calling and tool use guide.
- Anthropic — Tool use documentation.

---

## 13. Implementation status

| Component | Status |
|---|---|
| Diagram 1 (current architecture) | Reflects code on `feature/chat-streaming` branch |
| Diagram 2 (target architecture) | Designed; not yet implemented |
| Stable-prefix system instruction | ✓ Implemented ([orchestrator.py](../app/agent/orchestrator.py)) |
| Bilingual language rule | ✓ Implemented |
| Provider abstraction (types + ABC) | Pending — see Migration cost summary |
| `GeminiProvider` extraction from `LLMGateway` | Pending |
| `CachedContent` registration | Pending — to live in `GeminiProvider.register_cache()` |
| `OpenAIProvider` | Future work |
| `AnthropicProvider` | Future work |
