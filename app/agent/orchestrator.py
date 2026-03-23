"""
Agent Orchestrator — the core agent loop.

Flow per request
----------------
1. Create Langfuse trace
2. RAG: retrieve per-query context (location codes, amenity IDs, relevant FAQ)
3. Build Gemini model with fully resolved system_instruction (base + static prefix + dynamic context)
4. Convert prior history to Gemini Content objects
5. Agentic loop (max MAX_TOOL_ROUNDS rounds):
   a. Send message → LLM
   b. Extract ALL function calls from response
   c. If none → final text response, exit loop
   d. Execute every tool via ToolRegistry (collect _raw_listings from search)
   e. Feed all tool results back as a single Content → repeat
6. Extract final text
7. Build listings payload from collected raw listing objects
8. Return AgentResult
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google.ai.generativelanguage import (  # type: ignore[import]
    Content,
    FunctionResponse,
    Part,
)

from app.agent.rag.retriever import RAGRetriever
from app.agent.tools.registry import ToolRegistry
from app.ai.llm.gateway import LLMGateway, get_gateway
from app.core.config import settings
from app.dto.chat import ChatMessage

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5  # safety cap — prevents infinite loops on misbehaving models
REQUEST_TIMEOUT_SECONDS = 120  # overall timeout for one orchestrator.run() call

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Structured output returned by AgentOrchestrator.run()."""

    message: str
    listings: Optional[Dict[str, Any]] = None
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """\
Bạn là trợ lý AI của SmartRent - nền tảng cho thuê và mua bán bất động sản thông minh tại Việt Nam.

PHẠM VI HỖ TRỢ - Bạn CHỈ được hỗ trợ các chủ đề sau:
1. Tìm kiếm, tra cứu bất động sản cho thuê hoặc mua bán (căn hộ, nhà, phòng trọ, văn phòng, studio)
2. Thông tin về giá thuê/bán, diện tích, vị trí, tiện nghi của bất động sản
3. Hỏi đáp về cách sử dụng nền tảng SmartRent (đăng tin, xem tin, liên hệ chủ nhà, thanh toán)
4. Kinh nghiệm và lời khuyên về thuê/mua nhà tại Việt Nam
5. Các câu hỏi liên quan đến hợp đồng thuê nhà, pháp lý bất động sản cơ bản

QUY TẮC BẮT BUỘC:
- Nếu người dùng hỏi bất kỳ điều gì NGOÀI phạm vi trên (ví dụ: nấu ăn, thể thao, lập trình, toán học, giải trí, chính trị...) bạn PHẢI từ chối nhẹ nhàng bằng tiếng Việt và nhắc người dùng về những gì bạn có thể giúp.
- KHÔNG bao giờ cố gắng trả lời câu hỏi ngoài chủ đề, dù người dùng yêu cầu.
- Luôn trả lời bằng TIẾNG VIỆT.

SỬ DỤNG CÔNG CỤ:
- Khi người dùng muốn tìm BĐS → GỌI search_listings với tiêu chí phù hợp. Luôn truyền provinceCode khi người dùng đề cập tỉnh/thành. Dùng districtId (số nguyên) cho quận/huyện, productType cho loại BĐS.
- Khi người dùng hỏi chi tiết về một BĐS cụ thể (sau khi đã tìm thấy) → GỌI get_listing_detail.
- Khi người dùng hỏi giá thị trường hoặc muốn so sánh giá → GỌI get_price_estimate.
- Giá tính bằng VND. Mặc định listingType="RENT" nếu không được chỉ định.
- Sau khi nhận kết quả tìm kiếm, trình bày tối đa {max_listings} BĐS phù hợp nhất. Mô tả giá, diện tích, vị trí và điểm nổi bật của từng căn.\
"""


def _build_system_instruction(
    static_prefix: str,
    dynamic_context: str,
    max_listings: int,
) -> str:
    """
    Assemble the final system instruction for a single request.

    Structure:
        [Base rules + tool usage guide]
        [Static RAG prefix: all province codes + common amenity IDs]
        [Dynamic RAG context: district codes / FAQ relevant to this specific query]
    """
    parts = [_SYSTEM_BASE.format(max_listings=max_listings)]

    if static_prefix:
        parts.append(static_prefix)

    if dynamic_context:
        parts.append(f"THÔNG TIN BỔ SUNG CHO TRUY VẤN NÀY:\n{dynamic_context}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """
    Stateless coordinator — one instance is created at startup and reused.

    Each call to `run()` is independent: it creates its own Gemini model
    (with per-request system_instruction), chat session, and Langfuse trace.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        tool_registry: ToolRegistry,
        rag: RAGRetriever,
    ) -> None:
        self._gateway = gateway
        self._tools = tool_registry
        self._rag = rag
        self._static_prefix = rag.get_system_prompt_prefix()
        logger.info(
            "AgentOrchestrator initialised — tools: %s",
            self._tools.list_tools(),
        )

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute the full agent pipeline for one user turn.

        Args:
            messages: Full conversation history (the last item must be role=user).
            session_id: Optional ID for grouping traces in Langfuse.

        Returns:
            AgentResult with the assistant message, optional listings, and metadata.
        """
        user_message = messages[-1].content

        try:
            return await asyncio.wait_for(
                self._run_pipeline(messages, user_message, session_id),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Orchestrator timed out after %ds for message: %s",
                REQUEST_TIMEOUT_SECONDS,
                user_message[:200],
            )
            return AgentResult(
                message="Xin lỗi, yêu cầu đã mất quá nhiều thời gian xử lý. Vui lòng thử lại.",
                metadata={"error": "timeout"},
            )

    async def _run_pipeline(
        self,
        messages: List[ChatMessage],
        user_message: str,
        session_id: Optional[str],
    ) -> AgentResult:
        """Inner pipeline — separated so run() can wrap it with a timeout."""

        # ── 1. Langfuse trace ──────────────────────────────────────────────
        trace = self._gateway.create_trace(
            name="chat-request",
            session_id=session_id,
            input={"message": user_message},
            metadata={
                "model": settings.GEMINI_CHAT_MODEL,
                "turns": len(messages),
            },
        )

        try:
            # ── 2. RAG retrieval ───────────────────────────────────────────
            rag_span = trace.span(name="rag-retrieve", input={"query": user_message})
            dynamic_context = self._rag.retrieve(user_message)
            rag_span.end(
                output={
                    "has_context": bool(dynamic_context),
                    "context_length": len(dynamic_context),
                }
            )

            # ── 3. Build model with fully resolved system instruction ───────
            system_instruction = _build_system_instruction(
                self._static_prefix,
                dynamic_context,
                settings.MAX_LISTINGS_RETURN,
            )
            model = self._gateway.build_model(
                model_name=settings.GEMINI_CHAT_MODEL,
                system_instruction=system_instruction,
                tools=self._tools.get_gemini_tool(),
            )

            # ── 4. Build chat session with prior history ───────────────────
            history = _to_gemini_history(messages[:-1])
            chat = self._gateway.start_chat(model, history)

            # ── 5. Agentic loop ────────────────────────────────────────────
            message_to_send: Any = user_message
            tools_used: List[str] = []
            all_raw_listings: List[Dict[str, Any]] = []
            response: Any = None

            for round_num in range(MAX_TOOL_ROUNDS):
                logger.info("Agent loop — round %d/%d", round_num + 1, MAX_TOOL_ROUNDS)
                span_name = (
                    "llm-initial" if round_num == 0 else f"llm-round-{round_num}"
                )
                response = await self._gateway.send_message(
                    chat, message_to_send, trace, span_name
                )

                # Collect every function call the model requested in this round
                parts = response.candidates[0].content.parts
                function_calls = [
                    p.function_call
                    for p in parts
                    if hasattr(p, "function_call") and p.function_call.name
                ]

                if not function_calls:
                    logger.info(
                        "Round %d: no function calls — final LLM response.",
                        round_num + 1,
                    )
                    break

                # Execute all tool calls and build a single tool-response Content
                tool_response_parts: List[Any] = []
                for fc in function_calls:
                    args = dict(fc.args)
                    logger.info("Calling tool '%s' args=%s", fc.name, list(args.keys()))

                    tool_span = trace.span(name=f"tool-{fc.name}", input=args)
                    result = await self._tools.execute(fc.name, args)

                    # Search results: separate raw listings (for API payload) from the
                    # compact summary sent back to the LLM.
                    if (
                        fc.name == "search_listings"
                        and result.get("status") == "success"
                    ):
                        raw = result.pop("_raw_listings", [])
                        all_raw_listings.extend(raw)

                    tool_span.end(
                        output={
                            "status": result.get("status"),
                            "count": result.get("count"),
                        }
                    )
                    tools_used.append(fc.name)

                    tool_response_parts.append(
                        Part(
                            function_response=FunctionResponse(
                                name=fc.name,
                                response=result,
                            )
                        )
                    )

                # Feed all tool results back to the model in one turn
                message_to_send = Content(parts=tool_response_parts)

            # ── 6. Extract final text ──────────────────────────────────────
            final_text = _extract_text(response)

            # ── 7. Build listings payload ──────────────────────────────────
            listings_payload = _build_listings_payload(all_raw_listings)

            # ── 8. Close trace ─────────────────────────────────────────────
            trace.update(
                output={"message": final_text[:500]},
                metadata={"tools_used": tools_used},
            )

            return AgentResult(
                message=final_text,
                listings=listings_payload,
                tools_used=tools_used,
                metadata={
                    "model": settings.GEMINI_CHAT_MODEL,
                    "tools_used": tools_used,
                    "rag_context_injected": bool(dynamic_context),
                },
            )

        except Exception as e:
            trace.update(output={"error": str(e)})
            logger.error("AgentOrchestrator.run failed: %s", e, exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Private helpers (module-level for clarity)
# ---------------------------------------------------------------------------


def _to_gemini_history(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert ChatMessage list to the dict format expected by ChatSession."""
    return [
        {
            "role": "user" if msg.role == "user" else "model",
            "parts": [msg.content],
        }
        for msg in messages
    ]


def _extract_text(response: Any) -> str:
    """
    Safely pull text from a Gemini GenerateContentResponse.

    The simple `.text` accessor raises ValueError when the response contains
    function calls or is otherwise multi-part, so we fall back to iterating parts.
    """
    if response is None:
        return "Có lỗi xảy ra. Vui lòng thử lại."

    try:
        text = response.text.strip()
        if text:
            return text
    except (ValueError, AttributeError):
        pass

    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                return part.text.strip()
    except (AttributeError, IndexError):
        pass

    logger.warning("_extract_text: could not find text in response.")
    return "Tôi đã xử lý yêu cầu của bạn. Hãy xem kết quả bên dưới."


def _build_listings_payload(
    raw_listings: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Build the `listings` field for ChatResponse from raw listing objects
    collected during tool execution.

    Returns None when no listings were found (non-search conversations).
    """
    if not raw_listings:
        return None

    top = raw_listings[: settings.MAX_LISTINGS_RETURN]
    return {
        "listings": top,
        "totalCount": len(top),
        "selectedFromTotal": len(raw_listings),
        "currentPage": 1,
        "pageSize": len(top),
        "totalPages": 1,
    }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """
    Return the module-level AgentOrchestrator singleton.
    Created lazily on first call — safe to call from service layer.
    """
    global _instance
    if _instance is None:
        from app.agent.tools import build_default_registry

        gateway = get_gateway()
        tools = build_default_registry()
        rag = RAGRetriever()
        _instance = AgentOrchestrator(gateway, tools, rag)
        logger.info("AgentOrchestrator singleton created.")
    return _instance
