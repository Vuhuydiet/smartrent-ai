"""
Agent Orchestrator — the core agent loop.

Flow per request
----------------
1. Create Langfuse trace
2. RAG: retrieve per-query context (location codes, amenity IDs, relevant FAQ)
3. Build Gemini model with fully resolved system_instruction (base + static prefix + dynamic context)
4. Convert prior history to Vertex AI Content objects
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
import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from google.genai import types  # type: ignore[import]

from app.agent.rag.retriever import RAGRetriever
from app.agent.tools.registry import ToolRegistry
from app.ai.llm.gateway import LLMGateway, get_gateway
from app.core.config import settings
from app.dto.chat import ChatMessage, LastListingRef

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5  # safety cap — prevents infinite loops on misbehaving models
REQUEST_TIMEOUT_SECONDS = 120  # overall timeout for one orchestrator.run() call

# Token-budgeted sliding window — bounds LLM-bound history regardless of
# session length. Two complementary caps are enforced:
#   1. HISTORY_TOKEN_BUDGET — primary, character-based token estimate
#   2. MAX_HISTORY_MESSAGES — hard safety cap if estimation drifts
#
# Walking newest-first: always keep the current user message, then accumulate
# older messages until the token budget is exhausted. Entity continuity for
# follow-ups like "details of #2" is preserved separately via the
# `last_listings` field on ChatRequest, so we can drop old prose without
# losing referential meaning.
HISTORY_TOKEN_BUDGET = 4000  # ~3-4K tokens of history sent to LLM
MAX_HISTORY_MESSAGES = 20  # hard cap (safety)

# Listing-list compaction: assistant messages that enumerate listings as
# `1. [ID:35201] ... 2. [ID:35202] ...` are bulky and re-sent every turn.
# After the initial UI render, the listing IDs are the only thing the LLM
# needs for follow-up. We compact such messages in the LLM-bound history
# while keeping the FE-stored copy intact for the user-visible chat.
_LISTING_ID_RE = re.compile(r"\[ID:(\d+)\]")
_COMPACT_THRESHOLD = 2  # only compact when 2+ listing IDs present


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate. Conservative (~3 chars/token) — Vietnamese is
    slightly less efficient than English in Gemini's tokenizer, plus the
    extra +5 covers role prefix and message overhead per turn.
    """
    return (len(text) // 3) + 5


def _compact_assistant_message(content: str) -> str:
    """
    Compact assistant messages that enumerate listings. Preserves the leading
    summary sentence and the listing IDs (so the model can still resolve
    "cái thứ 2"-style references), drops the per-item titles/descriptions.

    Returns content unchanged when fewer than `_COMPACT_THRESHOLD` listing
    IDs are present.
    """
    ids = _LISTING_ID_RE.findall(content)
    if len(ids) < _COMPACT_THRESHOLD:
        return content

    # Find the first line that contains a [ID:...] — that's where the
    # numbered list begins. Take only the prose before that line as header.
    lines = content.split("\n")
    list_start = next(
        (i for i, line in enumerate(lines) if _LISTING_ID_RE.search(line)),
        len(lines),
    )
    header = "\n".join(lines[:list_start]).rstrip(" :\n").rstrip()
    if not header:
        header = "Đã trả về danh sách kết quả."

    return f"{header}\n[Đã hiển thị {len(ids)} tin: {', '.join(ids)}]"


def _trim_history(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    Token-budgeted sliding window over conversation history.

    Strategy:
      1. Always keep the current (last) user message — that's the query.
      2. Walk older messages newest-first, accumulating into the kept set
         until either the token budget or the hard message cap is reached.
      3. Ensure the trimmed window starts with a user message (Gemini chat
         history requires user-first alternation).

    The token budget is the primary cap; MAX_HISTORY_MESSAGES is a safety
    fallback in case the character-based estimate drifts (e.g. very long
    English passages tokenizing more efficiently than expected).
    """
    if not messages:
        return messages

    # Apply per-message compaction first so token accounting reflects what
    # actually goes to the LLM, not the FE-stored verbatim version.
    compacted = [
        ChatMessage(
            role=m.role,
            content=(
                _compact_assistant_message(m.content)
                if m.role == "assistant"
                else m.content
            ),
        )
        for m in messages
    ]

    current = compacted[-1]
    history = compacted[:-1]

    used_tokens = _estimate_tokens(current.content)
    kept: List[ChatMessage] = [current]

    for msg in reversed(history):
        if len(kept) >= MAX_HISTORY_MESSAGES:
            break
        msg_tokens = _estimate_tokens(msg.content)
        if used_tokens + msg_tokens > HISTORY_TOKEN_BUDGET:
            break
        used_tokens += msg_tokens
        kept.insert(0, msg)

    # Conversation must start with a user message (Gemini requirement).
    while kept and kept[0].role != "user":
        kept = kept[1:]

    return kept


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
3. Hỏi đáp về cách sử dụng nền tảng SmartRent: đăng tin, xem tin, liên hệ chủ nhà, thanh toán, gói VIP/membership, quản lý tài khoản, lưu tin yêu thích, báo cáo tin vi phạm, bộ lọc tìm kiếm, chia sẻ tin, thông báo, gia hạn tin, và mọi tính năng khác của nền tảng
4. Kinh nghiệm và lời khuyên về thuê/mua nhà tại Việt Nam, an toàn giao dịch
5. Các câu hỏi liên quan đến hợp đồng thuê nhà, pháp lý bất động sản cơ bản
6. Gợi ý BĐS phù hợp, thông tin tài khoản người dùng, tin đã lưu

QUY TẮC BẮT BUỘC:
- Nếu người dùng hỏi bất kỳ điều gì NGOÀI phạm vi trên (ví dụ: nấu ăn, thể thao, lập trình, toán học, giải trí, chính trị...) bạn PHẢI từ chối nhẹ nhàng bằng tiếng Việt và nhắc người dùng về những gì bạn có thể giúp.
- KHÔNG bao giờ cố gắng trả lời câu hỏi ngoài chủ đề, dù người dùng yêu cầu.
- Trả lời bằng đúng ngôn ngữ người dùng đang dùng. Nếu user viết tiếng Việt → trả lời tiếng Việt. Nếu user viết tiếng Anh → trả lời tiếng Anh. Match the user's language exactly: Vietnamese in → Vietnamese out, English in → English out.
- Khi có THÔNG TIN THAM KHẢO hoặc HƯỚNG DẪN SỬ DỤNG được cung cấp bên dưới, bạn PHẢI sử dụng thông tin đó để trả lời. KHÔNG ĐƯỢC nói "tôi không có thông tin" nếu thông tin đã được cung cấp.

SỬ DỤNG CÔNG CỤ:
- Khi người dùng muốn tìm BĐS → GỌI search_listings với tiêu chí phù hợp. Luôn truyền provinceCode khi người dùng đề cập tỉnh/thành. Dùng districtId (số nguyên) cho quận/huyện, productType cho loại BĐS.
- Khi người dùng hỏi chi tiết về một BĐS cụ thể (sau khi đã tìm thấy) → TỰ tra listingId từ kết quả search trước đó dựa trên tên, vị trí, hoặc thứ tự (ví dụ "cái đầu tiên", "phòng trọ ở Long Hòa") rồi GỌI get_listing_detail. KHÔNG BAO GIỜ hỏi lại user cung cấp ID.
- Khi người dùng hỏi thông tin liên hệ, số điện thoại, hoặc muốn liên hệ chủ nhà → GỌI get_listing_detail. Giao diện sẽ TỰ ĐỘNG hiển thị thẻ liên hệ từ dữ liệu trả về. Bạn CHỈ CẦN viết text ngắn gọn, ví dụ: "Đây là thông tin liên hệ của tin đăng này:" hoặc nếu contactAvailable=false thì nói "Chủ nhà chưa cung cấp thông tin liên hệ."
- Khi người dùng hỏi giá thị trường hoặc muốn so sánh giá → GỌI get_price_estimate.
- Giá tính bằng VND. Mặc định listingType="RENT" nếu không được chỉ định.
- Sau khi nhận kết quả tìm kiếm, viết 1-2 câu tổng quan ngắn gọn, sau đó LIỆT KÊ NGẮN GỌN danh sách kết quả theo thứ tự gồm listingId và tiêu đề, ví dụ:
  "Tìm thấy 90 kết quả ở Cần Thơ. Đây là {max_listings} BĐS phù hợp nhất:
  1. [ID:35201] Phòng trọ Bình Thạnh 17m²
  2. [ID:35202] Căn hộ Q1 50m²
  ..."
  KHÔNG liệt kê chi tiết (giá, diện tích, nội thất...) vì giao diện sẽ TỰ ĐỘNG hiển thị thẻ listing. Chỉ cần ID + tiêu đề ngắn để bạn có thể tra cứu chi tiết khi user hỏi "cái thứ 2", "trọ đầu tiên"...
- Khi user hỏi "chi tiết trọ thứ 2", "cái đầu tiên" → TRA listingId từ danh sách đã liệt kê ở tin nhắn trước rồi GỌI get_listing_detail. KHÔNG BAO GIỜ bịa listingId.

PHÂN TRANG:
- Khi người dùng nói "xem thêm", "tìm tiếp", "còn nữa không", "trang tiếp" → GỌI lại search_listings với cùng tiêu chí nhưng tăng page lên 1. Nhớ giữ nguyên tất cả filter từ lần search trước.
- Luôn cho user biết đang ở trang bao nhiêu và tổng số kết quả (ví dụ: "Trang 2/5, tổng 25 kết quả").

SO SÁNH:
- Khi người dùng muốn so sánh 2 hoặc nhiều BĐS → GỌI get_listing_detail cho TỪNG BĐS cần so sánh, sau đó trình bày bảng so sánh rõ ràng về: giá, diện tích, vị trí, số phòng, tiện nghi, nội thất.

SẮP XẾP:
- Khi người dùng muốn sắp xếp kết quả (giá thấp nhất, mới nhất, rẻ nhất...) → GỌI search_listings với sortBy phù hợp: PRICE_ASC (giá tăng), PRICE_DESC (giá giảm), NEWEST (mới nhất), OLDEST (cũ nhất).

TIỆN NGHI:
- Khi người dùng yêu cầu BĐS có tiện nghi cụ thể (WiFi, điều hòa, máy giặt, bãi đỗ xe...) → dùng amenityIds trong search_listings. Tra danh sách amenity ID từ thông tin hệ thống đã cung cấp.

ĐÁNH GIÁ GIÁ:
- Khi người dùng hỏi "giá này đắt hay rẻ?", "giá có hợp lý không?" về một BĐS cụ thể → GỌI get_listing_detail để lấy thông tin (giá, diện tích, vị trí, loại BĐS), sau đó GỌI get_price_estimate với askingPrice = giá BĐS đó để đánh giá so với thị trường.

LỊCH SỬ GIÁ:
- Khi người dùng hỏi "tin này có giảm giá không?", "lịch sử giá", "giá thay đổi thế nào?" → GỌI get_price_history với action="history" và listingId.
- Khi người dùng muốn xem thống kê giá (giá thấp nhất, cao nhất, trung bình) → GỌI get_price_history với action="statistics" và listingId.
- Khi người dùng hỏi "có tin nào mới giảm giá không?", "tin giảm giá gần đây" → GỌI get_price_history với action="recent_changes" và daysBack phù hợp (mặc định 7).
- Sau khi nhận danh sách listingIds từ recent_changes, GỌI search_listings hoặc get_listing_detail để lấy thông tin chi tiết các tin đó.

TÌM THEO VỊ TRÍ GẦN:
- Khi người dùng muốn tìm BĐS quanh một vị trí cụ thể (gần trường, gần chợ, tọa độ GPS) → dùng latitude, longitude và radiusKm trong search_listings.

TIN MỚI ĐĂNG:
- Khi người dùng muốn xem tin mới đăng gần đây → dùng postedWithinDays (ví dụ: 7 = trong 7 ngày qua) hoặc sortBy=NEWEST.

GỢI Ý BĐS:
- Khi người dùng muốn gợi ý, đề xuất, hoặc nói "gợi ý cho tôi", "tìm giúp tôi phòng phù hợp", "đề xuất phòng" → GỌI get_recommendations.
- Nếu người dùng đề cập một BĐS cụ thể và muốn tìm tương tự → GỌI get_recommendations với listingId.
- Nếu hệ thống trả về lỗi chưa đăng nhập → sử dụng thông tin sở thích từ cuộc hội thoại và dùng search_listings thay thế.

THÔNG TIN TÀI KHOẢN:
- Khi người dùng hỏi về tài khoản, hồ sơ cá nhân → GỌI get_user_info với infoType="profile".
- Khi người dùng hỏi về gói dịch vụ, membership, VIP → GỌI get_user_info với infoType="membership".
- Khi người dùng hỏi về tin đã lưu, tin yêu thích → GỌI get_user_info với infoType="saved_listings".
- Nếu chưa đăng nhập → nhắc người dùng đăng nhập.

LƯU TIN:
- Khi người dùng muốn lưu tin, thêm vào yêu thích → GỌI save_listing với action="save" và listingId từ kết quả search.
- Khi người dùng muốn bỏ lưu → GỌI save_listing với action="unsave".
- Nếu chưa đăng nhập → nhắc người dùng đăng nhập.

HỎI LẠI KHI THIẾU THÔNG TIN:
- Nếu người dùng yêu cầu tìm BĐS nhưng KHÔNG nêu vị trí (tỉnh/thành, quận/huyện) → HỎI LẠI vị trí trước khi search. Không bao giờ search mà không có ít nhất một tiêu chí vị trí hoặc keyword.
- Nếu yêu cầu quá mơ hồ (ví dụ: "tìm phòng") → hỏi thêm: vị trí nào? ngân sách bao nhiêu?

KHÔNG BỊA THÔNG TIN:
- Bạn KHÔNG biết hệ thống đang có listing ở những thành phố nào. KHÔNG BAO GIỜ tự liệt kê hay khẳng định danh sách thành phố có sẵn.
- Khi người dùng hỏi "hệ thống có listing ở đâu?" → trả lời: "Bạn có thể thử tìm kiếm ở tỉnh/thành phố cụ thể, tôi sẽ kiểm tra giúp bạn."
- KHÔNG BAO GIỜ bịa đặt thông tin mà bạn không có dữ liệu.

KHÔNG CÓ KẾT QUẢ:
- Khi search_listings trả về 0 kết quả → BÁO THẲNG cho user: "Hiện tại không tìm thấy BĐS phù hợp tại [vị trí]."
- KHÔNG TỰ Ý tìm ở thành phố/vị trí khác khi user đã chỉ định rõ vị trí.
- Chỉ gợi ý mở rộng tìm kiếm nếu user đồng ý: "Bạn có muốn tôi thử tìm ở khu vực lân cận không?"\
"""


@functools.lru_cache(maxsize=8)
def _build_static_system_instruction(
    base_prompt: str,
    static_prefix: str,
    max_listings: int,
) -> str:
    """
    Assemble the STABLE system instruction reused across every request.

    This deliberately excludes per-request data (RAG context, last_listings) so
    the resulting string is byte-identical across calls — letting Gemini's
    implicit cache hit and enabling explicit CachedContent registration later.

    Structure:
        [Base rules + tool usage guide]            — from Langfuse or local fallback
        [Static RAG prefix: provinces + amenities] — same for every user/turn

    Cached via lru_cache keyed on (base_prompt, static_prefix, max_listings).
    Both inputs are process-stable, so the cache effectively holds one entry.
    """
    resolved = base_prompt.replace("{{max_listings}}", str(max_listings))
    resolved = resolved.replace("{max_listings}", str(max_listings))
    parts = [resolved]
    if static_prefix:
        parts.append(static_prefix)
    return "\n\n".join(parts)


def _build_dynamic_context_block(
    dynamic_context: str,
    last_listings: Optional[List[LastListingRef]] = None,
) -> str:
    """
    Build the per-request context block prepended to the user's first message.

    Kept OUT of system_instruction so the system_instruction stays cache-stable.
    The model still sees this content — just on the user side of the turn.

    Returns an empty string when there is nothing to inject.
    """
    parts: List[str] = []

    if dynamic_context:
        parts.append(f"THÔNG TIN BỔ SUNG CHO TRUY VẤN NÀY:\n{dynamic_context}")

    if last_listings:
        lines = ["KẾT QUẢ TÌM KIẾM GẦN NHẤT (dùng listingId khi user hỏi chi tiết):"]
        for ref in last_listings:
            lines.append(f"  {ref.position}. listingId={ref.listingId} — {ref.title}")
        parts.append("\n".join(lines))

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
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        last_listings: Optional[List[LastListingRef]] = None,
    ) -> AgentResult:
        """
        Execute the full agent pipeline for one user turn.

        Args:
            messages: Full conversation history (the last item must be role=user).
            session_id: Optional ID for grouping traces in Langfuse.
            user_id: Optional authenticated user ID (for personalized features).
            auth_token: Optional JWT token (for calling authenticated backend APIs).

        Returns:
            AgentResult with the assistant message, optional listings, and metadata.
        """
        original_count = len(messages)
        messages = _trim_history(messages)
        kept_tokens = sum(_estimate_tokens(m.content) for m in messages)
        if len(messages) < original_count:
            logger.info(
                "Token-budgeted window: trimmed history %d → %d messages "
                "(~%d tokens, budget=%d)",
                original_count,
                len(messages),
                kept_tokens,
                HISTORY_TOKEN_BUDGET,
            )
        user_message = messages[-1].content

        try:
            return await asyncio.wait_for(
                self._run_pipeline(
                    messages,
                    user_message,
                    session_id,
                    user_id,
                    auth_token,
                    last_listings,
                ),
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
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        last_listings: Optional[List[LastListingRef]] = None,
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

            # ── 3. Fetch prompt from Langfuse (cached) & build STABLE system instruction
            base_prompt, prompt_obj = self._gateway.get_prompt(
                "smartrent-chat-system",
                label="production",
                fallback=_SYSTEM_BASE,
            )
            system_instruction = _build_static_system_instruction(
                base_prompt or _SYSTEM_BASE,
                self._static_prefix,
                settings.MAX_LISTINGS_RETURN,
            )
            # ── 4. Build chat session with prior history ───────────────────
            history = _to_vertex_history(messages[:-1])
            tools_obj = self._tools.get_tool()
            # Try to use Vertex CachedContent for the stable prefix.
            # Returns None on first call / cache miss / unavailable — start_chat
            # then falls back to passing system_instruction inline as before.
            cached_content = await self._gateway.get_or_create_cache(
                model_name=settings.GEMINI_CHAT_MODEL,
                system_instruction=system_instruction,
                tools=tools_obj,
            )
            chat = self._gateway.start_chat(
                model_name=settings.GEMINI_CHAT_MODEL,
                system_instruction=system_instruction,
                tools=tools_obj,
                history=history,
                cached_content=cached_content,
            )

            # ── 5. Agentic loop ────────────────────────────────────────────
            # Per-request RAG context rides on the user message so the
            # system_instruction prefix stays cache-stable across requests.
            context_block = _build_dynamic_context_block(dynamic_context, last_listings)
            message_to_send: Any = (
                f"{context_block}\n\n---\n\n{user_message}"
                if context_block
                else user_message
            )
            tools_used: List[str] = []
            all_raw_listings: List[Dict[str, Any]] = []
            response: Any = None

            for round_num in range(MAX_TOOL_ROUNDS):
                logger.info("Agent loop — round %d/%d", round_num + 1, MAX_TOOL_ROUNDS)
                span_name = (
                    "llm-initial" if round_num == 0 else f"llm-round-{round_num}"
                )
                response = await self._gateway.send_message(
                    chat,
                    message_to_send,
                    trace,
                    span_name,
                    prompt=prompt_obj if round_num == 0 else None,
                )

                # Collect every function call the model requested in this round
                parts = response.candidates[0].content.parts
                function_calls = [
                    p.function_call
                    for p in parts
                    if p.function_call is not None and p.function_call.name
                ]

                if not function_calls:
                    logger.info(
                        "Round %d: no function calls — final LLM response.",
                        round_num + 1,
                    )
                    break

                # Build execution context for tools that need user identity
                tool_context: Optional[Dict[str, Any]] = None
                if user_id or auth_token:
                    tool_context = {
                        "user_id": user_id,
                        "auth_token": auth_token,
                    }

                # Execute all tool calls and build a single tool-response Content
                tool_response_parts: List[Any] = []
                for fc in function_calls:
                    args = dict(fc.args) if fc.args else {}
                    logger.info("Calling tool '%s' args=%s", fc.name, list(args.keys()))

                    tool_span = trace.span(name=f"tool-{fc.name}", input=args)
                    result = await self._tools.execute(
                        fc.name, args, context=tool_context
                    )

                    # Extract raw listings for the API response payload.
                    # The compact summary stays in `result` and is sent back to the LLM.
                    if result.get("status") == "success":
                        if "_raw_listings" in result:
                            all_raw_listings.extend(result.pop("_raw_listings"))
                        if "_raw_listing" in result:
                            all_raw_listings.append(result.pop("_raw_listing"))

                    tool_span.end(
                        output={
                            "status": result.get("status"),
                            "count": result.get("count"),
                        }
                    )
                    tools_used.append(fc.name)

                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response=result,
                        )
                    )

                # Feed all tool results back to the model in one turn
                # New SDK: pass the list of Parts directly; chat session adds role=user
                message_to_send = tool_response_parts

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

    # ------------------------------------------------------------------
    # Streaming entry point (SSE — ChatService.process_chat_stream)
    # ------------------------------------------------------------------

    async def run_stream(  # noqa: C901
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        last_listings: Optional[List[LastListingRef]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming variant of run() — yields SSE-shaped events as the agent
        progresses through tool calls and LLM responses.

        Events yielded (dicts with "event" + "data"):
            {"event": "status",   "data": {"phase": "thinking", "round": int}}
            {"event": "status",   "data": {"phase": "tool_call", "tool": str}}
            {"event": "status",   "data": {"phase": "tool_result", "tool": str, "status": str, "error"?: str}}
            {"event": "text",     "data": {"delta": str}}
            {"event": "listings", "data": {...listings payload...}}
            {"event": "done",     "data": {"metadata": {...}, "tools_used": [...]}}
            {"event": "error",    "data": {"message": str}}
        """
        original_count = len(messages)
        messages = _trim_history(messages)
        kept_tokens = sum(_estimate_tokens(m.content) for m in messages)
        if len(messages) < original_count:
            logger.info(
                "Token-budgeted window: trimmed history %d → %d messages "
                "(~%d tokens, budget=%d)",
                original_count,
                len(messages),
                kept_tokens,
                HISTORY_TOKEN_BUDGET,
            )
        user_message = messages[-1].content

        trace = self._gateway.create_trace(
            name="chat-stream",
            session_id=session_id,
            input={"message": user_message},
            metadata={
                "model": settings.GEMINI_CHAT_MODEL,
                "turns_original": original_count,
                "turns_used": len(messages),
                "history_tokens_est": kept_tokens,
                "streamed": True,
            },
        )

        tools_used: List[str] = []
        all_raw_listings: List[Dict[str, Any]] = []

        try:
            # ── RAG ──────────────────────────────────────────────────────
            rag_span = trace.span(name="rag-retrieve", input={"query": user_message})
            dynamic_context = self._rag.retrieve(user_message)
            rag_span.end(
                output={
                    "has_context": bool(dynamic_context),
                    "context_length": len(dynamic_context),
                }
            )

            # ── System instruction + model + chat ────────────────────────
            base_prompt, prompt_obj = self._gateway.get_prompt(
                "smartrent-chat-system",
                label="production",
                fallback=_SYSTEM_BASE,
            )
            system_instruction = _build_static_system_instruction(
                base_prompt or _SYSTEM_BASE,
                self._static_prefix,
                settings.MAX_LISTINGS_RETURN,
            )
            history = _to_vertex_history(messages[:-1])
            tools_obj = self._tools.get_tool()
            cached_content = await self._gateway.get_or_create_cache(
                model_name=settings.GEMINI_CHAT_MODEL,
                system_instruction=system_instruction,
                tools=tools_obj,
            )
            chat = self._gateway.start_chat(
                model_name=settings.GEMINI_CHAT_MODEL,
                system_instruction=system_instruction,
                tools=tools_obj,
                history=history,
                cached_content=cached_content,
            )

            # ── Agentic loop ─────────────────────────────────────────────
            # Per-request RAG context rides on the user message so the
            # system_instruction prefix stays cache-stable across requests.
            context_block = _build_dynamic_context_block(dynamic_context, last_listings)
            message_to_send: Any = (
                f"{context_block}\n\n---\n\n{user_message}"
                if context_block
                else user_message
            )
            any_text_streamed = False

            for round_num in range(MAX_TOOL_ROUNDS):
                logger.info("Stream loop — round %d/%d", round_num + 1, MAX_TOOL_ROUNDS)
                yield {
                    "event": "status",
                    "data": {"phase": "thinking", "round": round_num + 1},
                }

                span_name = (
                    "llm-initial" if round_num == 0 else f"llm-round-{round_num}"
                )
                function_calls: List[Any] = []

                async for chunk in self._gateway.send_message_stream(
                    chat,
                    message_to_send,
                    trace,
                    span_name=span_name,
                    prompt=prompt_obj if round_num == 0 else None,
                ):
                    if chunk["type"] == "text_delta":
                        any_text_streamed = True
                        yield {
                            "event": "text",
                            "data": {"delta": chunk["delta"]},
                        }
                    elif chunk["type"] == "final":
                        function_calls = chunk["function_calls"]

                if not function_calls:
                    logger.info(
                        "Round %d: no function calls — end of stream.", round_num + 1
                    )
                    break

                # ── Tool execution ──────────────────────────────────────
                tool_context: Optional[Dict[str, Any]] = None
                if user_id or auth_token:
                    tool_context = {
                        "user_id": user_id,
                        "auth_token": auth_token,
                    }

                tool_response_parts: List[Any] = []
                for fc in function_calls:
                    args = dict(fc.args) if fc.args else {}
                    logger.info("Calling tool '%s' args=%s", fc.name, list(args.keys()))
                    yield {
                        "event": "status",
                        "data": {"phase": "tool_call", "tool": fc.name},
                    }

                    tool_span = trace.span(name=f"tool-{fc.name}", input=args)
                    result = await self._tools.execute(
                        fc.name, args, context=tool_context
                    )

                    if result.get("status") == "success":
                        if "_raw_listings" in result:
                            all_raw_listings.extend(result.pop("_raw_listings"))
                        if "_raw_listing" in result:
                            all_raw_listings.append(result.pop("_raw_listing"))

                    tool_span.end(
                        output={
                            "status": result.get("status"),
                            "count": result.get("count"),
                        }
                    )
                    tools_used.append(fc.name)

                    tool_result_data: Dict[str, Any] = {
                        "phase": "tool_result",
                        "tool": fc.name,
                        "status": result.get("status"),
                    }
                    # Surface the error message to FE when a tool fails so
                    # the UI can show it instead of a generic "tool failed".
                    if result.get("status") == "error" and result.get("error"):
                        tool_result_data["error"] = str(result["error"])[:500]
                    yield {"event": "status", "data": tool_result_data}

                    tool_response_parts.append(
                        types.Part.from_function_response(name=fc.name, response=result)
                    )

                # New SDK: pass the list of Parts directly; chat session adds role=user
                message_to_send = tool_response_parts

            # ── Fallback text if model ended without prose ───────────────
            if not any_text_streamed:
                fallback = "Tôi đã xử lý yêu cầu của bạn. Hãy xem kết quả bên dưới."
                yield {"event": "text", "data": {"delta": fallback}}

            # ── Listings payload ────────────────────────────────────────
            listings_payload = _build_listings_payload(all_raw_listings)
            if listings_payload:
                yield {"event": "listings", "data": listings_payload}

            # ── Done ────────────────────────────────────────────────────
            metadata = {
                "model": settings.GEMINI_CHAT_MODEL,
                "tools_used": tools_used,
                "rag_context_injected": bool(dynamic_context),
            }
            trace.update(metadata={"tools_used": tools_used})
            yield {
                "event": "done",
                "data": {"metadata": metadata, "tools_used": tools_used},
            }

        except asyncio.CancelledError:
            logger.info("run_stream cancelled (client disconnect)")
            trace.update(output={"cancelled": True})
            raise
        except Exception as e:
            trace.update(output={"error": str(e)})
            logger.error("AgentOrchestrator.run_stream failed: %s", e, exc_info=True)
            yield {
                "event": "error",
                "data": {
                    "message": "Có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại."
                },
            }


# ---------------------------------------------------------------------------
# Private helpers (module-level for clarity)
# ---------------------------------------------------------------------------


def _to_vertex_history(messages: List[ChatMessage]) -> List[Any]:
    """Convert ChatMessage list to google-genai Content objects."""
    history: List[Any] = []
    for msg in messages:
        role = "user" if msg.role == "user" else "model"
        history.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
        )
    return history


def _extract_text(response: Any) -> str:
    """
    Safely pull text from a Vertex AI GenerateContentResponse.

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

    Deduplicates by listingId — when the same listing appears from both
    search_listings and get_listing_detail, the later (more detailed) version wins.

    Returns None when no listings were found (non-search conversations).
    """
    if not raw_listings:
        return None

    # Deduplicate: later entries override earlier ones (detail > search card)
    seen: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for listing in raw_listings:
        lid = str(listing.get("listingId", id(listing)))
        if lid not in seen:
            order.append(lid)
        seen[lid] = listing  # later version wins (get_listing_detail overrides search)

    unique = [seen[lid] for lid in order]
    top = unique[: settings.MAX_LISTINGS_RETURN]
    return {
        "listings": top,
        "totalCount": len(top),
        "selectedFromTotal": len(unique),
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
