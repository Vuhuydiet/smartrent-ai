"""
Agent Orchestrator — built on the OpenAI Agents SDK.

Flow per request
----------------
1. Create Langfuse trace
2. RAG: retrieve per-query context (location codes, amenity IDs, FAQ)
3. Build a per-request `Agent` with:
       instructions = stable base prompt (cache-friendly across requests)
       model        = LiteLLM-backed model from agent_factory (provider-pluggable)
       tools        = @function_tool callables from app.agent.tools
4. Build conversation input as Responses-API messages; the dynamic RAG context
   is prepended to the user message so the system instructions stay byte-stable.
5. Run via Runner.run() (sync) or Runner.run_streamed() (SSE).
6. Tools push raw listings into ToolContext.collected_listings; the orchestrator
   builds the listings payload from that collection after the run.
7. Return AgentResult / yield SSE-shaped events.
"""

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from agents import Agent, Runner, RunResultStreaming  # type: ignore[import]
from agents.exceptions import MaxTurnsExceeded  # type: ignore[import]
from openai.types.responses import ResponseTextDeltaEvent  # type: ignore[import]

from app.agent.rag.retriever import RAGRetriever
from app.agent.suggestions import build_suggestions
from app.agent.tool_context import ToolContext
from app.agent.tools import get_chat_tools
from app.ai.llm.agent_factory import default_model_settings, make_model
from app.ai.llm.gateway import LLMGateway, get_gateway
from app.core.config import settings
from app.dto.chat import ChatMessage, LastListingRef

logger = logging.getLogger(__name__)

# Each tool round consumes ~2 turns (LLM call + tool result), so 12 is the
# rough equivalent of the old MAX_TOOL_ROUNDS=5 cap.
MAX_AGENT_TURNS = 12
REQUEST_TIMEOUT_SECONDS = 120

# Rough token budget for the conversation history sent to the LLM. Covers a
# few full turns of Vietnamese conversation while leaving headroom for the
# system prompt + RAG context. We use a char-based heuristic (no tokenizer
# dependency) — provider-side counting will differ but stays in the ballpark.
HISTORY_TOKEN_BUDGET = 6000
_CHARS_PER_TOKEN = 4  # rough average for Vietnamese text


def _estimate_tokens(text: str) -> int:
    """Approximate token count for a string (chars / 4)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _trim_history(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    Drop oldest messages until the running total fits under HISTORY_TOKEN_BUDGET.

    The last message (current user turn) is always kept. We accumulate from
    the end backwards and stop once we'd exceed the budget.
    """
    if not messages:
        return messages

    kept_reversed: List[ChatMessage] = []
    total = 0
    for msg in reversed(messages):
        cost = _estimate_tokens(msg.content)
        # Always keep the last message (current user turn) even if it alone
        # exceeds the budget — better to send something than nothing.
        if kept_reversed and total + cost > HISTORY_TOKEN_BUDGET:
            break
        kept_reversed.append(msg)
        total += cost
    return list(reversed(kept_reversed))


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
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """\
Bạn là trợ lý AI của SmartRent - nền tảng cho thuê và mua bán bất động sản thông minh tại Việt Nam.

PHONG CÁCH PHẢN HỒI — RẤT QUAN TRỌNG (ảnh hưởng tới UX):
- TRƯỚC khi gọi BẤT KỲ tool nào → viết 1 câu ngắn (5-15 từ tiếng Việt) giới thiệu việc bạn sắp làm. Vd:
  * "Để mình tìm thử các căn ở Bình Thạnh trong khoảng 5-10 triệu nhé..."
  * "Mình đang xem chi tiết tin số 35201..."
  * "Mình so sánh 3 tin này cho bạn..."
  * "Mình kiểm tra danh sách tin của bạn..."
  Câu này được STREAM ra ngay trước tool call → user thấy "AI đang gõ" thay vì im lặng. KHÔNG được bỏ qua bước này.
- SAU khi tool xong → tiếp tục viết bình thường (kết quả + giải thích).
- Quy tắc này áp dụng cho tool đầu tiên ở mỗi turn. Nếu cùng turn có nhiều tool liên tiếp (vd round 2 sau search), KHÔNG cần lặp lại — chỉ tự nhiên nối tiếp prose.

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
- Khi người dùng muốn tìm BĐS → GỌI search_listings với tiêu chí phù hợp. Luôn truyền provinceCode khi user đề cập tỉnh/thành. Dùng districtId (số nguyên) cho quận/huyện.
- LOẠI BĐS (productType vs productTypes) — quy tắc QUAN TRỌNG:
  * Từ CHÍNH XÁC, không mơ hồ → dùng `productType` đơn:
    - "căn hộ", "chung cư" → productType="APARTMENT"
    - "phòng trọ", "phòng đơn" (rõ là phòng nhỏ) → productType="ROOM"
    - "nhà nguyên căn", "nhà riêng" → productType="HOUSE"
    - "studio" → productType="STUDIO"
    - "văn phòng", "office" → productType="OFFICE"
  * Từ MƠ HỒ trong tiếng Việt → dùng `productTypes` mảng:
    - "nhà trọ", "trọ" → productTypes=["ROOM", "APARTMENT"]  (VN dùng "nhà trọ" cho cả phòng nhỏ lẫn căn hộ tầm trung — không rạch ròi như EN)
    - "thuê nhà", "tìm nhà" → productTypes=["ROOM", "APARTMENT", "HOUSE"]
  * Truy vấn HOÀN TOÀN MỞ → KHÔNG set cả productType lẫn productTypes:
    - "có gì cho thuê ở Q1?", "BĐS ở Bình Thạnh"
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
- Khi người dùng muốn so sánh 2-5 BĐS (vd "so sánh tin 1 và 3", "cái nào đáng thuê hơn?", "tin nào tốt nhất trong 3 cái này?") → GỌI compare_listings với mảng listingIds. Tự tra ID từ kết quả search trước đó dựa trên thứ tự ("cái thứ 2"), tên, hoặc vị trí — KHÔNG hỏi user cung cấp ID.
- compare_listings trả về `listings` (mảng row đã chuẩn hóa, có thêm `pricePerSqm`) + `callouts` (cheapest, largest, bestPricePerSqm, mostAmenities, priceRangeVnd, areaRangeSqm). Dùng các giá trị này viết bảng so sánh tiếng Việt + 1-2 câu kết luận khuyến nghị nên chọn cái nào và vì sao.
- KHÔNG gọi get_listing_detail riêng lẻ cho từng tin khi user yêu cầu so sánh — compare_listings đã fetch song song hiệu quả hơn.

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
- 1 tin → GỌI save_listing với action="save"/"unsave" và listingId từ context.
- Nhiều tin (vd "lưu cả 3 tin", "bỏ lưu hết tin trên") → GỌI bulk_save_listings với mảng listingIds + action. Tự suy IDs từ kết quả search trước, KHÔNG hỏi user.
- Khi user nói "cái thứ 2", "cái đầu" → tra ID từ danh sách hiển thị gần nhất rồi gọi save_listing.
- Nếu chưa đăng nhập → nhắc user đăng nhập.

TIN CỦA NGƯỜI DÙNG (OWNER DASHBOARD QUA CHAT):
- Khi user hỏi về tin của CHÍNH HỌ (vd "tin của tôi sao rồi", "tôi có bao nhiêu tin đang hiển thị", "tin nào sắp hết hạn", "có tin nào bị từ chối không") → GỌI my_listings_status với focus phù hợp (all|expiring|rejected|active).
- Phân biệt rõ với search_listings: my_listings_status chỉ trả về tin của user đang đăng nhập, dùng cho ngữ cảnh chủ tin/landlord. search_listings là tìm tin công khai.
- Tool trả về `statistics` (counts) + `needsAttention` (≤5 tin cần xử lý). Viết tiếng Việt: tóm tắt tổng (vd "Bạn có 12 tin: 8 đang hiển thị, 2 chờ duyệt, 1 bị từ chối, 1 sắp hết hạn"), sau đó liệt kê ngắn từng `needsAttention` item nếu có.

CẬP NHẬT GIÁ TIN CỦA MÌNH (OWNER):
- Khi user (chủ tin) muốn đổi giá tin của họ (vd "hạ giá tin 35201 xuống 5tr") → GỌI update_listing_price.
- Quy trình BẮT BUỘC 2 BƯỚC: (1) Gọi LẦN ĐẦU với confirmed=false để lấy preview (currentPrice + newPrice) → đọc lại cho user xác nhận; (2) CHỈ khi user trả lời rõ là đồng ý ("có", "ok", "đúng rồi") → gọi LẠI với confirmed=true để áp dụng.
- KHÔNG BAO GIỜ gọi với confirmed=true ngay từ đầu, dù user đã ghi giá trong câu đầu — vẫn phải xác nhận lại.
- newPrice luôn là số VND đầy đủ (5tr → 5000000).

THÔNG BÁO:
- Khi user hỏi "có gì mới không?", "đọc thông báo", "tóm tắt noti" → GỌI notifications_inbox. Nếu user nói rõ "đọc hết noti", "đánh dấu đã đọc" → set markAllRead=true.
- Tool trả về `byType` counts + `unread` count + `recent` items. Viết tiếng Việt: số noti mới + 1-3 noti gần nhất quan trọng.

BÁO CÁO TIN VI PHẠM:
- Khi user nói "tin này lừa đảo", "báo cáo tin", "nghi tin giả" → GỌI report_listing. Tự suy listingId từ context.
- Quy trình 2 BƯỚC: (1) Gọi LẦN ĐẦU với confirm=false để lấy danh sách lý do (`reasons`); đọc cho user chọn 1+ lý do; (2) Gọi LẠI với confirm=true + reasonIds đã chọn.

TRA CỨU ĐỊA CHỈ CŨ ↔ MỚI:
- Khi user hỏi tên quận/phường có chuyển đổi sau cải cách 1/7/2025 (vd "Quận Bình Thạnh giờ là phường nào?") → GỌI address_translator với `query` là tên user nói.
- Tool trả về `legacy` (province + districtId cũ nếu match) + `newMatches` (danh sách phường mới matching). Giải thích: 1 quận cũ thường tương ứng với nhiều phường mới.
- KHÔNG dùng tool này để search BĐS — chỉ để dịch tên địa chỉ.

HỎI LẠI KHI THIẾU THÔNG TIN:
- Nếu người dùng yêu cầu tìm BĐS nhưng KHÔNG nêu vị trí (tỉnh/thành, quận/huyện) → HỎI LẠI vị trí trước khi search. Không bao giờ search mà không có ít nhất một tiêu chí vị trí hoặc keyword.
- Nếu yêu cầu quá mơ hồ (ví dụ: "tìm phòng") → hỏi thêm: vị trí nào? ngân sách bao nhiêu?

KHÔNG BỊA THÔNG TIN:
- Bạn KHÔNG biết hệ thống đang có listing ở những thành phố nào. KHÔNG BAO GIỜ tự liệt kê hay khẳng định danh sách thành phố có sẵn.
- Khi người dùng hỏi "hệ thống có listing ở đâu?" → trả lời: "Bạn có thể thử tìm kiếm ở tỉnh/thành phố cụ thể, tôi sẽ kiểm tra giúp bạn."
- KHÔNG BAO GIỜ bịa đặt thông tin mà bạn không có dữ liệu.

KHÔNG CÓ KẾT QUẢ — QUY TRÌNH 2 BƯỚC:
- Bước 1: Khi search_listings trả về 0 kết quả VÀ bạn đã set `productType` HOẶC `productTypes` ở lần gọi đó → GỌI LẠI search_listings NGAY trong cùng turn với CÙNG mọi tiêu chí KHÁC nhưng BỎ HẾT cả productType và productTypes. Lý do: data thực tế có thể tag loại BĐS khác với từ user dùng. Đừng giải thích bước này cho user, cứ làm âm thầm.
- Bước 2: Nếu lần retry vẫn 0 kết quả (HOẶC lần đầu đã không có filter loại BĐS) → BÁO THẲNG cho user: "Hiện tại không tìm thấy BĐS phù hợp tại [vị trí] với tiêu chí này."
- KHÔNG TỰ Ý tìm ở thành phố/vị trí khác khi user đã chỉ định rõ vị trí.
- Chỉ gợi ý mở rộng tìm kiếm nếu user đồng ý: "Bạn có muốn tôi thử tìm ở khu vực lân cận hoặc nới giá không?"\
"""


@functools.lru_cache(maxsize=8)
def _build_static_instructions(
    base_prompt: str,
    static_prefix: str,
    max_listings: int,
) -> str:
    """
    Assemble the STABLE instructions reused across every request.

    Excludes per-request data (RAG context, last_listings) so the resulting
    string is byte-identical across calls — letting the provider's prompt
    cache (Gemini implicit cache, OpenAI prefix cache) hit.
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

    Kept OUT of the agent's instructions so the instructions stay cache-stable.
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


def _to_responses_input(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert ChatMessage list to Responses-API input items."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _tool_names_used(new_items: List[Any]) -> List[str]:
    """Extract names of tools the agent invoked, in order."""
    names: List[str] = []
    for item in new_items:
        if getattr(item, "type", None) == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            name = getattr(raw, "name", None)
            if name:
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Streaming UX: rich tool_call status events
# ---------------------------------------------------------------------------

# Short Vietnamese label per tool — surfaces in the SSE status event so the
# FE can show "Đang tìm BĐS ở Bình Thạnh..." instead of a generic spinner.
# Keep ≤25 chars; the FE may append derived params (location, price band).
_TOOL_LABELS: Dict[str, str] = {
    "search_listings": "Đang tìm BĐS",
    "get_listing_detail": "Đang xem chi tiết tin",
    "compare_listings": "Đang so sánh tin",
    "get_price_estimate": "Đang ước tính giá",
    "get_price_history": "Đang xem lịch sử giá",
    "get_recommendations": "Đang gợi ý tin phù hợp",
    "get_user_info": "Đang lấy thông tin tài khoản",
    "save_listing": "Đang xử lý lưu tin",
    "bulk_save_listings": "Đang lưu nhiều tin",
    "my_listings_status": "Đang kiểm tra tin của bạn",
    "address_translator": "Đang tra cứu địa chỉ",
    "update_listing_price": "Đang xử lý cập nhật giá",
    "notifications_inbox": "Đang xem thông báo",
    "report_listing": "Đang xử lý báo cáo",
}

# Filter heavy / token-burner fields out of the args dict the FE sees.
_ARG_DROP_KEYS = frozenset({"context", "auth_token"})


def _parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    """
    Best-effort parse of a tool_call item's raw arguments.

    The OpenAI Responses API returns `arguments` as a JSON string on the
    raw_item; fallback paths handle dict-shaped raw items and exceptions
    (logged at debug; UI degrades to no-args display).
    """
    args_raw = getattr(raw, "arguments", None)
    if args_raw is None and isinstance(raw, dict):
        args_raw = raw.get("arguments")
    if isinstance(args_raw, dict):
        parsed = args_raw
    elif isinstance(args_raw, str):
        try:
            import json

            parsed = json.loads(args_raw)
        except Exception:  # noqa: BLE001
            return {}
    else:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {k: v for k, v in parsed.items() if k not in _ARG_DROP_KEYS}


def _friendly_tool_summary(name: str, args: Dict[str, Any]) -> str:
    """
    Build a one-line Vietnamese summary of what the tool is about to do.

    Surfaces in the SSE `tool_call` status event so the FE can show a
    contentful spinner ("Đang tìm phòng ở Bình Thạnh giá 5-10tr...") rather
    than just the tool name. Falls back to the static label.
    """
    label = _TOOL_LABELS.get(name, "Đang xử lý")

    if name == "search_listings":
        bits: List[str] = []
        if args.get("districtId"):
            bits.append(f"quận {args['districtId']}")
        elif args.get("provinceCode"):
            bits.append(f"tỉnh {args['provinceCode']}")
        types = args.get("productTypes") or (
            [args["productType"]] if args.get("productType") else []
        )
        if types:
            label_map = {
                "ROOM": "phòng",
                "APARTMENT": "căn hộ",
                "HOUSE": "nhà",
                "STUDIO": "studio",
                "OFFICE": "văn phòng",
            }
            bits.append("/".join(label_map.get(t, t) for t in types))
        if args.get("minPrice") and args.get("maxPrice"):
            mn = int(args["minPrice"]) // 1_000_000
            mx = int(args["maxPrice"]) // 1_000_000
            bits.append(f"giá {mn}-{mx}tr")
        elif args.get("maxPrice"):
            mx = int(args["maxPrice"]) // 1_000_000
            bits.append(f"dưới {mx}tr")
        if bits:
            return f"{label}: " + " ".join(bits) + "..."

    if name == "get_listing_detail" and args.get("listingId"):
        return f"{label} #{args['listingId']}..."

    if name == "compare_listings":
        ids = args.get("listingIds") or []
        if isinstance(ids, list) and ids:
            return f"{label}: {len(ids)} tin..."

    if name == "save_listing":
        action = args.get("action", "save")
        verb = "Đang bỏ lưu" if action == "unsave" else "Đang lưu"
        if args.get("listingId"):
            return f"{verb} tin #{args['listingId']}..."
        return f"{verb} tin..."

    if name == "bulk_save_listings":
        ids = args.get("listingIds") or []
        action = args.get("action", "save")
        verb = "Đang bỏ lưu" if action == "unsave" else "Đang lưu"
        if isinstance(ids, list) and ids:
            return f"{verb} {len(ids)} tin..."

    if name == "address_translator" and args.get("query"):
        return f"{label}: {args['query']}..."

    if name == "my_listings_status":
        focus = args.get("focus") or "all"
        focus_map = {
            "expiring": "tin sắp hết hạn",
            "rejected": "tin bị từ chối",
            "active": "tin đang hiển thị",
        }
        if focus in focus_map:
            return f"Đang kiểm tra {focus_map[focus]}..."

    if name == "update_listing_price" and args.get("newPrice"):
        return f"{label} thành {int(args['newPrice']):,} VND..."

    return f"{label}..."


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """
    Stateless coordinator — one instance is created at startup and reused.

    Each call to `run()` is independent: it builds its own per-request Agent,
    ToolContext, and Langfuse trace.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        rag: RAGRetriever,
    ) -> None:
        self._gateway = gateway
        self._rag = rag
        self._static_prefix = rag.get_system_prompt_prefix()
        self._tools = get_chat_tools()
        logger.info(
            "AgentOrchestrator initialised — tools: %s",
            [getattr(t, "name", repr(t)) for t in self._tools],
        )

    # ------------------------------------------------------------------
    # Helpers shared by run() and run_stream()
    # ------------------------------------------------------------------

    def _build_agent(self, instructions: str) -> Agent[ToolContext]:
        return Agent[ToolContext](
            name="SmartRent Chat Agent",
            instructions=instructions,
            model=make_model(settings.LLM_CHAT_MODEL),
            model_settings=default_model_settings(temperature=0.7),
            tools=self._tools,
        )

    def _build_input(
        self,
        messages: List[ChatMessage],
        dynamic_context: str,
        last_listings: Optional[List[LastListingRef]],
    ) -> List[Dict[str, Any]]:
        """Build the Responses-API input list, prepending dynamic context to the last user message."""
        items = _to_responses_input(messages[:-1])
        last_user = messages[-1].content
        context_block = _build_dynamic_context_block(dynamic_context, last_listings)
        if context_block:
            last_user = f"{context_block}\n\n---\n\n{last_user}"
        items.append({"role": "user", "content": last_user})
        return items

    def _build_listings_payload(
        self, raw_listings: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Deduplicate raw listings by listingId (later entries win — detail >
        search card) and shape the response payload.
        """
        if not raw_listings:
            return None

        seen: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for listing in raw_listings:
            lid = str(listing.get("listingId", id(listing)))
            if lid not in seen:
                order.append(lid)
            seen[lid] = listing

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

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        last_listings: Optional[List[LastListingRef]] = None,
    ) -> AgentResult:
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
        user_id: Optional[str],
        auth_token: Optional[str],
        last_listings: Optional[List[LastListingRef]],
    ) -> AgentResult:
        # Keep the non-streaming path within the same token budget as run_stream
        # (run_stream trims at its own entry; _run_pipeline is the non-stream core).
        messages = _trim_history(messages)
        trace = self._gateway.create_trace(
            name="chat-request",
            session_id=session_id,
            input={"message": user_message},
            metadata={
                "model": settings.LLM_CHAT_MODEL,
                "provider": settings.LLM_PROVIDER,
                "turns": len(messages),
            },
        )

        try:
            # ── RAG ──────────────────────────────────────────────────
            rag_span = trace.span(name="rag-retrieve", input={"query": user_message})
            dynamic_context = self._rag.retrieve(user_message)
            rag_span.end(
                output={
                    "has_context": bool(dynamic_context),
                    "context_length": len(dynamic_context),
                }
            )

            # ── Build agent ──────────────────────────────────────────
            base_prompt, _prompt_obj = self._gateway.get_prompt(
                "smartrent-chat-system",
                label="production",
                fallback=_SYSTEM_BASE,
            )
            instructions = _build_static_instructions(
                base_prompt or _SYSTEM_BASE,
                self._static_prefix,
                settings.MAX_LISTINGS_RETURN,
            )
            agent = self._build_agent(instructions)

            # ── Run ──────────────────────────────────────────────────
            input_items = self._build_input(messages, dynamic_context, last_listings)
            tool_ctx = ToolContext(user_id=user_id, auth_token=auth_token)

            llm_span = trace.generation(
                name="agent-run",
                model=settings.LLM_CHAT_MODEL,
                input=str(input_items[-1])[:2000],
            )
            try:
                result = await Runner.run(
                    starting_agent=agent,
                    input=cast(Any, input_items),
                    context=tool_ctx,
                    max_turns=MAX_AGENT_TURNS,
                )
                llm_span.end(output=str(result.final_output)[:2000])
            except MaxTurnsExceeded as e:
                llm_span.end(level="ERROR", status_message=f"max_turns_exceeded: {e}")
                logger.warning("Agent run exceeded MAX_AGENT_TURNS=%d", MAX_AGENT_TURNS)
                final_text = (
                    "Xin lỗi, tôi cần thêm bước để hoàn tất yêu cầu này. Bạn có thể "
                    "cho tôi thêm chi tiết hoặc thử lại không?"
                )
                listings_payload = self._build_listings_payload(
                    tool_ctx.collected_listings
                )
                return AgentResult(
                    message=final_text,
                    listings=listings_payload,
                    metadata={"error": "max_turns_exceeded"},
                )

            # ── Extract output ──────────────────────────────────────
            final_text = (
                str(result.final_output).strip()
                if result.final_output
                else "Tôi đã xử lý yêu cầu của bạn. Hãy xem kết quả bên dưới."
            )
            tools_used = _tool_names_used(result.new_items)
            listings_payload = self._build_listings_payload(tool_ctx.collected_listings)

            trace.update(
                output={"message": final_text[:500]},
                metadata={"tools_used": tools_used},
            )

            return AgentResult(
                message=final_text,
                listings=listings_payload,
                tools_used=tools_used,
                metadata={
                    "model": settings.LLM_CHAT_MODEL,
                    "provider": settings.LLM_PROVIDER,
                    "tools_used": tools_used,
                    "rag_context_injected": bool(dynamic_context),
                },
            )

        except Exception as e:
            trace.update(output={"error": str(e)})
            logger.error("AgentOrchestrator.run failed: %s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Streaming entry point (SSE)
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
        Streaming variant of run() — yields SSE-shaped events:

            {"event": "status",   "data": {"phase": "thinking"|"tool_call"|"tool_result", ...}}
            {"event": "text",     "data": {"delta": str}}
            {"event": "listings", "data": {...}}
            {"event": "suggestions", "data": {"items": [{"label": str, "query": str}]}}
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
                "model": settings.LLM_CHAT_MODEL,
                "provider": settings.LLM_PROVIDER,
                "turns": len(messages),
                "streamed": True,
            },
        )

        tool_ctx = ToolContext(user_id=user_id, auth_token=auth_token)
        tools_used: List[str] = []
        any_text_streamed = False

        try:
            # ── RAG ──────────────────────────────────────────────────
            rag_span = trace.span(name="rag-retrieve", input={"query": user_message})
            dynamic_context = self._rag.retrieve(user_message)
            rag_span.end(
                output={
                    "has_context": bool(dynamic_context),
                    "context_length": len(dynamic_context),
                }
            )

            # ── Build agent ──────────────────────────────────────────
            base_prompt, _prompt_obj = self._gateway.get_prompt(
                "smartrent-chat-system",
                label="production",
                fallback=_SYSTEM_BASE,
            )
            instructions = _build_static_instructions(
                base_prompt or _SYSTEM_BASE,
                self._static_prefix,
                settings.MAX_LISTINGS_RETURN,
            )
            agent = self._build_agent(instructions)
            input_items = self._build_input(messages, dynamic_context, last_listings)

            yield {"event": "status", "data": {"phase": "thinking", "round": 1}}

            llm_span = trace.generation(
                name="agent-run-stream",
                model=settings.LLM_CHAT_MODEL,
                input=str(input_items[-1])[:2000],
            )

            stream: RunResultStreaming = Runner.run_streamed(
                starting_agent=agent,
                input=cast(Any, input_items),
                context=tool_ctx,
                max_turns=MAX_AGENT_TURNS,
            )

            async for event in stream.stream_events():
                etype = getattr(event, "type", None)

                if etype == "raw_response_event":
                    data = getattr(event, "data", None)
                    if isinstance(data, ResponseTextDeltaEvent):
                        delta = getattr(data, "delta", "")
                        if delta:
                            any_text_streamed = True
                            yield {"event": "text", "data": {"delta": delta}}
                    continue

                if etype == "run_item_stream_event":
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", None)
                    raw = getattr(item, "raw_item", None)

                    if item_type == "tool_call_item":
                        name = getattr(raw, "name", "") or ""
                        if name:
                            tools_used.append(name)
                            args = _parse_tool_arguments(raw)
                            yield {
                                "event": "status",
                                "data": {
                                    "phase": "tool_call",
                                    "tool": name,
                                    "summary": _friendly_tool_summary(name, args),
                                    "args": args,
                                },
                            }
                    elif item_type == "tool_call_output_item":
                        name = ""
                        # Tool name lives on the matching tool_call; best-effort
                        # extraction from raw output payload when available.
                        if isinstance(raw, dict):
                            name = raw.get("name", "")
                        status = "success"
                        try:
                            output = getattr(item, "output", None)
                            if isinstance(output, dict) and "status" in output:
                                status = str(output["status"])
                        except Exception:  # noqa: BLE001
                            pass
                        yield {
                            "event": "status",
                            "data": {
                                "phase": "tool_result",
                                "tool": name or (tools_used[-1] if tools_used else ""),
                                "status": status,
                            },
                        }
                    continue

                # AgentUpdatedStreamEvent / others — ignored

            # Stream complete — close generation span
            final_text = str(stream.final_output).strip() if stream.final_output else ""
            llm_span.end(output=final_text[:2000] if final_text else "(tool-only)")

            if not any_text_streamed and final_text:
                yield {"event": "text", "data": {"delta": final_text}}
            elif not any_text_streamed:
                yield {
                    "event": "text",
                    "data": {
                        "delta": "Tôi đã xử lý yêu cầu của bạn. Hãy xem kết quả bên dưới."
                    },
                }

            listings_payload = self._build_listings_payload(tool_ctx.collected_listings)
            if listings_payload:
                yield {"event": "listings", "data": listings_payload}

            try:
                suggestions = build_suggestions(
                    tools_used,
                    tool_ctx.collected_listings,
                    bool(auth_token),
                )
            except Exception:  # noqa: BLE001 — never let suggestions break the stream
                suggestions = []
            if suggestions:
                yield {"event": "suggestions", "data": {"items": suggestions}}

            metadata = {
                "model": settings.LLM_CHAT_MODEL,
                "provider": settings.LLM_PROVIDER,
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
        except MaxTurnsExceeded:
            trace.update(output={"error": "max_turns_exceeded"})
            yield {
                "event": "error",
                "data": {
                    "message": "Yêu cầu cần quá nhiều bước. Vui lòng thử lại với câu hỏi cụ thể hơn."
                },
            }
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
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Return the module-level AgentOrchestrator singleton."""
    global _instance
    if _instance is None:
        gateway = get_gateway()
        rag = RAGRetriever()
        _instance = AgentOrchestrator(gateway, rag)
        logger.info("AgentOrchestrator singleton created.")
    return _instance
