"""
Rule-based follow-up suggestions for the chat agent.

Given the tools the agent just used + the listings it collected, produce up to
four tappable follow-up chips. Pure and deterministic (no LLM call) so it adds
zero latency/cost and stays predictable for demos. Logic lives server-side so a
future upgrade to LLM-generated suggestions only touches this module.

Each suggestion is {"label": <chip text>, "query": <message sent on tap>}.
"""

from typing import Any, Dict, List

_MAX_SUGGESTIONS = 4

# Tools that define a recognised follow-up context, checked newest-first.
_CONTEXT_TOOLS = {
    "search_listings",
    "get_recommendations",
    "get_listing_detail",
    "compare_listings",
    "my_listings_status",
}

_STARTER = [
    {"label": "Tìm phòng Q1 dưới 5tr", "query": "tìm phòng quận 1 dưới 5 triệu"},
    {
        "label": "Nhà trọ gần ĐH Bách Khoa",
        "query": "tìm nhà trọ gần Đại học Bách Khoa",
    },
    {
        "label": "Căn hộ 2PN Bình Thạnh",
        "query": "tìm căn hộ 2 phòng ngủ ở Bình Thạnh",
    },
]

_ZERO_RESULT = [
    {"label": "Tăng ngân sách", "query": "tìm với ngân sách cao hơn"},
    {"label": "Đổi khu vực gần", "query": "tìm ở khu vực lân cận"},
    {"label": "Bỏ bớt tiêu chí", "query": "bỏ bớt điều kiện và tìm lại"},
]

_DETAIL = [
    {"label": "So sánh tất cả", "query": "so sánh tất cả các tin vừa tìm được"},
    {"label": "Lưu tin này", "query": "lưu tin này"},
    {"label": "Xung quanh có gì", "query": "xung quanh tin này có gì"},
    {"label": "Ước tính giá", "query": "ước tính giá tin này"},
]

_OWNER = [
    {"label": "Tin nào sắp hết hạn", "query": "tin nào của tôi sắp hết hạn"},
    {"label": "Hạ giá một tin", "query": "tôi muốn hạ giá một tin"},
    {"label": "Tin bị từ chối", "query": "có tin nào của tôi bị từ chối không"},
]


def _last_context_tool(tools_used: List[str]) -> str:
    for tool in reversed(tools_used or []):
        if tool in _CONTEXT_TOOLS:
            return tool
    return ""


def _result_head(listings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Detail + (conditional) compare chips shared by search and recommendation.

    Chip queries are phrased by position, never by listing ID: the user should
    never see or type one. The agent maps "căn đầu tiên" back to a listingId
    using the hidden result-set context block.
    """
    out: List[Dict[str, str]] = [
        {"label": "Xem chi tiết căn 1", "query": "xem chi tiết căn đầu tiên"}
    ]
    if len(listings) >= 2:
        # Comparison is all-or-nothing — there is no "compare these two".
        out.append({"label": "So sánh tất cả", "query": "so sánh tất cả các tin trên"})
    return out


def build_suggestions(
    tools_used: List[str], listings: List[Dict[str, Any]], has_auth: bool
) -> List[Dict[str, str]]:
    """Return up to _MAX_SUGGESTIONS follow-up chips for the current context."""
    last = _last_context_tool(tools_used)
    n = len(listings or [])

    if last == "search_listings":
        if n >= 1:
            out = _result_head(listings)
            out.append({"label": "Xem tiếp", "query": "xem tiếp"})
            out.append({"label": "Lọc rẻ hơn", "query": "lọc rẻ hơn"})
        else:
            out = list(_ZERO_RESULT)
    elif last == "get_recommendations" and n >= 1:
        out = _result_head(listings)
        out.append({"label": "Gợi ý thêm", "query": "gợi ý thêm cho tôi"})
    elif last in ("get_listing_detail", "compare_listings"):
        out = list(_DETAIL)
        if not has_auth:
            # "Lưu tin này" requires login — drop it for guests.
            out = [chip for chip in out if chip["query"] != "lưu tin này"]
    elif last == "my_listings_status":
        # Owner dashboard actions are auth-only.
        out = list(_OWNER) if has_auth else []
    elif not tools_used and n == 0:
        out = list(_STARTER)
    else:
        out = []

    return out[:_MAX_SUGGESTIONS]
