"""
Tool: my_listings_status — owner dashboard delivered through chat.

Triggered by phrases like "tin của tôi sao rồi", "tin nào sắp hết hạn",
"có tin nào bị từ chối không". Backend already returns a pre-computed
OwnerStatistics aggregate so this tool mostly normalises that for the LLM
and surfaces the small set of listings that need owner attention.
"""

import logging
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.enum_labels import (
    LISTING_STATUS_LABELS,
    MODERATION_STATUS_LABELS,
    localize_enum,
)
from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_LISTINGS_TO_SHOW = 5
_ATTENTION_STATUSES = {"EXPIRING_SOON", "EXPIRED", "REJECTED", "PENDING_PAYMENT"}
_ATTENTION_MODERATION = {"REJECTED", "REVISION_REQUIRED", "SUSPENDED"}

# Deep links into the seller's "Quản lý tin đăng" page. The chat summary is
# capped at _MAX_LISTINGS_TO_SHOW rows, so every focus gets a chip that opens
# the same subset on the page. Query params are the ones getFiltersFromQuery()
# reads on the frontend (listingStatus → the matching status tab).
_MANAGE_PATH = "/seller/listings"
_FOCUS_LINKS: Dict[str, Dict[str, str]] = {
    "all": {"label": "Xem tất cả tin đăng", "url": _MANAGE_PATH},
    "active": {
        "label": "Xem tất cả tin đang hoạt động",
        "url": f"{_MANAGE_PATH}?listingStatus=DISPLAYING",
    },
    "expiring": {
        "label": "Xem tất cả tin sắp hết hạn",
        "url": f"{_MANAGE_PATH}?listingStatus=EXPIRING_SOON",
    },
    "rejected": {
        "label": "Xem tất cả tin bị từ chối",
        "url": f"{_MANAGE_PATH}?listingStatus=REJECTED",
    },
}


def _attention_listings(listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick listings whose listingStatus or moderationStatus needs owner action."""
    flagged: List[Dict[str, Any]] = []
    for item in listings:
        ls = item.get("listingStatus")
        ms = item.get("moderationStatus")
        if ls in _ATTENTION_STATUSES or ms in _ATTENTION_MODERATION:
            addr = item.get("address") or {}
            flagged.append(
                {
                    "listingId": str(item.get("listingId", "")),
                    "title": item.get("title", ""),
                    # Vietnamese labels, not raw enums — the model echoes these
                    # verbatim, so "REVISION_REQUIRED" leaked straight to chat.
                    "listingStatus": localize_enum(ls, LISTING_STATUS_LABELS),
                    "moderationStatus": localize_enum(ms, MODERATION_STATUS_LABELS),
                    "expiryDate": item.get("expiryDate"),
                    "districtName": addr.get("districtName", ""),
                }
            )
        if len(flagged) >= _MAX_LISTINGS_TO_SHOW:
            break
    return flagged


async def _do_my_listings_status(token: str, focus: str) -> Dict[str, Any]:
    """Core logic — separated so it can be called directly in tests."""
    params: Dict[str, Any] = {"page": 1, "size": 50}
    if focus == "expiring":
        params["listingStatus"] = "EXPIRING_SOON"
    elif focus == "rejected":
        params["listingStatus"] = "REJECTED"
    elif focus == "active":
        params["listingStatus"] = "DISPLAYING"

    try:
        data = await backend_client.get_my_listings(params, token=token)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            return {
                "status": "error",
                "error": "Phiên đăng nhập không hợp lệ. Vui lòng đăng nhập lại.",
            }
        logger.error("my_listings_status HTTP %s", e.response.status_code)
        return {
            "status": "error",
            "error": f"Backend returned HTTP {e.response.status_code}",
        }
    except Exception as e:  # noqa: BLE001
        logger.error("my_listings_status failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}

    if "error" in data:
        return {"status": "error", "error": data["error"]}

    listings = data.get("listings") or []
    stats = data.get("statistics") or {}
    total_count = data.get("totalCount", len(listings))
    needs_attention = _attention_listings(listings)

    return {
        "status": "success",
        "focus": focus,
        "totalCount": total_count,
        # The frontend renders `manageUrl` as a tappable chip; tell the model so
        # it stops writing "vui lòng truy cập mục Quản lý tin đăng" by hand.
        "manageUrl": _FOCUS_LINKS.get(focus, _FOCUS_LINKS["all"])["url"],
        "shownCount": len(needs_attention),
        "moreAvailable": total_count > len(needs_attention),
        "statistics": {
            "active": stats.get("active", 0),
            "pendingVerification": stats.get("pendingVerification", 0),
            "rejected": stats.get("rejected", 0),
            "expired": stats.get("expired", 0),
            "drafts": stats.get("drafts", 0),
            "byVipTier": {
                "normal": stats.get("normalListings", 0),
                "silver": stats.get("silverListings", 0),
                "gold": stats.get("goldListings", 0),
                "diamond": stats.get("diamondListings", 0),
            },
        },
        "needsAttention": needs_attention,
    }


async def _dispatch_my_listings(
    ctx: RunContextWrapper[ToolContext], focus: Optional[str]
) -> Dict[str, Any]:
    """Auth gate + fetch + deep-link registration — called directly in tests."""
    token = ctx.context.auth_token
    if not token:
        return {
            "status": "error",
            "error": (
                "Người dùng chưa đăng nhập. Hãy hướng dẫn đăng nhập "
                "để xem tin của mình."
            ),
        }
    resolved_focus = focus or "all"
    result = await _do_my_listings_status(token, resolved_focus)
    if result.get("status") == "success":
        link = _FOCUS_LINKS.get(resolved_focus, _FOCUS_LINKS["all"])
        ctx.context.add_action_link(link["label"], link["url"])
    return result


@function_tool(
    name_override="my_listings_status",
    description_override=(
        "Get a chat-friendly summary of the logged-in user's OWN listings: "
        "active / pending / rejected / expired / expiring-soon counts plus "
        "the short list of listings that need attention. Use whenever the "
        "user asks about THEIR OWN listings (vd 'tin của tôi sao rồi', "
        "'tin nào sắp hết hạn'). Do NOT use for searching public listings. "
        "Only a few listings fit in chat: when moreAvailable is true, say so "
        "briefly — a 'Xem tất cả' button to manageUrl is added automatically, "
        "so do NOT write the link or the page name yourself."
    ),
)
async def my_listings_status(
    ctx: RunContextWrapper[ToolContext],
    focus: Annotated[
        Optional[str],
        Field(
            description=(
                "Filter when the user asks about a specific subset. Leave "
                "unset for an overall summary. Values: 'expiring' (only "
                "expiring/expired), 'rejected' (only rejected/revision-"
                "required), 'active' (only currently displayed), 'all' "
                "(default summary)."
            ),
            json_schema_extra={"enum": ["all", "expiring", "rejected", "active"]},
        ),
    ] = None,
) -> Dict[str, Any]:
    return await _dispatch_my_listings(ctx, focus)
