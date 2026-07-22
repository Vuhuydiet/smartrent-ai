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

# focus → the listingStatus the backend filters my-listings by. "all" fetches
# everything and lets the attention filter below pick the rows worth showing.
_FOCUS_STATUS: Dict[str, str] = {
    "active": "DISPLAYING",
    "pending": "IN_REVIEW",
    "expiring": "EXPIRING_SOON",
    "expired": "EXPIRED",
    "rejected": "REJECTED",
}

# Deep links into the seller's "Quản lý tin đăng" page. The chat summary is
# capped at _MAX_LISTINGS_TO_SHOW rows, so every focus gets a chip that opens
# the same subset on the page. Query params are the ones getFiltersFromQuery()
# reads on the frontend (listingStatus → the matching status tab).
_MANAGE_PATH = "/seller/listings"
_FOCUS_LABELS: Dict[str, str] = {
    "all": "Xem tất cả tin đăng",
    "active": "Xem tất cả tin đang hoạt động",
    "pending": "Xem tất cả tin chờ duyệt",
    "expiring": "Xem tất cả tin sắp hết hạn",
    "expired": "Xem tất cả tin hết hạn",
    "rejected": "Xem tất cả tin bị từ chối",
}


def _focus_link(focus: str) -> Dict[str, str]:
    """The {label, url} chip for a focus — plain manage page for 'all'."""
    status = _FOCUS_STATUS.get(focus)
    url = f"{_MANAGE_PATH}?listingStatus={status}" if status else _MANAGE_PATH
    return {"label": _FOCUS_LABELS.get(focus, _FOCUS_LABELS["all"]), "url": url}


def _row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Trim one backend listing down to the fields the model reads out."""
    addr = item.get("address") or {}
    return {
        "listingId": str(item.get("listingId", "")),
        "title": item.get("title", ""),
        # Vietnamese labels, not raw enums — the model echoes these verbatim,
        # so "REVISION_REQUIRED" leaked straight to chat.
        "listingStatus": localize_enum(
            item.get("listingStatus"), LISTING_STATUS_LABELS
        ),
        "moderationStatus": localize_enum(
            item.get("moderationStatus"), MODERATION_STATUS_LABELS
        ),
        "expiryDate": item.get("expiryDate"),
        "districtName": addr.get("districtName", ""),
    }


def _attention_listings(listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick listings whose listingStatus or moderationStatus needs owner action."""
    flagged: List[Dict[str, Any]] = []
    for item in listings:
        if (
            item.get("listingStatus") in _ATTENTION_STATUSES
            or item.get("moderationStatus") in _ATTENTION_MODERATION
        ):
            flagged.append(_row(item))
        if len(flagged) >= _MAX_LISTINGS_TO_SHOW:
            break
    return flagged


async def _do_my_listings_status(token: str, focus: str) -> Dict[str, Any]:
    """Core logic — separated so it can be called directly in tests."""
    params: Dict[str, Any] = {"page": 1, "size": 50}
    if focus in _FOCUS_STATUS:
        params["listingStatus"] = _FOCUS_STATUS[focus]

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

    # With a focus the backend already returned exactly that subset, so show it
    # as-is. Filtering it a second time through the attention set was silently
    # emptying the list for any status that needs no owner action — asking
    # "các tin của tôi chưa duyệt" answered with a bare count and no listings.
    if focus in _FOCUS_STATUS:
        rows = [_row(item) for item in listings[:_MAX_LISTINGS_TO_SHOW]]
    else:
        rows = _attention_listings(listings)

    return {
        "status": "success",
        "focus": focus,
        "totalCount": total_count,
        # The frontend renders `manageUrl` as a tappable chip; tell the model so
        # it stops writing "vui lòng truy cập mục Quản lý tin đăng" by hand.
        "manageUrl": _focus_link(focus)["url"],
        "shownCount": len(rows),
        "moreAvailable": total_count > len(rows),
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
        # "attention" = the mixed bag worth flagging on an overall summary;
        # "focus" = every listing matching the requested status.
        "listRole": "focus" if focus in _FOCUS_STATUS else "attention",
        "listings": rows,
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
        link = _focus_link(resolved_focus)
        ctx.context.add_action_link(link["label"], link["url"])
    return result


@function_tool(
    name_override="my_listings_status",
    description_override=(
        "Get a chat-friendly summary of the logged-in user's OWN listings: "
        "active / pending / rejected / expired / expiring-soon counts plus a "
        "short `listings` array. Use whenever the user asks about THEIR OWN "
        "listings (vd 'tin của tôi sao rồi', 'tin nào chưa duyệt'). Do NOT use "
        "for searching public listings. ALWAYS list the returned `listings` "
        "rows — with a focus they are exactly the listings the user asked "
        "about, so a bare count is not an answer. "
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
                "unset for an overall summary. Values: 'active' (đang hiển "
                "thị), 'pending' (chờ duyệt / chưa duyệt), 'expiring' (sắp "
                "hết hạn), 'expired' (đã hết hạn), 'rejected' (bị từ chối / "
                "cần chỉnh sửa / bị đình chỉ), 'all' (default summary)."
            ),
            json_schema_extra={
                "enum": [
                    "all",
                    "active",
                    "pending",
                    "expiring",
                    "expired",
                    "rejected",
                ]
            },
        ),
    ] = None,
) -> Dict[str, Any]:
    return await _dispatch_my_listings(ctx, focus)
