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

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_LISTINGS_TO_SHOW = 5
_ATTENTION_STATUSES = {"EXPIRING_SOON", "EXPIRED", "REJECTED", "PENDING_PAYMENT"}
_ATTENTION_MODERATION = {"REJECTED", "REVISION_REQUIRED"}


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
                    "listingStatus": ls,
                    "moderationStatus": ms,
                    "expiryDate": item.get("expiryDate"),
                    "districtName": addr.get("districtName", ""),
                }
            )
        if len(flagged) >= _MAX_LISTINGS_TO_SHOW:
            break
    return flagged


async def _do_my_listings_status(
    token: str, focus: str
) -> Dict[str, Any]:
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

    return {
        "status": "success",
        "focus": focus,
        "totalCount": data.get("totalCount", len(listings)),
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
        "needsAttention": _attention_listings(listings),
    }


@function_tool(
    name_override="my_listings_status",
    description_override=(
        "Get a chat-friendly summary of the logged-in user's OWN listings: "
        "active / pending / rejected / expired / expiring-soon counts plus "
        "the short list of listings that need attention. Use whenever the "
        "user asks about THEIR OWN listings (vd 'tin của tôi sao rồi', "
        "'tin nào sắp hết hạn'). Do NOT use for searching public listings."
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
    token = ctx.context.auth_token
    if not token:
        return {
            "status": "error",
            "error": (
                "Người dùng chưa đăng nhập. Hãy hướng dẫn đăng nhập "
                "để xem tin của mình."
            ),
        }
    return await _do_my_listings_status(token, focus or "all")
