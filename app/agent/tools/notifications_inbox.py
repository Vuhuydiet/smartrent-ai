"""
Tool: notifications_inbox — fetch + summarise + (optionally) mark-all-read.

Triggered by phrases like "có gì mới không?", "đọc thông báo", "tóm tắt
notification", "đọc hết noti giúp tôi". Returns recent items + counts by
type so the LLM can write a "you have X items, here are the most recent"
summary without making a second call. Mark-as-read happens only when the
user explicitly asks.
"""

import logging
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.enum_labels import NOTIFICATION_TYPE_LABELS, localize_enum
from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_RETURN = 10


def _summarise(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bucket notifications by type and pick the top few for display."""
    by_type: Dict[str, int] = {}
    unread = 0
    for n in items:
        raw_type = n.get("type") or n.get("notificationType") or "OTHER"
        # Bucket under the Vietnamese label — byType keys are read out loud by
        # the model, so raw "LISTING_REVISION_REQUIRED" leaked into the answer.
        label = localize_enum(raw_type, NOTIFICATION_TYPE_LABELS)
        by_type[label] = by_type.get(label, 0) + 1
        if not n.get("read", n.get("isRead", False)):
            unread += 1

    recent: List[Dict[str, Any]] = []
    for n in items[:_MAX_RETURN]:
        recent.append(
            {
                "id": n.get("id") or n.get("notificationId"),
                "title": n.get("title", ""),
                "message": (n.get("message") or n.get("content") or "")[:200],
                "type": localize_enum(
                    n.get("type") or n.get("notificationType"),
                    NOTIFICATION_TYPE_LABELS,
                ),
                "read": n.get("read", n.get("isRead", False)),
                "createdAt": n.get("createdAt") or n.get("created_at"),
            }
        )
    return {"byType": by_type, "unread": unread, "recent": recent}


async def _do_inbox(token: str, mark_all_read: bool) -> Dict[str, Any]:
    """Core logic — separated so it can be called directly in tests."""
    try:
        data = await backend_client.list_notifications(page=1, size=20, token=token)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            return {
                "status": "error",
                "error": "Phiên đăng nhập không hợp lệ. Vui lòng đăng nhập lại.",
            }
        return {"status": "error", "error": f"Backend HTTP {e.response.status_code}"}
    except Exception as e:  # noqa: BLE001
        logger.error("notifications_inbox failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}

    if "error" in data:
        return {"status": "error", "error": data["error"]}

    items = data.get("notifications") or data.get("content") or data.get("items") or []
    summary = _summarise(items)

    marked = False
    if mark_all_read and summary["unread"] > 0:
        try:
            await backend_client.mark_all_notifications_read(token=token)
            marked = True
        except Exception as e:  # noqa: BLE001
            logger.warning("mark_all_notifications_read failed: %s", e)

    return {
        "status": "success",
        "totalCount": data.get("totalCount", len(items)),
        "unread": summary["unread"],
        "byType": summary["byType"],
        "recent": summary["recent"],
        "markedAllRead": marked,
    }


@function_tool(
    name_override="notifications_inbox",
    description_override=(
        "Fetch the user's notification inbox + (optionally) mark all as "
        "read. Use when the user asks 'có gì mới không', 'đọc thông báo', "
        "'tóm tắt noti'. Set markAllRead=true ONLY when the user explicitly "
        "says to mark everything read. Returns recent items + counts by "
        "type. Write a Vietnamese summary that mentions unread count and "
        "the 1-3 most relevant items."
    ),
)
async def notifications_inbox(
    ctx: RunContextWrapper[ToolContext],
    markAllRead: Annotated[
        Optional[bool],
        Field(
            description=(
                "Set true to mark every notification as read AFTER fetching. "
                "Default false (read-only summary)."
            )
        ),
    ] = None,
) -> Dict[str, Any]:
    token = ctx.context.auth_token
    if not token:
        return {
            "status": "error",
            "error": "Người dùng chưa đăng nhập. Hãy hướng dẫn đăng nhập trước.",
        }
    return await _do_inbox(token, bool(markAllRead))
