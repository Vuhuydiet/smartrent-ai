"""
Tool: save_listing — save or unsave a listing to/from the user's favorites.
Requires a valid auth_token in ToolContext.
"""

import logging
from typing import Annotated, Any, Dict

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)


async def _do_save_unsave(token: str, listing_id: str, action: str) -> Dict[str, Any]:
    """Core save/unsave logic — separated so it can be called directly in tests."""
    try:
        if action == "unsave":
            data = await backend_client.unsave_listing(listing_id, token)
        else:
            data = await backend_client.save_listing(listing_id, token)

        if "error" in data:
            return {"status": "error", "error": data["error"]}

        return {
            "status": "success",
            "action": action,
            "listingId": listing_id,
            "message": (
                f"Đã lưu tin {listing_id} vào danh sách yêu thích."
                if action == "save"
                else f"Đã bỏ lưu tin {listing_id}."
            ),
        }

    except httpx.HTTPStatusError as e:
        logger.error("Backend HTTP %s for save_listing", e.response.status_code)
        return {
            "status": "error",
            "error": f"Backend returned HTTP {e.response.status_code}",
        }
    except Exception as e:
        logger.error("save_listing failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


async def _handle_save(
    ctx: RunContextWrapper[ToolContext], listing_id_str: str, action: str
) -> Dict[str, Any]:
    """Token guard + dispatch — separated so it can be called directly in tests."""
    token = ctx.context.auth_token
    if not token:
        return {
            "status": "error",
            "error": (
                "Người dùng chưa đăng nhập. "
                "Hãy hướng dẫn người dùng đăng nhập để lưu tin."
            ),
        }
    return await _do_save_unsave(token, listing_id_str, action)


@function_tool(
    name_override="save_listing",
    description_override=(
        "Save or unsave a property listing to/from the user's favorites. "
        "Use when the user says 'lưu tin này', 'thêm vào yêu thích', "
        "'bỏ lưu', or 'xóa khỏi yêu thích'."
    ),
)
async def save_listing(
    ctx: RunContextWrapper[ToolContext],
    listingId: Annotated[str, Field(description="The listing ID to save or unsave.")],
    action: Annotated[
        str,
        Field(
            description="'save' to add to favorites, 'unsave' to remove.",
            json_schema_extra={"enum": ["save", "unsave"]},
        ),
    ],
) -> Dict[str, Any]:
    try:
        listing_id = str(int(float(listingId)))
    except (ValueError, TypeError):
        listing_id = str(listingId)

    return await _handle_save(ctx, listing_id, action)
