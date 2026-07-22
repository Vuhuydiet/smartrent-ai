"""
Tool: save_listing — save or unsave a listing to/from the user's favorites.
Requires a valid auth_token in ToolContext.

Saving is not idempotent on the backend: saving twice is a 409 and unsaving
something that was never saved is a 404. Neither is a failure from the user's
point of view — the favorites list already holds the state they asked for — so
both are reported back as their own outcome instead of an error the model has
to improvise around.
"""

import logging
from typing import Annotated, Any, Dict, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

# DomainCodes from smartrent-backend (DomainCode.java, 24xxx = saved listing).
_CODE_ALREADY_SAVED = "24002"
_CODE_NOT_SAVED = "24003"

# Fallbacks for backends that predate those codes — the old build threw a bare
# RuntimeException, which the global handler turned into 500 UNKNOWN_ERROR with
# the raw message. Match on the message so the chatbot behaves correctly against
# both the current and the upgraded backend.
_ALREADY_SAVED_HINTS = ("already saved", "đã có trong danh sách")
_NOT_SAVED_HINTS = ("saved listing not found", "không có trong danh sách")


def _matches(message: Optional[str], hints: tuple) -> bool:
    lowered = (message or "").lower()
    return any(hint in lowered for hint in hints)


def classify_saved_conflict(
    action: str,
    status: Optional[int],
    code: Optional[str],
    message: Optional[str],
) -> Optional[str]:
    """Name the "already in that state" outcome, or None for a real error.

    Returns "already_saved" / "not_saved". Each is gated on the matching action:
    a 404 on `save` means the *listing* is gone, which is a genuine error, not
    a favorites-state no-op.
    """
    if action == "save" and (
        code == _CODE_ALREADY_SAVED
        or status == 409
        or _matches(message, _ALREADY_SAVED_HINTS)
    ):
        return "already_saved"
    if action == "unsave" and (
        code == _CODE_NOT_SAVED or status == 404 or _matches(message, _NOT_SAVED_HINTS)
    ):
        return "not_saved"
    return None


def _conflict_result(outcome: str, listing_id: str) -> Dict[str, Any]:
    """The tool payload for an already-in-that-state save/unsave."""
    if outcome == "already_saved":
        message = f"Tin {listing_id} đã có trong danh sách yêu thích của bạn từ trước."
    else:
        message = f"Tin {listing_id} vốn không có trong danh sách yêu thích của bạn."
    return {
        "status": outcome,
        "action": "save" if outcome == "already_saved" else "unsave",
        "listingId": listing_id,
        "message": message,
    }


async def _do_save_unsave(token: str, listing_id: str, action: str) -> Dict[str, Any]:
    """Core save/unsave logic — separated so it can be called directly in tests."""
    try:
        if action == "unsave":
            data = await backend_client.unsave_listing(listing_id, token)
        else:
            data = await backend_client.save_listing(listing_id, token)

        if "error" in data:
            # A 200 carrying a non-success envelope code — same conflicts can
            # surface here, so classify before calling it an error.
            outcome = classify_saved_conflict(action, None, None, data["error"])
            if outcome:
                return _conflict_result(outcome, listing_id)
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
        details = backend_client.error_details(e)
        outcome = classify_saved_conflict(
            action, details["status"], details["code"], details["message"]
        )
        if outcome:
            logger.info(
                "save_listing %s on %s → %s (HTTP %s)",
                action,
                listing_id,
                outcome,
                details["status"],
            )
            return _conflict_result(outcome, listing_id)

        logger.error(
            "Backend HTTP %s for save_listing: %s",
            details["status"],
            details["message"],
        )
        # Relay the backend's own message (e.g. the 50-listing cap) instead of
        # a bare status code the model can only paraphrase as "có lỗi".
        error = f"Backend returned HTTP {details['status']}"
        if details["message"]:
            error += f": {details['message']}"
        return {"status": "error", "error": error}
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
        "'bỏ lưu', or 'xóa khỏi yêu thích'. Never pre-check whether it is "
        "already saved — just call this. status='already_saved' / 'not_saved' "
        "means the favorites list already holds that state: relay `message` as "
        "a normal answer, NOT as an error or a failed attempt."
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
