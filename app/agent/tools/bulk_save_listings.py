"""
Tool: bulk_save_listings — save or unsave 2-10 listings in parallel.

Triggered when the user says "lưu cả 3 tin", "lưu hết", "bỏ lưu hết tin
trên". Single-listing operations stay on `save_listing`; this tool exists
because looping a single-target tool round-by-round through Gemini wastes
tokens and rounds. We fan out the HTTP calls in one orchestrator round.
"""

import asyncio
import logging
from typing import Annotated, Any, Dict, List, Tuple

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MIN_IDS = 2
_MAX_IDS = 10


def _coerce_id(raw: Any) -> str:
    try:
        return str(int(float(raw)))
    except (TypeError, ValueError):
        return str(raw)


async def _one(action: str, listing_id: str, token: str) -> Tuple[str, Dict[str, Any]]:
    try:
        if action == "unsave":
            data = await backend_client.unsave_listing(listing_id, token)
        else:
            data = await backend_client.save_listing(listing_id, token)
        if "error" in data:
            return listing_id, {"status": "error", "error": data["error"]}
        return listing_id, {"status": "success"}
    except httpx.HTTPStatusError as e:
        return listing_id, {
            "status": "error",
            "error": f"HTTP {e.response.status_code}",
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("bulk_save_listings: %s %s failed: %s", action, listing_id, e)
        return listing_id, {"status": "error", "error": str(e)}


async def _do_bulk_save(
    token: str, listing_ids: List[str], action: str
) -> Dict[str, Any]:
    """Core logic — separated so it can be called directly in tests."""
    if action not in ("save", "unsave"):
        return {
            "status": "error",
            "error": f"action phải là 'save' hoặc 'unsave', nhận được '{action}'.",
        }

    seen: set = set()
    ids: List[str] = []
    for raw in listing_ids:
        cid = _coerce_id(raw)
        if cid not in seen:
            ids.append(cid)
            seen.add(cid)

    if len(ids) < _MIN_IDS:
        return {
            "status": "error",
            "error": (
                f"Cần ít nhất {_MIN_IDS} listingId (nhận được {len(ids)}). "
                "Với 1 tin thì dùng save_listing."
            ),
        }
    if len(ids) > _MAX_IDS:
        ids = ids[:_MAX_IDS]

    logger.info("bulk_save_listings %s on %d listings: %s", action, len(ids), ids)

    results = await asyncio.gather(*(_one(action, lid, token) for lid in ids))

    succeeded: List[str] = []
    failed: List[Dict[str, str]] = []
    for lid, res in results:
        if res["status"] == "success":
            succeeded.append(lid)
        else:
            failed.append({"listingId": lid, "error": res["error"]})

    verb = "lưu" if action == "save" else "bỏ lưu"
    if not failed:
        message = f"Đã {verb} {len(succeeded)} tin."
    elif not succeeded:
        message = f"Không {verb} được tin nào."
    else:
        message = f"Đã {verb} {len(succeeded)}/{len(ids)} tin. {len(failed)} tin lỗi."

    return {
        "status": "success" if succeeded else "error",
        "action": action,
        "succeeded": succeeded,
        "failed": failed,
        "message": message,
    }


@function_tool(
    name_override="bulk_save_listings",
    description_override=(
        "Save or unsave 2-10 listings in one go. Use when the user says "
        "'lưu cả 3 tin', 'lưu hết tin trên', 'bỏ lưu hết'. Resolve "
        "listingIds from prior search results — never ask the user. For a "
        "single listing, prefer save_listing instead."
    ),
)
async def bulk_save_listings(
    ctx: RunContextWrapper[ToolContext],
    listingIds: Annotated[
        List[str],
        Field(description=f"Listing IDs ({_MIN_IDS}-{_MAX_IDS} items)."),
    ],
    action: Annotated[
        str,
        Field(
            description="'save' to add to favorites, 'unsave' to remove.",
            json_schema_extra={"enum": ["save", "unsave"]},
        ),
    ],
) -> Dict[str, Any]:
    token = ctx.context.auth_token
    if not token:
        return {
            "status": "error",
            "error": "Người dùng chưa đăng nhập. Hãy hướng dẫn đăng nhập trước.",
        }
    return await _do_bulk_save(token, listingIds, action)
