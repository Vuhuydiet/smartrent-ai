"""
Tool: get_user_info — fetches the authenticated user's profile, subscription,
or saved listings. Reads auth_token from ToolContext.
"""

import logging
from typing import Annotated, Any, Dict

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)


async def _get_profile(token: str) -> Dict[str, Any]:
    data = await backend_client.get_user_profile(token)
    if "error" in data:
        return {"status": "error", "error": data["error"]}
    return {
        "status": "success",
        "profile": {
            "name": f'{data.get("firstName", "")} {data.get("lastName", "")}'.strip(),
            "email": data.get("email", ""),
            "phone": data.get("contactPhoneNumber", ""),
            "phoneVerified": data.get("contactPhoneVerified", False),
        },
    }


async def _get_membership(token: str) -> Dict[str, Any]:
    data = await backend_client.get_user_membership(token)
    if "error" in data:
        return {
            "status": "success",
            "membership": {
                "active": False,
                "packageName": "",
                "packageLevel": "",
                "message": "Bạn chưa có gói membership nào đang hoạt động.",
            },
        }
    package = data.get("membershipPackage") or {}
    return {
        "status": "success",
        "membership": {
            "active": data.get("status") == "ACTIVE",
            "packageName": package.get("packageName", ""),
            "packageLevel": package.get("packageLevel", ""),
            "startDate": data.get("startDate", ""),
            "endDate": data.get("endDate", ""),
            "status": data.get("status", ""),
        },
    }


async def _get_saved(token: str) -> Dict[str, Any]:
    data = await backend_client.get_saved_listings(token, page=1, size=10)
    if "error" in data:
        return {"status": "error", "error": data["error"]}

    saved_items = data.get("data", data.get("listings", []))
    if isinstance(saved_items, list):
        listings = []
        for item in saved_items:
            listing = item.get("listing", item)
            addr = listing.get("address") or {}
            listings.append(
                {
                    "listingId": str(listing.get("listingId", "")),
                    "title": listing.get("title", ""),
                    "price": listing.get("price"),
                    "districtName": addr.get("districtName", ""),
                    "productType": listing.get("productType", ""),
                }
            )
        return {
            "status": "success",
            "count": len(listings),
            "savedListings": listings,
        }

    return {"status": "success", "count": 0, "savedListings": []}


async def _dispatch_user_info(
    ctx: RunContextWrapper[ToolContext], info_type: str
) -> Dict[str, Any]:
    """Core dispatch logic — separated so it can be called directly in tests."""
    token = ctx.context.auth_token
    if not token:
        return {
            "status": "error",
            "error": (
                "Người dùng chưa đăng nhập. "
                "Hãy hướng dẫn người dùng đăng nhập để xem thông tin tài khoản."
            ),
        }

    try:
        if info_type == "profile":
            return await _get_profile(token)
        if info_type == "membership":
            return await _get_membership(token)
        if info_type == "saved_listings":
            return await _get_saved(token)
        return {"status": "error", "error": f"Unknown info type: {info_type}"}
    except httpx.HTTPStatusError as e:
        logger.error("Backend HTTP %s for get_user_info", e.response.status_code)
        return {
            "status": "error",
            "error": f"Backend returned HTTP {e.response.status_code}",
        }
    except Exception as e:
        logger.error("get_user_info failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


@function_tool(
    name_override="get_user_info",
    description_override=(
        "Get the current user's account information. Use when the user asks "
        "about their profile, subscription/membership, or saved/bookmarked "
        "listings. Example: 'tài khoản của tôi', 'gói dịch vụ của tôi', "
        "'tin đã lưu'."
    ),
)
async def get_user_info(
    ctx: RunContextWrapper[ToolContext],
    infoType: Annotated[
        str,
        Field(
            description=(
                "Type of information to retrieve: 'profile' (user profile), "
                "'membership' (subscription/VIP status), "
                "'saved_listings' (bookmarked listings)."
            ),
            json_schema_extra={"enum": ["profile", "membership", "saved_listings"]},
        ),
    ],
) -> Dict[str, Any]:
    return await _dispatch_user_info(ctx, infoType)
