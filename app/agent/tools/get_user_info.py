"""
Tool: get_user_info

Fetches the authenticated user's profile, subscription, or saved listings
from the SmartRent backend. Requires a valid auth token in the execution context.
"""

import logging
from typing import Any, Dict

import httpx
from vertexai.generative_models import FunctionDeclaration  # type: ignore[import]

from app.agent.tools.base_tool import BaseTool
from app.core import backend_client

logger = logging.getLogger(__name__)


class GetUserInfoTool(BaseTool):
    name = "get_user_info"
    description = (
        "Get the current user's account information. "
        "Use when the user asks about their profile, subscription/membership, "
        "or saved/bookmarked listings. "
        "Example: 'tài khoản của tôi', 'gói dịch vụ của tôi', 'tin đã lưu'."
    )

    def to_function_declaration(self) -> Any:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "infoType": {
                        "type": "string",
                        "description": (
                            "Type of information to retrieve: "
                            "'profile' (user profile), "
                            "'membership' (subscription/VIP status), "
                            "'saved_listings' (bookmarked listings)."
                        ),
                        "enum": ["profile", "membership", "saved_listings"],
                    },
                },
                "required": ["infoType"],
            },
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        info_type = kwargs["infoType"]
        context = kwargs.get("context") or {}
        token = context.get("auth_token")

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
                return await self._get_profile(token)
            elif info_type == "membership":
                return await self._get_membership(token)
            elif info_type == "saved_listings":
                return await self._get_saved(token)
            else:
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

    async def _get_profile(self, token: str) -> Dict[str, Any]:
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

    async def _get_membership(self, token: str) -> Dict[str, Any]:
        data = await backend_client.get_user_membership(token)
        if "error" in data:
            # No active membership is not an error — it's a valid state
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

    async def _get_saved(self, token: str) -> Dict[str, Any]:
        data = await backend_client.get_saved_listings(token, page=1, size=10)
        if "error" in data:
            return {"status": "error", "error": data["error"]}

        # data is a page response with "data" list inside
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
