"""
Tool: save_listing

Save or unsave a listing to/from the user's favorites on SmartRent.
Requires a valid auth token in the execution context.
"""

import logging
from typing import Any, Dict

import httpx
from google.genai import types  # type: ignore[import]

from app.agent.tools.base_tool import BaseTool
from app.core import backend_client

logger = logging.getLogger(__name__)


class SaveListingTool(BaseTool):
    name = "save_listing"
    description = (
        "Save or unsave a property listing to/from the user's favorites. "
        "Use when the user says 'lưu tin này', 'thêm vào yêu thích', "
        "'bỏ lưu', or 'xóa khỏi yêu thích'."
    )

    def to_function_declaration(self) -> Any:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "listingId": {
                        "type": "string",
                        "description": "The listing ID to save or unsave.",
                    },
                    "action": {
                        "type": "string",
                        "description": "'save' to add to favorites, 'unsave' to remove.",
                        "enum": ["save", "unsave"],
                    },
                },
                "required": ["listingId", "action"],
            },
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        raw_id = kwargs["listingId"]
        try:
            listing_id = str(int(float(raw_id)))
        except (ValueError, TypeError):
            listing_id = str(raw_id)
        action = kwargs.get("action", "save")
        context = kwargs.get("context") or {}
        token = context.get("auth_token")

        if not token:
            return {
                "status": "error",
                "error": (
                    "Người dùng chưa đăng nhập. "
                    "Hãy hướng dẫn người dùng đăng nhập để lưu tin."
                ),
            }

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
