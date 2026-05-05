"""
Tool: get_recommendations

Returns personalized or similar listing recommendations via the SmartRent
backend recommendation engine.

Two modes:
1. **Similar listings** — when the user references a specific listing
   ("tìm phòng tương tự cái này"), calls GET /v1/recommendations/similar/{id}
2. **Personalized feed** — when the user asks for general suggestions
   ("gợi ý cho tôi"), calls GET /v1/recommendations/personalized.
   Falls back to search_listings if the user is not authenticated.
"""

import logging
from typing import Any, Dict

import httpx
from vertexai.generative_models import FunctionDeclaration  # type: ignore[import]

from app.agent.tools.base_tool import BaseTool
from app.core import backend_client

logger = logging.getLogger(__name__)


def _compact_recommendation_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fields the LLM needs from a recommendation listing."""
    addr = item.get("address") or {}
    summary: Dict[str, Any] = {
        "listingId": str(item.get("listingId", "")),
        "title": item.get("title", ""),
        "price": item.get("price"),
        "priceUnit": item.get("priceUnit", ""),
        "area": item.get("area"),
        "districtName": addr.get("districtName", ""),
        "wardName": addr.get("wardName", ""),
        "productType": item.get("productType", ""),
        "listingType": item.get("listingType", ""),
    }
    for key in ("bedrooms", "bathrooms", "furnishing"):
        val = item.get(key)
        if val is not None:
            summary[key] = val
    return summary


class GetRecommendationsTool(BaseTool):
    name = "get_recommendations"
    description = (
        "Get property recommendations for the user. "
        "Use this when the user asks for suggestions, recommendations, or says "
        "'gợi ý cho tôi', 'đề xuất phòng', 'tìm phòng phù hợp'. "
        "Can also find listings similar to a specific listing the user likes."
    )

    def to_function_declaration(self) -> Any:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "listingId": {
                        "type": "string",
                        "description": (
                            "Optional: ID of a listing to find similar ones. "
                            "Use when user says 'tìm phòng tương tự' or likes a specific listing."
                        ),
                    },
                    "topN": {
                        "type": "integer",
                        "description": "Number of recommendations to return (default 5, max 20).",
                    },
                },
            },
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        raw_id = kwargs.get("listingId")
        if raw_id:
            try:
                listing_id = str(int(float(raw_id)))
            except (ValueError, TypeError):
                listing_id = str(raw_id)
        else:
            listing_id = None
        top_n = max(1, min(int(kwargs.get("topN", 5)), 20))

        # Extract auth token from execution context if available
        context = kwargs.get("context") or {}
        token = context.get("auth_token")

        try:
            if listing_id:
                return await self._similar(int(listing_id), top_n, token)
            else:
                return await self._personalized(top_n, token)
        except httpx.HTTPStatusError as e:
            logger.error("Backend HTTP %s for recommendations", e.response.status_code)
            return {
                "status": "error",
                "error": f"Backend returned HTTP {e.response.status_code}",
            }
        except Exception as e:
            logger.error("get_recommendations failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _similar(
        self, listing_id: int, top_n: int, token: str | None
    ) -> Dict[str, Any]:
        """Get similar listings from the backend recommendation engine."""
        data = await backend_client.get_similar_listings(listing_id, top_n, token)

        if "error" in data:
            return {"status": "error", "error": data["error"]}

        listings = data.get("listings", [])
        return {
            "status": "success",
            "mode": data.get("mode", "similar"),
            "count": len(listings),
            "listings": [_compact_recommendation_item(item) for item in listings],
            "_raw_listings": listings,
        }

    async def _personalized(self, top_n: int, token: str | None) -> Dict[str, Any]:
        """Get personalized recommendations. Falls back to search if not authenticated."""
        if not token:
            return {
                "status": "error",
                "error": (
                    "Người dùng chưa đăng nhập. Không thể lấy gợi ý cá nhân hóa. "
                    "Hãy hỏi người dùng về sở thích (vị trí, giá, loại BĐS) rồi "
                    "dùng search_listings thay thế."
                ),
            }

        data = await backend_client.get_personalized_recommendations(top_n, token)

        if "error" in data:
            return {"status": "error", "error": data["error"]}

        listings = data.get("listings", [])
        return {
            "status": "success",
            "mode": data.get("mode", "personalized"),
            "coldStart": data.get("coldStart", False),
            "count": len(listings),
            "listings": [_compact_recommendation_item(item) for item in listings],
            "_raw_listings": listings,
        }
