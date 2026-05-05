"""
Tool: get_recommendations — personalized or similar listing recommendations.

Two modes:
1. **Similar listings** — when the user references a specific listing
   ("tìm phòng tương tự cái này"), calls GET /v1/recommendations/similar/{id}
2. **Personalized feed** — when the user asks for general suggestions
   ("gợi ý cho tôi"), calls GET /v1/recommendations/personalized.
   Returns a hint to fall back to search_listings if not authenticated.
"""

import logging
from typing import Annotated, Any, Dict, Optional

import httpx
from google.genai import types  # type: ignore[import]

from app.agent.tool_context import ToolContext
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
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={  # type: ignore[arg-type]
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
    ctx.context.collected_listings.extend(listings)
    return {
        "status": "success",
        "mode": data.get("mode", "similar"),
        "count": len(listings),
        "listings": [_compact_recommendation_item(item) for item in listings],
    }


async def _personalized(
    ctx: RunContextWrapper[ToolContext], top_n: int, token: Optional[str]
) -> Dict[str, Any]:
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
    ctx.context.collected_listings.extend(listings)
    return {
        "status": "success",
        "mode": data.get("mode", "personalized"),
        "coldStart": data.get("coldStart", False),
        "count": len(listings),
        "listings": [_compact_recommendation_item(item) for item in listings],
    }


@function_tool(
    name_override="get_recommendations",
    description_override=(
        "Get property recommendations for the user. Use this when the user "
        "asks for suggestions, recommendations, or says 'gợi ý cho tôi', "
        "'đề xuất phòng', 'tìm phòng phù hợp'. Can also find listings similar "
        "to a specific listing the user likes."
    ),
)
async def get_recommendations(
    ctx: RunContextWrapper[ToolContext],
    listingId: Annotated[
        Optional[str],
        Field(
            description=(
                "Optional: ID of a listing to find similar ones. "
                "Use when user says 'tìm phòng tương tự' or likes a specific listing."
            )
        ),
    ] = None,
    topN: Annotated[
        Optional[int],
        Field(description="Number of recommendations to return (default 5, max 20)."),
    ] = None,
) -> Dict[str, Any]:
    if listingId:
        try:
            listing_id: Optional[int] = int(float(listingId))
        except (ValueError, TypeError):
            listing_id = None
    else:
        listing_id = None

    top_n = max(1, min(int(topN or 5), 20))
    token = ctx.context.auth_token

    try:
        if listing_id is not None:
            return await _similar(ctx, listing_id, top_n, token)
        return await _personalized(ctx, top_n, token)
    except httpx.HTTPStatusError as e:
        logger.error("Backend HTTP %s for recommendations", e.response.status_code)
        return {
            "status": "error",
            "error": f"Backend returned HTTP {e.response.status_code}",
        }
    except Exception as e:
        logger.error("get_recommendations failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}
