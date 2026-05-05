"""
Tool: get_price_history

Fetches pricing history and statistics for listings from the SmartRent backend.

Three modes:
1. **Price history** — full price change timeline for a specific listing
2. **Price statistics** — min/max/avg and change counts for a listing
3. **Recent price changes** — find listings that recently dropped/increased in price
"""

import logging
from typing import Any, Dict, List

import httpx
from vertexai.generative_models import FunctionDeclaration  # type: ignore[import]

from app.agent.tools.base_tool import BaseTool
from app.core import backend_client

logger = logging.getLogger(__name__)


def _compact_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields the LLM needs from a pricing history record."""
    return {
        "oldPrice": entry.get("oldPrice"),
        "newPrice": entry.get("newPrice"),
        "changeType": entry.get("changeType", ""),
        "changePercentage": entry.get("changePercentage"),
        "changedAt": entry.get("changedAt", ""),
    }


class GetPriceHistoryTool(BaseTool):
    name = "get_price_history"
    description = (
        "Get pricing history, price statistics, or recent price changes for listings. "
        "Use when the user asks about price trends, whether a listing's price has "
        "changed, or wants to find listings with recent price drops. "
        "Example: 'tin này có giảm giá không?', 'lịch sử giá', 'tin nào mới giảm giá?'."
    )

    def to_function_declaration(self) -> Any:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "'history' — full price change timeline for a listing. "
                            "'statistics' — min/max/avg price and change counts. "
                            "'recent_changes' — find listings with recent price changes."
                        ),
                        "enum": ["history", "statistics", "recent_changes"],
                    },
                    "listingId": {
                        "type": "string",
                        "description": (
                            "Listing ID to check price history/statistics for. "
                            "Required for 'history' and 'statistics' actions."
                        ),
                    },
                    "daysBack": {
                        "type": "integer",
                        "description": (
                            "Number of days to look back for 'recent_changes' action. "
                            "Default 7. Example: 30 = price changes in the last month."
                        ),
                    },
                },
                "required": ["action"],
            },
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        action = kwargs["action"]
        raw_id = kwargs.get("listingId")
        if raw_id:
            try:
                listing_id = str(int(float(raw_id)))
            except (ValueError, TypeError):
                listing_id = str(raw_id)
        else:
            listing_id = None
        days_back = max(1, min(int(kwargs.get("daysBack", 7)), 365))

        try:
            if action == "history":
                return await self._get_history(listing_id)
            elif action == "statistics":
                return await self._get_statistics(listing_id)
            elif action == "recent_changes":
                return await self._get_recent_changes(days_back)
            else:
                return {"status": "error", "error": f"Unknown action: {action}"}

        except httpx.HTTPStatusError as e:
            logger.error(
                "Backend HTTP %s for get_price_history", e.response.status_code
            )
            return {
                "status": "error",
                "error": f"Backend returned HTTP {e.response.status_code}",
            }
        except Exception as e:
            logger.error("get_price_history failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _get_history(self, listing_id: str | None) -> Dict[str, Any]:
        if not listing_id:
            return {
                "status": "error",
                "error": "listingId is required for 'history' action.",
            }

        data = await backend_client.get_pricing_history(int(listing_id))
        if "error" in data:
            return {"status": "error", "error": data["error"]}

        history: List[Dict[str, Any]] = data.get("history", [])
        # Filter out ADJUSTED synthetic points — not useful for user-facing answers
        real_changes = [
            _compact_history_entry(e)
            for e in history
            if e.get("changeType") != "ADJUSTED"
        ]

        if not real_changes:
            return {
                "status": "success",
                "listingId": listing_id,
                "totalChanges": 0,
                "message": "Tin đăng này chưa có lịch sử thay đổi giá.",
            }

        return {
            "status": "success",
            "listingId": listing_id,
            "totalChanges": len(real_changes),
            "currentPrice": real_changes[-1].get("newPrice"),
            "history": real_changes[-5:],  # Last 5 changes to save tokens
        }

    async def _get_statistics(self, listing_id: str | None) -> Dict[str, Any]:
        if not listing_id:
            return {
                "status": "error",
                "error": "listingId is required for 'statistics' action.",
            }

        data = await backend_client.get_price_statistics(int(listing_id))
        if "error" in data:
            return {"status": "error", "error": data["error"]}

        return {
            "status": "success",
            "listingId": listing_id,
            "minPrice": data.get("minPrice"),
            "maxPrice": data.get("maxPrice"),
            "avgPrice": data.get("avgPrice"),
            "totalChanges": data.get("totalChanges", 0),
            "priceIncreases": data.get("priceIncreases", 0),
            "priceDecreases": data.get("priceDecreases", 0),
        }

    async def _get_recent_changes(self, days_back: int) -> Dict[str, Any]:
        data = await backend_client.get_recent_price_changes(
            days_back=days_back, page=1, size=20
        )
        if "error" in data:
            return {"status": "error", "error": data["error"]}

        listing_ids = data.get("data", [])
        return {
            "status": "success",
            "daysBack": days_back,
            "totalListings": data.get("totalElements", len(listing_ids)),
            "listingIds": listing_ids[:10],  # Top 10 to save tokens
            "message": (
                f"Có {data.get('totalElements', len(listing_ids))} tin đăng "
                f"thay đổi giá trong {days_back} ngày qua."
            ),
        }
