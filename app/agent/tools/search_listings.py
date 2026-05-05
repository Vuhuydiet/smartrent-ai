"""
Tool: search_listings

Calls the SmartRent backend listing search API and returns a compact summary
for the LLM plus the full raw listing objects for the API response layer.
"""

import logging
from typing import Any, Dict

import httpx
from vertexai.generative_models import FunctionDeclaration  # type: ignore[import]

from app.agent.tools.base_tool import BaseTool
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_SIZE = 50  # hard cap — prevents overloading the backend


def _compact_search_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fields the LLM needs, handling nested address."""
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
    # Optional fields — only include when present to save tokens
    for key in ("bedrooms", "bathrooms", "furnishing", "direction"):
        val = item.get(key)
        if val is not None:
            summary[key] = val
    return summary


class SearchListingsTool(BaseTool):
    name = "search_listings"
    description = (
        "Search for real estate listings on SmartRent. "
        "Use this when the user wants to find properties to rent or buy based on "
        "location, price, size, furnishing, or other criteria. "
        "Always call this before presenting any listings to the user."
    )

    def to_function_declaration(self) -> Any:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Free-text search (title, address, description). Vietnamese supported.",
                    },
                    "provinceCode": {
                        "type": "string",
                        "description": (
                            "Province/city code. Common values: "
                            "01=Hà Nội, 79=TP. Hồ Chí Minh, 48=Đà Nẵng, "
                            "92=Cần Thơ, 31=Hải Phòng."
                        ),
                    },
                    "provinceId": {
                        "type": "string",
                        "description": (
                            "Province/city ID (same value as provinceCode). "
                            "Must be sent together with provinceCode for backward compatibility."
                        ),
                    },
                    "districtId": {
                        "type": "integer",
                        "description": "District ID (integer). E.g. 760=Quận 1, 765=Bình Thạnh.",
                    },
                    "productType": {
                        "type": "string",
                        "description": (
                            "ROOM (phòng trọ), APARTMENT (chung cư), "
                            "HOUSE (nhà nguyên căn), OFFICE (văn phòng), "
                            "STUDIO (căn hộ studio)."
                        ),
                    },
                    "listingType": {
                        "type": "string",
                        "description": "RENT for rental, SALE for sale, SHARE for shared. Default context is RENT.",
                    },
                    "minPrice": {
                        "type": "number",
                        "description": "Minimum price in VND (e.g. 5000000 = 5 triệu).",
                    },
                    "maxPrice": {
                        "type": "number",
                        "description": "Maximum price in VND.",
                    },
                    "minArea": {
                        "type": "number",
                        "description": "Minimum area in m².",
                    },
                    "maxArea": {
                        "type": "number",
                        "description": "Maximum area in m².",
                    },
                    "minBedrooms": {
                        "type": "integer",
                        "description": "Minimum number of bedrooms.",
                    },
                    "maxBedrooms": {
                        "type": "integer",
                        "description": "Maximum number of bedrooms.",
                    },
                    "bedrooms": {
                        "type": "integer",
                        "description": "Exact number of bedrooms.",
                    },
                    "bathrooms": {
                        "type": "integer",
                        "description": "Exact number of bathrooms.",
                    },
                    "furnishing": {
                        "type": "string",
                        "description": (
                            "FULLY_FURNISHED (đầy đủ nội thất), "
                            "SEMI_FURNISHED (nội thất cơ bản), "
                            "UNFURNISHED (không nội thất)."
                        ),
                    },
                    "direction": {
                        "type": "string",
                        "description": (
                            "Facing direction: NORTH, SOUTH, EAST, WEST, "
                            "NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST."
                        ),
                    },
                    "amenityIds": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of amenity IDs to filter by. E.g. [1,2] for WiFi + Điều hòa.",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude for location-based search.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude for location-based search.",
                    },
                    "radiusKm": {
                        "type": "number",
                        "description": "Search radius in kilometers (used with latitude/longitude).",
                    },
                    "postedWithinDays": {
                        "type": "integer",
                        "description": "Only return listings posted within the last N days (e.g. 7).",
                    },
                    "sortBy": {
                        "type": "string",
                        "description": "Sort order: DEFAULT, PRICE_ASC, PRICE_DESC, NEWEST, OLDEST.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-based, default 1).",
                    },
                    "size": {
                        "type": "integer",
                        "description": f"Results per page (default 5, max {_MAX_SIZE}).",
                    },
                },
            },
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {**kwargs}
        params.setdefault("size", 5)
        if params["size"] > _MAX_SIZE:
            params["size"] = _MAX_SIZE

        # Cast fields that the backend expects as integers.
        for int_field in (
            "districtId",
            "minBedrooms",
            "maxBedrooms",
            "bedrooms",
            "bathrooms",
            "page",
            "size",
            "postedWithinDays",
        ):
            if int_field in params and isinstance(params[int_field], float):
                params[int_field] = int(params[int_field])

        # Backend checks both old and new address fields — keep them in sync.
        if "provinceCode" in params:
            params.setdefault("provinceId", params["provinceCode"])
        elif "provinceId" in params:
            params.setdefault("provinceCode", params["provinceId"])

        try:
            logger.info("search_listings request params: %s", params)
            data = await backend_client.search_listings(params)
            raw_listings = data.get("listings", [])
            listing_ids = [str(item.get("listingId", "?")) for item in raw_listings]
            logger.info(
                "search_listings response: %d listings, totalCount=%s, IDs=%s",
                len(raw_listings),
                data.get("totalCount"),
                listing_ids,
            )

            if "error" in data:
                return {
                    "status": "error",
                    "error": data["error"],
                    "code": data.get("code"),
                }

            listings: list = data.get("listings", [])

            return {
                "status": "success",
                "count": len(listings),
                "totalCount": data.get("totalCount", len(listings)),
                "currentPage": params.get("page", 1),
                "pageSize": params["size"],
                "listings": [_compact_search_item(item) for item in listings],
                "_raw_listings": listings,
            }

        except httpx.HTTPStatusError as e:
            logger.error("Backend HTTP %s for search_listings", e.response.status_code)
            return {
                "status": "error",
                "error": f"Backend returned HTTP {e.response.status_code}",
            }
        except Exception as e:
            logger.error("search_listings failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}
