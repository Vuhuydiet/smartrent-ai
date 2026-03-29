"""
Tool: search_listings

Calls the SmartRent backend listing search API and returns a compact summary
for the LLM plus the full raw listing objects for the API response layer.
"""

import logging
from typing import Any, Dict

import httpx
from google.ai.generativelanguage import (  # type: ignore[import]
    FunctionDeclaration,
    Schema,
    Type,
)

from app.agent.tools.base_tool import BaseTool
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_SIZE = 50  # hard cap — prevents overloading the backend


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
            parameters=Schema(
                type=Type.OBJECT,
                properties={
                    "keyword": Schema(
                        type=Type.STRING,
                        description="Free-text search (title, address, description). Vietnamese supported.",
                    ),
                    "provinceCode": Schema(
                        type=Type.STRING,
                        description=(
                            "Province/city code. Common values: "
                            "01=Hà Nội, 79=TP. Hồ Chí Minh, 48=Đà Nẵng, "
                            "92=Cần Thơ, 31=Hải Phòng."
                        ),
                    ),
                    "provinceId": Schema(
                        type=Type.STRING,
                        description=(
                            "Province/city ID (same value as provinceCode). "
                            "Must be sent together with provinceCode for backward compatibility."
                        ),
                    ),
                    "districtId": Schema(
                        type=Type.INTEGER,
                        description="District ID (integer). E.g. 760=Quận 1, 765=Bình Thạnh.",
                    ),
                    "productType": Schema(
                        type=Type.STRING,
                        description=(
                            "ROOM (phòng trọ), APARTMENT (chung cư), "
                            "HOUSE (nhà nguyên căn), OFFICE (văn phòng), "
                            "STUDIO (căn hộ studio)."
                        ),
                    ),
                    "listingType": Schema(
                        type=Type.STRING,
                        description="RENT for rental, SALE for sale, SHARE for shared. Default context is RENT.",
                    ),
                    "minPrice": Schema(
                        type=Type.NUMBER,
                        description="Minimum price in VND (e.g. 5000000 = 5 triệu).",
                    ),
                    "maxPrice": Schema(
                        type=Type.NUMBER,
                        description="Maximum price in VND.",
                    ),
                    "minArea": Schema(
                        type=Type.NUMBER,
                        description="Minimum area in m².",
                    ),
                    "maxArea": Schema(
                        type=Type.NUMBER,
                        description="Maximum area in m².",
                    ),
                    "minBedrooms": Schema(
                        type=Type.INTEGER,
                        description="Minimum number of bedrooms.",
                    ),
                    "maxBedrooms": Schema(
                        type=Type.INTEGER,
                        description="Maximum number of bedrooms.",
                    ),
                    "bedrooms": Schema(
                        type=Type.INTEGER,
                        description="Exact number of bedrooms.",
                    ),
                    "bathrooms": Schema(
                        type=Type.INTEGER,
                        description="Exact number of bathrooms.",
                    ),
                    "furnishing": Schema(
                        type=Type.STRING,
                        description=(
                            "FULLY_FURNISHED (đầy đủ nội thất), "
                            "SEMI_FURNISHED (nội thất cơ bản), "
                            "UNFURNISHED (không nội thất)."
                        ),
                    ),
                    "direction": Schema(
                        type=Type.STRING,
                        description=(
                            "Facing direction: NORTH, SOUTH, EAST, WEST, "
                            "NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST."
                        ),
                    ),
                    "amenityIds": Schema(
                        type=Type.ARRAY,
                        items=Schema(type=Type.INTEGER),
                        description="List of amenity IDs to filter by. E.g. [1,2] for WiFi + Điều hòa.",
                    ),
                    "sortBy": Schema(
                        type=Type.STRING,
                        description="Sort order: DEFAULT, PRICE_ASC, PRICE_DESC, NEWEST, OLDEST.",
                    ),
                    "page": Schema(
                        type=Type.INTEGER,
                        description="Page number (1-based, default 1).",
                    ),
                    "size": Schema(
                        type=Type.INTEGER,
                        description=f"Results per page (default 5, max {_MAX_SIZE}).",
                    ),
                },
            ),
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {**kwargs}
        params.setdefault("size", 5)
        if params["size"] > _MAX_SIZE:
            params["size"] = _MAX_SIZE

        # Gemini returns all numbers as floats (protobuf Value.number_value).
        # Cast fields that the backend expects as integers.
        for int_field in (
            "districtId", "minBedrooms", "maxBedrooms", "bedrooms",
            "bathrooms", "page", "size",
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
            logger.info(
                "search_listings response: %d listings, totalCount=%s",
                len(data.get("listings", [])),
                data.get("totalCount"),
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
                "listings": [
                    {
                        "listingId": item.get("listingId"),
                        "title": item.get("title", ""),
                        "price": item.get("price"),
                        "area": item.get("area"),
                        "bedrooms": item.get("bedrooms"),
                        "bathrooms": item.get("bathrooms"),
                        "districtName": item.get("districtName", ""),
                        "wardName": item.get("wardName", ""),
                        "productType": item.get("productType", ""),
                        "furnishing": item.get("furnishing", ""),
                        "listingType": item.get("listingType", ""),
                    }
                    for item in listings
                ],
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
