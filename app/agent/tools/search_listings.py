"""
Tool: search_listings — calls the SmartRent backend and returns a compact
summary for the LLM. Raw listings are appended to ToolContext.collected_listings
so the orchestrator can return them to the frontend.
"""

import logging
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
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
    for key in ("bedrooms", "bathrooms", "furnishing", "direction"):
        val = item.get(key)
        if val is not None:
            summary[key] = val
    return summary


async def _do_search(
    ctx: RunContextWrapper[ToolContext], params: Dict[str, Any]
) -> Dict[str, Any]:
    """Core search logic — separated so it can be called directly in tests."""
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
        ctx.context.collected_listings.extend(listings)

        return {
            "status": "success",
            "count": len(listings),
            "totalCount": data.get("totalCount", len(listings)),
            "currentPage": params.get("page", 1),
            "pageSize": params["size"],
            "listings": [_compact_search_item(item) for item in listings],
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


@function_tool(
    name_override="search_listings",
    description_override=(
        "Search for real estate listings on SmartRent. Use this when the user "
        "wants to find properties to rent or buy based on location, price, "
        "size, furnishing, or other criteria. Always call this before "
        "presenting any listings to the user."
    ),
)
async def search_listings(
    ctx: RunContextWrapper[ToolContext],
    keyword: Annotated[
        Optional[str],
        Field(
            description="Free-text search (title, address, description). Vietnamese supported."
        ),
    ] = None,
    provinceCode: Annotated[
        Optional[str],
        Field(
            description=(
                "Province/city code. Common values: 01=Hà Nội, 79=TP. Hồ Chí Minh, "
                "48=Đà Nẵng, 92=Cần Thơ, 31=Hải Phòng."
            )
        ),
    ] = None,
    provinceId: Annotated[
        Optional[str],
        Field(
            description=(
                "Province/city ID (same value as provinceCode). Sent together "
                "with provinceCode for backward compatibility."
            )
        ),
    ] = None,
    districtId: Annotated[
        Optional[int],
        Field(description="District ID (integer). E.g. 760=Quận 1, 765=Bình Thạnh."),
    ] = None,
    productType: Annotated[
        Optional[str],
        Field(
            description=(
                "ROOM (phòng trọ), APARTMENT (chung cư), HOUSE (nhà nguyên căn), "
                "OFFICE (văn phòng), STUDIO (căn hộ studio)."
            )
        ),
    ] = None,
    listingType: Annotated[
        Optional[str],
        Field(
            description="RENT for rental, SALE for sale, SHARE for shared. Default context is RENT."
        ),
    ] = None,
    minPrice: Annotated[
        Optional[float],
        Field(description="Minimum price in VND (e.g. 5000000 = 5 triệu)."),
    ] = None,
    maxPrice: Annotated[
        Optional[float], Field(description="Maximum price in VND.")
    ] = None,
    minArea: Annotated[
        Optional[float], Field(description="Minimum area in m².")
    ] = None,
    maxArea: Annotated[
        Optional[float], Field(description="Maximum area in m².")
    ] = None,
    minBedrooms: Annotated[
        Optional[int], Field(description="Minimum number of bedrooms.")
    ] = None,
    maxBedrooms: Annotated[
        Optional[int], Field(description="Maximum number of bedrooms.")
    ] = None,
    bedrooms: Annotated[
        Optional[int], Field(description="Exact number of bedrooms.")
    ] = None,
    bathrooms: Annotated[
        Optional[int], Field(description="Exact number of bathrooms.")
    ] = None,
    furnishing: Annotated[
        Optional[str],
        Field(
            description=(
                "FULLY_FURNISHED (đầy đủ nội thất), SEMI_FURNISHED (nội thất "
                "cơ bản), UNFURNISHED (không nội thất)."
            )
        ),
    ] = None,
    direction: Annotated[
        Optional[str],
        Field(
            description=(
                "Facing direction: NORTH, SOUTH, EAST, WEST, NORTHEAST, "
                "NORTHWEST, SOUTHEAST, SOUTHWEST."
            )
        ),
    ] = None,
    amenityIds: Annotated[
        Optional[List[int]],
        Field(
            description="List of amenity IDs to filter by. E.g. [1,2] for WiFi + Điều hòa."
        ),
    ] = None,
    latitude: Annotated[
        Optional[float], Field(description="Latitude for location-based search.")
    ] = None,
    longitude: Annotated[
        Optional[float], Field(description="Longitude for location-based search.")
    ] = None,
    radiusKm: Annotated[
        Optional[float],
        Field(
            description="Search radius in kilometers (used with latitude/longitude)."
        ),
    ] = None,
    postedWithinDays: Annotated[
        Optional[int],
        Field(
            description="Only return listings posted within the last N days (e.g. 7)."
        ),
    ] = None,
    sortBy: Annotated[
        Optional[str],
        Field(
            description="Sort order: DEFAULT, PRICE_ASC, PRICE_DESC, NEWEST, OLDEST."
        ),
    ] = None,
    page: Annotated[
        Optional[int], Field(description="Page number (1-based, default 1).")
    ] = None,
    size: Annotated[
        Optional[int],
        Field(description=f"Results per page (default 5, max {_MAX_SIZE})."),
    ] = None,
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "keyword": keyword,
        "provinceCode": provinceCode,
        "provinceId": provinceId,
        "districtId": districtId,
        "productType": productType,
        "listingType": listingType,
        "minPrice": minPrice,
        "maxPrice": maxPrice,
        "minArea": minArea,
        "maxArea": maxArea,
        "minBedrooms": minBedrooms,
        "maxBedrooms": maxBedrooms,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "furnishing": furnishing,
        "direction": direction,
        "amenityIds": amenityIds,
        "latitude": latitude,
        "longitude": longitude,
        "radiusKm": radiusKm,
        "postedWithinDays": postedWithinDays,
        "sortBy": sortBy,
        "page": page,
        "size": size,
    }
    params: Dict[str, Any] = {k: v for k, v in raw.items() if v is not None}
    params.setdefault("size", 5)
    if params["size"] > _MAX_SIZE:
        params["size"] = _MAX_SIZE

    # Backend checks both old and new address fields — keep them in sync.
    if "provinceCode" in params:
        params.setdefault("provinceId", params["provinceCode"])
    elif "provinceId" in params:
        params.setdefault("provinceCode", params["provinceId"])

    return await _do_search(ctx, params)
