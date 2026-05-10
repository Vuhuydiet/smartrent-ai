"""
Tool: search_listings — calls the SmartRent backend and returns a compact
summary for the LLM. Raw listings are appended to ToolContext.collected_listings
so the orchestrator can return them to the frontend.

Supports a single `productType` (exact term) OR a `productTypes` array (for
ambiguous Vietnamese terms like "nhà trọ" that span multiple backend enum
values). When multiple types are passed, we fan out one backend call per
type in parallel and merge results — the backend's `productType` filter
is a single-enum match, so this is the only way to get a union without a
backend change.
"""

import asyncio
import logging
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_SIZE = 50  # hard cap — prevents overloading the backend
_MAX_PRODUCT_TYPES = 5
_VALID_PRODUCT_TYPES = frozenset(
    {"ROOM", "APARTMENT", "HOUSE", "STUDIO", "OFFICE"}
)


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


async def _do_multi_type_search(
    ctx: RunContextWrapper[ToolContext],
    params: Dict[str, Any],
    product_types: List[str],
) -> Dict[str, Any]:
    """
    Fan out one backend call per `productType` in `product_types`, then merge.

    Pagination semantics: each underlying call asks for `size` items from
    page `page`. We dedupe by listingId and keep insertion order across
    types. totalCount returned is SUM across types (slight over-count when
    a listing somehow appears in multiple — rare since the backend's
    productType is a strict single-enum match).
    """
    size = params.get("size", 5)
    logger.info(
        "search_listings multi-type fan-out: types=%s size=%d",
        product_types,
        size,
    )

    async def _one(pt: str) -> Dict[str, Any]:
        sub = {**params, "productType": pt}
        try:
            return await backend_client.search_listings(sub)
        except httpx.HTTPStatusError as e:
            logger.warning("multi-type %s HTTP %s", pt, e.response.status_code)
            return {"error": f"HTTP {e.response.status_code}", "listings": []}
        except Exception as e:  # noqa: BLE001
            logger.warning("multi-type %s failed: %s", pt, e)
            return {"error": str(e), "listings": []}

    results = await asyncio.gather(*(_one(pt) for pt in product_types))

    merged_raw: List[Dict[str, Any]] = []
    seen_ids: set = set()
    total_count = 0
    per_type_counts: Dict[str, int] = {}
    for pt, data in zip(product_types, results):
        if "error" in data and not data.get("listings"):
            per_type_counts[pt] = 0
            continue
        sub_listings = data.get("listings") or []
        per_type_counts[pt] = len(sub_listings)
        total_count += int(data.get("totalCount") or 0)
        for item in sub_listings:
            lid = str(item.get("listingId", ""))
            if lid and lid not in seen_ids:
                seen_ids.add(lid)
                merged_raw.append(item)

    merged_raw = merged_raw[:size]
    ctx.context.collected_listings.extend(merged_raw)

    return {
        "status": "success",
        "count": len(merged_raw),
        "totalCount": total_count,
        "currentPage": params.get("page", 1),
        "pageSize": size,
        "productTypes": product_types,
        "perTypeCount": per_type_counts,
        "listings": [_compact_search_item(item) for item in merged_raw],
    }


def _normalise_product_types(raw: Optional[List[str]]) -> List[str]:
    """Uppercase, dedupe, validate against enum, cap at _MAX_PRODUCT_TYPES."""
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for t in raw:
        tt = str(t).upper().strip()
        if tt in _VALID_PRODUCT_TYPES and tt not in out:
            out.append(tt)
    return out[:_MAX_PRODUCT_TYPES]


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
                "Province code — works for BOTH old (63-province, pre-2025-07) "
                "and new (34-province, post-reform) structures; backend resolves "
                "bidirectionally. PREFERRED location parameter. Values: "
                "01=Hà Nội, 79=TP. Hồ Chí Minh, 48=Đà Nẵng, 92=Cần Thơ, "
                "31=Hải Phòng. Always set when the user mentions a province."
            )
        ),
    ] = None,
    provinceId: Annotated[
        Optional[str],
        Field(
            description=(
                "LEGACY province ID (pre-reform). Same numeric value as "
                "provinceCode for major cities. Set alongside provinceCode "
                "for backward compatibility with old-structure listings."
            )
        ),
    ] = None,
    districtId: Annotated[
        Optional[int],
        Field(
            description=(
                "LEGACY district ID (pre-2025-07 3-tier structure). Use when "
                "the user names a district like 'Bình Thạnh', 'Quận 1', 'Cầu "
                "Giấy' — districts no longer exist in the new 2-tier structure, "
                "but the backend reverse-maps districtId to new ward codes via "
                "address_mapping. Examples: 760=Quận 1, 765=Bình Thạnh, "
                "1=Ba Đình."
            )
        ),
    ] = None,
    newWardCode: Annotated[
        Optional[str],
        Field(
            description=(
                "NEW ward code (post-2025-07 2-tier structure). Use ONLY when "
                "the user explicitly names a post-reform ward AND you have its "
                "exact code. For ordinary queries by district name, prefer "
                "districtId — backend handles the mapping."
            )
        ),
    ] = None,
    wardId: Annotated[
        Optional[str],
        Field(
            description=(
                "LEGACY ward ID (pre-reform). Rarely needed; prefer districtId "
                "or newWardCode. Backend resolves to new ward codes via "
                "address_mapping."
            )
        ),
    ] = None,
    productType: Annotated[
        Optional[str],
        Field(
            description=(
                "Single property type — use ONLY for unambiguous Vietnamese "
                "terms. For ambiguous terms like 'nhà trọ', use `productTypes` "
                "(array) instead. Exact mapping:\n"
                "  ROOM       ← 'phòng trọ', 'phòng đơn', 'phòng cho thuê'\n"
                "  APARTMENT  ← 'căn hộ', 'chung cư'\n"
                "  HOUSE      ← 'nhà nguyên căn', 'nhà riêng'\n"
                "  STUDIO     ← 'studio', 'căn hộ studio'\n"
                "  OFFICE     ← 'văn phòng', 'office'\n"
                "If `productTypes` is also set, this single value is ignored."
            )
        ),
    ] = None,
    productTypes: Annotated[
        Optional[List[str]],
        Field(
            description=(
                "Multiple property types to search across (results merged + "
                "deduped). Use for AMBIGUOUS Vietnamese terms where intent "
                "spans types:\n"
                "  'nhà trọ', 'trọ' → ['ROOM', 'APARTMENT'] (VN usage covers "
                "both small rentals and apartments)\n"
                "  'thuê nhà', 'tìm nhà' → ['ROOM', 'APARTMENT', 'HOUSE'] "
                "(broad rental search, exclude OFFICE/STUDIO)\n"
                "Leave UNSET (and leave productType UNSET) for fully open "
                "queries like 'có gì cho thuê ở Q1?'. Max 5 types."
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
        "newWardCode": newWardCode,
        "wardId": wardId,
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

    # productTypes (array) wins over productType (single). When 2+ types are
    # passed, fan out and merge. A 1-element array collapses to a single call.
    types_list = _normalise_product_types(productTypes)
    if len(types_list) >= 2:
        params.pop("productType", None)
        return await _do_multi_type_search(ctx, params, types_list)
    if len(types_list) == 1:
        params["productType"] = types_list[0]

    return await _do_search(ctx, params)
