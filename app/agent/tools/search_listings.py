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
from typing import Annotated, Any, Dict, List, Literal, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.enum_labels import localize_listing_enums
from app.agent.tool_context import ToolContext
from app.core import backend_client
from app.core.config import settings

logger = logging.getLogger(__name__)

_MAX_SIZE = 50  # hard cap — prevents overloading the backend
_MAX_PRODUCT_TYPES = 5
_VALID_PRODUCT_TYPES = frozenset({"ROOM", "APARTMENT", "HOUSE", "STUDIO", "OFFICE"})

# A search must be scoped by at least one of these (or a keyword); otherwise it
# would scan the whole dataset. Guards against the model calling search with no
# criteria instead of asking the user where they want to look.
_LOCATION_KEYS = (
    "provinceCode",
    "provinceId",
    "districtCode",
    "newWardCode",
    "wardId",
    "latitude",
)
_NO_CRITERIA_ERROR = (
    "Cần ít nhất một tiêu chí vị trí (provinceCode/districtCode/newWardCode/...) "
    "hoặc keyword để tìm kiếm. Hãy hỏi người dùng khu vực họ muốn tìm trước khi gọi lại."
)
_ZERO_RESULT_HINT = (
    "0 kết quả với bộ lọc hiện tại. KHÔNG tự đổi sang khu vực khác. Báo người dùng "
    "không tìm thấy, rồi hỏi xem họ có muốn nới lỏng bộ lọc (giá, loại BĐS, khu vực "
    "lân cận) không."
)


def _has_search_criteria(params: Dict[str, Any]) -> bool:
    """True when the call is scoped by a location or a keyword."""
    has_location = any(params.get(k) for k in _LOCATION_KEYS)
    return has_location or bool(params.get("keyword"))


# Provinces large enough that a province-only query returns tens of thousands of
# listings — a "phòng trọ ở TP.HCM" search that is both slow to run and useless
# to show. For these we require at least one narrowing signal before hitting the
# backend. Data-driven: TP.HCM (~35k) and Hà Nội (~20k) dominate the catalogue;
# smaller provinces (e.g. Cần Thơ ~tens of results) are fine province-only, so
# they are deliberately excluded. Extend this set as the dataset grows.
_MAJOR_METRO_PROVINCE_CODES = frozenset({"79", "01"})

# Filters that meaningfully cut a metro result set below "the whole city". A bare
# province — even WITH a productType — is not enough: "phòng trọ ở TP.HCM" still
# spans tens of thousands of rows. So productType / productTypes / listingType /
# sortBy / page / size are intentionally NOT counted as narrowing here.
_NARROWING_KEYS = (
    "districtCode",
    "newWardCode",
    "wardId",
    "latitude",
    "keyword",
    "minPrice",
    "maxPrice",
    "minArea",
    "maxArea",
    "minBedrooms",
    "maxBedrooms",
    "bedrooms",
    "bathrooms",
    "amenityIds",
)

_NEED_NARROWING_MSG = (
    "Khu vực này có RẤT nhiều tin nên tìm kiểu chỉ-có-tỉnh sẽ ra hàng chục nghìn "
    "kết quả không sát nhu cầu. ĐỪNG gọi lại search vội. HÃY HỎI người dùng thêm "
    "ít nhất MỘT tiêu chí để lọc: khu vực cụ thể hơn (quận/huyện) HOẶC khoảng "
    "ngân sách (giá). Chỉ gọi lại search khi người dùng đã cung cấp thêm."
)


def _needs_narrowing(params: Dict[str, Any]) -> bool:
    """True when a major-metro search is scoped only to the province (too broad).

    Catches the slow, useless "whole big-city" query — a call on TP.HCM / Hà Nội
    with no district, budget, or other narrowing filter. The model should ask the
    user for one more criterion first, instead of dumping tens of thousands of
    rows. productType alone does NOT count (see ``_NARROWING_KEYS``).
    """
    province = params.get("provinceCode") or params.get("provinceId")
    if str(province) not in _MAJOR_METRO_PROVINCE_CODES:
        return False
    return not any(params.get(k) for k in _NARROWING_KEYS)


# The backend's ListingFilterRequest takes price / area / bedroom filters as a
# SINGLE `from..to` string (either side optional) — it has no minPrice/maxPrice
# fields. Sending those made Jackson silently drop them, so e.g. a maxPrice
# ceiling was ignored and over-budget listings came back. Each tuple maps the
# tool's (minKey, maxKey) to the backend's combined range field.
_RANGE_PARAM_MAP = (
    ("minPrice", "maxPrice", "price"),
    ("minArea", "maxArea", "area"),
    ("minBedrooms", "maxBedrooms", "bedroomsRange"),
)


def _fmt_bound(n: Any) -> str:
    """Render a numeric bound without a trailing '.0' for whole numbers.

    Bedroom bounds feed the backend's Integer range parser, which rejects
    '2.0'; price/area go through decimal/float parsers that tolerate it, but a
    clean integer string keeps the range readable in logs either way.
    """
    if isinstance(n, bool):  # bool is an int subclass — guard against True/False
        return str(n)
    if isinstance(n, float) and n.is_integer():
        return str(int(n))
    return str(n)


def _apply_range_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite min/max pairs into the backend's combined `from..to` fields.

    Mutates and returns `params`. Idempotent and safe when neither bound is
    present (the pair is left untouched). Applied at the backend boundary so
    both the single-type and multi-type search paths are covered.
    """
    for lo_key, hi_key, dst in _RANGE_PARAM_MAP:
        lo = params.pop(lo_key, None)
        hi = params.pop(hi_key, None)
        if lo is None and hi is None:
            continue
        lo_s = "" if lo is None else _fmt_bound(lo)
        hi_s = "" if hi is None else _fmt_bound(hi)
        params[dst] = f"{lo_s}..{hi_s}"
    return params


def _compact_search_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fields the LLM needs, handling nested address."""
    addr = item.get("address") or {}
    listing_id = str(item.get("listingId", ""))
    summary: Dict[str, Any] = {
        "listingId": listing_id,
        # Canonical share URL — the LLM must echo this verbatim, never build its
        # own domain/path.
        "url": f"{settings.FRONTEND_URL}/listing-detail/{listing_id}",
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
    return localize_listing_enums(summary)


async def _do_search(
    ctx: RunContextWrapper[ToolContext], params: Dict[str, Any]
) -> Dict[str, Any]:
    """Core search logic — separated so it can be called directly in tests."""
    _apply_range_params(params)
    if not _has_search_criteria(params):
        return {"status": "error", "error": _NO_CRITERIA_ERROR}
    try:
        logger.info("search_listings request params: %s", params)
        data = await backend_client.search_listings(
            params, token=ctx.context.auth_token
        )
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

        result: Dict[str, Any] = {
            "status": "success",
            "count": len(listings),
            "totalCount": data.get("totalCount", len(listings)),
            "currentPage": params.get("page", 1),
            "pageSize": params["size"],
            "listings": [_compact_search_item(item) for item in listings],
        }
        if not listings:
            result["hint"] = _ZERO_RESULT_HINT
        return result

    except httpx.HTTPStatusError as e:
        # Surface the backend's own message (e.g. an invalid-param 400) so the
        # model can self-correct, instead of an opaque "HTTP 400".
        status = e.response.status_code
        backend_msg = None
        try:
            body = e.response.json()
            backend_msg = body.get("message") or body.get("error")
        except Exception:
            backend_msg = (e.response.text or "").strip()[:300] or None
        logger.error("Backend HTTP %s for search_listings: %s", status, backend_msg)
        error = f"Backend returned HTTP {status}"
        if backend_msg:
            error += f": {backend_msg}"
        return {"status": "error", "error": error}
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
    _apply_range_params(params)
    if not _has_search_criteria(params):
        return {"status": "error", "error": _NO_CRITERIA_ERROR}

    size = params.get("size", 5)
    logger.info(
        "search_listings multi-type fan-out: types=%s size=%d",
        product_types,
        size,
    )

    token = ctx.context.auth_token

    async def _one(pt: str) -> Dict[str, Any]:
        sub = {**params, "productType": pt}
        try:
            return await backend_client.search_listings(sub, token=token)
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

    result: Dict[str, Any] = {
        "status": "success",
        "count": len(merged_raw),
        "totalCount": total_count,
        "currentPage": params.get("page", 1),
        "pageSize": size,
        "productTypes": product_types,
        "perTypeCount": per_type_counts,
        "listings": [_compact_search_item(item) for item in merged_raw],
    }
    if not merged_raw:
        result["hint"] = _ZERO_RESULT_HINT
    return result


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
    districtCode: Annotated[
        Optional[str],
        Field(
            description=(
                "District GSO administrative code (string) — use when the user "
                "names a district like 'Bình Thạnh', 'Quận 1', 'Cầu Giấy'. The "
                "backend resolves it to its internal district id and reverse-maps "
                "to new ward codes via address_mapping. Take the value from the "
                "MÃ ĐỊA ĐIỂM reference in the prompt. Examples: 760=Quận 1, "
                "765=Bình Thạnh, 005=Cầu Giấy. Do NOT pass a guessed integer "
                "districtId — only the GSO code is valid."
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
        Optional[Literal["ROOM", "APARTMENT", "HOUSE", "STUDIO", "OFFICE"]],
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
        Optional[Literal["RENT", "SALE", "SHARE"]],
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
        Optional[Literal["FULLY_FURNISHED", "SEMI_FURNISHED", "UNFURNISHED"]],
        Field(
            description=(
                "FULLY_FURNISHED (đầy đủ nội thất), SEMI_FURNISHED (nội thất "
                "cơ bản), UNFURNISHED (không nội thất)."
            )
        ),
    ] = None,
    direction: Annotated[
        Optional[
            Literal[
                "NORTH",
                "SOUTH",
                "EAST",
                "WEST",
                "NORTHEAST",
                "NORTHWEST",
                "SOUTHEAST",
                "SOUTHWEST",
            ]
        ],
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
        Optional[Literal["DEFAULT", "PRICE_ASC", "PRICE_DESC", "NEWEST", "OLDEST"]],
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
        "districtCode": districtCode,
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

    # Backend checks both old and new address fields — mirror provinceCode into
    # the legacy provinceId so old-structure listings are matched too. provinceId
    # is no longer model-facing (it only duplicated provinceCode and invited the
    # model to fill it wrongly).
    if "provinceCode" in params:
        params.setdefault("provinceId", params["provinceCode"])

    # Too-broad guard: a major-metro province with no narrowing filter would scan
    # the whole city (tens of thousands of rows — slow and useless). Ask the user
    # to narrow BEFORE calling the backend, rather than returning a giant set.
    # Runs before the product-type dispatch so it covers both the single- and
    # multi-type paths (e.g. "phòng trọ ở TP.HCM" → productTypes but no district).
    if _needs_narrowing(params):
        logger.info(
            "search_listings: metro province-only query (province=%s) — asking to narrow",
            params.get("provinceCode"),
        )
        return {"status": "need_narrowing", "message": _NEED_NARROWING_MSG}

    # productTypes (array) wins over productType (single). When 2+ types are
    # passed, fan out and merge. A 1-element array collapses to a single call.
    types_list = _normalise_product_types(productTypes)
    if len(types_list) >= 2:
        params.pop("productType", None)
        return await _do_multi_type_search(ctx, params, types_list)
    if len(types_list) == 1:
        params["productType"] = types_list[0]

    return await _do_search(ctx, params)
