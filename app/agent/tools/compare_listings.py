"""
Tool: compare_listings — compare EVERY listing in the user's current result set
side-by-side and return normalised rows the LLM turns into a table + verdict.

Takes no arguments on purpose. Comparison is a whole-result-set operation: the
user finds listings, then compares all of them. The LLM decides *whether* to
compare, never *which* listings — the target set is resolved server-side from
(in priority order):

  1. ToolContext.collected_listings — a search ran earlier in this same turn
     ("tìm phòng Q1 rồi so sánh giúp mình")
  2. ToolContext.last_listing_ids — the set shown in the previous turn, echoed
     back by the frontend as `last_listings`

That keeps listing IDs entirely out of the conversation: the user never types
one, and the model never has to resolve "tin 1 và 3" into IDs it could get
wrong or hallucinate.

Returning structured rows + headline callouts (cheapest, largest, best
price/m², most amenities) keeps the verdict grounded — the model only writes
prose about numbers it sees in the tool result.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]

from app.agent.enum_labels import localize_listing_enums
from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MIN_LISTINGS = 2
_MAX_LISTINGS = 5
_DESC_TRUNCATE = 200


def _price_per_sqm(price: Optional[float], area: Optional[float]) -> Optional[int]:
    """VND/m² rounded to int. Returns None if either input missing/zero."""
    try:
        if price is None or area is None:
            return None
        p, a = float(price), float(area)
        if a <= 0:
            return None
        return int(round(p / a))
    except (TypeError, ValueError):
        return None


def _normalise_for_comparison(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull fields that matter when comparing — keep token usage low."""
    addr = data.get("address") or {}
    price = data.get("price")
    area = data.get("area")
    amenities = [a.get("name") for a in (data.get("amenities") or []) if a.get("name")]
    row: Dict[str, Any] = {
        "listingId": str(data.get("listingId", "")),
        "title": data.get("title", ""),
        "price": price,
        "priceUnit": data.get("priceUnit", ""),
        "area": area,
        "pricePerSqm": _price_per_sqm(price, area),
        "districtName": addr.get("districtName", ""),
        "wardName": addr.get("wardName", ""),
        "fullAddress": addr.get("fullAddress", ""),
        "productType": data.get("productType", ""),
        "listingType": data.get("listingType", ""),
        "amenities": amenities,
        "amenityCount": len(amenities),
        "shortDescription": (data.get("description") or "")[:_DESC_TRUNCATE],
    }
    for key in (
        "bedrooms",
        "bathrooms",
        "furnishing",
        "direction",
        "waterPrice",
        "electricityPrice",
        "internetPrice",
        "serviceFee",
    ):
        val = data.get(key)
        if val is not None:
            row[key] = val
    return localize_listing_enums(row)


def _summary_callouts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute headline insights so the LLM doesn't have to reason about numbers.

    Keys in output (all listingId strings; missing data → key absent):
      cheapest, largest, bestPricePerSqm, mostAmenities,
      priceRangeVnd, areaRangeSqm
    """
    out: Dict[str, Any] = {}

    priced = [r for r in rows if isinstance(r.get("price"), (int, float))]
    if priced:
        cheapest = min(priced, key=lambda r: r["price"])
        out["cheapest"] = cheapest["listingId"]
        out["priceRangeVnd"] = [
            min(r["price"] for r in priced),
            max(r["price"] for r in priced),
        ]

    sized = [r for r in rows if isinstance(r.get("area"), (int, float))]
    if sized:
        largest = max(sized, key=lambda r: r["area"])
        out["largest"] = largest["listingId"]
        out["areaRangeSqm"] = [
            min(r["area"] for r in sized),
            max(r["area"] for r in sized),
        ]

    pps = [r for r in rows if r.get("pricePerSqm") is not None]
    if pps:
        best = min(pps, key=lambda r: r["pricePerSqm"])
        out["bestPricePerSqm"] = best["listingId"]

    if rows:
        most = max(rows, key=lambda r: r.get("amenityCount", 0))
        if most.get("amenityCount", 0) > 0:
            out["mostAmenities"] = most["listingId"]

    return out


async def _fetch_one(listing_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch one listing; on error return a sentinel row the LLM can flag."""
    try:
        data = await backend_client.get_listing(listing_id, token=token)
        if "error" in data:
            return {"listingId": listing_id, "_error": data["error"]}
        return data
    except httpx.HTTPStatusError as e:
        return {"listingId": listing_id, "_error": f"HTTP {e.response.status_code}"}
    except Exception as e:  # noqa: BLE001
        logger.warning("compare_listings: fetch %s failed: %s", listing_id, e)
        return {"listingId": listing_id, "_error": str(e)}


def _dedupe_ids(raw_ids: List[Any]) -> List[str]:
    """Coerce to clean string IDs, dropping blanks and duplicates, keeping order."""
    seen: set = set()
    out: List[str] = []
    for raw in raw_ids:
        try:
            cid = str(int(float(raw)))
        except (TypeError, ValueError):
            cid = str(raw or "").strip()
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _resolve_target_ids(ctx: RunContextWrapper[ToolContext]) -> Tuple[List[str], str]:
    """
    Resolve which listings to compare, without consulting the LLM.

    Returns (ids, source). Listings collected during THIS turn win over the
    previous turn's set — if the user searched and then asked to compare in one
    breath, the fresh results are what they mean.
    """
    collected = _dedupe_ids(
        [item.get("listingId") for item in ctx.context.collected_listings]
    )
    if len(collected) >= _MIN_LISTINGS:
        return collected, "current_turn"

    previous = _dedupe_ids(ctx.context.last_listing_ids)
    if len(previous) >= _MIN_LISTINGS:
        return previous, "previous_turn"

    # Neither source is usable on its own — prefer whichever had anything at all
    # so the error message can be specific about how little we had.
    return (collected or previous), "insufficient"


async def _do_compare(ctx: RunContextWrapper[ToolContext]) -> Dict[str, Any]:
    """Core compare logic — separated so it can be called directly in tests."""
    unique_ids, source = _resolve_target_ids(ctx)

    if len(unique_ids) < _MIN_LISTINGS:
        return {
            "status": "error",
            "error": (
                "Chưa có đủ tin trong danh sách kết quả để so sánh "
                f"(cần ít nhất {_MIN_LISTINGS}, hiện có {len(unique_ids)}). "
                "Hãy tìm kiếm BĐS trước, rồi so sánh toàn bộ kết quả."
            ),
        }

    available = len(unique_ids)
    if available > _MAX_LISTINGS:
        unique_ids = unique_ids[:_MAX_LISTINGS]

    logger.info(
        "compare_listings comparing %d/%d listings from %s: %s",
        len(unique_ids),
        available,
        source,
        unique_ids,
    )

    raw_results = await asyncio.gather(
        *(_fetch_one(i, ctx.context.auth_token) for i in unique_ids)
    )

    rows: List[Dict[str, Any]] = []
    raw_listings: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for listing_id, data in zip(unique_ids, raw_results):
        if "_error" in data:
            errors.append({"listingId": listing_id, "error": data["_error"]})
            continue
        rows.append(_normalise_for_comparison(data))
        raw_listings.append(data)

    if not rows:
        return {
            "status": "error",
            "error": "Không lấy được chi tiết listing nào để so sánh.",
            "errors": errors,
        }

    ctx.context.collected_listings.extend(raw_listings)

    result: Dict[str, Any] = {
        "status": "success",
        "count": len(rows),
        "listings": rows,
        "callouts": _summary_callouts(rows),
        "errors": errors,
    }
    if available > _MAX_LISTINGS:
        # Never let a cap pass silently — the model must tell the user it
        # compared a subset rather than implying it covered everything.
        result["availableCount"] = available
        result["truncated"] = True
    return result


@function_tool(
    name_override="compare_listings",
    description_override=(
        "Compare ALL listings in the user's current result set side-by-side. "
        "Takes no arguments — the listings are resolved server-side from the "
        "most recent search results. Call this whenever the user asks to "
        "compare, weigh up, or pick between the listings they were just shown "
        "(e.g. 'so sánh tất cả', 'cái nào đáng thuê hơn?', 'nên chọn cái nào?') "
        "— including when they phrase it as comparing only some of them, since "
        "comparison always covers the whole result set. Returns structured rows "
        "+ headline callouts (cheapest, largest, best price-per-m², etc.). "
        "Write a Vietnamese prose verdict based on those values, referring to "
        "listings by position and title — never by ID."
    ),
)
async def compare_listings(ctx: RunContextWrapper[ToolContext]) -> Dict[str, Any]:
    return await _do_compare(ctx)
