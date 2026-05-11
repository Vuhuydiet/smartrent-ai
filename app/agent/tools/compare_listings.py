"""
Tool: compare_listings — fetch 2-5 listings in parallel and return normalised
side-by-side rows the LLM turns into a comparison table + verdict.

Triggered by user phrases like "so sánh tin 1 và 3", "cái nào đáng thuê hơn",
"compare these two". The LLM resolves ordinal references to listingIds from
the conversation context and passes them in.

Returning structured rows + headline callouts (cheapest, largest, best
price/m², most amenities) keeps the verdict grounded — the model only writes
prose about numbers it sees in the tool result.
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
    return row


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


async def _fetch_one(listing_id: str) -> Dict[str, Any]:
    """Fetch one listing; on error return a sentinel row the LLM can flag."""
    try:
        data = await backend_client.get_listing(listing_id)
        if "error" in data:
            return {"listingId": listing_id, "_error": data["error"]}
        return data
    except httpx.HTTPStatusError as e:
        return {"listingId": listing_id, "_error": f"HTTP {e.response.status_code}"}
    except Exception as e:  # noqa: BLE001
        logger.warning("compare_listings: fetch %s failed: %s", listing_id, e)
        return {"listingId": listing_id, "_error": str(e)}


async def _do_compare(
    ctx: RunContextWrapper[ToolContext], listing_ids: List[str]
) -> Dict[str, Any]:
    """Core compare logic — separated so it can be called directly in tests."""
    # Dedupe while preserving order, coerce to clean string IDs.
    seen: set = set()
    unique_ids: List[str] = []
    for raw in listing_ids:
        try:
            cid = str(int(float(raw)))
        except (TypeError, ValueError):
            cid = str(raw)
        if cid and cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)

    if len(unique_ids) < _MIN_LISTINGS:
        return {
            "status": "error",
            "error": (
                f"Cần ít nhất {_MIN_LISTINGS} listingId để so sánh "
                f"(nhận được {len(unique_ids)})."
            ),
        }
    if len(unique_ids) > _MAX_LISTINGS:
        unique_ids = unique_ids[:_MAX_LISTINGS]

    logger.info(
        "compare_listings fetching %d listings: %s", len(unique_ids), unique_ids
    )

    raw_results = await asyncio.gather(*(_fetch_one(i) for i in unique_ids))

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

    return {
        "status": "success",
        "count": len(rows),
        "listings": rows,
        "callouts": _summary_callouts(rows),
        "errors": errors,
    }


@function_tool(
    name_override="compare_listings",
    description_override=(
        "Compare 2 to 5 listings side-by-side. Use this when the user asks to "
        "compare specific listings (e.g. 'so sánh tin 1 và 3', 'cái nào đáng "
        "thuê hơn?'). Resolve ordinal references like 'cái thứ 2' to "
        "listingIds from the most recent search results in conversation "
        "context — never ask the user for IDs. Returns structured rows + "
        "headline callouts (cheapest, largest, best price-per-m², etc.). "
        "Write a Vietnamese prose verdict based on those values."
    ),
)
async def compare_listings(
    ctx: RunContextWrapper[ToolContext],
    listingIds: Annotated[
        List[str],
        Field(
            description=(
                f"Listing IDs to compare ({_MIN_LISTINGS}-{_MAX_LISTINGS} items). "
                "Resolve ordinal phrases to IDs from the prior turn's search "
                "results."
            )
        ),
    ],
) -> Dict[str, Any]:
    return await _do_compare(ctx, listingIds)
