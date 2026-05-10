"""
Tool: get_listing_detail — fetches full details for a single listing.
Use when the user asks for more information about a specific property
(e.g. "tell me more about the first one", "what's the contact number?").
"""

import logging
from typing import Annotated, Any, Dict

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_DESCRIPTION_LENGTH = 500  # chars sent to LLM — full version goes to frontend


async def _fetch_detail(
    ctx: RunContextWrapper[ToolContext], listing_id_str: str
) -> Dict[str, Any]:
    """Core detail-fetch logic — separated so it can be called directly in tests."""
    try:
        data = await backend_client.get_listing(listing_id_str)

        if "error" in data:
            return {"status": "error", "error": data["error"]}

        addr = data.get("address") or {}
        listing_for_llm: Dict[str, Any] = {
            "listingId": str(data.get("listingId", "")),
            "title": data.get("title", ""),
            "description": (data.get("description") or "")[:_MAX_DESCRIPTION_LENGTH],
            "price": data.get("price"),
            "priceUnit": data.get("priceUnit", ""),
            "area": data.get("area"),
            "address": addr.get("fullAddress", ""),
            "wardName": addr.get("wardName", ""),
            "districtName": addr.get("districtName", ""),
            "provinceName": addr.get("provinceName", ""),
            "productType": data.get("productType", ""),
            "listingType": data.get("listingType", ""),
            "amenities": [
                a.get("name") for a in data.get("amenities", []) if a.get("name")
            ],
            "contactName": data.get("contactName") or "",
            "contactPhone": data.get("contactPhone") or "",
            "contactAvailable": data.get("contactAvailable", False),
            "postDate": data.get("postDate", ""),
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
            "ownerZaloLink",
        ):
            val = data.get(key)
            if val is not None:
                listing_for_llm[key] = val

        ctx.context.collected_listings.append(data)
        return {"status": "success", "listing": listing_for_llm}

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {
                "status": "error",
                "error": f"Listing {listing_id_str} was not found.",
            }
        logger.error(
            "Backend HTTP %s fetching listing %s",
            e.response.status_code,
            listing_id_str,
        )
        return {
            "status": "error",
            "error": f"Backend returned HTTP {e.response.status_code}",
        }
    except Exception as e:
        logger.error(
            "get_listing_detail failed for %s: %s", listing_id_str, e, exc_info=True
        )
        return {"status": "error", "error": str(e)}


@function_tool(
    name_override="get_listing_detail",
    description_override=(
        "Fetch complete details for a specific property listing, including full "
        "description, amenities, contact information, and address. Use this "
        "when the user asks for more details about a listing they have seen."
    ),
)
async def get_listing_detail(
    ctx: RunContextWrapper[ToolContext],
    listingId: Annotated[
        str, Field(description="The listing ID returned by search_listings.")
    ],
) -> Dict[str, Any]:
    # Models may pass numeric IDs as float (123.0) — normalize to clean string.
    try:
        normalized_id = str(int(float(listingId)))
    except (ValueError, TypeError):
        normalized_id = str(listingId)

    return await _fetch_detail(ctx, normalized_id)
