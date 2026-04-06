"""
Tool: get_listing_detail

Fetches full details for a single listing from the SmartRent backend.
Use when the user asks for more information about a specific property
(e.g. "tell me more about the first one", "what's the contact number?").
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

_MAX_DESCRIPTION_LENGTH = 500  # chars sent to LLM — full version kept in raw listing


class GetListingDetailTool(BaseTool):
    name = "get_listing_detail"
    description = (
        "Fetch complete details for a specific property listing, including full "
        "description, amenities, contact information, and address. "
        "Use this when the user asks for more details about a listing they have seen."
    )

    def to_function_declaration(self) -> Any:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=Schema(
                type=Type.OBJECT,
                properties={
                    "listingId": Schema(
                        type=Type.STRING,
                        description="The listing ID returned by search_listings.",
                    ),
                },
                required=["listingId"],
            ),
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        listingId: str = kwargs["listingId"]  # noqa: N806
        try:
            data = await backend_client.get_listing(listingId)

            if "error" in data:
                return {"status": "error", "error": data["error"]}

            # Compact summary for LLM — keeps token usage low
            listing_for_llm = {
                "listingId": data.get("listingId"),
                "title": data.get("title", ""),
                "description": (data.get("description") or "")[
                    :_MAX_DESCRIPTION_LENGTH
                ],
                "price": data.get("price"),
                "priceUnit": data.get("priceUnit", ""),
                "area": data.get("area"),
                "bedrooms": data.get("bedrooms"),
                "bathrooms": data.get("bathrooms"),
                "address": data.get("address", ""),
                "wardName": data.get("wardName", ""),
                "districtName": data.get("districtName", ""),
                "provinceName": data.get("provinceName", ""),
                "propertyType": data.get("propertyType", ""),
                "listingType": data.get("listingType", ""),
                "furnishing": data.get("furnishing", ""),
                "direction": data.get("direction", ""),
                "floor": data.get("floor"),
                "totalFloors": data.get("totalFloors"),
                "amenities": data.get("amenities", []),
                "contactName": data.get("contactName") or "",
                "contactPhone": data.get("contactPhone") or "",
                "contactAvailable": data.get("contactAvailable", False),
                "postedAt": data.get("postedAt", ""),
                "expiredAt": data.get("expiredAt", ""),
            }

            return {
                "status": "success",
                "listing": listing_for_llm,
                "_raw_listing": data,  # full backend object for frontend
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {
                    "status": "error",
                    "error": f"Listing {listingId} was not found.",
                }
            logger.error(
                "Backend HTTP %s fetching listing %s",
                e.response.status_code,
                listingId,
            )
            return {
                "status": "error",
                "error": f"Backend returned HTTP {e.response.status_code}",
            }
        except Exception as e:
            logger.error(
                "get_listing_detail failed for %s: %s", listingId, e, exc_info=True
            )
            return {"status": "error", "error": str(e)}
