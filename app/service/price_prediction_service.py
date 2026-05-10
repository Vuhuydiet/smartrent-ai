"""
Price prediction service — uses an OpenAI Agents SDK Agent that calls the
shared `search_listings` function tool to find comparable listings, then
returns a structured price range.

Falls back to a rule-based estimate when the AI agent fails.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import (  # type: ignore[import]
    Agent,
    RunContextWrapper,
    Runner,
    function_tool,
)
from pydantic import Field

from app.ai.llm.agent_factory import default_model_settings, make_model
from app.ai.llm.gateway import get_gateway
from app.core import backend_client
from app.core.config import settings
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = """\
You are a real estate price prediction expert for Vietnam rental market.

You have access to the SmartRent backend listing database through the
`search_comparable_listings` tool.

Your task:
1. Use the tool to find similar rental properties in the requested location.
2. Search within 2km radius of the coordinates.
3. Filter by property type and area (±30% range).
4. Analyze the prices of similar listings.
5. Calculate a realistic price range based on market data.

Return ONLY a JSON object with this exact format:
{
    "min_price": <number in VND>,
    "max_price": <number in VND>,
    "listings_found": <number of listings analyzed>,
    "confidence": <"high" | "medium" | "low">
}

If no listings found, use estimation based on Vietnam rental market standards:
- Hanoi: 150k-200k VND/m²/month
- Ho Chi Minh: 180k-250k VND/m²/month
- Da Nang: 130k-180k VND/m²/month"""


@dataclass
class _PriceCtx:
    """Context for the price-prediction agent."""

    last_listings: List[Dict[str, Any]] = field(default_factory=list)


@function_tool(
    name_override="search_comparable_listings",
    description_override="Search for rental property listings in SmartRent database.",
)
async def _search_comparable_listings(
    ctx: RunContextWrapper[_PriceCtx],
    listing_type: Annotated[
        str,
        Field(
            description="Type of listing",
            json_schema_extra={"enum": ["RENT", "SELL"]},
        ),
    ],
    latitude: Annotated[float, Field(description="Latitude coordinate")],
    longitude: Annotated[float, Field(description="Longitude coordinate")],
    radius_km: Annotated[
        Optional[float], Field(description="Search radius in kilometers")
    ] = None,
    product_type: Annotated[
        Optional[str],
        Field(
            description="Property type",
            json_schema_extra={
                "enum": ["APARTMENT", "HOUSE", "VILLA", "OFFICE", "ROOM"]
            },
        ),
    ] = None,
    min_area: Annotated[
        Optional[float], Field(description="Minimum area in m²")
    ] = None,
    max_area: Annotated[
        Optional[float], Field(description="Maximum area in m²")
    ] = None,
    size: Annotated[
        Optional[int], Field(description="Number of results to return")
    ] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "listingType": listing_type,
        "latitude": latitude,
        "longitude": longitude,
        "excludeExpired": True,
    }
    if radius_km is not None:
        params["radiusKm"] = radius_km
    if product_type is not None:
        params["propertyType"] = product_type
    if min_area is not None:
        params["minArea"] = min_area
    if max_area is not None:
        params["maxArea"] = max_area
    if size is not None:
        params["size"] = size

    try:
        data = await backend_client.search_listings(params)
        listings = data.get("listings", [])
        ctx.context.last_listings.extend(listings)
        logger.info("price-search returned %d results", len(listings))
        return {"listings": listings, "total": len(listings)}
    except httpx.HTTPStatusError as e:
        logger.error("Backend HTTP %s in price-search", e.response.status_code)
        return {"listings": [], "total": 0, "error": str(e)}
    except Exception as e:
        logger.error("price-search failed: %s", e, exc_info=True)
        return {"listings": [], "total": 0, "error": str(e)}


class PricePredictionService:
    """Service for rental price prediction using the unified Agents SDK."""

    def __init__(self) -> None:
        self._gateway = get_gateway()

    async def predict_price(
        self, request: PriceSuggestionRequest
    ) -> PriceSuggestionResponse:
        try:
            prompt = (
                f"Now analyze rental prices for this property:\n"
                f"- Location: {request.ward}, {request.district}, {request.city}\n"
                f"- Coordinates: {request.latitude}, {request.longitude}\n"
                f"- Property Type: {request.property_type}\n"
                f"- Area: {request.area or 'not specified'} m²\n\n"
                f"Search for similar listings within 2km radius and provide a realistic price range."
            )

            trace = self._gateway.create_trace(
                name="price-prediction",
                input={"location": f"{request.district}, {request.city}"},
                metadata={
                    "property_type": request.property_type,
                    "model": settings.LLM_PRICE_MODEL,
                    "provider": settings.LLM_PROVIDER,
                },
            )

            agent = Agent[_PriceCtx](
                name="Price Prediction Agent",
                instructions=_SYSTEM_INSTRUCTION,
                model=make_model(settings.LLM_PRICE_MODEL),
                model_settings=default_model_settings(temperature=0.3),
                tools=[_search_comparable_listings],
            )

            generation = trace.generation(
                name="price-agent-run",
                model=settings.LLM_PRICE_MODEL,
                input=prompt[:2000],
            )
            try:
                run_result = await Runner.run(
                    starting_agent=agent,
                    input=prompt,
                    context=_PriceCtx(),
                    max_turns=12,
                )
                result_text = str(run_result.final_output or "")
                generation.end(output=result_text[:2000])
            except Exception as e:
                generation.end(level="ERROR", status_message=str(e))
                raise

            trace.update(output={"response": result_text[:500]})
            result = self._parse_json(result_text)

            return PriceSuggestionResponse(
                price_range={
                    "min": int(result["min_price"]),
                    "max": int(result["max_price"]),
                },
                location=f"{request.district}, {request.city}",
                property_type=request.property_type,
                currency="VND",
            )

        except Exception as e:
            logger.error("Error in AI price prediction: %s", e)
            price_range = self._estimate_price_range(request)
            return PriceSuggestionResponse(
                price_range=price_range,
                location=f"{request.district}, {request.city}",
                property_type=request.property_type,
                currency="VND",
            )

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("\n```", 1)[0]
            text = text.strip()
        return json.loads(text)

    @staticmethod
    def _estimate_price_range(request: PriceSuggestionRequest) -> Dict[str, int]:
        """Fallback estimation when AI analysis fails."""
        city = request.city.lower()
        property_type = request.property_type.lower()
        area = request.area or 30
        district = request.district.lower()

        base_rent_per_m2 = {
            "hanoi": {"high": 200_000, "medium": 150_000, "low": 100_000},
            "ho chi minh": {"high": 250_000, "medium": 180_000, "low": 120_000},
            "da nang": {"high": 180_000, "medium": 130_000, "low": 90_000},
        }
        type_multipliers = {
            "apartment": 1.0,
            "house": 1.1,
            "villa": 1.5,
            "office": 1.2,
            "room": 0.8,
            "studio": 0.9,
        }

        city_rents = base_rent_per_m2.get(
            city, {"high": 180_000, "medium": 130_000, "low": 90_000}
        )

        tier = "medium"
        high_districts = [
            "hoan kiem",
            "ba dinh",
            "district 1",
            "quan 1",
            "hai chau",
            "tay ho",
            "district 3",
        ]
        low_districts = [
            "ha dong",
            "thanh tri",
            "thu duc",
            "binh thanh",
            "binh tan",
            "go vap",
        ]
        if any(x in district for x in high_districts):
            tier = "high"
        elif any(x in district for x in low_districts):
            tier = "low"

        rent_per_m2 = city_rents[tier]
        type_key = next(
            (k for k in type_multipliers if k in property_type), "apartment"
        )
        multiplier = type_multipliers.get(type_key, 1.0)
        monthly_rent = rent_per_m2 * multiplier * area

        return {
            "min": int(monthly_rent * 0.80),
            "max": int(monthly_rent * 1.20),
        }
