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
from app.dto.house_pricing import (
    PriceConfidence,
    PriceSuggestionRequest,
    PriceSuggestionResponse,
)

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


# --------------------------------------------------------------------------
# Rule-based fallback tables.
#
# Keys must be matched as substrings against the *raw* names the frontend
# sends, which are Vietnamese with diacritics ("Hà Nội", "Hoàn Kiếm").
# ASCII spellings are kept alongside for callers that send transliterated
# names. These mirror the tables in app/agent/tools/get_price_estimate.py.
# --------------------------------------------------------------------------
_CITY_RENT: Dict[str, Dict[str, int]] = {
    "hà nội": {"high": 220_000, "medium": 160_000, "low": 110_000},
    "hanoi": {"high": 220_000, "medium": 160_000, "low": 110_000},
    "ha noi": {"high": 220_000, "medium": 160_000, "low": 110_000},
    "hồ chí minh": {"high": 270_000, "medium": 190_000, "low": 130_000},
    "ho chi minh": {"high": 270_000, "medium": 190_000, "low": 130_000},
    "đà nẵng": {"high": 190_000, "medium": 140_000, "low": 90_000},
    "da nang": {"high": 190_000, "medium": 140_000, "low": 90_000},
}
_DEFAULT_RENT: Dict[str, int] = {"high": 180_000, "medium": 130_000, "low": 90_000}

_HIGH_TIER_DISTRICTS = (
    "hoàn kiếm",
    "hoan kiem",
    "ba đình",
    "ba dinh",
    "tây hồ",
    "tay ho",
    "quận 1",
    "quan 1",
    "district 1",
    "quận 3",
    "quan 3",
    "district 3",
    "hải châu",
    "hai chau",
)
_LOW_TIER_DISTRICTS = (
    "hà đông",
    "ha dong",
    "thanh trì",
    "thanh tri",
    "gia lâm",
    "gia lam",
    "thủ đức",
    "thu duc",
    "bình tân",
    "binh tan",
    "gò vấp",
    "go vap",
)

_PROPERTY_MULTIPLIERS: Dict[str, float] = {
    "apartment": 1.0,
    "house": 1.1,
    "villa": 1.5,
    "office": 1.2,
    "room": 0.8,
    "studio": 0.9,
}


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
            # Keep our own handle on the context so we can count the listings the
            # agent actually retrieved, rather than trusting the number it reports.
            ctx = _PriceCtx()
            try:
                run_result = await Runner.run(
                    starting_agent=agent,
                    input=prompt,
                    context=ctx,
                    max_turns=12,
                )
                result_text = str(run_result.final_output or "")
                generation.end(output=result_text[:2000])
            except Exception as e:
                generation.end(level="ERROR", status_message=str(e))
                raise

            trace.update(output={"response": result_text[:500]})
            result = self._parse_json(result_text)
            price_range = self._validated_range(result)

            listings_found = self._count_unique_listings(ctx.last_listings)
            confidence = self._resolve_confidence(
                result.get("confidence"), listings_found
            )

            return PriceSuggestionResponse(
                price_range=price_range,
                location=f"{request.district}, {request.city}",
                property_type=request.property_type,
                currency="VND",
                source="ai_comparables",
                listings_found=listings_found,
                confidence=confidence,
            )

        except Exception as e:
            logger.warning(
                "AI price prediction failed, using rule-based fallback: %s",
                e,
                exc_info=True,
            )
            price_range = self._estimate_price_range(request)
            return PriceSuggestionResponse(
                price_range=price_range,
                location=f"{request.district}, {request.city}",
                property_type=request.property_type,
                currency="VND",
                source="rule_based_fallback",
                listings_found=0,
                confidence="low",
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

    # Sanity bounds for a monthly rent in VND. Anything outside this is treated
    # as a hallucinated figure (e.g. the model answering in millions or in USD)
    # and drops the whole response to the rule-based fallback.
    _MIN_PLAUSIBLE_VND = 300_000
    _MAX_PLAUSIBLE_VND = 5_000_000_000

    @classmethod
    def _validated_range(cls, result: Dict[str, Any]) -> Dict[str, int]:
        """Validate the agent's price range, raising if it is not usable.

        The LLM does the arithmetic, so nothing guarantees the numbers are
        present, integral, ordered or even denominated in VND. Raising here
        routes the request to the rule-based fallback instead of returning a
        nonsensical range to the user.
        """
        try:
            minimum = int(result["min_price"])
            maximum = int(result["max_price"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"agent returned an unusable price range: {result}") from e

        if minimum > maximum:
            minimum, maximum = maximum, minimum

        if not (
            cls._MIN_PLAUSIBLE_VND <= minimum
            and maximum <= cls._MAX_PLAUSIBLE_VND
            and minimum > 0
        ):
            raise ValueError(
                f"agent price range out of plausible VND bounds: {minimum}-{maximum}"
            )

        return {"min": minimum, "max": maximum}

    @staticmethod
    def _count_unique_listings(listings: List[Dict[str, Any]]) -> int:
        """Count distinct listings retrieved across all tool calls.

        The agent may search several times, so the same listing can be appended
        more than once; de-duplicate by id and fall back to counting entries
        that carry no id at all.
        """
        seen: set = set()
        unidentified = 0
        for listing in listings:
            listing_id = listing.get("id") or listing.get("listingId")
            if listing_id is None:
                unidentified += 1
            else:
                seen.add(listing_id)
        return len(seen) + unidentified

    @staticmethod
    def _resolve_confidence(reported: Any, listings_found: int) -> PriceConfidence:
        """Trust the agent's confidence only as far as the evidence supports it."""
        if listings_found == 0:
            return "low"
        candidate = str(reported).lower() if reported is not None else ""
        if candidate not in ("high", "medium", "low"):
            return "medium"
        # A handful of comparables cannot justify a "high" claim.
        if candidate == "high":
            return "high" if listings_found >= 5 else "medium"
        return "low" if candidate == "low" else "medium"

    @staticmethod
    def _estimate_price_range(request: PriceSuggestionRequest) -> Dict[str, int]:
        """Fallback estimation when AI analysis fails."""
        city = request.city.lower()
        property_type = request.property_type.lower()
        area = request.area or 30
        district = request.district.lower()

        city_rents = _DEFAULT_RENT
        for key, rents in _CITY_RENT.items():
            if key in city:
                city_rents = rents
                break

        tier = "medium"
        if any(x in district for x in _HIGH_TIER_DISTRICTS):
            tier = "high"
        elif any(x in district for x in _LOW_TIER_DISTRICTS):
            tier = "low"

        rent_per_m2 = city_rents[tier]
        multiplier = next(
            (v for k, v in _PROPERTY_MULTIPLIERS.items() if k in property_type), 1.0
        )
        monthly_rent = rent_per_m2 * multiplier * area

        return {
            "min": int(monthly_rent * 0.80),
            "max": int(monthly_rent * 1.20),
        }
