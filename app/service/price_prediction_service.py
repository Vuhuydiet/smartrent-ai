"""
Price prediction service — an OpenAI Agents SDK Agent chooses the comparable-
search criteria and calls the `get_price_comparables` tool, which returns price
statistics computed server-side in SQL. The service builds the final range from
those statistics (interquartile band) — the model never does the arithmetic.

Falls back to a rule-based estimate when the AI agent fails or finds no
comparables.
"""

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

You have access to the SmartRent backend through the `get_price_comparables`
tool. The tool does the maths for you: given your chosen criteria it filters
comparable listings and returns price statistics computed in SQL
(min/p25/median/p75/max/avg and median price per m²). You are responsible for
CHOOSING GOOD CRITERIA — never for computing the range yourself.

Your task:
1. Call `get_price_comparables` for the requested property: same product_type,
   listing_type RENT, price_unit MONTH, within a 2km radius, and an area band of
   roughly ±30% around the target area (min_area/max_area).
2. Look at `sampleSize`. If it is small (< 8), widen the search and call again:
   first grow the radius (e.g. 3–5 km), then loosen the area band. Prefer the
   query that yields the most comparables while staying representative.
3. Once you have a query with enough comparables, you are done — the backend has
   already produced the statistics. Reply with a one-line confirmation such as
   "Found N comparables." Do NOT invent or recompute any price; the service reads
   the SQL-computed statistics directly from your tool calls.

If every query returns sampleSize 0, say "No comparables found." and stop — the
service will fall back to a rule-based estimate."""


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
    """Context for the price-prediction agent.

    Holds every comparables aggregate the agent pulled during the run so the
    service can build the final range from the SQL-computed statistics of the
    best-supported query, rather than trusting any number the model writes out.
    """

    stats_calls: List[Dict[str, Any]] = field(default_factory=list)


@function_tool(
    name_override="get_price_comparables",
    description_override=(
        "Return deterministic price statistics (min/p25/median/p75/max/avg and "
        "median price per m²) for comparable rental listings near a point. The "
        "backend filters by geo radius + type + area and computes the numbers in "
        "SQL — you choose the criteria; you do NOT compute the range yourself."
    ),
)
async def _get_price_comparables(
    ctx: RunContextWrapper[_PriceCtx],
    latitude: Annotated[float, Field(description="Center latitude")],
    longitude: Annotated[float, Field(description="Center longitude")],
    product_type: Annotated[
        str,
        Field(
            description="Property type",
            json_schema_extra={
                "enum": ["ROOM", "APARTMENT", "HOUSE", "OFFICE", "STUDIO", "STORE"]
            },
        ),
    ],
    listing_type: Annotated[
        str,
        Field(
            description="Listing type (usually RENT)",
            json_schema_extra={"enum": ["RENT", "SALE", "SHARE"]},
        ),
    ] = "RENT",
    radius_km: Annotated[
        Optional[float],
        Field(description="Search radius in km (default 2, max 20)"),
    ] = None,
    price_unit: Annotated[
        str,
        Field(
            description="Price unit to compare within (usually MONTH)",
            json_schema_extra={"enum": ["MONTH", "DAY", "YEAR"]},
        ),
    ] = "MONTH",
    min_area: Annotated[
        Optional[float], Field(description="Minimum comparable area in m²")
    ] = None,
    max_area: Annotated[
        Optional[float], Field(description="Maximum comparable area in m²")
    ] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "productType": product_type,
        "listingType": listing_type,
        "priceUnit": price_unit,
    }
    if radius_km is not None:
        params["radiusKm"] = radius_km
    if min_area is not None:
        params["minArea"] = min_area
    if max_area is not None:
        params["maxArea"] = max_area

    try:
        data = await backend_client.get_price_comparables(params)
        if "error" in data:
            logger.error("price-comparables backend error: %s", data.get("error"))
            return {"sampleSize": 0, "error": data["error"]}
        ctx.context.stats_calls.append(data)
        logger.info("price-comparables returned sampleSize=%s", data.get("sampleSize"))
        return data
    except httpx.HTTPStatusError as e:
        logger.error("Backend HTTP %s in price-comparables", e.response.status_code)
        return {"sampleSize": 0, "error": str(e)}
    except Exception as e:
        logger.error("price-comparables failed: %s", e, exc_info=True)
        return {"sampleSize": 0, "error": str(e)}


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
                tools=[_get_price_comparables],
            )

            generation = trace.generation(
                name="price-agent-run",
                model=settings.LLM_PRICE_MODEL,
                input=prompt[:2000],
            )
            # Keep our own handle on the context so we can read the SQL-computed
            # statistics the agent's tool calls produced, rather than trusting any
            # number the model writes into its final message.
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

            # The range is built from the backend statistics of the best-supported
            # query the agent ran — never from the model's prose. Raises (→ fallback)
            # when no query returned any comparables.
            best = self._best_stats(ctx.stats_calls)
            price_range = self._range_from_stats(best)
            listings_found = int(best.get("sampleSize") or 0)
            confidence = self._confidence_from_stats(best)

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

    # Sanity bounds for a monthly rent in VND. A backend aggregate outside this
    # band means the comparables were mispriced/misfiltered, so we drop to the
    # rule-based fallback rather than surfacing an absurd range.
    _MIN_PLAUSIBLE_VND = 300_000
    _MAX_PLAUSIBLE_VND = 5_000_000_000

    @staticmethod
    def _best_stats(stats_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick the comparables query with the most evidence behind it.

        The agent may widen its search across several calls; the one with the
        largest sample is the most representative. Raises when no call returned
        any comparables so the caller falls back to a rule-based estimate.
        """
        usable = [c for c in stats_calls if int(c.get("sampleSize") or 0) > 0]
        if not usable:
            raise ValueError("no comparables returned by any query")
        return max(usable, key=lambda c: int(c.get("sampleSize") or 0))

    @classmethod
    def _range_from_stats(cls, stats: Dict[str, Any]) -> Dict[str, int]:
        """Build a price range from the backend statistics.

        Uses the interquartile band (p25–p75) as the range: robust to the
        outliers that raw min/max would drag in. Guarantees min < max even for a
        tiny or single-price sample, and validates the numbers sit in a plausible
        VND band (raising → rule-based fallback otherwise).
        """
        lower = cls._as_int(stats.get("p25")) or cls._as_int(stats.get("min"))
        upper = cls._as_int(stats.get("p75")) or cls._as_int(stats.get("max"))
        if lower is None or upper is None:
            raise ValueError(f"comparables stats missing price bounds: {stats}")

        if lower > upper:
            lower, upper = upper, lower

        # Degenerate band (all comparables at one price, or n==1): spread ±10%
        # around the point so the UI never shows an identical min and max.
        if lower == upper:
            lower = int(lower * 0.9)
            upper = int(upper * 1.1)

        if not (
            cls._MIN_PLAUSIBLE_VND <= lower
            and upper <= cls._MAX_PLAUSIBLE_VND
            and lower > 0
        ):
            raise ValueError(
                f"comparables price range out of plausible VND bounds: {lower}-{upper}"
            )

        return {"min": lower, "max": upper}

    @classmethod
    def _confidence_from_stats(cls, stats: Dict[str, Any]) -> PriceConfidence:
        """Derive confidence from sample size and price dispersion.

        Deterministic — driven by how much market evidence there is and how
        tightly the comparables agree, not by anything the model asserts.
        """
        sample = int(stats.get("sampleSize") or 0)
        if sample == 0:
            return "low"

        median = cls._as_int(stats.get("median"))
        p25 = cls._as_int(stats.get("p25"))
        p75 = cls._as_int(stats.get("p75"))
        # Coefficient-of-dispersion proxy: interquartile width over the median.
        spread = (
            (p75 - p25) / median
            if median and p25 is not None and p75 is not None and median > 0
            else 1.0
        )

        if sample >= 20 and spread <= 0.4:
            return "high"
        if sample >= 8 and spread <= 0.8:
            return "medium"
        return "low"

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        """Coerce a backend numeric field to int, tolerating None/strings."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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
