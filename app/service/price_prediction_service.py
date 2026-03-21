import json
import logging
from typing import Any, Dict

from google.ai.generativelanguage import (  # type: ignore[import]
    Content,
    FunctionDeclaration,
    FunctionResponse,
    Part,
    Schema,
    Tool,
    Type,
)

from app.ai.llm.gateway import get_gateway
from app.core import backend_client
from app.core.config import settings
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = """\
You are a real estate price prediction expert for Vietnam rental market.

You have access to the SmartRent backend listing database through tools.

Your task:
1. Use the `search_listings` tool to find similar rental properties in the requested location
2. Search within 2km radius of the coordinates
3. Filter by property type and area (±30% range)
4. Analyze the prices of similar listings
5. Calculate a realistic price range based on market data

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


def _get_search_tool() -> Any:
    """Build Gemini Tool declaration for search_listings."""
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name="search_listings",
                description="Search for rental property listings in SmartRent database.",
                parameters=Schema(
                    type=Type.OBJECT,
                    properties={
                        "listing_type": Schema(
                            type=Type.STRING,
                            description="Type of listing",
                            enum=["RENT", "SELL"],
                        ),
                        "latitude": Schema(
                            type=Type.NUMBER, description="Latitude coordinate"
                        ),
                        "longitude": Schema(
                            type=Type.NUMBER, description="Longitude coordinate"
                        ),
                        "radius_km": Schema(
                            type=Type.NUMBER, description="Search radius in kilometers"
                        ),
                        "product_type": Schema(
                            type=Type.STRING,
                            description="Property type",
                            enum=["APARTMENT", "HOUSE", "VILLA", "OFFICE", "ROOM"],
                        ),
                        "min_area": Schema(
                            type=Type.NUMBER, description="Minimum area in m²"
                        ),
                        "max_area": Schema(
                            type=Type.NUMBER, description="Maximum area in m²"
                        ),
                        "size": Schema(
                            type=Type.INTEGER,
                            description="Number of results to return",
                        ),
                    },
                    required=["listing_type", "latitude", "longitude"],
                ),
            )
        ]
    )


class PricePredictionService:
    """Service for rental price prediction using Gemini AI with backend integration."""

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
                metadata={"property_type": request.property_type},
            )

            # Build model with function calling
            model = self._gateway.build_model(
                model_name=settings.GEMINI_PRICE_MODEL,
                system_instruction=_SYSTEM_INSTRUCTION,
                tools=_get_search_tool(),
            )
            chat = self._gateway.start_chat(model)

            # First call
            response = await self._gateway.send_message(
                chat, prompt, trace, span_name="price-initial"
            )

            # Handle function calls
            round_num = 0
            while round_num < 5:
                parts = response.candidates[0].content.parts
                function_calls = [
                    p.function_call
                    for p in parts
                    if hasattr(p, "function_call") and p.function_call.name
                ]
                if not function_calls:
                    break

                tool_response_parts = []
                for fc in function_calls:
                    args = dict(fc.args)
                    logger.info("Price prediction calling tool: %s", fc.name)
                    result = await self._execute_tool(fc.name, args)
                    tool_response_parts.append(
                        Part(
                            function_response=FunctionResponse(
                                name=fc.name,
                                response={"result": result},
                            )
                        )
                    )

                response = await self._gateway.send_message(
                    chat,
                    Content(parts=tool_response_parts),
                    trace,
                    span_name=f"price-round-{round_num}",
                )
                round_num += 1

            # Parse final response
            result_text = response.text
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

    async def _execute_tool(
        self, function_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool call using the shared backend_client."""
        if function_name == "search_listings":
            try:
                # Map MCP-style args to backend API params
                params: Dict[str, Any] = {}
                if "listing_type" in args:
                    params["listingType"] = args["listing_type"]
                if "latitude" in args:
                    params["latitude"] = args["latitude"]
                if "longitude" in args:
                    params["longitude"] = args["longitude"]
                if "radius_km" in args:
                    params["radiusKm"] = args["radius_km"]
                if "product_type" in args:
                    params["propertyType"] = args["product_type"]
                if "min_area" in args:
                    params["minArea"] = args["min_area"]
                if "max_area" in args:
                    params["maxArea"] = args["max_area"]
                if "size" in args:
                    params["size"] = args["size"]
                params.setdefault("excludeExpired", True)

                data = await backend_client.search_listings(params)
                listings = data.get("listings", [])
                logger.info("search_listings returned %d results", len(listings))
                return {"listings": listings, "total": len(listings)}
            except Exception as e:
                logger.error("Error executing search_listings: %s", e)
                return {"listings": [], "total": 0, "error": str(e)}

        return {"error": f"Unknown function: {function_name}"}

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences."""
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
