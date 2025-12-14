import json
import logging
from typing import Any, Dict

import google.generativeai as genai  # type: ignore

from app.core.config import settings
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse

logger = logging.getLogger(__name__)


class PricePredictionService:
    """Service for rental price prediction using Gemini AI with MCP backend integration."""

    def __init__(self) -> None:
        """Initialize the price prediction service with Gemini AI."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")

        genai.configure(api_key=settings.GEMINI_API_KEY)

        # System instruction for price prediction with MCP
        self.system_instruction = """You are a real estate price prediction expert for Vietnam rental market.

You have access to the SmartRent backend listing database through MCP tools.

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

        # Initialize model with function calling (without system_instruction in constructor)
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash-exp", tools=[self._get_mcp_search_tool()]
        )

    def _get_mcp_search_tool(self):
        """Get MCP search_listings tool declaration for Gemini."""
        from google.ai.generativelanguage_v1beta.types import (
            FunctionDeclaration,
            Schema,
            Tool,
            Type,
        )

        search_function = FunctionDeclaration(
            name="search_listings",
            description="Search for rental property listings in SmartRent database. Use this to find similar properties and analyze market prices.",
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
                        type=Type.INTEGER, description="Number of results to return"
                    ),
                },
                required=["listing_type", "latitude", "longitude"],
            ),
        )

        return Tool(function_declarations=[search_function])

    async def predict_price(
        self, request: PriceSuggestionRequest
    ) -> PriceSuggestionResponse:
        """
        Predict property price using Gemini AI with MCP backend integration.

        Args:
            request: Price prediction request with property details

        Returns:
            Price suggestion response with AI-analyzed price range
        """
        try:
            # Build prompt for Gemini with system instruction
            prompt = f"""{self.system_instruction}

Now analyze rental prices for this property:
- Location: {request.ward}, {request.district}, {request.city}
- Coordinates: {request.latitude}, {request.longitude}
- Property Type: {request.property_type}
- Area: {request.area or 'not specified'} m²

Search for similar listings within 2km radius and provide a realistic price range."""

            # Call Gemini with MCP tool access
            chat = self.model.start_chat()
            response = chat.send_message(prompt)

            # Handle function calls (MCP tools)
            while response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                logger.info(f"Gemini called MCP tool: {function_call.name}")

                # Execute MCP function
                function_result = await self._execute_mcp_function(
                    function_call.name, dict(function_call.args)
                )

                # Send result back to Gemini
                from google.ai.generativelanguage_v1beta.types import (
                    Content,
                    FunctionResponse,
                    Part,
                )

                response = chat.send_message(
                    Content(
                        parts=[
                            Part(
                                function_response=FunctionResponse(
                                    name=function_call.name,
                                    response={"result": function_result},
                                )
                            )
                        ]
                    )
                )

            # Parse final response
            result_text = response.text
            logger.info(f"Gemini final response: {result_text}")

            result = json.loads(result_text)

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
            logger.error(f"Error in AI price prediction: {str(e)}")
            # Fallback to estimation
            price_range = self._estimate_price_range(request)
            return PriceSuggestionResponse(
                price_range=price_range,
                location=f"{request.district}, {request.city}",
                property_type=request.property_type,
                currency="VND",
            )

    async def _execute_mcp_function(
        self, function_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute MCP function call by calling backend API.

        Args:
            function_name: Name of the MCP function
            args: Function arguments

        Returns:
            Function result
        """
        if function_name == "search_listings":
            import httpx

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/search",
                        json=args,
                        timeout=15.0,
                    )
                    response.raise_for_status()
                    result = response.json()

                    if result.get("code") == "999999" and "data" in result:
                        listings = result["data"].get("listings", [])
                        logger.info(
                            f"MCP search_listings returned {len(listings)} results"
                        )
                        return {"listings": listings, "total": len(listings)}
                    else:
                        logger.warning(f"Backend API error: {result.get('message')}")
                        return {
                            "listings": [],
                            "total": 0,
                            "error": result.get("message"),
                        }

            except Exception as e:
                logger.error(f"Error executing MCP function: {str(e)}")
                return {"listings": [], "total": 0, "error": str(e)}

        return {"error": f"Unknown function: {function_name}"}

    def _estimate_price_range(self, request: PriceSuggestionRequest) -> Dict[str, int]:
        """
        Estimate RENTAL price range for a property (monthly rent in VND).
        Fallback estimation when no backend data available.

        Args:
            request: Price prediction request

        Returns:
            Dictionary with min and max MONTHLY RENT price in VND
        """
        city = request.city.lower()
        district = request.district.lower()
        property_type = request.property_type.lower()
        area = request.area or 30  # Default 30m² for typical room

        # Base RENTAL price per m² per MONTH in VND
        base_rent_per_m2 = {
            "hanoi": {"high": 200_000, "medium": 150_000, "low": 100_000},
            "ho chi minh": {"high": 250_000, "medium": 180_000, "low": 120_000},
            "da nang": {"high": 180_000, "medium": 130_000, "low": 90_000},
        }

        # Property type rent multipliers
        type_multipliers = {
            "apartment": 1.0,
            "house": 1.1,
            "villa": 1.5,
            "office": 1.2,
            "room": 0.8,
            "studio": 0.9,
        }

        # Select city base rent
        city_rents = base_rent_per_m2.get(
            city, {"high": 180_000, "medium": 130_000, "low": 90_000}
        )

        # Determine tier based on district
        tier = "medium"
        if any(
            x in district
            for x in [
                "hoan kiem",
                "ba dinh",
                "district 1",
                "quan 1",
                "hai chau",
                "tay ho",
                "district 3",
            ]
        ):
            tier = "high"
        elif any(
            x in district
            for x in [
                "ha dong",
                "thanh tri",
                "thu duc",
                "binh thanh",
                "binh tan",
                "go vap",
            ]
        ):
            tier = "low"

        rent_per_m2 = city_rents[tier]

        # Apply property type multiplier
        type_key = next(
            (k for k in type_multipliers if k in property_type), "apartment"
        )
        multiplier = type_multipliers.get(type_key, 1.0)

        # Calculate monthly rent
        monthly_rent = rent_per_m2 * multiplier * area

        # Return price range
        return {
            "min": int(monthly_rent * 0.80),
            "max": int(monthly_rent * 1.20),
        }
