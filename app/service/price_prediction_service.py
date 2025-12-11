import logging
from typing import Any, Dict

import google.generativeai as genai  # type: ignore

from app.core.config import settings
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse

logger = logging.getLogger(__name__)


class PricePredictionService:
    """Service for AI-powered price prediction using Gemini."""

    def __init__(self) -> None:
        """Initialize the price prediction service with Gemini."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")

        genai.configure(api_key=settings.GEMINI_API_KEY)  # type: ignore

        # System instruction for price prediction
        system_instruction = """You are an expert real estate price advisor for SmartRent in Vietnam.

Your role is to provide accurate price suggestions for rental and sale properties based on location, property type, and area.

Key guidelines:
1. Understand Vietnamese real estate market dynamics
2. Consider location hierarchy: City > District > Ward
3. Property types: House, Apartment, Villa, Office, Land, etc.
4. Prices are in Vietnamese Dong (VND)
5. Use coordinates (latitude, longitude) to refine estimates
6. Provide realistic price ranges with min/max values
7. Consider market conditions and property characteristics

When users ask for price suggestions:
1. Extract all property details (location, type, area, coordinates)
2. Use the predict_price tool to get market-based estimates
3. Explain the price range in context of the location and property type
4. Provide insights about the market in that area if possible"""

        # Define predict_price tool using dictionary format
        predict_price_declaration: Dict[str, Any] = {
            "name": "predict_price",
            "description": "Predict real estate price based on property characteristics and location in Vietnam. Returns estimated price range in VND.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City or province name (e.g., 'Hanoi', 'Ho Chi Minh', 'Da Nang')",
                    },
                    "district": {
                        "type": "string",
                        "description": "District or county name within the city",
                    },
                    "ward": {
                        "type": "string",
                        "description": "Ward or commune name within the district",
                    },
                    "property_type": {
                        "type": "string",
                        "description": "Type of property: House, Apartment, Villa, Office, Land, etc.",
                    },
                    "area": {
                        "type": "number",
                        "description": "Property area in square meters (m²)",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Property latitude coordinate in decimal degrees",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Property longitude coordinate in decimal degrees",
                    },
                },
                "required": ["city", "district", "ward", "property_type", "latitude", "longitude"],
            },
        }

        # Initialize model with function calling
        self.model = genai.GenerativeModel(  # type: ignore[call-arg]
            model_name="gemini-2.0-flash-exp",
            tools=[predict_price_declaration],  # type: ignore[arg-type]
        )
        self.system_instruction = system_instruction

    async def predict_price(
        self, request: PriceSuggestionRequest
    ) -> PriceSuggestionResponse:
        """
        Predict property price using AI.

        Args:
            request: Price prediction request with property details

        Returns:
            Price suggestion response with estimated range
        """
        try:
            # Build the user query
            query = f"""Please predict the price for this property:
- Location: {request.ward}, {request.district}, {request.city}
- Property Type: {request.property_type}
- Area: {f'{request.area} m²' if request.area else 'Not specified'}
- Coordinates: {request.latitude}, {request.longitude}

Provide a realistic price range for this property in the current Vietnamese real estate market."""

            # Start chat session
            chat = self.model.start_chat(history=[])  # type: ignore

            # Send message
            response = chat.send_message(query)  # type: ignore

            # Check for function calls
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_call = part.function_call

                            if function_call.name == "predict_price":
                                # Call the actual prediction function
                                result = await self._call_predict_price(dict(function_call.args))

                                # Send function response back to model
                                function_response = genai.protos.FunctionResponse(  # type: ignore
                                    name="predict_price",
                                    response={"result": result}
                                )

                                response = chat.send_message(  # type: ignore
                                    genai.protos.Part(function_response=function_response)  # type: ignore
                                )

            # For now, return a mock response based on location
            # TODO: Integrate with actual price predictor model
            price_range = self._estimate_price_range(request)

            return PriceSuggestionResponse(
                price_range=price_range,
                location=f"{request.district}, {request.city}",
                property_type=request.property_type,
                currency="VND",
            )

        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            raise

    async def _call_predict_price(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the price prediction logic.

        Args:
            params: Prediction parameters from function call

        Returns:
            Prediction result with price range
        """
        # TODO: Integrate with RealEstatePricePredictorModel
        # For now, return estimated ranges based on location

        city = params.get("city", "").lower()
        district = params.get("district", "").lower()
        property_type = params.get("property_type", "").lower()
        area = params.get("area")

        # Base price per m² (VND millions) by city
        base_prices = {
            "hanoi": {"high": 80, "medium": 50, "low": 30},
            "ho chi minh": {"high": 100, "medium": 60, "low": 35},
            "da nang": {"high": 60, "medium": 40, "low": 25},
        }

        # Property type multipliers
        type_multipliers = {
            "apartment": 1.0,
            "house": 1.2,
            "villa": 1.8,
            "office": 1.5,
            "land": 0.8,
        }

        # Select base price
        city_prices = base_prices.get(city, base_prices["hanoi"])

        # Determine district tier (simplified)
        tier = "medium"
        if "1" in district or "central" in district or "hoan kiem" in district:
            tier = "high"
        elif any(x in district for x in ["suburb", "ngoại", "outer"]):
            tier = "low"

        base_price_per_m2 = city_prices[tier]

        # Apply property type multiplier
        type_key = next((k for k in type_multipliers if k in property_type), "apartment")
        multiplier = type_multipliers[type_key]

        price_per_m2 = base_price_per_m2 * multiplier

        # Calculate total price if area is provided
        if area and area > 0:
            total_price = price_per_m2 * area
            min_price = int(total_price * 0.85)  # -15%
            max_price = int(total_price * 1.15)  # +15%
        else:
            # Default ranges for typical properties
            min_price = int(price_per_m2 * 50 * 0.85)  # Assume 50m²
            max_price = int(price_per_m2 * 50 * 1.15)

        # Convert to VND (from millions)
        return {
            "price_range": {
                "min": min_price * 1_000_000,
                "max": max_price * 1_000_000,
            },
            "currency": "VND",
            "confidence": 0.75,
            "base_price_per_m2": price_per_m2,
        }

    def _estimate_price_range(self, request: PriceSuggestionRequest) -> Dict[str, int]:
        """
        Estimate price range for a property (fallback method).

        Args:
            request: Price prediction request

        Returns:
            Dictionary with min and max price in VND
        """
        city = request.city.lower()
        district = request.district.lower()
        property_type = request.property_type.lower()
        area = request.area or 50  # Default 50m² if not provided

        # Base price per m² (VND millions)
        base_prices = {
            "hanoi": {"high": 80, "medium": 50, "low": 30},
            "ho chi minh": {"high": 100, "medium": 60, "low": 35},
            "da nang": {"high": 60, "medium": 40, "low": 25},
        }

        # Property type multipliers
        type_multipliers = {
            "apartment": 1.0,
            "house": 1.2,
            "villa": 1.8,
            "office": 1.5,
            "land": 0.8,
        }

        # Select city base price
        city_prices = base_prices.get(city, {"high": 50, "medium": 35, "low": 20})

        # Determine tier
        tier = "medium"
        if "1" in district or "central" in district:
            tier = "high"

        base_price_per_m2 = city_prices[tier]

        # Apply multiplier
        type_key = next((k for k in type_multipliers if k in property_type), "apartment")
        multiplier = type_multipliers[type_key]

        price_per_m2 = base_price_per_m2 * multiplier
        total_price = price_per_m2 * area

        return {
            "min": int(total_price * 0.85 * 1_000_000),
            "max": int(total_price * 1.15 * 1_000_000),
        }
