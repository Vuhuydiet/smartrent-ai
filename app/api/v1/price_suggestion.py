from typing import Any, cast

import httpx

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse

router = APIRouter()


@router.post("/get-price-suggestion", response_model=PriceSuggestionResponse)
async def get_price_suggestion(
    request: PriceSuggestionRequest,
) -> PriceSuggestionResponse:
    """
    Get price range suggestion for real estate properties

    This endpoint provides AI-powered price range suggestions based on:
    - Property location (city, district, ward)
    - Property type (House, Apartment, etc.)
    - Property area (optional)
    - Geographic coordinates (latitude, longitude)

    Similar to popular Vietnamese real estate platforms like batdongsan.com,
    this API helps users understand market price ranges for their properties.

    Returns:
        PriceSuggestionResponse: Contains price range in VND, location info, and property type

    Example:
        Request: Property in My Tho, Tien Giang (60m² house)
        Response: Price range 9.5M - 43.1M VND
    """
    try:
        # Call SmartRent backend house pricing API via MCP
        payload = {
            "city": request.city,
            "district": request.district,
            "ward": request.ward,
            "property_type": request.property_type,
            "latitude": request.latitude,
            "longitude": request.longitude,
        }

        if request.area is not None:
            payload["area"] = request.area

        # Use configured backend URL
        backend_url = settings.SMARTRENT_AI_URL.rstrip("/")
        pricing_endpoint = f"{backend_url}/api/v1/house-pricing/get-price-range"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                pricing_endpoint,
                json=payload,
                timeout=60.0,  # Longer timeout for ML prediction
            )
            response.raise_for_status()
            result = cast(dict[str, Any], response.json())

        # Extract price range from response
        price_range = result.get("price_range", {})
        range_min = price_range.get("min", 0)  # VND
        range_max = price_range.get("max", 0)  # VND

        return PriceSuggestionResponse(
            price_range={"min": range_min, "max": range_max},
            location=f"{request.district}, {request.city}",
            property_type=request.property_type,
            currency="VND",
        )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Backend API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting price suggestion: {str(e)}"
        )
