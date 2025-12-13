import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse
from app.service.price_prediction_service import PricePredictionService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_price_prediction_service() -> PricePredictionService:
    """Dependency to get price prediction service instance."""
    try:
        return PricePredictionService()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price prediction service not available: {str(e)}",
        )


@router.post(
    "/get-price-suggestion",
    response_model=PriceSuggestionResponse,
    status_code=status.HTTP_200_OK,
)
async def get_price_suggestion(
    request: PriceSuggestionRequest,
    service: PricePredictionService = Depends(get_price_prediction_service),
) -> PriceSuggestionResponse:
    """
    Get AI-powered price prediction for a property.

    Provide property details to get an estimated price range based on:
    - Location (city, district, ward)
    - Property type (House, Apartment, Villa, Office, etc.)
    - Area in square meters
    - Geographic coordinates

    The AI analyzes market data and provides realistic price ranges in VND.

    - **city**: City or province name (e.g., 'Hanoi', 'Ho Chi Minh')
    - **district**: District name within the city
    - **ward**: Ward name within the district
    - **property_type**: Type of property
    - **area**: Property area in m² (optional)
    - **latitude**: Latitude coordinate
    - **longitude**: Longitude coordinate
    """
    try:
        return await service.predict_price(request)
    except Exception as e:
        logger.error(f"Price prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price prediction failed: {str(e)}",
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Health check endpoint for price prediction service."""
    return {"status": "healthy", "service": "price_prediction"}
