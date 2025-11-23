from fastapi import APIRouter, HTTPException

from app.ai.house_pricing.price_predictor import RealEstatePricePredictorModel
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse

router = APIRouter()

_model_instance = None


def get_model() -> RealEstatePricePredictorModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = RealEstatePricePredictorModel()
        _model_instance.load_and_train_from_sql()
    return _model_instance


@router.post("/get-price-range", response_model=PriceSuggestionResponse)
async def get_price_range(
    request: PriceSuggestionRequest,
) -> PriceSuggestionResponse:
    """
    Get price range for real estate properties

    This endpoint provides AI-powered price range predictions based on:
    - Property location (city, district, ward)
    - Property type (APARTMENT, HOUSE, ROOM, STUDIO)
    - Property area (optional)
    - Geographic coordinates (latitude, longitude)

    Returns:
        PriceSuggestionResponse: Contains price range in VND, location info, and property type

    Example:
        Request: Property in Ba Dinh, Hanoi (APARTMENT)
        Response: Price range 15M - 50M VND
    """
    try:
        # Load model
        model = get_model()

        # Prepare input data
        input_data = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "property_type": request.property_type.value,  # Use enum value
            "city": request.city,
            "district": request.district,
            "ward": request.ward,
            "post_date": "2025-11-07T00:00:00",  # Current date
        }

        # Get price prediction
        result = model.predict_price_range(input_data)
        range_min = result["price_range"]["min"]  # triệu VND
        range_max = result["price_range"]["max"]  # triệu VND

        # Convert to VND for response
        total_price_min = int(range_min * 1_000_000)  # VND
        total_price_max = int(range_max * 1_000_000)  # VND

        return PriceSuggestionResponse(
            price_range={"min": total_price_min, "max": total_price_max},
            location=f"{request.district}, {request.city}",
            property_type=request.property_type.value,
            currency="VND",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting price range: {str(e)}"
        )
