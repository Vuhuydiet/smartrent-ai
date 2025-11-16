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
        # Load model
        model = get_model()

        # Prepare input data
        input_data = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "property_type": request.property_type,
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
            property_type=request.property_type,
            currency="VND",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting price suggestion: {str(e)}"
        )
