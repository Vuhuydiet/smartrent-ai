import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.ai.house_pricing.price_predictor import RealEstatePricePredictorModel
from app.dto.house_pricing import PriceRangeResponse, PropertyData

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance
_model_instance = None
_model_loaded_at = None


def get_model() -> RealEstatePricePredictorModel:
    """
    Get trained model instance
    """
    global _model_instance, _model_loaded_at

    if _model_instance is None:
        logger.info("Initializing model on first request...")
        try:
            _model_instance = RealEstatePricePredictorModel()
            _model_instance.load_and_train_from_sql()
            _model_loaded_at = datetime.now()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Model initialization failed: {str(e)}"
            )

    return _model_instance


@router.post(
    "/get-price-range",
    response_model=PriceRangeResponse,
    tags=["House Pricing"],
    summary="Get Property Price Range",
    description="Main API endpoint for getting AI-powered property price predictions",
    response_description="Property price range with confidence interval",
)
async def get_property_price_range(
    property_data: PropertyData,
    model: RealEstatePricePredictorModel = Depends(get_model),
) -> PriceRangeResponse:
    """
    Get property price range prediction based on location and property type.

    Processes property details to provide AI-powered price predictions for Vietnamese properties.

    Args:
        property_data: Property details including coordinates, type, and address

    Returns:
        PriceRangeResponse: Predicted price range with confidence interval
    """
    try:
        logger.info(
            f"Processing price range request for {property_data.property_type} in {property_data.district}, {property_data.city}"
        )

        # Convert PropertyData to dict for model
        property_dict = {
            "latitude": property_data.latitude,
            "longitude": property_data.longitude,
            "property_type": property_data.property_type,
            "city": property_data.city,
            "district": property_data.district,
            "ward": property_data.ward,
            "post_date": property_data.post_date or "2025-11-01T00:00:00",
        }

        # Get price prediction and range
        prediction = model.predict_price_range(property_dict)

        # Return formatted response for frontend
        response = PriceRangeResponse(
            address=f"{property_data.district}, {property_data.city}",
            property_type=property_data.property_type,
            predicted_price=prediction["predicted_price"],
            price_range={
                "min_price": prediction["price_range"]["min"],
                "max_price": prediction["price_range"]["max"],
            },
            confidence=prediction["confidence_interval"],
            currency="VND_millions",
        )

        logger.info(
            f"Price range calculated successfully: {response.predicted_price:.1f} million VND"
        )
        return response

    except Exception as e:
        logger.error(f"Price range calculation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Price range calculation failed: {str(e)}"
        )


@router.get(
    "/health",
    tags=["System"],
    summary="API Health Check",
    description="Check if the API and AI model are running properly",
    response_description="Service status and model availability",
)
async def health_check() -> Dict[str, Any]:
    """
    # 🔍 API Health Check

    **System Status Monitoring**

    This endpoint allows frontend applications to verify that the SmartRent AI service is operational and the machine learning model is loaded and ready to process requests.

    ## 📊 Response Information
    - **status**: Service operational status
    - **service**: Service name and description
    - **model_loaded**: Whether AI model is initialized
    - **timestamp**: Current server time

    ## 🚀 Usage
    - Use for application startup verification
    - Implement health monitoring in production
    - Debug connectivity issues
    - Verify model availability before making predictions
    """
    return {
        "status": "healthy",
        "service": "SmartRent AI - House Pricing",
        "model_loaded": _model_instance is not None,
        "timestamp": datetime.now().isoformat(),
    }
