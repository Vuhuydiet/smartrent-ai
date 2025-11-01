import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from fastapi import APIRouter, Depends, HTTPException

from app.ai.house_pricing.price_predictor import RealEstatePricePredictorModel
from app.dto.house_pricing import (
    PriceEvaluationRequest,
    PriceEvaluationResponse,
    PriceRangeResponse,
    PropertyData,
    PropertyListing,
    TrainingDataRequest,
    TrainingDataResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance và data storage
_model_instance = None
_model_loaded_at = None
_training_data = None
_data_received_at = None


@router.post("/initialize-model")
async def initialize_model(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Initialize model from static training data (SQL file)

    - **force_retrain**: Whether to force model retraining
    """
    global _model_instance, _model_loaded_at, _training_data, _data_received_at

    try:
        if _model_instance is None or force_retrain:
            logger.info("Initializing model from static training data...")

            try:
                _model_instance = RealEstatePricePredictorModel()
                # Load from SQL file
                _model_instance.load_and_train_from_sql()
                _model_loaded_at = datetime.now()
                _data_received_at = datetime.now()

                logger.info("Model initialized successfully from SQL data")

                return {
                    "success": True,
                    "message": "Model initialized successfully",
                    "model_loaded_at": _model_loaded_at.isoformat(),
                    "training_samples": len(_model_instance.training_data)
                    if hasattr(_model_instance, "training_data")
                    else 0,
                }

            except Exception as training_error:
                logger.error(f"Model initialization failed: {str(training_error)}")
                _model_instance = None
                raise HTTPException(
                    status_code=400,
                    detail=f"Model initialization failed: {str(training_error)}",
                )
        else:
            return {
                "success": True,
                "message": "Model already initialized",
                "model_loaded_at": _model_loaded_at.isoformat()
                if _model_loaded_at
                else None,
                "training_samples": len(_model_instance.training_data)
                if hasattr(_model_instance, "training_data")
                else 0,
            }

    except Exception as e:
        logger.error(f"Failed to process training data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Training data processing failed: {str(e)}"
        )


def get_model() -> RealEstatePricePredictorModel:
    """
    Get trained model instance
    """
    if _model_instance is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Please send training data first via /receive-training-data",
        )
    return _model_instance


@router.post("/get-price-range", response_model=PriceRangeResponse)
async def get_property_price_range(
    property_data: PropertyData,
    model: RealEstatePricePredictorModel = Depends(get_model),
) -> PriceRangeResponse:
    """
    Get price range for property based on address and type
    Frontend sends address + type → Backend gets price range from AI

    - **property_data**: Property address and type from frontend
    """
    try:
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

        # Return simplified response for backend
        return PriceRangeResponse(
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

    except Exception as e:
        logger.error(f"Price range calculation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Price range calculation failed: {str(e)}"
        )


@router.post("/evaluate-price", response_model=PriceEvaluationResponse)
async def evaluate_property_price(
    request: PriceEvaluationRequest,
    model: RealEstatePricePredictorModel = Depends(get_model),
) -> PriceEvaluationResponse:
    """
    Evaluate property price against market rates

    - **property_data**: Property information (location, type, etc.)
    - **asking_price**: Price to evaluate (million VND)

    Returns: Detailed evaluation with recommendation
    """
    try:
        logger.info(
            f"Evaluating price for property in {request.property_data.district}, {request.property_data.city}"
        )

        # Convert PropertyData to dict for model
        property_dict = {
            "latitude": request.property_data.latitude,
            "longitude": request.property_data.longitude,
            "property_type": request.property_data.property_type,
            "city": request.property_data.city,
            "district": request.property_data.district,
            "ward": request.property_data.ward,
            "post_date": request.property_data.post_date or datetime.now().isoformat(),
        }

        # Get price evaluation from model
        evaluation = model.evaluate_price_vs_market(property_dict, request.asking_price)

        # Convert to response format
        response = PriceEvaluationResponse(
            asking_price=evaluation["asking_price"],
            predicted_price=evaluation["predicted_price"],
            price_difference_percentage=evaluation["price_difference_percentage"],
            price_difference_absolute=evaluation["asking_price"]
            - evaluation["predicted_price"],
            market_evaluation=evaluation["market_evaluation"],
            price_range=evaluation["price_range"],
            confidence_interval=evaluation["confidence_interval"],
            currency=evaluation["currency"],
            model_loaded_at=_model_loaded_at.isoformat() if _model_loaded_at else None,
        )

        logger.info(
            f"Price evaluation completed: {evaluation['market_evaluation']['category']}"
        )
        return response

    except Exception as e:
        logger.error(f"Price evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Price evaluation failed: {str(e)}"
        )


@router.post("/predict-property")
async def predict_user_property(
    property_data: PropertyData,
    model: RealEstatePricePredictorModel = Depends(get_model),
) -> Dict[str, Any]:
    """
    Predict property price from user input (address, property type)

    - **property_data**: User input (address, property type, etc.)
    """
    try:
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

        # Get price prediction
        prediction = model.predict_price_range(property_dict)

        return prediction

    except Exception as e:
        logger.error(f"Price prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Price prediction failed: {str(e)}"
        )


@router.post("/predict-price")
async def predict_property_price(
    property_data: PropertyData,
    model: RealEstatePricePredictorModel = Depends(get_model),
) -> Dict[str, Any]:
    """
    Predict property price based on location and characteristics

    Returns: Predicted price and confidence interval
    """
    try:
        logger.info(
            f"Predicting price for property in {property_data.district}, {property_data.city}"
        )

        # Convert to dict for model
        property_dict = {
            "latitude": property_data.latitude,
            "longitude": property_data.longitude,
            "property_type": property_data.property_type,
            "city": property_data.city,
            "district": property_data.district,
            "ward": property_data.ward,
            "post_date": property_data.post_date or datetime.now().isoformat(),
        }

        # Get prediction
        prediction = model.predict_price_range(property_dict)

        logger.info(
            f"Price prediction completed: {prediction['predicted_price']:.1f} triệu VND"
        )
        return {
            **prediction,
            "model_loaded_at": _model_loaded_at.isoformat()
            if _model_loaded_at
            else None,
        }

    except Exception as e:
        logger.error(f"Price prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Price prediction failed: {str(e)}"
        )


@router.get("/model-status")
async def get_model_status() -> Dict[str, Any]:
    """
    Get model and training data status
    """
    return {
        "is_trained": _model_instance is not None,
        "training_samples": len(_training_data) if _training_data is not None else 0,
        "last_training_time": _model_loaded_at.isoformat()
        if _model_loaded_at
        else None,
        "data_received_at": _data_received_at.isoformat()
        if _data_received_at
        else None,
        "features": [
            "latitude",
            "longitude",
            "year",
            "month",
            "property_type_encoded",
            "city_encoded",
            "district_encoded",
            "ward_encoded",
            "knn_mean_price",
        ]
        if _model_instance
        else [],
        "status": "ready" if _model_instance else "waiting_for_data",
    }


@router.get("/training-data/info")
async def get_training_data_info() -> Dict[str, Any]:
    """
    Get information about current training data
    """
    if _training_data is None:
        raise HTTPException(status_code=404, detail="No training data available")

    try:
        # Thống kê cơ bản về data
        stats = {
            "total_samples": len(_training_data),
            "data_received_at": _data_received_at.isoformat()
            if _data_received_at
            else None,
            "price_stats": {
                "min": float(_training_data["price"].min()),
                "max": float(_training_data["price"].max()),
                "mean": float(_training_data["price"].mean()),
                "median": float(_training_data["price"].median()),
            },
            "city_distribution": _training_data["city"].value_counts().head().to_dict(),
            "property_type_distribution": _training_data["property_type"]
            .value_counts()
            .to_dict(),
            "date_range": {
                "earliest": _training_data["post_date"].min()
                if "post_date" in _training_data.columns
                else None,
                "latest": _training_data["post_date"].max()
                if "post_date" in _training_data.columns
                else None,
            },
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to get training data info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training data info: {str(e)}"
        )
