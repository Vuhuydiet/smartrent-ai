from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PropertyData(BaseModel):
    """Property information for AI price prediction."""

    latitude: float = Field(
        ...,
        description="Property latitude coordinate (Vietnam range: 8.0-23.5)",
        ge=8.0,
        le=23.5,
    )
    longitude: float = Field(
        ...,
        description="Property longitude coordinate (Vietnam range: 102.0-110.0)",
        ge=102.0,
        le=110.0,
    )
    property_type: str = Field(..., description="Type of real estate property")
    city: str = Field(
        ..., description="Major city name (Hanoi, Ho Chi Minh, Da Nang, etc.)"
    )
    district: str = Field(..., description="District/Quan within the city")
    ward: str = Field(..., description="Ward/Phuong within the district")
    post_date: Optional[str] = Field(
        None, description="Property post date (ISO format, optional)"
    )


class PriceEvaluationRequest(BaseModel):
    """Request for property price evaluation."""

    property_data: PropertyData = Field(..., description="Property information")
    asking_price: float = Field(..., gt=0, description="Asking price (million VND)")


class MarketEvaluation(BaseModel):
    """Market evaluation result."""

    category: str = Field(
        ...,
        description="Evaluation category (very_low, low, reasonable, high, very_high)",
    )
    message: str = Field(..., description="Evaluation message")
    recommendation: str = Field(..., description="Price recommendation")


class PriceRange(BaseModel):
    """Price confidence range."""

    min: float = Field(..., description="Minimum price")
    max: float = Field(..., description="Maximum price")


class PriceEvaluationResponse(BaseModel):
    """Property price evaluation response."""

    asking_price: float = Field(..., description="User asking price")
    predicted_price: float = Field(..., description="Model predicted price")
    price_difference_percentage: float = Field(..., description="Percentage difference")
    price_difference_absolute: float = Field(..., description="Absolute difference")
    market_evaluation: MarketEvaluation = Field(..., description="Market evaluation")
    price_range: PriceRange = Field(..., description="Price confidence range")
    confidence_interval: float = Field(..., description="Confidence level (0.9 = 90%)")
    currency: str = Field(..., description="Currency unit")
    model_loaded_at: Optional[str] = Field(None, description="Model loading timestamp")


class PropertyListing(BaseModel):
    """Property listing data from backend."""

    id: str = Field(..., description="Listing ID")
    title: str = Field(..., description="Property title")
    description: Optional[str] = Field(None, description="Detailed description")
    address: str = Field(..., description="Full address")
    city: str = Field(..., description="City name")
    district: str = Field(..., description="District/County")
    ward: str = Field(..., description="Ward/Commune")
    property_type: str = Field(..., description="Property type")
    price: float = Field(..., gt=0, description="Price (million VND)")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    post_date: str = Field(..., description="Post date")
    area: Optional[float] = Field(None, description="Area (m²)")
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[int] = Field(None, description="Number of bathrooms")


class TrainingDataRequest(BaseModel):
    """Request to receive training data from backend."""

    listings: List[PropertyListing] = Field(
        ..., description="List of property listings"
    )
    force_retrain: bool = Field(False, description="Whether to force model retraining")


class TrainingDataResponse(BaseModel):
    """Response after receiving training data."""

    total_listings: int = Field(..., description="Total number of listings")
    processed_listings: int = Field(..., description="Number of processed listings")
    model_trained: bool = Field(..., description="Whether model was trained")
    training_completed_at: str = Field(..., description="Training completion timestamp")
    model_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Model performance metrics"
    )


class PriceRangeResponse(BaseModel):
    """AI-powered price range response for frontend applications."""

    address: str = Field(..., description="Formatted property address for display")
    property_type: str = Field(..., description="Type of property that was evaluated")
    predicted_price: float = Field(
        ..., description="AI predicted price in million VND", gt=0
    )
    price_range: Dict[str, float] = Field(
        ..., description="Confidence interval with min/max prices"
    )
    confidence: float = Field(
        ...,
        description="Prediction confidence level (0.0-1.0, where 0.9 = 90%)",
        ge=0.0,
        le=1.0,
    )
    currency: str = Field(..., description="Currency unit for all price values")
