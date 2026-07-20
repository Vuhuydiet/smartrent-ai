from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

PriceConfidence = Literal["high", "medium", "low"]
PriceSource = Literal["ai_comparables", "rule_based_fallback"]


class PriceSuggestionRequest(BaseModel):
    """Request for AI-powered real estate price suggestion"""

    city: str = Field(
        ...,
        description="City or province name (e.g., 'Hanoi', 'Ho Chi Minh', 'Tien Giang')",
    )
    district: str = Field(
        ...,
        description="District or county name (e.g., 'Ba Dinh', 'District 1', 'My Tho')",
    )
    ward: str = Field(
        ..., description="Ward or commune name (e.g., 'Dien Bien Ward', 'Ward 1')"
    )
    property_type: str = Field(
        ..., description="Type of property (House, Apartment, Villa, Office, etc.)"
    )
    area: Optional[float] = Field(
        None, description="Property area in square meters (m²) - optional", gt=0
    )
    latitude: float = Field(
        ..., description="Property latitude coordinate (decimal degrees)"
    )
    longitude: float = Field(
        ..., description="Property longitude coordinate (decimal degrees)"
    )


class PriceSuggestionResponse(BaseModel):
    """AI-powered price suggestion response"""

    price_range: Dict[str, int] = Field(
        ...,
        description="Estimated price range in Vietnamese Dong (VND) with min and max values",
    )
    location: str = Field(..., description="Formatted location string (District, City)")
    property_type: str = Field(..., description="Type of property being evaluated")
    currency: str = Field(default="VND", description="Currency code (Vietnamese Dong)")
    source: PriceSource = Field(
        default="ai_comparables",
        description=(
            "How the range was produced. 'ai_comparables' = derived from real "
            "listings retrieved from the backend; 'rule_based_fallback' = the "
            "AI path failed and a hardcoded per-m² table was used instead."
        ),
    )
    listings_found: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of comparable listings actually retrieved from the backend "
            "and used as evidence. 0 means the range is not backed by market data."
        ),
    )
    confidence: PriceConfidence = Field(
        default="low",
        description=(
            "Confidence in the estimate. Forced to 'low' whenever no comparable "
            "listings were retrieved or the rule-based fallback was used."
        ),
    )
