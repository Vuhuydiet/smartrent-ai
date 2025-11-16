from typing import Dict, Optional

from pydantic import BaseModel, Field


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
