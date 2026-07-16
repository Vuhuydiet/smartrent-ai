from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl

from app.core.config import settings


class HousingPropertyType(str, Enum):
    """Housing property type enumeration"""

    APARTMENT = "APARTMENT"
    HOUSE = "HOUSE"
    ROOM = "ROOM"
    STUDIO = "STUDIO"
    OFFICE = "OFFICE"
    STORE = "STORE"


class ListingType(str, Enum):
    """Listing transaction type"""

    RENT = "RENT"
    SALE = "SALE"
    SHARE = "SHARE"


class PriceUnit(str, Enum):
    """Unit the listing price is denominated in"""

    MONTH = "MONTH"
    DAY = "DAY"
    YEAR = "YEAR"


class VerificationSuggestedStatus(str, Enum):
    """Suggested status from AI analysis"""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ListingImage(BaseModel):
    """Model for listing images"""

    url: HttpUrl


class ListingVideo(BaseModel):
    """Model for listing videos"""

    url: HttpUrl


class PropertyMetadata(BaseModel):
    """Model for additional property metadata"""

    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floor: Optional[int] = None
    total_floors: Optional[int] = None


class ListingVerificationRequest(BaseModel):
    """Request model for listing verification"""

    title: str = Field(..., min_length=1, max_length=500, description="Property title")
    description: str = Field(
        ..., min_length=10, max_length=50000, description="Property description"
    )
    price: float = Field(
        ...,
        gt=0,
        description="Listing price, interpreted using listing_type and price_unit",
    )
    price_unit: Optional[PriceUnit] = Field(
        None, description="Unit the price is denominated in (MONTH, DAY, YEAR)"
    )
    listing_type: Optional[ListingType] = Field(
        None,
        description="Transaction type: RENT (cho thue), SALE (ban), SHARE (o ghep)",
    )
    area: Optional[float] = Field(
        None, gt=0, description="Property area in square meters"
    )
    address: str = Field(
        ..., min_length=5, max_length=500, description="Property address"
    )
    amenities: List[str] = Field(
        default_factory=list, description="List of property amenities"
    )
    images: List[str] = Field(
        default_factory=list, max_length=20, description="Property image URLs"
    )
    videos: List[ListingVideo] = Field(
        default_factory=list, max_length=5, description="Property videos"
    )
    metadata: Optional[PropertyMetadata] = None
    property_type: Optional[HousingPropertyType] = Field(
        None, description="Type of property"
    )
    direction: Optional[str] = Field(None, description="House/room direction")
    furnishing: Optional[str] = Field(None, description="Furnishing status")
    room_capacity: Optional[int] = Field(
        None, description="Max occupants the room supports"
    )
    water_price: Optional[str] = Field(None, description="Water cost")
    electricity_price: Optional[str] = Field(None, description="Electricity cost")
    internet_price: Optional[str] = Field(None, description="Internet cost")
    service_fee: Optional[str] = Field(None, description="Service/management fee")


class Violation(BaseModel):
    """Model for listing violations"""

    category: str = Field(..., description="Category of violation")
    severity: str = Field(
        ..., description="Severity level (low, medium, high, critical)"
    )
    message: str = Field(..., description="Violation description")
    field: Optional[str] = Field(
        None, description="Specific field that has the violation"
    )


class Suggestion(BaseModel):
    """Model for improvement suggestions"""

    category: str = Field(..., description="Category of suggestion")
    message: str = Field(..., description="Suggestion description")
    field: Optional[str] = Field(None, description="Specific field to improve")
    priority: str = Field(..., description="Priority level (low, medium, high)")


class ImageValidation(BaseModel):
    """Model for image validation results"""

    is_valid: bool = Field(..., description="Whether images are valid")
    total_images: int = Field(..., description="Total number of images")
    valid_images: int = Field(..., description="Number of valid images")
    issues: List[str] = Field(
        default_factory=list, description="Image validation issues"
    )
    quality_score: float = Field(
        ..., ge=0, le=1, description="Overall image quality score"
    )


class VideoValidation(BaseModel):
    """Model for video validation results"""

    is_valid: bool = Field(..., description="Whether videos are valid")
    total_videos: int = Field(..., description="Total number of videos")
    valid_videos: int = Field(..., description="Number of valid videos")
    issues: List[str] = Field(
        default_factory=list, description="Video validation issues"
    )
    quality_score: float = Field(
        ..., ge=0, le=1, description="Overall video quality score"
    )


class ContentValidation(BaseModel):
    """Model for content validation results"""

    is_rental_related: bool = Field(
        ..., description="Whether content is rental-related"
    )
    category_match: bool = Field(
        ..., description="Whether content matches rental category"
    )
    content_score: float = Field(..., ge=0, le=1, description="Content relevance score")
    issues: List[str] = Field(
        default_factory=list, description="Content validation issues"
    )


class CompletenessValidation(BaseModel):
    """Model for completeness validation results"""

    is_complete: bool = Field(..., description="Whether listing is complete")
    completeness_score: float = Field(..., ge=0, le=1, description="Completeness score")
    missing_fields: List[str] = Field(
        default_factory=list, description="Missing required fields"
    )
    quality_issues: List[str] = Field(
        default_factory=list, description="Quality issues"
    )


class StructuredReason(BaseModel):
    """Structured reasons for AI decisions"""

    blurriness_issue: bool = Field(default=False, description="Image is too blurry")
    missing_fields: List[str] = Field(
        default_factory=list, description="Missing important fields"
    )
    inconsistent_info: bool = Field(
        default=False, description="Text and image info do not match"
    )
    watermark_or_phone: bool = Field(
        default=False, description="Contains watermarks or phone numbers in images"
    )
    stock_photo: bool = Field(
        default=False,
        description="Images appear to be stock photos or from the internet",
    )
    details: str = Field(..., description="A detailed explanation of the reasons")


class ListingVerificationResponse(BaseModel):
    """Response model for listing verification"""

    is_valid: bool = Field(..., description="Overall validity of the listing")
    score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in the assessment"
    )

    # Detailed validation results
    image_validation: ImageValidation
    video_validation: VideoValidation
    content_validation: ContentValidation
    completeness_validation: CompletenessValidation

    # Structured reasons and violations
    reason: StructuredReason = Field(
        ..., description="Detailed structured reasons for the assessment"
    )
    violation_codes: List[str] = Field(
        default_factory=list,
        description="Critical violation codes like SCAM, INAPPROPRIATE_CONTENT",
    )

    # Issues and suggestions
    violations: List[Violation] = Field(default_factory=list)
    suggestions: List[Suggestion] = Field(default_factory=list)

    # Metadata
    suggested_status: VerificationSuggestedStatus = Field(
        default=VerificationSuggestedStatus.NEEDS_REVIEW,
        description="Suggested action for the backend",
    )
    verification_timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(default_factory=lambda: settings.LLM_VISION_MODEL)
    processing_time_seconds: Optional[float] = None


class ListingVerificationError(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
