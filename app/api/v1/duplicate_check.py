import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, status

from app.service.duplicate_detection_service import DuplicateDetectionService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


class DuplicateCheckRequest(BaseModel):
    """Request body for duplicate listing check."""

    title: str
    description: str
    price: float
    area: Optional[float] = None
    productType: str  # ROOM, APARTMENT, HOUSE, STUDIO, OFFICE
    provinceCode: str
    districtId: Optional[int] = None
    address: Optional[str] = None
    imageUrls: Optional[List[str]] = None
    # ID of the listing being checked — used to exclude it from its own
    # candidate set (defensive against self-match on re-moderation).
    listingId: Optional[Any] = None


class SuspiciousMatch(BaseModel):
    listingId: Any
    title: str = ""
    score: float
    titleSimilarity: float = 0
    descriptionSimilarity: float = 0
    addressSimilarity: float = 0
    priceSimilarity: float = 0
    imageSimilarity: float = 0
    llmScore: Optional[float] = None
    llmReason: Optional[str] = None


class DuplicateCheckResponse(BaseModel):
    isDuplicate: bool
    highestScore: float
    decision: str  # PASS, SUSPICIOUS, DUPLICATE
    suspiciousMatches: List[SuspiciousMatch]


# ---------------------------------------------------------------------------
# Service singleton
# ---------------------------------------------------------------------------

_service: Optional[DuplicateDetectionService] = None


def _get_service() -> DuplicateDetectionService:
    global _service
    if _service is None:
        _service = DuplicateDetectionService()
    return _service


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/check-duplicate",
    response_model=DuplicateCheckResponse,
    status_code=status.HTTP_200_OK,
)
async def check_duplicate(request: DuplicateCheckRequest) -> DuplicateCheckResponse:
    """
    Check if a listing is a duplicate of any existing listing.

    Called by the Spring Boot backend during listing creation/update.
    Returns a decision (PASS/SUSPICIOUS/DUPLICATE) with similarity details.
    """
    try:
        service = _get_service()
        listing_data: Dict[str, Any] = {
            "title": request.title,
            "description": request.description,
            "price": request.price,
            "area": request.area,
            "productType": request.productType,
            "provinceCode": request.provinceCode,
            "districtId": request.districtId,
            "address": request.address or "",
            "imageUrls": request.imageUrls or [],
            "listingId": request.listingId,
        }

        result = await service.check_duplicate(listing_data)
        return DuplicateCheckResponse(**result)

    except Exception as e:
        logger.error("Duplicate check failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Duplicate check error: {type(e).__name__}: {str(e)}",
        )


@router.get("/check-duplicate/health")
async def duplicate_check_health() -> Dict[str, str]:
    return {"status": "healthy", "service": "duplicate-detection"}
