import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.dto.listing_verification import (
    ListingVerificationError,
    ListingVerificationRequest,
    ListingVerificationResponse,
)
from app.service.listing_verification_service import ListingVerificationService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/verify-listing",
    response_model=ListingVerificationResponse,
    responses={
        400: {"model": ListingVerificationError},
        500: {"model": ListingVerificationError},
    },
    summary="Verify Rental Listing",
    description="""
    Verify a rental property listing using AI analysis.

    This endpoint analyzes:
    1. **Image Quality**: Checks if images are appropriate, clear, and show actual property spaces
    2. **Content Relevance**: Verifies that the content is rental-related and matches the category
    3. **Completeness**: Evaluates if all necessary information is provided

    The AI returns a comprehensive analysis with validation scores, detected violations,
    and suggestions for improvement.
    """,
)
async def verify_listing(
    request: ListingVerificationRequest,
) -> ListingVerificationResponse:
    """
    Verify a rental listing using Gemini AI multimodal analysis.

    Args:
        request: The listing data to verify

    Returns:
        ListingVerificationResponse: Comprehensive verification results

    Raises:
        HTTPException: If verification fails due to invalid input or system errors
    """
    try:
        logger.info(f"Starting verification for listing: {request.title[:50]}...")

        # Initialize verification service
        service = ListingVerificationService()

        # Perform verification
        result = await service.verify_listing(request)

        logger.info(
            f"Verification completed for listing: {request.title[:50]}. "
            f"Score: {result.score:.2f}, Valid: {result.is_valid}"
        )

        from fastapi.encoders import jsonable_encoder

        return JSONResponse(
            status_code=status.HTTP_200_OK, content=jsonable_encoder(result)
        )

    except ValueError as e:
        logger.warning(f"Invalid request data: {str(e)}")
        error = ListingVerificationError(
            error="validation_error",
            message=f"Invalid request data: {str(e)}",
            details={},
        )
        from fastapi.encoders import jsonable_encoder

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=jsonable_encoder(error),
        )

    except Exception as e:
        logger.error(f"Unexpected error during listing verification: {str(e)}")
        error = ListingVerificationError(
            error="internal_error",
            message="An unexpected error occurred during verification",
            details={"error_type": type(e).__name__},
        )
        from fastapi.encoders import jsonable_encoder

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=jsonable_encoder(error),
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Check if the listing verification service is operational",
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the listing verification service.

    Returns:
        dict: Service status and version information
    """
    try:
        # Basic service health check
        return {
            "status": "healthy",
            "service": "listing_verification",
            "version": "1.0.0",
            "ai_model": settings.GEMINI_VISION_MODEL,
            "capabilities": [
                "image_analysis",
                "content_verification",
                "completeness_check",
                "multimodal_analysis",
            ],
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": "listing_verification",
            },
        )
