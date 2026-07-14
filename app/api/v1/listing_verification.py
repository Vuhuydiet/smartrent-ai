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
) -> Any:
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
        import traceback

        tb = traceback.format_exc()
        logger.error(f"Unexpected error during listing verification:\n{tb}")
        error = ListingVerificationError(
            error="internal_error",
            message="An unexpected error occurred during verification",
            details={"error_type": type(e).__name__, "traceback": tb},
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
async def health_check(probe: bool = False) -> Dict[str, Any]:
    """
    Health check for the listing verification service.

    By default this verifies the LLM is actually *configured* — a cheap,
    no-network check that constructs the model and surfaces the "no credentials
    configured" case that previously stayed hidden until every analysis silently
    fell back to basic rules while this endpoint still reported "healthy".

    Pass ``?probe=true`` for a real round-trip to the LLM (a tiny completion) when
    you need to confirm the provider is reachable and authenticated — kept opt-in
    because it costs a token and adds latency, so it must not run on every poll.
    """
    base = {
        "service": "listing_verification",
        "version": "1.0.0",
        "ai_model": settings.LLM_VISION_MODEL,
        "capabilities": [
            "image_analysis",
            "content_verification",
            "completeness_check",
            "multimodal_analysis",
        ],
    }

    # 1. Configuration check (no network). Catches the missing-credential case.
    try:
        from app.ai.llm.agent_factory import make_model

        make_model(settings.LLM_VISION_MODEL)
    except Exception as e:
        logger.error("Health check: LLM not configured: %s", e)
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={**base, "status": "unhealthy", "ai_available": False,
                     "error_code": "LLM_NOT_CONFIGURED", "error": str(e)},
        )

    if not probe:
        return {**base, "status": "healthy", "ai_available": True}

    # 2. Live probe (opt-in): actually call the LLM.
    try:
        service = ListingVerificationService()
        result = await service.gemini_helper.analyze_text_content(
            text_content="ping",
            analysis_prompt="Reply with a JSON object: {\"ok\": true}",
        )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        return {**base, "status": "healthy", "ai_available": True, "probed": True}
    except Exception as e:
        logger.error("Health check: LLM probe failed: %s", e)
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={**base, "status": "unhealthy", "ai_available": False,
                     "error_code": "LLM_ERROR", "error": str(e), "probed": True},
        )
