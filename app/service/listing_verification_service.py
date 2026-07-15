import logging
import time
from typing import Any, Dict

from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper
from app.dto.listing_verification import (
    CompletenessValidation,
    ContentValidation,
    ImageValidation,
    ListingVerificationRequest,
    ListingVerificationResponse,
    Suggestion,
    VideoValidation,
    Violation,
)

logger = logging.getLogger(__name__)


class AiAnalysisUnavailableError(Exception):
    """Raised when the LLM call fails and no real analysis can be produced.

    Deliberately NOT swallowed into a fake 200 response with basic-rule scores —
    a caller (admin, or the backend's background pre-computation worker) needs to
    know verification did not happen, not be handed placeholder numbers that look
    like a verdict. The API layer maps this to a proper HTTP error; the backend's
    background worker already treats any exception here as "retry later".
    """

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


class ListingVerificationService:
    """Service for verifying rental listings using Gemini AI"""

    def __init__(self) -> None:
        """Initialize the listing verification service"""
        self.gemini_helper = GeminiListingVerificationHelper()

    async def verify_listing(
        self, listing_data: ListingVerificationRequest
    ) -> ListingVerificationResponse:
        """
        Verify a rental listing using Gemini AI multimodal capabilities

        Args:
            listing_data: The listing data to verify

        Returns:
            ListingVerificationResponse: Verification results
        """
        start_time = time.time()

        try:
            # Prepare text content for analysis
            text_content = self._prepare_text_content(listing_data)

            # Extract assets
            image_urls = listing_data.images if listing_data.images else []
            video_urls = (
                [str(video.url) for video in listing_data.videos]
                if listing_data.videos
                else []
            )

            # Perform comprehensive analysis using Gemini
            if image_urls or video_urls:
                analysis_result = await self.gemini_helper.analyze_multimodal(
                    image_urls=image_urls,
                    video_urls=video_urls,
                    text_content=text_content,
                    analysis_prompt=self.gemini_helper.create_analysis_prompt(),
                )
            else:
                # Text-only analysis
                analysis_result = await self.gemini_helper.analyze_text_content(
                    text_content=text_content,
                    analysis_prompt=self.gemini_helper.create_analysis_prompt(),
                )

            # Process results
            response = self._process_analysis_result(analysis_result, listing_data)
            response.processing_time_seconds = time.time() - start_time

            logger.info(
                f"Listing verification completed in {response.processing_time_seconds:.2f}s. "
                f"Score: {response.score:.2f}, Valid: {response.is_valid}"
            )

            return response

        except Exception as e:
            logger.error(f"Error during listing verification: {str(e)}")
            raise

    def _prepare_text_content(self, listing_data: ListingVerificationRequest) -> str:
        """Prepare text content for analysis"""
        metadata_str = "None"
        if listing_data.metadata:
            metadata_parts = []
            if listing_data.metadata.bedrooms:
                metadata_parts.append(f"Bedrooms: {listing_data.metadata.bedrooms}")
            if listing_data.metadata.bathrooms:
                metadata_parts.append(f"Bathrooms: {listing_data.metadata.bathrooms}")
            if listing_data.metadata.floor:
                metadata_parts.append(f"Floor: {listing_data.metadata.floor}")

            metadata_str = ", ".join(metadata_parts) if metadata_parts else "None"

        text_content = f"""
### LISTING INFORMATION:
- **Title**: {listing_data.title}
- **Description**: {listing_data.description}
- **Price**: {listing_data.price} (VND per month)
- **Area**: {listing_data.area or 'Not specified'} m2
- **Address**: {listing_data.address}
- **Property Type**: {listing_data.property_type.value if listing_data.property_type else 'Not specified'}
- **Amenities**: {', '.join(listing_data.amenities) if listing_data.amenities else 'None'}

### MEDIA STATS:
- **Number of Images Attached**: {len(listing_data.images)}
- **Number of Videos**: {len(listing_data.videos)}

### METADATA:
- {metadata_str}
"""
        return text_content

    # Removed _create_text_only_analysis_prompt as it's handled by system instruction

    def _process_analysis_result(
        self, analysis_result: Dict[str, Any], listing_data: ListingVerificationRequest
    ) -> ListingVerificationResponse:
        """Process the analysis result and create response"""

        # Handle error cases: raise rather than fabricate a response. A basic-rule
        # fallback here would return 200 with a score and claims like
        # is_rental_related=true that nothing actually verified — indistinguishable
        # from a real analysis to anything downstream that isn't specifically
        # checking for it.
        if "error" in analysis_result:
            raise AiAnalysisUnavailableError(
                error_code=analysis_result.get("error_code", "LLM_ERROR"),
                message=analysis_result["error"],
            )

        # Extract analysis components
        image_validation_data = analysis_result.get("image_validation", {})
        video_validation_data = analysis_result.get("video_validation", {})
        content_validation_data = analysis_result.get("content_validation", {})
        completeness_validation_data = analysis_result.get(
            "completeness_validation", {}
        )
        reason_data = analysis_result.get("reason", {})
        violation_codes = analysis_result.get("violation_codes", [])

        # Create validation objects
        image_validation = ImageValidation(
            is_valid=image_validation_data.get(
                "is_valid", len(listing_data.images) == 0
            ),
            total_images=len(listing_data.images),
            valid_images=image_validation_data.get("valid_images", 0),
            issues=image_validation_data.get("issues", []),
            quality_score=image_validation_data.get(
                "quality_score", 1.0 if len(listing_data.images) == 0 else 0.5
            ),
        )

        video_validation = VideoValidation(
            is_valid=video_validation_data.get("is_valid", True)
            if len(listing_data.videos) == 0
            else video_validation_data.get("is_valid", False),
            total_videos=len(listing_data.videos),
            valid_videos=video_validation_data.get("valid_videos", 0),
            issues=video_validation_data.get("issues", []),
            quality_score=video_validation_data.get(
                "quality_score", 1.0 if len(listing_data.videos) == 0 else 0.0
            ),
        )

        content_validation = ContentValidation(
            is_rental_related=content_validation_data.get("is_rental_related", True),
            category_match=content_validation_data.get("category_match", True),
            content_score=content_validation_data.get("content_score", 0.5),
            issues=content_validation_data.get("issues", []),
        )

        completeness_validation = CompletenessValidation(
            is_complete=completeness_validation_data.get("is_complete", False),
            completeness_score=completeness_validation_data.get(
                "completeness_score", 0.5
            ),
            missing_fields=completeness_validation_data.get("missing_fields", []),
            quality_issues=completeness_validation_data.get("quality_issues", []),
        )

        from app.dto.listing_verification import StructuredReason

        reason = StructuredReason(
            blurriness_issue=reason_data.get("blurriness_issue", False),
            missing_fields=reason_data.get("missing_fields", []),
            inconsistent_info=reason_data.get("inconsistent_info", False),
            watermark_or_phone=reason_data.get("watermark_or_phone", False),
            stock_photo=reason_data.get("stock_photo", False),
            details=reason_data.get("details", "AI Assessment"),
        )

        # Extract violations and suggestions
        violations = []
        for violation_data in analysis_result.get("violations", []):
            violations.append(
                Violation(
                    category=violation_data.get("category", "unknown"),
                    severity=violation_data.get("severity", "low"),
                    message=violation_data.get("message", "Violation detected"),
                    field=violation_data.get("field")
                    if violation_data.get("field")
                    else "",
                )
            )

        suggestions = []
        for suggestion_data in analysis_result.get("suggestions", []):
            suggestions.append(
                Suggestion(
                    category=suggestion_data.get("category", "improvement"),
                    message=suggestion_data.get("message", "Improvement suggested"),
                    field=suggestion_data.get("field")
                    if suggestion_data.get("field")
                    else "",
                    priority=suggestion_data.get("priority", "medium"),
                )
            )

        # Use overall assessment scores if available, otherwise calculate
        overall_score = analysis_result.get(
            "score",
            (
                image_validation.quality_score * 0.3
                + content_validation.content_score * 0.4
                + completeness_validation.completeness_score * 0.3
            ),
        )

        is_valid = analysis_result.get(
            "is_valid",
            (
                content_validation.is_rental_related
                and content_validation.category_match
                and completeness_validation.is_complete
                and overall_score >= 0.6
            ),
        )

        confidence = analysis_result.get("confidence", 0.8)
        suggested_status = analysis_result.get("suggested_status", "NEEDS_REVIEW")

        return ListingVerificationResponse(
            is_valid=is_valid,
            score=overall_score,
            confidence=confidence,
            suggested_status=suggested_status,
            image_validation=image_validation,
            video_validation=video_validation,
            content_validation=content_validation,
            completeness_validation=completeness_validation,
            violations=violations,
            suggestions=suggestions,
            reason=reason,
            violation_codes=violation_codes,
        )
