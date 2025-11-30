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
                # For now, we'll analyze images with text. Video support can be added later
                if image_urls:
                    analysis_result = await self.gemini_helper.analyze_images_with_text(
                        images=image_urls,
                        text_content=text_content,
                        analysis_prompt=self.gemini_helper.create_comprehensive_analysis_prompt(),
                    )
                else:
                    # Only videos, fallback to text analysis for now
                    analysis_result = await self.gemini_helper.analyze_text_content(
                        text_content=text_content,
                        analysis_prompt=self._create_text_only_analysis_prompt(),
                    )
            else:
                # Text-only analysis
                analysis_result = await self.gemini_helper.analyze_text_content(
                    text_content=text_content,
                    analysis_prompt=self._create_text_only_analysis_prompt(),
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
        Title: {listing_data.title}
        Description: {listing_data.description}
        Price: ${listing_data.price}/month
        Area: {listing_data.area or 'Not specified'} sq meters
        Address: {listing_data.address}
        Property Type: {listing_data.property_type or 'Not specified'}
        Amenities: {', '.join(listing_data.amenities) if listing_data.amenities else 'None'}
        Number of Images: {len(listing_data.images)}
        Number of Videos: {len(listing_data.videos)}
        Additional Info: {metadata_str}
"""
        return text_content

    def _create_text_only_analysis_prompt(self) -> str:
        """Create analysis prompt for text-only content"""
        return """
You are an AI expert in rental property listing verification. Analyze the provided text content for a rental property listing.

Evaluate across these categories:

1. **CONTENT RELEVANCE**:
   - Is this clearly a rental property listing?
   - Does the content match rental property category?
   - Is the information coherent and professional?
   - Are there any inappropriate or suspicious elements?

2. **COMPLETENESS & QUALITY**:
   - Is essential information provided (price, location, description)?
   - Is the description detailed and informative?
   - Are important details missing?
   - Is the overall quality sufficient for a good listing?

Return a JSON response with this structure (keep messages brief, max 80 chars each):
{
    "content_analysis": {
        "is_rental_related": boolean,
        "category_match": boolean,
        "content_score": float (0-1),
        "issues": ["max 2 brief issues"],
        "violations": [{"category": "string", "severity": "low|medium|high|critical", "message": "brief message"}]
    },
    "completeness_analysis": {
        "is_complete": boolean,
        "completeness_score": float (0-1),
        "missing_fields": ["max 3 field names"],
        "quality_issues": ["max 2 brief issues"],
        "suggestions": [{"category": "string", "message": "brief suggestion", "priority": "low|medium|high"}]
    },
    "overall_assessment": {
        "is_valid": boolean,
        "overall_score": float (0-1),
        "confidence": float (0-1),
        "major_concerns": ["max 2 primary issues"],
        "recommendations": ["max 2 brief recommendations"]
    }
}
"""

    def _process_analysis_result(
        self, analysis_result: Dict[str, Any], listing_data: ListingVerificationRequest
    ) -> ListingVerificationResponse:
        """Process the analysis result and create response"""

        # Handle error cases
        if "error" in analysis_result:
            return self._create_fallback_response(
                listing_data, analysis_result["error"]
            )

        # Extract analysis components
        image_analysis = analysis_result.get("image_analysis", {})
        content_analysis = analysis_result.get("content_analysis", {})
        completeness_analysis = analysis_result.get("completeness_analysis", {})
        overall_assessment = analysis_result.get("overall_assessment", {})

        # Create validation objects
        image_validation = ImageValidation(
            is_valid=image_analysis.get("is_valid", len(listing_data.images) == 0),
            total_images=len(listing_data.images),
            valid_images=image_analysis.get("total_images_analyzed", 0),
            issues=image_analysis.get("issues", []),
            quality_score=image_analysis.get(
                "quality_score", 1.0 if len(listing_data.images) == 0 else 0.5
            ),
        )

        # Extract video analysis
        video_analysis = analysis_result.get("video_analysis", {})
        video_validation = VideoValidation(
            is_valid=video_analysis.get("is_valid", len(listing_data.videos) == 0),
            total_videos=len(listing_data.videos),
            valid_videos=video_analysis.get(
                "total_videos_analyzed",
                len(listing_data.videos) if listing_data.videos else 0,
            ),
            issues=video_analysis.get("issues", []),
            quality_score=video_analysis.get(
                "quality_score", 1.0 if len(listing_data.videos) == 0 else 0.7
            ),
        )

        content_validation = ContentValidation(
            is_rental_related=content_analysis.get("is_rental_related", True),
            category_match=content_analysis.get("category_match", True),
            content_score=content_analysis.get("content_score", 0.5),
            issues=content_analysis.get("issues", []),
        )

        completeness_validation = CompletenessValidation(
            is_complete=completeness_analysis.get("is_complete", False),
            completeness_score=completeness_analysis.get("completeness_score", 0.5),
            missing_fields=completeness_analysis.get("missing_fields", []),
            quality_issues=completeness_analysis.get("quality_issues", []),
        )

        # Extract violations and suggestions
        violations = []
        for violation_data in content_analysis.get("violations", []):
            violations.append(
                Violation(
                    category=violation_data.get("category", "unknown"),
                    severity=violation_data.get("severity", "low"),
                    message=violation_data.get("message", "Violation detected"),
                    field=violation_data.get("field"),
                )
            )

        suggestions = []
        for suggestion_data in completeness_analysis.get("suggestions", []):
            suggestions.append(
                Suggestion(
                    category=suggestion_data.get("category", "improvement"),
                    message=suggestion_data.get("message", "Improvement suggested"),
                    field=suggestion_data.get("field"),
                    priority=suggestion_data.get("priority", "medium"),
                )
            )

        # Add general suggestions from overall assessment
        for recommendation in overall_assessment.get("recommendations", []):
            suggestions.append(
                Suggestion(
                    category="general",
                    message=recommendation,
                    priority="medium",
                )
            )

        # Use overall assessment scores if available, otherwise calculate
        overall_score = overall_assessment.get(
            "overall_score",
            (
                image_validation.quality_score * 0.3
                + content_validation.content_score * 0.4
                + completeness_validation.completeness_score * 0.3
            ),
        )

        is_valid = overall_assessment.get(
            "is_valid",
            (
                content_validation.is_rental_related
                and content_validation.category_match
                and completeness_validation.is_complete
                and overall_score >= 0.6
            ),
        )

        confidence = overall_assessment.get("confidence", 0.8)

        return ListingVerificationResponse(
            is_valid=is_valid,
            score=overall_score,
            confidence=confidence,
            image_validation=image_validation,
            video_validation=video_validation,
            content_validation=content_validation,
            completeness_validation=completeness_validation,
            violations=violations,
            suggestions=suggestions,
        )

    def _create_fallback_response(
        self, listing_data: ListingVerificationRequest, error_msg: str
    ) -> ListingVerificationResponse:
        """Create a fallback response when AI analysis fails"""

        logger.warning(f"Creating fallback response due to error: {error_msg}")

        # Clean up error message for user-friendly display
        if "quota exceeded" in error_msg.lower() or "429" in error_msg:
            clean_error_msg = "AI service temporarily unavailable due to quota limits. Using basic validation."
        elif (
            "invalid api key" in error_msg.lower()
            or "unauthorized" in error_msg.lower()
        ):
            clean_error_msg = "AI service configuration issue. Using basic validation."
        else:
            clean_error_msg = "AI analysis unavailable. Using basic validation rules."

        # Basic fallback validation
        missing_fields = []
        if not listing_data.area:
            missing_fields.append("area")
        if len(listing_data.description) < 50:
            missing_fields.append("detailed_description")
        if len(listing_data.images) < 1:  # Changed from 2 to 1
            missing_fields.append("sufficient_images")

        basic_score = max(0.6, 1.0 - len(missing_fields) * 0.15)  # Less penalty

        return ListingVerificationResponse(
            is_valid=len(missing_fields) <= 1
            and basic_score >= 0.6,  # Allow 1 missing field
            score=basic_score,
            confidence=0.5,  # Low confidence for fallback
            image_validation=ImageValidation(
                is_valid=len(listing_data.images) >= 1,  # Changed from 2 to 1
                total_images=len(listing_data.images),
                valid_images=len(listing_data.images),
                issues=[clean_error_msg]
                if len(listing_data.images) < 1
                else [],  # Changed from 2 to 1
                quality_score=0.7
                if len(listing_data.images) >= 1
                else 0.3,  # Changed from 2 to 1
            ),
            video_validation=VideoValidation(
                is_valid=True,  # Fallback assumes videos are valid if present
                total_videos=len(listing_data.videos),
                valid_videos=len(listing_data.videos),
                issues=[clean_error_msg] if len(listing_data.videos) > 0 else [],
                quality_score=0.8 if len(listing_data.videos) > 0 else 1.0,
            ),
            content_validation=ContentValidation(
                is_rental_related=True,
                category_match=True,
                content_score=basic_score,
                issues=[clean_error_msg],
            ),
            completeness_validation=CompletenessValidation(
                is_complete=len(missing_fields) == 0,
                completeness_score=basic_score,
                missing_fields=missing_fields,
                quality_issues=[clean_error_msg] if len(missing_fields) > 0 else [],
            ),
            violations=[],
            suggestions=[
                Suggestion(
                    category="system",
                    message="AI analysis failed - manual review recommended",
                    priority="high",
                )
            ],
        )
