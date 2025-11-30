from unittest.mock import patch

import pytest

from app.dto.listing_verification import (
    ListingVerificationRequest,
    ListingVerificationResponse,
    ListingVideo,
    PropertyMetadata,
)
from app.service.listing_verification_service import ListingVerificationService


class TestListingVerificationService:
    """Test cases for ListingVerificationService"""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing"""
        return ListingVerificationService()

    @pytest.fixture
    def sample_listing_request(self):
        """Create a sample listing request for testing"""
        return ListingVerificationRequest(
            title="Beautiful 2BR Apartment in Downtown",
            description="Modern 2-bedroom apartment with great city views. Fully furnished with high-end appliances. Located in the heart of downtown with easy access to public transportation.",
            price=1500.0,
            area=80.0,
            address="123 Main Street, Downtown City, State 12345",
            amenities=["WiFi", "Air Conditioning", "Parking", "Gym", "Pool"],
            images=[
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
                "https://example.com/image3.jpg",
            ],
            videos=[
                ListingVideo(url="https://example.com/video1.mp4"),
            ],
            metadata=PropertyMetadata(
                bedrooms=2,
                bathrooms=2,
                floor=5,
                total_floors=10,
            ),
            property_type="apartment",
        )

    @pytest.fixture
    def minimal_listing_request(self):
        """Create a minimal listing request for testing"""
        return ListingVerificationRequest(
            title="Room for Rent",
            description="Small room available for rent",
            price=500.0,
            address="456 Side Street, City",
        )

    @pytest.fixture
    def mock_successful_analysis_result(self):
        """Mock successful analysis result from Gemini"""
        return {
            "image_analysis": {
                "is_valid": True,
                "quality_score": 0.85,
                "issues": [],
                "total_images_analyzed": 3,
            },
            "content_analysis": {
                "is_rental_related": True,
                "category_match": True,
                "content_score": 0.9,
                "issues": [],
                "violations": [],
            },
            "completeness_analysis": {
                "is_complete": True,
                "completeness_score": 0.88,
                "missing_fields": [],
                "quality_issues": [],
                "suggestions": [
                    {
                        "category": "improvement",
                        "message": "Consider adding more interior photos",
                        "priority": "low",
                    }
                ],
            },
            "overall_assessment": {
                "is_valid": True,
                "overall_score": 0.87,
                "confidence": 0.92,
                "major_concerns": [],
                "recommendations": ["Great listing overall"],
            },
        }

    @pytest.fixture
    def mock_failed_analysis_result(self):
        """Mock failed analysis result from Gemini"""
        return {
            "image_analysis": {
                "is_valid": False,
                "quality_score": 0.3,
                "issues": ["Images are too dark", "Low resolution"],
                "total_images_analyzed": 2,
            },
            "content_analysis": {
                "is_rental_related": False,
                "category_match": False,
                "content_score": 0.2,
                "issues": ["Content appears to be commercial"],
                "violations": [
                    {
                        "category": "content_type",
                        "severity": "high",
                        "message": "Content does not appear to be residential rental",
                    }
                ],
            },
            "completeness_analysis": {
                "is_complete": False,
                "completeness_score": 0.4,
                "missing_fields": ["detailed_description", "contact_info"],
                "quality_issues": ["Description too short"],
                "suggestions": [
                    {
                        "category": "completeness",
                        "message": "Add more detailed description",
                        "priority": "high",
                    }
                ],
            },
            "overall_assessment": {
                "is_valid": False,
                "overall_score": 0.3,
                "confidence": 0.6,
                "major_concerns": ["Not suitable for rental platform"],
                "recommendations": ["Review content and images"],
            },
        }

    @pytest.mark.asyncio
    async def test_verify_listing_success(
        self, service, sample_listing_request, mock_successful_analysis_result
    ):
        """Test successful listing verification"""
        with patch.object(
            service.gemini_helper,
            "analyze_images_with_text",
            return_value=mock_successful_analysis_result,
        ):
            result = await service.verify_listing(sample_listing_request)

            assert isinstance(result, ListingVerificationResponse)
            assert result.is_valid is True
            assert result.score == 0.87
            assert result.confidence == 0.92
            assert result.processing_time_seconds is not None
            assert result.processing_time_seconds >= 0

            # Check image validation
            assert result.image_validation.is_valid is True
            assert result.image_validation.quality_score == 0.85
            assert result.image_validation.total_images == 3

            # Check content validation
            assert result.content_validation.is_rental_related is True
            assert result.content_validation.category_match is True
            assert result.content_validation.content_score == 0.9

            # Check completeness validation
            assert result.completeness_validation.is_complete is True
            assert result.completeness_validation.completeness_score == 0.88

            # Check suggestions
            assert len(result.suggestions) >= 1
            assert (
                result.suggestions[0].message == "Consider adding more interior photos"
            )

    @pytest.mark.asyncio
    async def test_verify_listing_failure(
        self, service, sample_listing_request, mock_failed_analysis_result
    ):
        """Test listing verification with failures"""
        with patch.object(
            service.gemini_helper,
            "analyze_images_with_text",
            return_value=mock_failed_analysis_result,
        ):
            result = await service.verify_listing(sample_listing_request)

            assert isinstance(result, ListingVerificationResponse)
            assert result.is_valid is False
            assert result.score == 0.3
            assert result.confidence == 0.6

            # Check violations
            assert len(result.violations) > 0
            violation_categories = [v.category for v in result.violations]
            assert "content_type" in violation_categories

            # Check suggestions
            assert len(result.suggestions) > 0
            suggestion_messages = [s.message for s in result.suggestions]
            assert any("detailed description" in msg for msg in suggestion_messages)

    @pytest.mark.asyncio
    async def test_verify_listing_no_images(
        self, service, minimal_listing_request, mock_successful_analysis_result
    ):
        """Test listing verification with no images"""
        # Modify the mock to reflect no images
        mock_result = mock_successful_analysis_result.copy()
        mock_result["content_analysis"]["content_score"] = 0.7
        mock_result["overall_assessment"]["overall_score"] = 0.7

        with patch.object(
            service.gemini_helper,
            "analyze_text_content",
            return_value=mock_result,
        ):
            result = await service.verify_listing(minimal_listing_request)

            assert isinstance(result, ListingVerificationResponse)
            assert result.image_validation.total_images == 0
            # When using mock data, the response comes from the mock rather than fallback logic
            assert (
                result.image_validation.quality_score >= 0.8
            )  # Should be high quality from mock

    @pytest.mark.asyncio
    async def test_verify_listing_gemini_error(self, service, sample_listing_request):
        """Test listing verification when Gemini returns an error"""
        error_result = {"error": "API rate limit exceeded", "analysis_completed": False}

        with patch.object(
            service.gemini_helper,
            "analyze_images_with_text",
            return_value=error_result,
        ):
            result = await service.verify_listing(sample_listing_request)

            # Should return fallback response
            assert isinstance(result, ListingVerificationResponse)
            assert result.confidence == 0.5  # Low confidence for fallback
            assert any(
                "ai analysis failed" in issue
                for issue in result.image_validation.issues
            )
            assert len(result.suggestions) > 0
            assert any(
                "manual review recommended" in s.message.lower()
                for s in result.suggestions
            )

    @pytest.mark.asyncio
    async def test_verify_listing_exception(self, service, sample_listing_request):
        """Test listing verification when an exception occurs"""
        with patch.object(
            service.gemini_helper,
            "analyze_images_with_text",
            side_effect=Exception("Network error"),
        ):
            with pytest.raises(Exception) as exc_info:
                await service.verify_listing(sample_listing_request)

            assert "Network error" in str(exc_info.value)

    def test_prepare_text_content(self, service, sample_listing_request):
        """Test text content preparation"""
        text_content = service._prepare_text_content(sample_listing_request)

        assert "Beautiful 2BR Apartment" in text_content
        assert "$1500.0/month" in text_content
        assert "80.0 sq meters" in text_content
        assert "Downtown City" in text_content
        assert "WiFi, Air Conditioning" in text_content
        assert "Bedrooms: 2" in text_content
        assert "Bathrooms: 2" in text_content
        assert "Furnished: True" in text_content

    def test_prepare_text_content_minimal(self, service, minimal_listing_request):
        """Test text content preparation with minimal data"""
        text_content = service._prepare_text_content(minimal_listing_request)

        assert "Room for Rent" in text_content
        assert "$500.0/month" in text_content
        assert "Not specified" in text_content  # For missing area
        assert "None" in text_content  # For missing amenities

    def test_create_text_only_analysis_prompt(self, service):
        """Test text-only analysis prompt creation"""
        prompt = service._create_text_only_analysis_prompt()

        assert "rental property listing verification" in prompt.lower()
        assert "content_analysis" in prompt
        assert "completeness_analysis" in prompt
        assert "overall_assessment" in prompt
        assert "json" in prompt

    def test_process_analysis_result_success(
        self, service, sample_listing_request, mock_successful_analysis_result
    ):
        """Test processing of successful analysis result"""
        result = service._process_analysis_result(
            mock_successful_analysis_result, sample_listing_request
        )

        assert isinstance(result, ListingVerificationResponse)
        assert result.is_valid is True
        assert result.score == 0.87
        assert result.confidence == 0.92
        assert len(result.suggestions) >= 1

    def test_process_analysis_result_with_error(self, service, sample_listing_request):
        """Test processing of analysis result with error"""
        error_result = {"error": "Processing failed", "raw_response": "Error occurred"}

        result = service._process_analysis_result(error_result, sample_listing_request)

        # Should return fallback response
        assert isinstance(result, ListingVerificationResponse)
        assert result.confidence == 0.5
        assert any(
            "ai analysis failed" in issue for issue in result.image_validation.issues
        )

    def test_create_fallback_response(self, service, sample_listing_request):
        """Test fallback response creation"""
        result = service._create_fallback_response(
            sample_listing_request, "Test error message"
        )

        assert isinstance(result, ListingVerificationResponse)
        assert result.confidence == 0.5
        assert "Test error message" in result.image_validation.issues[0]
        assert any(
            "manual review recommended" in s.message.lower() for s in result.suggestions
        )

    def test_create_fallback_response_missing_fields(
        self, service, minimal_listing_request
    ):
        """Test fallback response with missing fields"""
        result = service._create_fallback_response(
            minimal_listing_request, "Test error"
        )

        assert isinstance(result, ListingVerificationResponse)
        assert "area" in result.completeness_validation.missing_fields
        assert "detailed_description" in result.completeness_validation.missing_fields
        assert "sufficient_images" in result.completeness_validation.missing_fields
        assert result.score < 0.6  # Should be invalid due to missing fields

    @pytest.mark.asyncio
    async def test_verify_listing_with_metadata_none(self, service):
        """Test listing verification with None metadata"""
        request = ListingVerificationRequest(
            title="Test Listing",
            description="Test description for rental property with all necessary details",
            price=1000.0,
            address="Test Address",
            metadata=None,  # Explicitly None
        )

        with patch.object(
            service.gemini_helper,
            "analyze_text_content",
            return_value={
                "content_analysis": {
                    "is_rental_related": True,
                    "category_match": True,
                    "content_score": 0.8,
                    "issues": [],
                    "violations": [],
                },
                "completeness_analysis": {
                    "is_complete": True,
                    "completeness_score": 0.8,
                    "missing_fields": [],
                    "quality_issues": [],
                    "suggestions": [],
                },
                "overall_assessment": {
                    "is_valid": True,
                    "overall_score": 0.8,
                    "confidence": 0.8,
                    "major_concerns": [],
                    "recommendations": [],
                },
            },
        ):
            result = await service.verify_listing(request)

            assert isinstance(result, ListingVerificationResponse)
            text_content = service._prepare_text_content(request)
            assert "Additional Info: None" in text_content

    def test_metadata_formatting(self, service):
        """Test metadata formatting in text content"""
        request = ListingVerificationRequest(
            title="Test",
            description="Test description for rental property with all necessary details",
            price=1000.0,
            address="Test Address",
            metadata=PropertyMetadata(
                bedrooms=2,
                bathrooms=None,  # Test with None value
                furnished=False,
                pet_friendly=True,
                parking_available=None,  # Test with None value
            ),
        )

        text_content = service._prepare_text_content(request)
        assert "Bedrooms: 2" in text_content
        assert "Furnished: False" in text_content
        assert "Pet friendly: True" in text_content
        # None values should not appear
        assert "Bathrooms:" not in text_content
        assert "Parking:" not in text_content
