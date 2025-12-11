# Listing Verification Module

## Overview

The Listing Verification module provides AI-powered verification of rental property listings using Google's Gemini multimodal AI. This module analyzes text content and images to ensure listings meet quality standards and are appropriate for the rental platform.

## Features

### 🔍 Comprehensive Analysis
- **Image Quality Assessment**: Analyzes property images for clarity, appropriateness, and relevance
- **Video Quality Assessment**: Analyzes property videos for quality and relevance
- **Content Verification**: Ensures listings are rental-related and categorized correctly
- **Completeness Check**: Validates that all necessary information is provided
- **Multimodal Analysis**: Combines text, image, and video analysis for comprehensive evaluation

### 📊 Detailed Scoring
- Overall quality score (0-1)
- Individual component scores for images, content, and completeness
- Confidence rating for the assessment
- Specific violation detection and improvement suggestions

### 🛡️ Safety & Quality Controls
- Inappropriate content detection
- Category mismatch identification
- Missing information highlighting
- Quality improvement recommendations

## API Endpoints

### POST `/ai/verify-listing`
Verify a rental property listing.

**Request Body:**
```json
{
    "title": "Beautiful 2BR Apartment in Downtown",
    "description": "Modern apartment with city views...",
    "price": 1500.0,
    "area": 80.0,
    "address": "123 Main Street, City",
    "amenities": ["WiFi", "AC", "Parking"],
    "images": [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg"
    ],
    "videos": [
        {
            "url": "https://example.com/video1.mp4"
        }
    ],
    "metadata": {
        "bedrooms": 2,
        "bathrooms": 2,
        "floor": 5,
        "total_floors": 10
    },
    "property_type": "APARTMENT"
}
```

**Response:**
```json
{
    "is_valid": true,
    "score": 0.87,
    "confidence": 0.92,
    "image_validation": {
        "is_valid": true,
        "total_images": 3,
        "valid_images": 3,
        "quality_score": 0.85,
        "issues": []
    },
    "video_validation": {
        "is_valid": true,
        "total_videos": 1,
        "valid_videos": 1,
        "quality_score": 0.9,
        "issues": []
    },
    "content_validation": {
        "is_rental_related": true,
        "category_match": true,
        "content_score": 0.9,
        "issues": []
    },
    "completeness_validation": {
        "is_complete": true,
        "completeness_score": 0.88,
        "missing_fields": [],
        "quality_issues": []
    },
    "violations": [],
    "suggestions": [
        {
            "category": "improvement",
            "message": "Consider adding more interior photos",
            "priority": "low"
        }
    ],
    "verification_timestamp": "2025-11-30T10:30:00Z",
    "model_used": "gemini-2.5-flash",
    "processing_time_seconds": 2.34
}
```

### GET `/ai/health`
Check service health and capabilities.

## Data Models

### ListingVerificationRequest
- `title`: Property title (required, 1-200 chars)
- `description`: Detailed description (required, 10-5000 chars)
- `price`: Monthly rent (required, > 0)
- `area`: Property area in square meters (optional)
- `address`: Property address (required, 5-500 chars)
- `amenities`: List of amenities (optional)
- `images`: Array of image URLs (max 20)
- `videos`: Array of video objects with URL (max 5)
- `metadata`: Additional property information (bedrooms, bathrooms, floor, total_floors)
- `property_type`: Type of property (APARTMENT, HOUSE, ROOM, STUDIO)

### ListingVerificationResponse
- `is_valid`: Overall validity boolean
- `score`: Quality score (0-1)
- `confidence`: Assessment confidence (0-1)
- `image_validation`: Image analysis results
- `video_validation`: Video analysis results
- `content_validation`: Content analysis results
- `completeness_validation`: Completeness analysis results
- `violations`: Array of detected violations
- `suggestions`: Array of improvement suggestions
- `verification_timestamp`: When verification was performed
- `model_used`: AI model used for analysis
- `processing_time_seconds`: Time taken for analysis

## Usage Examples

### Python Client Example
```python
import httpx
import asyncio

async def verify_listing():
    listing_data = {
        "title": "Cozy Studio Apartment",
        "description": "Beautiful studio in downtown area with modern amenities...",
        "price": 800.0,
        "address": "456 Oak Street, Downtown",
        "images": [
            "https://example.com/photo1.jpg",
            "https://example.com/photo2.jpg"
        ],
        "videos": [
            {"url": "https://example.com/tour.mp4"}
        ],
        "metadata": {
            "bedrooms": 0,
            "bathrooms": 1
        },
        "property_type": "STUDIO"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/ai/verify-listing",
            json=listing_data,
            timeout=60.0
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Valid: {result['is_valid']}")
            print(f"Score: {result['score']:.2f}")
            return result
        else:
            print(f"Error: {response.status_code}")

# Run the verification
asyncio.run(verify_listing())
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/ai/verify-listing" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Modern 1BR Apartment",
    "description": "Spacious one-bedroom apartment with modern amenities in prime location...",
    "price": 1200.0,
    "area": 65.0,
    "address": "789 Pine Street, City Center",
    "amenities": ["WiFi", "AC", "Gym"],
    "images": [
      "https://example.com/apt1.jpg",
      "https://example.com/apt2.jpg"
    ],
    "videos": [
      {"url": "https://example.com/apartment-tour.mp4"}
    ],
    "metadata": {
      "bedrooms": 1,
      "bathrooms": 1
    },
    "property_type": "APARTMENT"
  }'
```

## Testing

Run the included test script to verify the service:

```bash
# Install test dependencies
pip install httpx

# Run the test script
python examples/test_listing_verification.py
```

### Unit Tests
```bash
# Run all tests
pytest tests/test_listing_verification.py -v

# Run specific test
pytest tests/test_listing_verification.py::TestListingVerificationService::test_verify_listing_success -v

# Run with coverage
pytest tests/test_listing_verification.py --cov=app.service.listing_verification_service
```

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Model Configuration
The service uses:
- **Gemini 2.0 Flash Experimental** for multimodal analysis (images + text)
- **Gemini 2.5 Pro** for text-only analysis
- Low temperature (0.1) for consistent results
- JSON response format for structured output

## Architecture

```
├── app/
│   ├── dto/
│   │   └── listing_verification.py    # Pydantic models
│   ├── service/
│   │   └── listing_verification_service.py  # Core business logic
│   ├── ai/llm/
│   │   └── gemini_listing_helper.py   # Gemini AI client
│   └── api/v1/
│       └── listing_verification.py    # FastAPI router
├── tests/
│   └── test_listing_verification.py   # Unit tests
└── examples/
    └── test_listing_verification.py   # Example usage
```

## Error Handling

The service includes comprehensive error handling:

- **Validation Errors**: Invalid request data returns 400 with details
- **AI Service Errors**: Falls back to basic validation rules
- **Network Errors**: Gracefully handles image download failures
- **Rate Limiting**: Respects Gemini API rate limits
- **Timeout Handling**: Prevents hanging requests

## Performance Considerations

- **Image Optimization**: Automatically resizes large images
- **Batch Processing**: Analyzes multiple images efficiently
- **Caching**: Results can be cached by clients
- **Timeouts**: 60-second default timeout for complex analysis
- **Rate Limiting**: Respects API rate limits

## Security

- **Input Validation**: All inputs are validated using Pydantic
- **URL Validation**: Only HTTPS URLs accepted for images/videos
- **Content Filtering**: AI detects inappropriate content
- **Error Sanitization**: Sensitive error details are not exposed

## Monitoring & Logging

The service provides detailed logging:
- Request processing times
- AI analysis results
- Error conditions
- Performance metrics

## Future Enhancements

- [ ] Support for additional image formats
- [ ] Batch listing verification
- [ ] Custom verification rules per client
- [ ] Integration with image storage services
- [ ] Advanced analytics and reporting
- [ ] Webhook notifications for verification results

## Support

For issues or questions:
1. Check the logs for detailed error information
2. Verify your Gemini API key is configured correctly
3. Ensure all required dependencies are installed
4. Review the test examples for proper usage patterns
