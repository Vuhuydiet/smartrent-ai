# 📋 Listing Verification API Usage Guide

## 🔗 Endpoint
```
POST http://localhost:8000/ai/verify-listing
Content-Type: application/json
```

## 📤 Request Format

### Basic Structure
```json
{
  "title": "string (1-200 chars, required)",
  "description": "string (10-5000 chars, required)",
  "price": "number (>0, required)",
  "address": "string (5-500 chars, required)",
  "property_type": "enum (optional)",
  "area": "number (>0, optional)",
  "amenities": ["array of strings (optional)"],
  "images": ["array of image objects (max 20, optional)"],
  "videos": ["array of video objects (max 5, optional)"],
  "metadata": "object (optional)"
}
```

### Property Type Enum
```typescript
type HousingPropertyType = 'APARTMENT' | 'HOUSE' | 'ROOM' | 'STUDIO'
```

### Image Object Structure
```json
{
  "url": "https://example.com/image.jpg",
  "caption": "Image description (optional)",
  "is_primary": true/false
}
```

### Video Object Structure
```json
{
  "url": "https://example.com/video.mp4",
  "thumbnail_url": "https://example.com/thumb.jpg (optional)",
  "duration_seconds": 60,
  "caption": "Video description (optional)"
}
```

### Metadata Object Structure
```json
{
  "bedrooms": 2,
  "bathrooms": 1,
  "floor": 5,
  "total_floors": 10,
  "furnished": true,
  "pet_friendly": false,
  "parking_available": true
}
```

## 📥 Response Format

```json
{
  "is_valid": true,
  "score": 0.85,
  "confidence": 0.9,
  "image_validation": {
    "is_valid": true,
    "total_images": 3,
    "valid_images": 3,
    "issues": [],
    "quality_score": 0.8
  },
  "content_validation": {
    "is_rental_related": true,
    "category_match": true,
    "content_score": 0.9,
    "issues": []
  },
  "completeness_validation": {
    "is_complete": true,
    "completeness_score": 0.8,
    "missing_fields": [],
    "quality_issues": []
  },
  "violations": [],
  "suggestions": [
    {
      "category": "images",
      "message": "Add more photos of bathroom",
      "field": "images",
      "priority": "medium"
    }
  ],
  "verification_timestamp": "2025-11-23T15:30:00Z",
  "model_used": "gemini-2.5-pro",
  "processing_time_seconds": 6.2
}
```

## 🚀 Code Examples

### JavaScript/TypeScript
```typescript
// Type definitions
type HousingPropertyType = 'APARTMENT' | 'HOUSE' | 'ROOM' | 'STUDIO';

interface ListingRequest {
  title: string;
  description: string;
  price: number;
  address: string;
  property_type?: HousingPropertyType;
  area?: number;
  amenities?: string[];
  images?: Array<{
    url: string;
    caption?: string;
    is_primary: boolean;
  }>;
  videos?: Array<{
    url: string;
    thumbnail_url?: string;
    duration_seconds?: number;
    caption?: string;
  }>;
  metadata?: {
    bedrooms?: number;
    bathrooms?: number;
    floor?: number;
    total_floors?: number;
    furnished?: boolean;
    pet_friendly?: boolean;
    parking_available?: boolean;
  };
}

// API call function
async function verifyListing(listing: ListingRequest) {
  const response = await fetch('http://localhost:8000/ai/verify-listing', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(listing)
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }

  return await response.json();
}

// Example usage
const result = await verifyListing({
  title: "Cho thuê căn hộ 2PN Vinhomes Central Park",
  description: "Căn hộ 2 phòng ngủ, đầy đủ nội thất, view sông đẹp. Vị trí thuận tiện, gần trung tâm thương mại và trường học.",
  price: 20000000,
  address: "208 Nguyễn Hữu Cảnh, Bình Thạnh, TP.HCM",
  property_type: "APARTMENT",
  area: 75,
  images: [
    {
      url: "https://example.com/living-room.jpg",
      caption: "Phòng khách",
      is_primary: true
    },
    {
      url: "https://example.com/bedroom.jpg",
      caption: "Phòng ngủ",
      is_primary: false
    }
  ],
  metadata: {
    bedrooms: 2,
    bathrooms: 2,
    furnished: true,
    parking_available: true
  }
});

console.log('Verification result:', result);
```

### Python
```python
import requests
from typing import Optional, List, Dict, Any
from enum import Enum

class HousingPropertyType(str, Enum):
    APARTMENT = "APARTMENT"
    HOUSE = "HOUSE"
    ROOM = "ROOM"
    STUDIO = "STUDIO"

def verify_listing(
    title: str,
    description: str,
    price: float,
    address: str,
    property_type: Optional[HousingPropertyType] = None,
    area: Optional[float] = None,
    images: Optional[List[Dict]] = None,
    videos: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    amenities: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Call listing verification API"""

    payload = {
        "title": title,
        "description": description,
        "price": price,
        "address": address
    }

    # Add optional fields
    if property_type:
        payload["property_type"] = property_type.value
    if area:
        payload["area"] = area
    if images:
        payload["images"] = images
    if videos:
        payload["videos"] = videos
    if metadata:
        payload["metadata"] = metadata
    if amenities:
        payload["amenities"] = amenities

    response = requests.post(
        "http://localhost:8000/ai/verify-listing",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    response.raise_for_status()
    return response.json()

# Example usage
result = verify_listing(
    title="Cho thuê nhà nguyên căn 3 tầng Q1",
    description="Nhà nguyên căn 3 tầng, 4 phòng ngủ, đầy đủ nội thất. Vị trí trung tâm, gần chợ Bến Thành.",
    price=50000000,
    address="Lê Lợi, Quận 1, TP.HCM",
    property_type=HousingPropertyType.HOUSE,
    area=120,
    images=[
        {
            "url": "https://example.com/house-front.jpg",
            "caption": "Mặt tiền nhà",
            "is_primary": True
        }
    ],
    metadata={
        "bedrooms": 4,
        "bathrooms": 3,
        "furnished": True,
        "parking_available": True
    }
)

print("Verification result:", result)
```

### cURL
```bash
curl -X POST http://localhost:8000/ai/verify-listing \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Cho thuê studio Landmark 81 view sông",
    "description": "Studio cao cấp, đầy đủ nội thất, ban công view sông Sài Gòn tuyệt đẹp. Tòa nhà có đầy đủ tiện ích hiện đại.",
    "price": 15000000,
    "address": "Landmark 81, Vinhomes Central Park, Bình Thạnh, TP.HCM",
    "property_type": "STUDIO",
    "area": 35,
    "images": [
      {
        "url": "https://example.com/studio.jpg",
        "caption": "Không gian studio",
        "is_primary": true
      }
    ],
    "metadata": {
      "bedrooms": 1,
      "bathrooms": 1,
      "furnished": true
    }
  }'
```

## ✅ Validation Rules

### Required Fields
- `title`: 1-200 characters
- `description`: 10-5000 characters
- `price`: Must be > 0
- `address`: 5-500 characters

### Optional Fields
- `property_type`: Must be one of enum values
- `area`: Must be > 0 if provided
- `images`: Max 20 images
- `videos`: Max 5 videos

## 🚨 **COMMON ERROR FIXES**

### 422 Unprocessable Entity - Field Name Issues

❌ **WRONG field names that cause 422 errors:**
```json
{
  "metadata": {
    "petAllowed": false,      // ❌ Wrong - use "pet_friendly"
    "parkingAvailable": true  // ❌ Wrong - use "parking_available"
  }
}
```

✅ **CORRECT field names:**
```json
{
  "metadata": {
    "pet_friendly": false,        // ✅ Correct
    "parking_available": true     // ✅ Correct
  }
}
```

### Complete Valid Example for Testing:
```json
{
  "title": "Cho thuê căn hộ 2PN view đẹp",
  "description": "Căn hộ 2 phòng ngủ, đầy đủ nội thất cao cấp, view sông Sài Gòn tuyệt đẹp. Tòa nhà hiện đại với đầy đủ tiện ích.",
  "price": 20000000,
  "address": "208 Nguyễn Hữu Cảnh, Bình Thạnh, TP.HCM",
  "property_type": "APARTMENT",
  "area": 75,
  "amenities": ["WIFI", "AIR_CONDITIONING"],
  "images": [
    {
      "url": "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2",
      "caption": "Phòng khách",
      "is_primary": true
    }
  ],
  "metadata": {
    "bedrooms": 2,
    "bathrooms": 1,
    "furnished": true,
    "pet_friendly": false,
    "parking_available": true
  }
}
```

## ❌ Error Responses

### 422 Validation Error
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "description"],
      "msg": "String should have at least 10 characters",
      "input": "Short"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error occurred during listing verification"
}
```

## 📊 Response Score Interpretation

- **Score 0.8-1.0**: Excellent listing quality ✅
- **Score 0.6-0.79**: Good, minor improvements needed ✅
- **Score 0.4-0.59**: Average, significant improvements needed ⚠️
- **Score 0.0-0.39**: Poor quality, major issues ❌

## 🔧 Integration Notes

1. **API Base URL**: Replace `localhost:8000` with your actual server URL
2. **Authentication**: Currently no auth required (add if needed)
3. **Rate Limiting**: Consider implementing rate limits for production
4. **Timeouts**: Set appropriate timeout (6-15 seconds recommended)
5. **Error Handling**: Always handle validation and server errors
6. **Image URLs**: Ensure images are publicly accessible
