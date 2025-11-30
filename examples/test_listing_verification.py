"""
Example script to test the Listing Verification API

This script demonstrates how to use the listing verification endpoint
with sample data.
"""

import asyncio
from typing import Any, Dict

import httpx

# Sample listing data for testing
SAMPLE_LISTING_DATA = {
    "title": "Beautiful 2BR Apartment in Downtown Hanoi",
    "description": """
    Modern 2-bedroom apartment with stunning city views located in the heart of downtown Hanoi.
    This fully furnished unit features:

    - Spacious living room with floor-to-ceiling windows
    - Modern kitchen with high-end appliances
    - Two comfortable bedrooms with built-in wardrobes
    - Two full bathrooms with modern fixtures
    - Private balcony overlooking the city
    - Central air conditioning and heating
    - High-speed WiFi included

    The building offers excellent amenities including:
    - 24/7 security and concierge
    - Fitness center and swimming pool
    - Rooftop garden and BBQ area
    - Covered parking space

    Located just 5 minutes walk from Hoan Kiem Lake and close to restaurants,
    shopping, and public transportation. Perfect for professionals or couples
    looking for a premium living experience in the city center.

    Available for immediate move-in. One-year lease preferred.
    """,
    "price": 1200.0,
    "area": 85.0,
    "address": "123 Hang Bong Street, Hoan Kiem District, Hanoi, Vietnam",
    "amenities": [
        "WiFi",
        "Air Conditioning",
        "Heating",
        "Furnished",
        "Parking",
        "Security",
        "Elevator",
        "Gym",
        "Pool",
        "Balcony",
    ],
    "images": [
        {
            "url": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=800",
            "caption": "Living room with city view",
            "is_primary": True,
        },
        {
            "url": "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?w=800",
            "caption": "Modern kitchen",
        },
        {
            "url": "https://images.unsplash.com/photo-1571624436279-b272aff752b5?w=800",
            "caption": "Master bedroom",
        },
        {
            "url": "https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?w=800",
            "caption": "Bathroom",
        },
    ],
    "videos": [
        {
            "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "thumbnail_url": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=400",
            "duration_seconds": 30,
            "caption": "Apartment walkthrough",
        }
    ],
    "metadata": {
        "bedrooms": 2,
        "bathrooms": 2,
        "floor": 12,
        "total_floors": 25,
        "furnished": True,
        "pet_friendly": False,
        "parking_available": True,
    },
    "property_type": "apartment",
}

# Example with issues for testing
PROBLEMATIC_LISTING_DATA = {
    "title": "Room",
    "description": "Room for rent. Contact me.",
    "price": 300.0,
    "address": "Somewhere in the city",
    "amenities": [],
    "images": [],
    "videos": [],
    "property_type": "room",
}


async def test_listing_verification(
    listing_data: Dict[str, Any], base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Test the listing verification endpoint

    Args:
        listing_data: The listing data to verify
        base_url: The base URL of the API

    Returns:
        The verification response
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/ai/verify-listing",
                json=listing_data,
                headers={"Content-Type": "application/json"},
                timeout=60.0,  # Allow time for AI processing
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            print(f"Request failed: {str(e)}")
            return {"error": str(e)}


async def test_health_check(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test the health check endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/ai/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def print_verification_result(
    result: Dict[str, Any], title: str = "Verification Result"
):
    """Pretty print verification result"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"✅ Valid: {result.get('is_valid', False)}")
    print(f"📊 Score: {result.get('score', 0):.2f}/1.00")
    print(f"🎯 Confidence: {result.get('confidence', 0):.2f}/1.00")
    print(f"⏱️  Processing Time: {result.get('processing_time_seconds', 0):.2f}s")

    # Image validation
    img_validation = result.get("image_validation", {})
    print(
        f"\n📸 Images: {img_validation.get('valid_images', 0)}/{img_validation.get('total_images', 0)} valid"
    )
    print(f"   Quality Score: {img_validation.get('quality_score', 0):.2f}")
    if img_validation.get("issues"):
        print(f"   Issues: {', '.join(img_validation['issues'])}")

    # Content validation
    content_validation = result.get("content_validation", {})
    print(f"\n📝 Content Score: {content_validation.get('content_score', 0):.2f}")
    print(f"   Rental Related: {content_validation.get('is_rental_related', False)}")
    print(f"   Category Match: {content_validation.get('category_match', False)}")
    if content_validation.get("issues"):
        print(f"   Issues: {', '.join(content_validation['issues'])}")

    # Completeness validation
    completeness = result.get("completeness_validation", {})
    print(f"\n✅ Completeness Score: {completeness.get('completeness_score', 0):.2f}")
    print(f"   Complete: {completeness.get('is_complete', False)}")
    if completeness.get("missing_fields"):
        print(f"   Missing: {', '.join(completeness['missing_fields'])}")

    # Violations
    violations = result.get("violations", [])
    if violations:
        print(f"\n⚠️  Violations ({len(violations)}):")
        for v in violations[:3]:  # Show first 3
            print(
                f"   • {v.get('category', 'unknown')} ({v.get('severity', 'unknown')}): {v.get('message', '')}"
            )

    # Suggestions
    suggestions = result.get("suggestions", [])
    if suggestions:
        print(f"\n💡 Suggestions ({len(suggestions)}):")
        for s in suggestions[:3]:  # Show first 3
            print(
                f"   • {s.get('message', '')} ({s.get('priority', 'medium')} priority)"
            )


async def main() -> None:
    """Main function to run tests"""
    print("🏠 Listing Verification API Test")
    print("================================\n")

    # Test health check
    print("Testing health check...")
    health_result = await test_health_check()
    if "error" in health_result:
        print(f"❌ Health check failed: {health_result['error']}")
        print("Make sure the server is running on localhost:8000")
        return
    else:
        print(f"✅ Service is healthy: {health_result.get('status', 'unknown')}")

    # Test good listing
    print("\n🧪 Testing high-quality listing...")
    good_result = await test_listing_verification(SAMPLE_LISTING_DATA)
    print_verification_result(good_result, "High-Quality Listing Result")

    # Test problematic listing
    print("\n🧪 Testing problematic listing...")
    bad_result = await test_listing_verification(PROBLEMATIC_LISTING_DATA)
    print_verification_result(bad_result, "Problematic Listing Result")

    print(f"\n{'='*50}")
    print("🎉 Testing completed!")
    print("💡 Check the results above to see how the AI evaluated each listing.")


if __name__ == "__main__":
    # Install required packages first:
    # pip install httpx

    asyncio.run(main())
