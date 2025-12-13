"""
Example script to test the Price Prediction API

This script demonstrates how to use the price prediction endpoint to get
AI-powered property price estimates.
"""

import asyncio
from typing import Any, Dict

import httpx


async def predict_price(
    property_data: Dict[str, Any], base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Get price prediction for a property

    Args:
        property_data: Dictionary containing property information
        base_url: The base URL of the API

    Returns:
        The price prediction response
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/api/v1/predict-price",
                json=property_data,
                headers={"Content-Type": "application/json"},
                timeout=60.0,  # Allow time for AI processing
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}", "detail": response.text}

        except Exception as e:
            print(f"Request failed: {str(e)}")
            return {"error": str(e)}


async def test_health_check(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test the price prediction health check endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/api/v1/predict-price/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def print_prediction_result(result: Dict[str, Any], title: str = "Price Prediction"):
    """Pretty print prediction result"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "detail" in result:
            print(f"   Detail: {result['detail']}")
        return

    price_range = result.get("price_range", {})
    min_price = price_range.get("min", 0)
    max_price = price_range.get("max", 0)

    print(f"📍 Location: {result.get('location', 'N/A')}")
    print(f"🏠 Property Type: {result.get('property_type', 'N/A')}")
    print(f"💰 Currency: {result.get('currency', 'VND')}")

    print(f"\n💵 Price Range:")
    print(f"   Min: {min_price:,} VND ({min_price/1_000_000:,.1f} million VND)")
    print(f"   Max: {max_price:,} VND ({max_price/1_000_000:,.1f} million VND)")
    print(f"   Average: {(min_price + max_price)/2:,.0f} VND ({(min_price + max_price)/2/1_000_000:,.1f} million VND)")


async def test_hanoi_apartment() -> None:
    """Test price prediction for Hanoi apartment"""
    print("\n🧪 Test 1: Apartment in Hoan Kiem, Hanoi")
    print("-" * 60)

    property_data = {
        "city": "Hanoi",
        "district": "Hoan Kiem",
        "ward": "Hang Bong",
        "property_type": "Apartment",
        "area": 85.0,
        "latitude": 21.0285,
        "longitude": 105.8542,
    }

    print(f"📋 Property Details:")
    print(f"   - Location: {property_data['ward']}, {property_data['district']}, {property_data['city']}")
    print(f"   - Type: {property_data['property_type']}")
    print(f"   - Area: {property_data['area']} m²")
    print(f"   - Coordinates: ({property_data['latitude']}, {property_data['longitude']})")

    result = await predict_price(property_data)
    print_prediction_result(result, "Hanoi Apartment - Price Prediction")


async def test_hcm_house() -> None:
    """Test price prediction for Ho Chi Minh house"""
    print("\n🧪 Test 2: House in District 1, Ho Chi Minh City")
    print("-" * 60)

    property_data = {
        "city": "Ho Chi Minh",
        "district": "District 1",
        "ward": "Ben Nghe Ward",
        "property_type": "House",
        "area": 120.0,
        "latitude": 10.7756,
        "longitude": 106.7019,
    }

    print(f"📋 Property Details:")
    print(f"   - Location: {property_data['ward']}, {property_data['district']}, {property_data['city']}")
    print(f"   - Type: {property_data['property_type']}")
    print(f"   - Area: {property_data['area']} m²")
    print(f"   - Coordinates: ({property_data['latitude']}, {property_data['longitude']})")

    result = await predict_price(property_data)
    print_prediction_result(result, "HCMC House - Price Prediction")


async def test_danang_villa() -> None:
    """Test price prediction for Da Nang villa"""
    print("\n🧪 Test 3: Villa in Hai Chau, Da Nang")
    print("-" * 60)

    property_data = {
        "city": "Da Nang",
        "district": "Hai Chau",
        "ward": "Thach Thang",
        "property_type": "Villa",
        "area": 250.0,
        "latitude": 16.0544,
        "longitude": 108.2022,
    }

    print(f"📋 Property Details:")
    print(f"   - Location: {property_data['ward']}, {property_data['district']}, {property_data['city']}")
    print(f"   - Type: {property_data['property_type']}")
    print(f"   - Area: {property_data['area']} m²")
    print(f"   - Coordinates: ({property_data['latitude']}, {property_data['longitude']})")

    result = await predict_price(property_data)
    print_prediction_result(result, "Da Nang Villa - Price Prediction")


async def test_without_area() -> None:
    """Test price prediction without area specified"""
    print("\n🧪 Test 4: Apartment without area (Hanoi)")
    print("-" * 60)

    property_data = {
        "city": "Hanoi",
        "district": "Ba Dinh",
        "ward": "Dien Bien",
        "property_type": "Apartment",
        "latitude": 21.0369,
        "longitude": 105.8195,
    }

    print(f"📋 Property Details:")
    print(f"   - Location: {property_data['ward']}, {property_data['district']}, {property_data['city']}")
    print(f"   - Type: {property_data['property_type']}")
    print(f"   - Area: Not specified")
    print(f"   - Coordinates: ({property_data['latitude']}, {property_data['longitude']})")

    result = await predict_price(property_data)
    print_prediction_result(result, "Apartment (No Area) - Price Prediction")


async def test_invalid_request() -> None:
    """Test error handling with missing required fields"""
    print("\n🧪 Test 5: Invalid Request (Missing Coordinates)")
    print("-" * 60)

    property_data = {
        "city": "Hanoi",
        "district": "Hoan Kiem",
        "ward": "Hang Bong",
        "property_type": "Apartment",
    }

    print("📋 Property Details: Missing latitude and longitude")

    result = await predict_price(property_data)
    print_prediction_result(result, "Invalid Request - Error Expected")


async def main() -> None:
    """Main function to run tests"""
    print("💰 Price Prediction API Test")
    print("=" * 60)

    # Test health check
    print("\n🏥 Testing health check...")
    health_result = await test_health_check()
    if "error" in health_result:
        print(f"❌ Health check failed: {health_result['error']}")
        print("Make sure the server is running on localhost:8000")
        print("Also ensure GEMINI_API_KEY is set in your .env file")
        return
    else:
        print(f"✅ Service is healthy: {health_result.get('status', 'unknown')}")

    # Run test scenarios
    await test_hanoi_apartment()
    await test_hcm_house()
    await test_danang_villa()
    await test_without_area()
    await test_invalid_request()

    print(f"\n{'='*60}")
    print("🎉 Testing completed!")
    print("💡 The AI uses property characteristics and location data")
    print("   to estimate realistic price ranges in the Vietnamese market.")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Install required packages first:
    # uv add httpx (if using uv)
    # or: pip install httpx

    asyncio.run(main())
