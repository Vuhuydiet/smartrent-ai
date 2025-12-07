"""Test script for MCP house pricing integration."""
import asyncio
import json

from app.mcp.house_pricing_client import HousePricingClient


async def test_house_pricing() -> None:
    """Test the house pricing client."""
    client = HousePricingClient(base_url="http://localhost:8000")

    # Test case 1: Hanoi Apartment
    print("Test 1: Hanoi Apartment")
    print("-" * 50)
    try:
        result = await client.get_price_range(
            city="Hanoi",
            district="Ba Dinh",
            ward="Dien Bien",
            property_type="APARTMENT",
            latitude=21.0285,
            longitude=105.8342,
            area=75.5,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n")

    # Test case 2: Ho Chi Minh House
    print("Test 2: Ho Chi Minh House")
    print("-" * 50)
    try:
        result = await client.get_price_range(
            city="Ho Chi Minh",
            district="District 1",
            ward="Ben Nghe Ward",
            property_type="HOUSE",
            latitude=10.7769,
            longitude=106.7009,
            area=120,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

    print("\n")

    # Test case 3: Room without area
    print("Test 3: Room without area")
    print("-" * 50)
    try:
        result = await client.get_price_range(
            city="Hanoi",
            district="Cau Giay",
            ward="Dich Vong",
            property_type="ROOM",
            latitude=21.0313,
            longitude=105.7937,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing House Pricing Client")
    print("=" * 50)
    print("Make sure the AI service is running on port 8000:")
    print("  uvicorn app.main:app --reload --port 8000")
    print("=" * 50)
    print("\n")

    asyncio.run(test_house_pricing())
