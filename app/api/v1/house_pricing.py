"""House pricing prediction API endpoint - queries backend database."""
from typing import Any, cast

import httpx

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse

router = APIRouter()


async def query_similar_listings(
    city: str,
    district: str,
    property_type: str,
    latitude: float,
    longitude: float,
) -> list[dict[str, Any]]:
    """Query smartrent-backend for similar listings to calculate price range.

    Args:
        city: City name
        district: District name
        property_type: Property type (APARTMENT, HOUSE, ROOM, STUDIO)
        latitude: Property latitude
        longitude: Property longitude

    Returns:
        List of similar listing data with prices
    """
    backend_url = settings.SMARTRENT_BACKEND_URL.rstrip("/")

    # Search for similar listings in the same area
    # Using the comprehensive search API from ListingController
    search_payload = {
        "listingType": "RENT",  # Rental listings
        "productType": property_type.upper(),  # APARTMENT, HOUSE, ROOM, STUDIO, OFFICE
        "verified": True,  # Only verified listings
        "excludeExpired": True,  # Exclude expired listings
        "page": 0,
        "size": 100,  # Get more listings for better price range calculation
        "sortBy": "NEWEST",
    }

    # Note: The backend API uses provinceId/districtId/wardId for filtering
    # Since we only have city/district names, we'll rely on the keyword search
    # or let the backend return all matching property types and filter client-side

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{backend_url}/v1/listings/search",  # Note: /listings/ with 's'
            json=search_payload,
            timeout=30.0,
        )
        response.raise_for_status()
        result = cast(dict[str, Any], response.json())

        # Extract listing data from paginated response
        # Backend returns: { "data": { "content": [...listings...], "totalElements": N, ... } }
        if "data" in result and isinstance(result["data"], dict):
            data = result["data"]
            if "content" in data:
                listings = cast(list[dict[str, Any]], data["content"])

                # Filter by city/district on client-side since we don't have IDs
                filtered_listings = []
                for listing in listings:
                    address = listing.get("address", {})
                    # Check if address matches our search criteria
                    # Address structure may vary, so we check multiple possible fields
                    listing_city = address.get("city") or address.get("province") or ""
                    listing_district = address.get("district") or ""

                    if (
                        city.lower() in listing_city.lower()
                        and district.lower() in listing_district.lower()
                    ):
                        filtered_listings.append(listing)

                return filtered_listings

        return []


def calculate_price_range(listings: list[dict[str, Any]]) -> tuple[int, int]:
    """Calculate price range from similar listings.

    Args:
        listings: List of listing data with price information

    Returns:
        Tuple of (min_price, max_price) in millions VND
    """
    if not listings:
        # Return default range if no data
        return (0, 0)

    # Extract prices and convert to millions VND
    prices = []
    for listing in listings:
        price = listing.get("price")
        if price and isinstance(price, (int, float)):
            # Price in database is in VND, convert to millions
            price_in_millions = price / 1_000_000
            prices.append(price_in_millions)

    if not prices:
        return (0, 0)

    # Calculate percentile-based range (10th to 90th percentile)
    prices.sort()
    n = len(prices)

    if n == 1:
        # Only one listing
        price = int(prices[0])
        return (price, price)

    # 10th percentile for min
    min_idx = int(n * 0.1)
    min_price = int(prices[min_idx])

    # 90th percentile for max
    max_idx = int(n * 0.9)
    max_price = int(prices[max_idx])

    return (min_price, max_price)


@router.post("/get-price-range", response_model=PriceSuggestionResponse)
async def get_price_range(
    request: PriceSuggestionRequest,
) -> PriceSuggestionResponse:
    """
    Get price range by querying similar listings from smartrent-backend database.

    This endpoint:
    1. Queries the backend database for similar listings (same city, district, property type)
    2. Calculates price range based on actual market data (10th-90th percentile)
    3. Returns price range in millions VND (triệu VND)

    Returns:
        PriceSuggestionResponse: Price range based on real market data
    """
    try:
        # Query backend for similar listings
        listings = await query_similar_listings(
            city=request.city,
            district=request.district,
            property_type=request.property_type,
            latitude=request.latitude,
            longitude=request.longitude,
        )

        # Calculate price range from listings
        min_price, max_price = calculate_price_range(listings)

        return PriceSuggestionResponse(
            price_range={"min": min_price, "max": max_price},
            location=f"{request.district}, {request.city}",
            property_type=request.property_type,
            currency="triệu VND",
        )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Backend API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting price range: {str(e)}"
        )
