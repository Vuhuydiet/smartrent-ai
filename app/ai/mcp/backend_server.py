"""
MCP Server for SmartRent Listing Search

This MCP server provides tools to search property listings from the SmartRent backend API.
It uses FastMCP to expose listing search capabilities to AI assistants.
"""

import httpx
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP

from app.core.config import settings

# Initialize FastMCP server
mcp = FastMCP("smartrent-backend")


@mcp.tool()
async def search_listings(
    # User & Ownership Filters
    user_id: Optional[str] = None,
    is_draft: Optional[bool] = None,
    verified: Optional[bool] = None,
    is_verify: Optional[bool] = None,
    expired: Optional[bool] = None,
    exclude_expired: Optional[bool] = True,
    status: Optional[str] = None,
    listing_status: Optional[str] = None,

    # Location Filters
    province_id: Optional[str] = None,
    province_code: Optional[str] = None,
    district_id: Optional[int] = None,
    ward_id: Optional[str] = None,
    new_ward_code: Optional[str] = None,
    street_id: Optional[int] = None,
    is_legacy: Optional[bool] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_km: Optional[float] = None,

    # Category & Type Filters
    category_id: Optional[int] = None,
    listing_type: Optional[str] = None,
    vip_type: Optional[str] = None,
    product_type: Optional[str] = None,

    # Property Specs Filters
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    price_unit: Optional[str] = None,
    has_price_reduction: Optional[bool] = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    min_bedrooms: Optional[int] = None,
    max_bedrooms: Optional[int] = None,
    min_bathrooms: Optional[int] = None,
    max_bathrooms: Optional[int] = None,
    furnishing: Optional[str] = None,
    direction: Optional[str] = None,
    min_room_capacity: Optional[int] = None,
    max_room_capacity: Optional[int] = None,

    # Utility Price Filters
    water_price: Optional[str] = None,
    electricity_price: Optional[str] = None,
    internet_price: Optional[str] = None,
    service_fee: Optional[str] = None,

    # Amenities & Media Filters
    amenity_ids: Optional[List[int]] = None,
    amenity_match_mode: Optional[str] = "ALL",
    has_media: Optional[bool] = None,
    min_media_count: Optional[int] = None,

    # Content Search
    keyword: Optional[str] = None,

    # Contact Filters
    owner_phone_verified: Optional[bool] = None,

    # Time Filters
    posted_within_days: Optional[int] = None,
    updated_within_days: Optional[int] = None,

    # Pagination & Sorting
    page: int = 1,
    size: int = 20,
    sort_by: Optional[str] = None,
    sort_direction: str = "DESC"
) -> Dict[str, Any]:
    """
    Search property listings from SmartRent backend.

    This tool allows searching for rental and sale properties with various filters including:
    - Location (province, district, ward, radius-based)
    - Property type (apartment, house, room, office, studio)
    - Listing type (rent, sale, share)
    - VIP tier (normal, silver, gold, diamond)
    - Price range and property specifications
    - Amenities and media
    - Time-based filters

    Args:
        user_id: Filter by user ID (for "my listings")
        is_draft: Filter by draft status (true/false/null)
        verified: Filter by verified status
        is_verify: Filter by verification pending status
        expired: Filter by expired status
        exclude_expired: Exclude expired listings (default true)
        status: Filter by post status (ACTIVE, EXPIRED, PENDING, DRAFT)
        listing_status: Filter for owner view (EXPIRED, EXPIRING_SOON, DISPLAYING, IN_REVIEW, etc.)

        province_id: Province ID (old structure)
        province_code: Province code (new structure - 34 provinces)
        district_id: District ID
        ward_id: Ward ID
        new_ward_code: Ward code (new structure)
        street_id: Street ID
        is_legacy: Use legacy address structure
        latitude: Latitude for location-based search
        longitude: Longitude for location-based search
        radius_km: Search radius in kilometers

        category_id: Category ID to filter by
        listing_type: RENT, SALE, or SHARE
        vip_type: NORMAL, SILVER, GOLD, or DIAMOND
        product_type: ROOM, APARTMENT, HOUSE, OFFICE, or STUDIO

        min_price: Minimum price in VND
        max_price: Maximum price in VND
        price_unit: MONTH, DAY, or YEAR
        has_price_reduction: Filter listings with price reductions
        min_area: Minimum area in square meters
        max_area: Maximum area in square meters
        bedrooms: Exact number of bedrooms
        bathrooms: Exact number of bathrooms
        min_bedrooms: Minimum bedrooms
        max_bedrooms: Maximum bedrooms
        min_bathrooms: Minimum bathrooms
        max_bathrooms: Maximum bathrooms
        furnishing: FULLY_FURNISHED, SEMI_FURNISHED, or UNFURNISHED
        direction: NORTH, SOUTH, EAST, WEST, NORTHEAST, etc.
        min_room_capacity: Minimum room capacity
        max_room_capacity: Maximum room capacity

        water_price: Water price filter (LOW, MEDIUM, HIGH)
        electricity_price: Electricity price filter
        internet_price: Internet price filter (FREE, LOW, MEDIUM, HIGH)
        service_fee: Service fee filter

        amenity_ids: List of amenity IDs to filter by
        amenity_match_mode: ALL (must have all) or ANY (at least one)
        has_media: Only show listings with media
        min_media_count: Minimum number of media items

        keyword: Search in title and description

        owner_phone_verified: Only verified owner phone numbers

        posted_within_days: Listings posted within last X days
        updated_within_days: Listings updated within last X days

        page: Page number (1-based)
        size: Page size (max 100)
        sort_by: Sort field (DEFAULT, PRICE_ASC, PRICE_DESC, NEWEST, OLDEST)
        sort_direction: ASC or DESC

    Returns:
        Dictionary containing:
        - listings: List of matching property listings
        - totalCount: Total number of matching listings
        - currentPage: Current page number
        - pageSize: Page size
        - totalPages: Total number of pages
        - recommendations: Recommended listings (if any)
        - filterCriteria: Applied filter criteria
    """

    # Build request payload, excluding None values
    filter_request: Dict[str, Any] = {}

    # User & Ownership
    if user_id is not None:
        filter_request["userId"] = user_id
    if is_draft is not None:
        filter_request["isDraft"] = is_draft
    if verified is not None:
        filter_request["verified"] = verified
    if is_verify is not None:
        filter_request["isVerify"] = is_verify
    if expired is not None:
        filter_request["expired"] = expired
    if exclude_expired is not None:
        filter_request["excludeExpired"] = exclude_expired
    if status is not None:
        filter_request["status"] = status
    if listing_status is not None:
        filter_request["listingStatus"] = listing_status

    # Location
    if province_id is not None:
        filter_request["provinceId"] = province_id
    if province_code is not None:
        filter_request["provinceCode"] = province_code
    if district_id is not None:
        filter_request["districtId"] = district_id
    if ward_id is not None:
        filter_request["wardId"] = ward_id
    if new_ward_code is not None:
        filter_request["newWardCode"] = new_ward_code
    if street_id is not None:
        filter_request["streetId"] = street_id
    if is_legacy is not None:
        filter_request["isLegacy"] = is_legacy
    if latitude is not None:
        filter_request["latitude"] = latitude
    if longitude is not None:
        filter_request["longitude"] = longitude
    if radius_km is not None:
        filter_request["radiusKm"] = radius_km

    # Category & Type
    if category_id is not None:
        filter_request["categoryId"] = category_id
    if listing_type is not None:
        filter_request["listingType"] = listing_type
    if vip_type is not None:
        filter_request["vipType"] = vip_type
    if product_type is not None:
        filter_request["productType"] = product_type

    # Property Specs
    if min_price is not None:
        filter_request["minPrice"] = min_price
    if max_price is not None:
        filter_request["maxPrice"] = max_price
    if price_unit is not None:
        filter_request["priceUnit"] = price_unit
    if has_price_reduction is not None:
        filter_request["hasPriceReduction"] = has_price_reduction
    if min_area is not None:
        filter_request["minArea"] = min_area
    if max_area is not None:
        filter_request["maxArea"] = max_area
    if bedrooms is not None:
        filter_request["bedrooms"] = bedrooms
    if bathrooms is not None:
        filter_request["bathrooms"] = bathrooms
    if min_bedrooms is not None:
        filter_request["minBedrooms"] = min_bedrooms
    if max_bedrooms is not None:
        filter_request["maxBedrooms"] = max_bedrooms
    if min_bathrooms is not None:
        filter_request["minBathrooms"] = min_bathrooms
    if max_bathrooms is not None:
        filter_request["maxBathrooms"] = max_bathrooms
    if furnishing is not None:
        filter_request["furnishing"] = furnishing
    if direction is not None:
        filter_request["direction"] = direction
    if min_room_capacity is not None:
        filter_request["minRoomCapacity"] = min_room_capacity
    if max_room_capacity is not None:
        filter_request["maxRoomCapacity"] = max_room_capacity

    # Utility Prices
    if water_price is not None:
        filter_request["waterPrice"] = water_price
    if electricity_price is not None:
        filter_request["electricityPrice"] = electricity_price
    if internet_price is not None:
        filter_request["internetPrice"] = internet_price
    if service_fee is not None:
        filter_request["serviceFee"] = service_fee

    # Amenities & Media
    if amenity_ids is not None:
        filter_request["amenityIds"] = amenity_ids
    if amenity_match_mode is not None:
        filter_request["amenityMatchMode"] = amenity_match_mode
    if has_media is not None:
        filter_request["hasMedia"] = has_media
    if min_media_count is not None:
        filter_request["minMediaCount"] = min_media_count

    # Content Search
    if keyword is not None:
        filter_request["keyword"] = keyword

    # Contact
    if owner_phone_verified is not None:
        filter_request["ownerPhoneVerified"] = owner_phone_verified

    # Time
    if posted_within_days is not None:
        filter_request["postedWithinDays"] = posted_within_days
    if updated_within_days is not None:
        filter_request["updatedWithinDays"] = updated_within_days

    # Pagination & Sorting
    filter_request["page"] = page
    filter_request["size"] = size
    if sort_by is not None:
        filter_request["sortBy"] = sort_by
    filter_request["sortDirection"] = sort_direction

    # Make API request
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/search",
                json=filter_request,
                timeout=30.0
            )
            response.raise_for_status()

            result = response.json()

            # Extract data from API response
            if result.get("code") == "999999" and "data" in result:
                return result["data"]
            else:
                return {
                    "error": result.get("message", "Unknown error"),
                    "code": result.get("code")
                }

        except httpx.HTTPError as e:
            return {
                "error": f"HTTP error occurred: {str(e)}",
                "status": "failed"
            }
        except Exception as e:
            return {
                "error": f"Error searching listings: {str(e)}",
                "status": "failed"
            }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
