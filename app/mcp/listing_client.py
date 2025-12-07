"""HTTP client for SmartRent backend listing APIs."""
from typing import Any, Optional, cast

import httpx


class ListingClient:
    """Client for interacting with SmartRent backend listing APIs."""

    def __init__(self, base_url: str) -> None:
        """Initialize the listing client.

        Args:
            base_url: Base URL of the SmartRent backend (e.g., http://localhost:8080)
        """
        self.base_url = base_url.rstrip("/")
        self.listings_endpoint = f"{self.base_url}/v1/listings"

    async def get_listing(self, listing_id: int) -> dict[str, Any]:
        """Get a single listing by ID.

        Args:
            listing_id: The ID of the listing to retrieve

        Returns:
            Listing data as a dictionary

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.listings_endpoint}/{listing_id}",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def search_listings(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Search and filter listings.

        This is the main search API that supports comprehensive filtering:
        - Location (province, district, ward, GPS coordinates)
        - Price range and price reduction
        - Area range
        - Property features (bedrooms, bathrooms, furnishing, direction)
        - Utilities pricing
        - Listing type and VIP status
        - Amenities

        Args:
            filters: Dictionary of filter parameters

        Returns:
            Search results with listings and pagination info

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.listings_endpoint}/search",
                json=filters,
                timeout=30.0,
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    async def list_listings(
        self,
        ids: Optional[list[int]] = None,
        page: int = 0,
        size: int = 20,
    ) -> dict[str, Any]:
        """List listings with pagination or get specific listings by IDs.

        Args:
            ids: Optional list of listing IDs to fetch explicitly
            page: Zero-based page index
            size: Page size (max 100)

        Returns:
            List of listings or paginated results

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        params: dict[str, Any] = {}

        if ids:
            params["ids"] = ",".join(map(str, ids))
        else:
            params["page"] = page
            params["size"] = min(size, 100)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.listings_endpoint,
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
