"""HTTP client for SmartRent backend house pricing APIs."""
from typing import Any, cast

import httpx


class HousePricingClient:
    """Client for interacting with SmartRent AI house pricing APIs."""

    def __init__(self, base_url: str) -> None:
        """Initialize the house pricing client.

        Args:
            base_url: Base URL of the SmartRent AI backend (e.g., http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        self.pricing_endpoint = f"{self.base_url}/v1/house-pricing"

    async def get_price_range(
        self,
        city: str,
        district: str,
        ward: str,
        property_type: str,
        latitude: float,
        longitude: float,
        area: float | None = None,
    ) -> dict[str, Any]:
        """Get predicted price range for a property.

        Args:
            city: City or province name (e.g., 'Hanoi', 'Ho Chi Minh')
            district: District or county name
            ward: Ward or commune name
            property_type: Type of property (APARTMENT, HOUSE, ROOM, STUDIO)
            latitude: Property latitude coordinate
            longitude: Property longitude coordinate
            area: Optional property area in square meters

        Returns:
            Price range prediction with min/max values in VND

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        payload = {
            "city": city,
            "district": district,
            "ward": ward,
            "property_type": property_type,
            "latitude": latitude,
            "longitude": longitude,
        }

        if area is not None:
            payload["area"] = area

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.pricing_endpoint}/get-price-range",
                json=payload,
                timeout=60.0,  # Longer timeout for ML prediction
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
