"""
Tool: get_nearby_places

Find nearby points of interest (schools, hospitals, supermarkets, etc.)
and calculate distances from a listing's coordinates.

Two modes:
1. Google Places API (when GOOGLE_MAPS_API_KEY is configured) — real POI data
2. Haversine fallback (no API key) — straight-line distance between two coordinates
"""

import logging
import math
from typing import Any, Dict

import httpx
from vertexai.generative_models import FunctionDeclaration  # type: ignore[import]

from app.agent.tools.base_tool import BaseTool
from app.core.config import settings

logger = logging.getLogger(__name__)

_PLACES_API_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Mapping of user-friendly types to Google Places API types
_PLACE_TYPE_MAP = {
    "school": "school",
    "university": "university",
    "hospital": "hospital",
    "supermarket": "supermarket",
    "convenience_store": "convenience_store",
    "bus_station": "bus_station",
    "subway_station": "subway_station",
    "park": "park",
    "bank": "bank",
    "atm": "atm",
    "pharmacy": "pharmacy",
    "restaurant": "restaurant",
    "cafe": "cafe",
    "gym": "gym",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate straight-line distance between two coordinates in km."""
    r = 6371  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _compact_place(
    place: Dict[str, Any], origin_lat: float, origin_lon: float
) -> Dict[str, Any]:
    """Extract key fields from a Google Places result."""
    loc = place.get("geometry", {}).get("location", {})
    place_lat = loc.get("lat", 0)
    place_lon = loc.get("lng", 0)
    distance = _haversine_km(origin_lat, origin_lon, place_lat, place_lon)
    return {
        "name": place.get("name", ""),
        "address": place.get("vicinity", ""),
        "distance_km": round(distance, 2),
        "rating": place.get("rating"),
        "open_now": place.get("opening_hours", {}).get("open_now"),
    }


class GetNearbyPlacesTool(BaseTool):
    name = "get_nearby_places"
    description = (
        "Find nearby places (schools, hospitals, supermarkets, bus stations, etc.) "
        "around a property listing's location and calculate distances. "
        "Use when the user asks 'gần trường nào?', 'cách bệnh viện bao xa?', "
        "'xung quanh có gì?', or wants to know about nearby amenities."
    )

    def to_function_declaration(self) -> Any:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the property.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the property.",
                    },
                    "placeType": {
                        "type": "string",
                        "description": (
                            "Type of place to search for: "
                            "school, university, hospital, supermarket, "
                            "convenience_store, bus_station, subway_station, "
                            "park, bank, atm, pharmacy, restaurant, cafe, gym."
                        ),
                    },
                    "radiusMeters": {
                        "type": "integer",
                        "description": "Search radius in meters (default 1000, max 5000).",
                    },
                    "targetLatitude": {
                        "type": "number",
                        "description": (
                            "Optional: latitude of a specific place to calculate "
                            "distance to (e.g. a specific university)."
                        ),
                    },
                    "targetLongitude": {
                        "type": "number",
                        "description": "Optional: longitude of the target place.",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        lat = float(kwargs["latitude"])
        lon = float(kwargs["longitude"])
        place_type = kwargs.get("placeType", "")
        radius = max(100, min(int(kwargs.get("radiusMeters", 1000)), 5000))
        target_lat = kwargs.get("targetLatitude")
        target_lon = kwargs.get("targetLongitude")

        # Mode 1: Direct distance calculation between two points
        if target_lat is not None and target_lon is not None:
            distance = _haversine_km(lat, lon, float(target_lat), float(target_lon))
            return {
                "status": "success",
                "mode": "distance",
                "distance_km": round(distance, 2),
                "distance_text": (
                    f"{round(distance * 1000)}m"
                    if distance < 1
                    else f"{round(distance, 1)}km"
                ),
            }

        # Mode 2: Find nearby places via Google Places API
        if settings.GOOGLE_MAPS_API_KEY:
            return await self._search_places(lat, lon, place_type, radius)

        # Mode 3: No API key — return helpful message
        return {
            "status": "error",
            "error": (
                "Google Maps API chưa được cấu hình. "
                "Không thể tìm địa điểm xung quanh. "
                "Hãy hỏi người dùng về địa điểm cụ thể họ muốn tính khoảng cách."
            ),
        }

    async def _search_places(
        self,
        lat: float,
        lon: float,
        place_type: str,
        radius: int,
    ) -> Dict[str, Any]:
        """Search nearby places using Google Places API."""
        google_type = _PLACE_TYPE_MAP.get(place_type, place_type)

        params: Dict[str, Any] = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "key": settings.GOOGLE_MAPS_API_KEY,
            "language": "vi",
        }
        if google_type:
            params["type"] = google_type

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(_PLACES_API_URL, params=params)
                response.raise_for_status()
                data = response.json()

            if data.get("status") != "OK":
                error_msg = data.get("status", "UNKNOWN_ERROR")
                if error_msg == "ZERO_RESULTS":
                    return {
                        "status": "success",
                        "count": 0,
                        "places": [],
                        "message": f"Không tìm thấy {place_type or 'địa điểm'} nào trong bán kính {radius}m.",
                    }
                return {"status": "error", "error": f"Google Places API: {error_msg}"}

            results = data.get("results", [])
            places = [_compact_place(p, lat, lon) for p in results[:5]]  # Top 5 closest
            # Sort by distance
            places.sort(key=lambda p: p["distance_km"])

            return {
                "status": "success",
                "placeType": place_type or "all",
                "radiusMeters": radius,
                "count": len(places),
                "places": places,
            }

        except httpx.HTTPStatusError as e:
            logger.error("Google Places API HTTP %s", e.response.status_code)
            return {
                "status": "error",
                "error": f"Google API HTTP {e.response.status_code}",
            }
        except Exception as e:
            logger.error("get_nearby_places failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}
