"""
Shared async HTTP client for the SmartRent Spring Boot backend.

All backend API calls go through this module so that:
- Base URL and timeout are configured in one place
- Error handling is consistent
- Easy to add auth headers, retries, or circuit breakers later
"""

import logging
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0


async def search_listings(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /v1/listings/search

    Returns the raw `data` payload on success, or an error dict.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/search",
            json=params,
        )
        response.raise_for_status()
        result = response.json()

    logger.info(
        "Backend search raw response: code=%s, message=%s",
        result.get("code"),
        result.get("message"),
    )

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {
        "error": result.get("message", "Backend returned an unexpected response"),
        "code": result.get("code"),
    }


async def get_listing(listing_id: str) -> Dict[str, Any]:
    """
    GET /v1/listings/{listing_id}

    Returns the raw `data` payload on success, or an error dict.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/{listing_id}",
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {
        "error": result.get("message", "Listing not found"),
        "code": result.get("code"),
    }


# ---------------------------------------------------------------------------
# Recommendation endpoints
# ---------------------------------------------------------------------------


async def get_similar_listings(
    listing_id: int, top_n: int = 8, token: Optional[str] = None
) -> Dict[str, Any]:
    """
    GET /v1/recommendations/similar/{listingId}?topN=N

    Returns similar listings. Auth optional (provides personalization if present).
    """
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/recommendations/similar/{listing_id}",
            params={"topN": top_n},
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {
        "error": result.get("message", "Failed to get similar listings"),
        "code": result.get("code"),
    }


async def get_personalized_recommendations(
    top_n: int = 20, token: Optional[str] = None
) -> Dict[str, Any]:
    """
    GET /v1/recommendations/personalized?topN=N

    Returns personalized feed based on user's browsing history. Requires auth.
    """
    headers = _auth_headers(token)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/recommendations/personalized",
            params={"topN": top_n},
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {
        "error": result.get("message", "Failed to get personalized recommendations"),
        "code": result.get("code"),
    }


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------


async def get_user_profile(token: str) -> Dict[str, Any]:
    """
    GET /v1/users

    Returns authenticated user's profile.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/users",
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {"error": result.get("message", "Failed to get user profile")}


async def get_saved_listings(
    token: str, page: int = 1, size: int = 10
) -> Dict[str, Any]:
    """
    GET /v1/saved-listings/my-saved

    Returns paginated saved listings for authenticated user.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/saved-listings/my-saved",
            params={"page": page, "size": size},
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {"error": result.get("message", "Failed to get saved listings")}


async def save_listing(listing_id: Any, token: str) -> Dict[str, Any]:
    """
    POST /v1/saved-listings

    Save a listing to user's favorites.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/saved-listings",
            json={"listingId": listing_id},
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999":
        return result.get("data", {"status": "saved"})

    return {"error": result.get("message", "Failed to save listing")}


async def unsave_listing(listing_id: Any, token: str) -> Dict[str, Any]:
    """
    DELETE /v1/saved-listings/{listingId}

    Remove a listing from user's favorites.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.delete(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/saved-listings/{listing_id}",
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999":
        return result.get("data", {"status": "unsaved"})

    return {"error": result.get("message", "Failed to unsave listing")}


async def get_user_membership(token: str) -> Dict[str, Any]:
    """
    GET /v1/memberships/my-membership

    Returns current active membership/subscription.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/memberships/my-membership",
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {"error": result.get("message", "No active membership found")}


# ---------------------------------------------------------------------------
# Pricing history endpoints
# ---------------------------------------------------------------------------


async def get_pricing_history(listing_id: int) -> Dict[str, Any]:
    """
    GET /v1/listings/{listingId}/pricing-history

    Returns full price change history for a listing.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/{listing_id}/pricing-history",
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return {"history": result["data"]}

    return {"error": result.get("message", "Failed to get pricing history")}


async def get_price_statistics(listing_id: int) -> Dict[str, Any]:
    """
    GET /v1/listings/{listingId}/price-statistics

    Returns min/max/avg price and change counts for a listing.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/{listing_id}/price-statistics",
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {"error": result.get("message", "Failed to get price statistics")}


async def get_recent_price_changes(
    days_back: int = 7, page: int = 1, size: int = 20
) -> Dict[str, Any]:
    """
    GET /v1/listings/recent-price-changes

    Returns listing IDs with recent price changes.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/recent-price-changes",
            params={"daysBack": days_back, "page": page, "size": size},
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]

    return {"error": result.get("message", "Failed to get recent price changes")}


# ---------------------------------------------------------------------------
# Owner / dashboard endpoints
# ---------------------------------------------------------------------------


async def get_my_listings(
    params: Optional[Dict[str, Any]] = None, token: Optional[str] = None
) -> Dict[str, Any]:
    """
    POST /v1/listings/my-listings

    Owner-scoped paginated list with pre-computed `statistics` summary
    (drafts/pendingVerification/rejected/active/expired counts + VIP tier
    breakdown). Auth required.
    """
    body: Dict[str, Any] = {**(params or {})}
    body.setdefault("page", 1)
    body.setdefault("size", 20)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/my-listings",
            json=body,
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]
    return {
        "error": result.get("message", "Failed to fetch my listings"),
        "code": result.get("code"),
    }


async def update_listing_price(
    listing_id: str,
    new_price: float,
    token: Optional[str] = None,
    effective_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    PUT /v1/listings/{listingId}/price

    Owner-only. Records a new entry in pricing history. Auth required.
    """
    body: Dict[str, Any] = {"newPrice": new_price}
    if effective_at:
        body["effectiveAt"] = effective_at

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.put(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/{listing_id}/price",
            json=body,
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999":
        return result.get("data") or {"updated": True}
    return {
        "error": result.get("message", "Failed to update price"),
        "code": result.get("code"),
    }


# ---------------------------------------------------------------------------
# Address translator endpoint
# ---------------------------------------------------------------------------


async def search_new_address(
    keyword: str, page: int = 1, limit: int = 10
) -> Dict[str, Any]:
    """
    GET /v1/addresses/search-new-address

    Search across NEW (post-2025-07) provinces and wards by keyword. Public.
    Used by the address_translator tool to find new-structure codes for a
    district/ward name the user mentions.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/addresses/search-new-address",
            params={"keyword": keyword, "page": page, "limit": limit},
        )
        response.raise_for_status()
        result = response.json()

    if result.get("code") == "999999" and "data" in result:
        return result["data"]
    return {
        "error": result.get("message", "Failed to search new address"),
        "code": result.get("code"),
    }


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


async def list_notifications(
    page: int = 1, size: int = 20, token: Optional[str] = None
) -> Dict[str, Any]:
    """GET /v1/notifications — paginated. Auth required."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/notifications",
            params={"page": page, "size": size},
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()
    if result.get("code") == "999999" and "data" in result:
        return result["data"]
    return {"error": result.get("message", "Failed to list notifications")}


async def mark_all_notifications_read(token: Optional[str] = None) -> Dict[str, Any]:
    """PATCH /v1/notifications/read-all. Auth required."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.patch(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/notifications/read-all",
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()
    if result.get("code") == "999999":
        return result.get("data") or {"updated": True}
    return {"error": result.get("message", "Failed to mark all read")}


# ---------------------------------------------------------------------------
# Listing reports
# ---------------------------------------------------------------------------


async def get_report_reasons() -> Dict[str, Any]:
    """GET /v1/listings/reports/reasons — public list of report categories."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/reports/reasons",
        )
        response.raise_for_status()
        result = response.json()
    if result.get("code") == "999999" and "data" in result:
        return result["data"]
    return {"error": result.get("message", "Failed to fetch report reasons")}


async def submit_listing_report(
    listing_id: str,
    body: Dict[str, Any],
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    POST /v1/listings/{listingId}/reports

    Body shape per backend: reasonIds[], otherFeedback, reporterName,
    reporterPhone, reporterEmail. Auth optional (anonymous reports allowed).
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/{listing_id}/reports",
            json=body,
            headers=_auth_headers(token),
        )
        response.raise_for_status()
        result = response.json()
    if result.get("code") == "999999":
        return result.get("data") or {"submitted": True}
    return {"error": result.get("message", "Failed to submit report")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Build Authorization header if token is provided."""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}
