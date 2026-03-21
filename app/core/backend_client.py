"""
Shared async HTTP client for the SmartRent Spring Boot backend.

All backend API calls go through this module so that:
- Base URL and timeout are configured in one place
- Error handling is consistent
- Easy to add auth headers, retries, or circuit breakers later
"""

import logging
from typing import Any, Dict, List, Optional

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
