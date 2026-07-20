import logging
import time
import unicodedata
from typing import Optional, Tuple

from cachetools import TTLCache

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.core.security import require_internal_key
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse
from app.service.price_prediction_service import PricePredictionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Coordinate rounding precision for the cache key, in decimal degrees. 3
# places is ~110m at the equator: coarse enough that repeated map-picker
# clicks within the same spot collide, fine enough not to blur distinct
# wards together.
_CACHE_COORD_PRECISION = 3

# Area bucket width in m². Requests within the same 5m² band share an entry
# — the agent's own comparable search already tolerates a ±30% area range,
# so this loses no meaningful precision.
_CACHE_AREA_BUCKET_M2 = 5.0

# (city, district, ward, property_type, area_bucket, lat, lon), all
# normalized. `area_bucket` is `None` when the request omitted area — kept
# distinct from any numeric bucket (including 0.0) so "unspecified" never
# collides with a real value.
CacheKey = Tuple[str, str, str, str, Optional[float], float, float]

# Module-level, process-wide cache. Rebuilt lazily by `_get_cache` whenever
# the configured size/TTL changes, so the values below are placeholders only.
_cache: Optional["TTLCache[CacheKey, PriceSuggestionResponse, float]"] = None
_cache_config: Optional[Tuple[int, int]] = None


def _get_cache() -> "TTLCache[CacheKey, PriceSuggestionResponse, float]":
    """Return the shared response cache, (re)building it from current settings.

    Settings are read here — at request time, not at import time — so ops
    can size/tune the cache via env vars without code changes, and so tests
    can monkeypatch `settings` directly. A rebuild only happens when the
    resolved (maxsize, ttl) actually changes, which in production is a
    one-time event at startup, not a per-request cost.
    """
    global _cache, _cache_config
    config = (
        settings.PRICE_SUGGESTION_CACHE_MAXSIZE,
        settings.PRICE_SUGGESTION_CACHE_TTL_SECONDS,
    )
    if _cache is None or _cache_config != config:
        # `timer` passed explicitly (mypy's cachetools stubs otherwise can't
        # infer the TTLCache[K, V, T] type params through the maxsize/ttl-only
        # overload).
        _cache = TTLCache(maxsize=config[0], ttl=config[1], timer=time.monotonic)
        _cache_config = config
    return _cache


def _normalize_text(value: str) -> str:
    """Collapse diacritics/whitespace/case noise in free-text Vietnamese
    location fields (e.g. " Hà Nội " vs "ha noi") so equivalent requests hit
    the same cache entry. NFC first so precomposed and decomposed diacritic
    forms of the same character compare equal."""
    return unicodedata.normalize("NFC", value).strip().casefold()


def _cache_key(request: PriceSuggestionRequest) -> CacheKey:
    """Build a normalized cache key for a price-suggestion request.

    Location text and coordinates are noisy for what is semantically the
    same query (whitespace, diacritics, sub-meter GPS jitter, near-identical
    areas), so normalizing here is what makes the cache actually hit instead
    of missing on every request.
    """
    area_bucket: Optional[float]
    if request.area is None:
        area_bucket = None
    else:
        area_bucket = (
            round(request.area / _CACHE_AREA_BUCKET_M2) * _CACHE_AREA_BUCKET_M2
        )

    return (
        _normalize_text(request.city),
        _normalize_text(request.district),
        _normalize_text(request.ward),
        _normalize_text(request.property_type),
        area_bucket,
        round(request.latitude, _CACHE_COORD_PRECISION),
        round(request.longitude, _CACHE_COORD_PRECISION),
    )


def get_price_prediction_service() -> PricePredictionService:
    """Dependency to get price prediction service instance."""
    try:
        return PricePredictionService()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price prediction service not available: {str(e)}",
        )


@router.post(
    "/get-price-suggestion",
    response_model=PriceSuggestionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_internal_key)],
)
async def get_price_suggestion(
    request: PriceSuggestionRequest,
    service: PricePredictionService = Depends(get_price_prediction_service),
) -> PriceSuggestionResponse:
    """
    Get AI-powered price prediction for a property.

    Provide property details to get an estimated price range based on:
    - Location (city, district, ward)
    - Property type (House, Apartment, Villa, Office, etc.)
    - Area in square meters
    - Geographic coordinates

    The AI analyzes market data and provides realistic price ranges in VND.

    - **city**: City or province name (e.g., 'Hanoi', 'Ho Chi Minh')
    - **district**: District name within the city
    - **ward**: Ward name within the district
    - **property_type**: Type of property
    - **area**: Property area in m² (optional)
    - **latitude**: Latitude coordinate
    - **longitude**: Longitude coordinate
    """
    cache = _get_cache()
    key = _cache_key(request)
    cached = cache.get(key)
    if cached is not None:
        logger.info("price-suggestion cache hit")
        return cached

    try:
        response = await service.predict_price(request)
    except Exception as e:
        logger.error(f"Price prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price prediction failed: {str(e)}",
        )

    # Only cache real market evidence. `rule_based_fallback` means the AI
    # path failed and a hardcoded per-m² table stood in — caching that would
    # pin a degraded answer for the whole TTL and defeat the point of the
    # `source` field, which exists precisely so callers can tell the two
    # apart.
    if response.source == "ai_comparables":
        cache[key] = response

    return response


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Health check endpoint for price prediction service."""
    return {"status": "healthy", "service": "price_prediction"}
