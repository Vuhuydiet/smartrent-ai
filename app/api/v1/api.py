from fastapi import APIRouter

from . import (
    chat,
    completion,
    duplicate_check,
    listing_verification,
    price_suggestion,
    recommendation,
    users,
)

api_router = APIRouter()
api_router.include_router(users.router, prefix="/api/v1/users", tags=["users"])
api_router.include_router(chat.router, prefix="/api/v1", tags=["chat"])
api_router.include_router(
    completion.router, prefix="/api/v1/completion", tags=["completion"]
)
api_router.include_router(
    price_suggestion.router,
    prefix="/api/v1/price-suggestion",
    tags=["Price Suggestion"],
)
api_router.include_router(
    listing_verification.router,
    prefix="/ai",
    tags=["AI Listing Verification"],
)
api_router.include_router(
    recommendation.router,
    prefix="/api/v1/recommendations",
    tags=["Recommendations"],
)
api_router.include_router(
    duplicate_check.router,
    prefix="/api/v1/listings",
    tags=["Duplicate Detection"],
)
