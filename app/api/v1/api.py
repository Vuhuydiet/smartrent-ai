from fastapi import APIRouter

from . import chat, completion, listing_verification, price_suggestion, users

api_router = APIRouter()
api_router.include_router(users.router, prefix="/api/v1/users", tags=["users"])
api_router.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
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
