from fastapi import APIRouter

from . import chat, house_pricing_frontend, users

api_router = APIRouter()
api_router.include_router(users.router, prefix="/api/v1/users", tags=["users"])
api_router.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
api_router.include_router(
    house_pricing_frontend.router,
    prefix="/api/v1/house-pricing",
    tags=["House Pricing"],
)
