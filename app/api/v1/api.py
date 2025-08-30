from fastapi import APIRouter

from app.api.v1 import users, chat

api_router = APIRouter()
api_router.include_router(users.router, prefix="/api/v1/users", tags=["users"])
api_router.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
