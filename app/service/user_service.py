from typing import Optional

from app.dto.user import User, UserCreate
from app.repository.user_repository import UserRepository


class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def create_user(self, user_data: UserCreate) -> User:
        # Check if user already exists

        return await self.user_repository.create(user_data)

    async def delete_user(self, user_id: int) -> bool:
        return await self.user_repository.delete(user_id)

    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        return None
