from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dto.user import UserCreate
from app.entity.user import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    async def create(self, user_data: UserCreate) -> User:
        db_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password="",
            is_active=user_data.is_active,
            is_superuser=user_data.is_superuser,
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        stmt = select(User).offset(skip).limit(limit)
        result = self.db.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, user_id: int) -> bool:
        stmt = select(User).where(User.id == user_id)
        result = self.db.execute(stmt)
        db_user = result.scalar_one_or_none()
        if not db_user:
            return False

        self.db.delete(db_user)
        self.db.commit()
        return True
