from typing import Optional, List
from sqlalchemy.orm import Session
from app.entity.user import User
from app.dto.user import UserCreate


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    async def create(self, user_data: UserCreate) -> User:
        db_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password='',
            is_active=user_data.is_active,
            is_superuser=user_data.is_superuser
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    async def get_by_id(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    async def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.db.query(User).offset(skip).limit(limit).all()

    async def delete(self, user_id: int) -> bool:
        db_user = self.db.query(User).filter(User.id == user_id).first()
        if not db_user:
            return False
        
        self.db.delete(db_user)
        self.db.commit()
        return True
