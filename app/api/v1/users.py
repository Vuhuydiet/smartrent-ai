from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException, status

from app.database.database import get_db
from app.dto.user import User, UserCreate
from app.repository.user_repository import UserRepository
from app.service.user_service import UserService

router = APIRouter()


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    user_repository = UserRepository(db)
    return UserService(user_repository)


@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate, user_service: UserService = Depends(get_user_service)
) -> User:
    try:
        return await user_service.create_user(user_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{user_id}")
async def delete_user(
    user_id: int, user_service: UserService = Depends(get_user_service)
) -> dict[str, str]:
    success = await user_service.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return {"message": "User deleted successfully"}
