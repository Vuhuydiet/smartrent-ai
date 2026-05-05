from typing import List

from fastapi import APIRouter

from app.dto.recommendation import (
    PersonalizedFeedRequest,
    RecommendationItem,
    SimilarListingRequest,
)
from app.service.recommendation_service import RecommendationService

router = APIRouter()
recommendation_service = RecommendationService()


@router.post("/similar", response_model=List[RecommendationItem])
async def get_similar_listings(request: SimilarListingRequest):
    return await recommendation_service.get_similar_listings(request)


@router.post("/personalized", response_model=List[RecommendationItem])
async def get_personalized_feed(request: PersonalizedFeedRequest):
    return await recommendation_service.get_personalized_feed(request)
