from typing import List, Optional

from pydantic import BaseModel


class ListingFeature(BaseModel):
    listing_id: int
    product_type: str  # ROOM | APARTMENT | HOUSE | STUDIO | OFFICE
    listing_type: str  # RENT | SALE | SHARE
    price: float
    area: Optional[float] = 0.0
    bedrooms: Optional[int] = 0
    province_code: str  # legacyProvinceId string or new province code
    district_id: Optional[int] = None
    ward_id: Optional[int] = None
    ward_code: Optional[str] = None
    vip_type: str  # NORMAL | SILVER | GOLD | DIAMOND
    post_date_days_ago: int  # freshness
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class InteractionEntry(BaseModel):
    user_id: str
    listing_id: int
    weight: float  # 3.0 for Saved, 2.5 for PhoneClick, 1.0 for View


class SimilarListingRequest(BaseModel):
    target: ListingFeature
    candidates: List[ListingFeature]
    top_n: int = 8
    user_interactions: Optional[
        List[InteractionEntry]
    ] = None  # Optional: user interaction weights
    interaction_features: Optional[
        List[ListingFeature]
    ] = None  # Data for historical listings to build profile vector


class PersonalizedFeedRequest(BaseModel):
    user_id: str
    user_interactions: List[InteractionEntry]  # explicit interactions of this user
    all_interactions: List[
        InteractionEntry
    ]  # interactions for candidate items from all users
    candidates: List[ListingFeature]
    top_n: int = 20
    interaction_features: Optional[List[ListingFeature]] = None


class RecommendationItem(BaseModel):
    listing_id: int
    score: float
    cf_score: float
    cbf_score: float
