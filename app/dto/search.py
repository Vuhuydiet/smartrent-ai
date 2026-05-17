from typing import Optional

from pydantic import BaseModel, Field


class SearchParseRequest(BaseModel):
    query: str


class AiParsedCriteriaDto(BaseModel):
    propertyType: Optional[str] = None
    listingType: Optional[str] = None
    minPrice: Optional[float] = None
    maxPrice: Optional[float] = None
    minArea: Optional[float] = None
    maxArea: Optional[float] = None
    bedrooms: Optional[int] = None
    province: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    amenities: list[str] = Field(default_factory=list)
    keyword: Optional[str] = None
    phoneticKeyword: Optional[str] = None


class SearchSuggestionRequest(BaseModel):
    query: str
    limit: int = 5


class SearchSuggestionResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list)
    normalizedQuery: Optional[str] = None
