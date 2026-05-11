from typing import Optional
from pydantic import BaseModel

class SearchParseRequest(BaseModel):
    query: str

class AiParsedCriteriaDto(BaseModel):
    propertyType: Optional[str] = None
    listingType: Optional[str] = None
    minPrice: Optional[float] = None
    maxPrice: Optional[float] = None
    province: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    keyword: Optional[str] = None
    phoneticKeyword: Optional[str] = None