from typing import List, Optional

from pydantic import BaseModel, Field


class SearchParseRequest(BaseModel):
    query: str


class AppliedFilters(BaseModel):
    """
    Backend-ready search filter keys resolved from a free-text query.

    Unlike the loose text fields on :class:`AiParsedCriteriaDto`, every key
    here is something a consumer can hand straight to ``POST /v1/listings/search``
    (or, via the Java suggestion passthrough, to the frontend filter panel) —
    location names are resolved to legacy ids, amenities to ids, the property
    type to backend enums. The whole point is that the frontend never has to
    fall back to a raw ``keyword`` FULLTEXT search for a query that clearly
    expressed structured intent.

    Resolution reuses the AI chatbox knowledge base (``RAGRetriever``) so this
    stays consistent with what the chatbox's ``search_listings`` tool would do.
    """

    # Property type. `productType` is the single primary enum (what the FE
    # filter panel can apply today); `productTypes` is the multi-type set the
    # chatbox uses for ambiguous Vietnamese terms ("trọ" → ROOM + APARTMENT)
    # and is what the natural-language search / JPA path should fan out on.
    productType: Optional[str] = None
    productTypes: List[str] = Field(default_factory=list)
    listingType: Optional[str] = None

    minPrice: Optional[float] = None
    maxPrice: Optional[float] = None
    minArea: Optional[float] = None
    maxArea: Optional[float] = None
    bedrooms: Optional[int] = None

    # Legacy (pre-2025-07) ids/codes — the listing search filter expects these;
    # the backend reverse-maps them to new ward codes on every query.
    provinceCode: Optional[str] = None
    districtCode: Optional[str] = None
    legacyProvinceId: Optional[int] = None
    legacyDistrictId: Optional[int] = None

    amenityIds: List[int] = Field(default_factory=list)
    amenities: List[str] = Field(default_factory=list)
    amenityMatchMode: Optional[str] = None

    # NOTE: intentionally NO `keyword` / `locationText`. appliedFilters is a
    # STRUCTURED-ONLY payload — a residual/location keyword would make the
    # consumer run a title FULLTEXT search off an error-prone parse, which is
    # exactly what this feature exists to avoid. When nothing structured can
    # be resolved the resolver returns None instead of a keyword-only object.


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
    # Resolved, ready-to-apply filters (location → legacy ids, amenities → ids,
    # type → backend enum). Consumers should prefer this over the loose text
    # fields above. Null only when nothing structured could be extracted.
    appliedFilters: Optional[AppliedFilters] = None


class SearchSuggestionRequest(BaseModel):
    query: str
    limit: int = 5


class SearchSuggestionResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list)
    normalizedQuery: Optional[str] = None
    # Same resolved filter payload as /parse — lets the Java suggestion
    # endpoint forward backend-ready filters to the frontend instead of the
    # frontend FULLTEXT-searching the raw query.
    appliedFilters: Optional[AppliedFilters] = None
