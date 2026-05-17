"""
Tests for app.agent.search_filter_resolver.

Focus: a free-text query expressing structured intent must resolve to
backend-ready filter keys (location → legacy ids, amenity → ids, type → enum,
price/area numbers) and must NOT fall back to dumping the whole query into
`keyword`. The headline case is the one the feature was reported broken on:
"tìm trọ tân bình dưới 5tr có máy lạnh".
"""

from app.agent.search_filter_resolver import resolve_applied_filters
from app.dto.search import AiParsedCriteriaDto


def test_reported_query_resolves_to_filters_not_keyword():
    af = resolve_applied_filters("tìm trọ tân bình dưới 5tr có máy lạnh")
    assert af is not None
    # "trọ" is ambiguous → multi-type like the chatbox; primary stays ROOM.
    assert af.productType == "ROOM"
    assert af.productTypes == ["ROOM", "APARTMENT"]
    # "dưới 5tr" → 5,000,000 VND upper bound.
    assert af.maxPrice == 5_000_000
    # "tân bình" → HCM (79) / legacy district 766.
    assert af.provinceCode == "79"
    assert af.legacyProvinceId == 79
    assert af.districtCode == "766"
    assert af.legacyDistrictId == 766
    # "máy lạnh" → amenity id 2 (Điều hòa).
    assert af.amenityIds == [2]
    assert af.amenityMatchMode == "ALL"
    # The whole sentence must NOT be echoed back as a keyword.
    assert not af.keyword
    assert af.locationText is None


def test_price_range_and_bedrooms():
    af = resolve_applied_filters("căn hộ quận 7 từ 5 đến 10 triệu 2pn")
    assert af is not None
    assert af.productType == "APARTMENT"
    assert af.productTypes == ["APARTMENT"]
    assert af.minPrice == 5_000_000
    assert af.maxPrice == 10_000_000
    assert af.bedrooms == 2
    assert af.provinceCode == "79"
    assert af.legacyDistrictId == 778  # Quận 7


def test_area_extracted_before_price():
    # "30m2" must be area, never misread as 30 triệu price.
    af = resolve_applied_filters("cho thuê văn phòng quận 1 dưới 30m2")
    assert af is not None
    assert af.productType == "OFFICE"
    assert af.listingType == "RENT"
    assert af.maxArea == 30
    assert af.maxPrice is None
    assert af.legacyDistrictId == 760  # Quận 1


def test_exact_type_is_single_ambiguous_is_multi():
    assert resolve_applied_filters("phòng trọ thủ đức").productTypes == ["ROOM"]
    assert resolve_applied_filters("nhà trọ quận 1").productTypes == [
        "ROOM",
        "APARTMENT",
    ]


def test_province_only_resolves_without_district():
    af = resolve_applied_filters("nhà nguyên căn hà nội trên 10 triệu")
    assert af is not None
    assert af.productType == "HOUSE"
    assert af.minPrice == 10_000_000
    assert af.provinceCode == "01"
    assert af.legacyProvinceId == 1
    assert af.legacyDistrictId is None


def test_pure_free_text_stays_keyword_not_location():
    # No structured signal at all → plain keyword search, never a bogus
    # district filter (that was the downstream bug for the /parse fallback).
    af = resolve_applied_filters("hello world random text")
    assert af is not None
    assert af.keyword == "hello world random text"
    assert af.locationText is None
    assert af.provinceCode is None
    assert not af.productTypes


def test_empty_query_returns_none():
    assert resolve_applied_filters("") is None
    assert resolve_applied_filters("   ") is None


def test_llm_criteria_overlay_wins_and_resolves_names():
    crit = AiParsedCriteriaDto(
        propertyType="APARTMENT",
        maxPrice=8_000_000,
        district="Bình Thạnh",
        amenities=["wifi"],
    )
    af = resolve_applied_filters("tìm chỗ ở giá tốt", crit)
    assert af is not None
    assert af.productTypes == ["APARTMENT"]
    assert af.maxPrice == 8_000_000
    assert af.legacyDistrictId == 765  # Bình Thạnh, resolved from criteria
    assert af.amenityIds == [1]  # wifi
