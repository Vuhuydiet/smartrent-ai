"""Tests for backend listing enum → Vietnamese label mapping."""

from app.agent.enum_labels import localize_listing_enums


def test_translates_all_enum_fields():
    out = localize_listing_enums(
        {
            "furnishing": "SEMI_FURNISHED",
            "direction": "NORTHEAST",
            "productType": "APARTMENT",
            "listingType": "RENT",
        }
    )
    assert out["furnishing"] == "Nội thất cơ bản"
    assert out["direction"] == "Đông Bắc"
    assert out["productType"] == "Căn hộ"
    assert out["listingType"] == "Cho thuê"


def test_unknown_value_kept_as_is():
    out = localize_listing_enums({"productType": "PENTHOUSE", "furnishing": "WAT"})
    assert out["productType"] == "PENTHOUSE"
    assert out["furnishing"] == "WAT"


def test_missing_and_non_string_fields_untouched():
    out = localize_listing_enums({"title": "Căn hộ", "price": 5000000, "bedrooms": 2})
    assert out == {"title": "Căn hộ", "price": 5000000, "bedrooms": 2}


def test_lookup_is_case_insensitive():
    assert localize_listing_enums({"direction": "northeast"})["direction"] == "Đông Bắc"


def test_every_backend_enum_value_has_a_label():
    # Guard: each value from the Listing.java enums maps to a non-raw label.
    cases = {
        "furnishing": ["FULLY_FURNISHED", "SEMI_FURNISHED", "UNFURNISHED"],
        "direction": [
            "NORTH",
            "SOUTH",
            "EAST",
            "WEST",
            "NORTHEAST",
            "NORTHWEST",
            "SOUTHEAST",
            "SOUTHWEST",
        ],
        "productType": ["ROOM", "APARTMENT", "HOUSE", "OFFICE", "STUDIO", "STORE"],
        "listingType": ["RENT", "SALE", "SHARE"],
    }
    for field, values in cases.items():
        for value in values:
            assert localize_listing_enums({field: value})[field] != value
