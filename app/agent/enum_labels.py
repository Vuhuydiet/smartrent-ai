"""Vietnamese labels for backend listing enums.

The backend returns raw enum values (e.g. ``SEMI_FURNISHED``, ``NORTHEAST``).
Agent tools translate them here before handing listing data to the LLM, so the
model renders human labels instead of echoing raw enums — which it did
inconsistently, sometimes leaking or mistranslating the value.

Source of truth: the Listing.java enums in smartrent-backend
(ListingType / ProductType / Direction / Furnishing).
"""

from typing import Any, Dict

FURNISHING_LABELS: Dict[str, str] = {
    "FULLY_FURNISHED": "Đầy đủ nội thất",
    "SEMI_FURNISHED": "Nội thất cơ bản",
    "UNFURNISHED": "Không nội thất",
}

DIRECTION_LABELS: Dict[str, str] = {
    "NORTH": "Bắc",
    "SOUTH": "Nam",
    "EAST": "Đông",
    "WEST": "Tây",
    "NORTHEAST": "Đông Bắc",
    "NORTHWEST": "Tây Bắc",
    "SOUTHEAST": "Đông Nam",
    "SOUTHWEST": "Tây Nam",
}

PRODUCT_TYPE_LABELS: Dict[str, str] = {
    "ROOM": "Phòng trọ",
    "APARTMENT": "Căn hộ",
    "HOUSE": "Nhà nguyên căn",
    "OFFICE": "Văn phòng",
    "STUDIO": "Studio",
    "STORE": "Cửa hàng",
}

LISTING_TYPE_LABELS: Dict[str, str] = {
    "RENT": "Cho thuê",
    "SALE": "Bán",
    "SHARE": "Ở ghép",
}

_FIELD_LABELS = (
    ("furnishing", FURNISHING_LABELS),
    ("direction", DIRECTION_LABELS),
    ("productType", PRODUCT_TYPE_LABELS),
    ("listingType", LISTING_TYPE_LABELS),
)


def localize_listing_enums(listing: Dict[str, Any]) -> Dict[str, Any]:
    """Translate raw enum values in an LLM-facing listing dict to Vietnamese.

    Mutates and returns ``listing``. Unknown / renamed enum values are kept
    as-is so a new backend enum degrades to its raw value rather than vanishing.
    """
    for key, mapping in _FIELD_LABELS:
        val = listing.get(key)
        if isinstance(val, str) and val:
            listing[key] = mapping.get(val.upper(), val)
    return listing
