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

# Owner-facing lifecycle enums (ListingStatus.java / ModerationStatus.java).
# Wording mirrors the seller UI (messages/vi.json → seller.listings.status) so
# the chatbot and the "Quản lý tin đăng" page name the same thing identically.
LISTING_STATUS_LABELS: Dict[str, str] = {
    "EXPIRED": "Hết hạn",
    "EXPIRING_SOON": "Sắp hết hạn",
    "DISPLAYING": "Đang hiển thị",
    "IN_REVIEW": "Chờ duyệt",
    "PENDING_PAYMENT": "Chờ thanh toán",
    "REJECTED": "Bị từ chối",
    "VERIFIED": "Đã xác thực",
    "RESUBMITTED": "Đã gửi lại chờ duyệt",
}

MODERATION_STATUS_LABELS: Dict[str, str] = {
    "PENDING_REVIEW": "Chờ duyệt",
    "APPROVED": "Đã duyệt",
    "REJECTED": "Admin từ chối",
    "REVISION_REQUIRED": "Cần chỉnh sửa",
    "RESUBMITTED": "Đã gửi lại",
    "SUSPENDED": "Bị đình chỉ",
    "REMOVED": "Đã gỡ vĩnh viễn",
}

# NotificationType.java — what the inbox tool buckets counts by. Legacy values
# (PHONE_CLICK / VIEW_MILESTONE / NEW_FOLLOWER) still exist on old rows.
NOTIFICATION_TYPE_LABELS: Dict[str, str] = {
    "NEW_REPORT": "Báo cáo mới",
    "REPORT_RESOLVED": "Báo cáo đã xử lý",
    "REPORT_REJECTED": "Báo cáo bị từ chối",
    "REPORT_ACTION_REQUIRED": "Báo cáo cần xử lý",
    "REPORT_LISTING_REMOVED": "Tin bị gỡ do vi phạm",
    "LISTING_APPROVED": "Tin được duyệt",
    "LISTING_REJECTED": "Tin bị từ chối",
    "LISTING_REVISION_REQUIRED": "Tin cần chỉnh sửa",
    "LISTING_SUSPENDED": "Tin bị đình chỉ",
    "LISTING_RESUBMITTED": "Tin đã gửi lại",
    "LISTING_PENDING_REVIEW": "Tin chờ duyệt",
    "LISTING_DUPLICATE_DETECTED": "Phát hiện tin trùng lặp",
    "NEW_LISTING_PENDING_REVIEW": "Tin mới chờ duyệt",
    "BROKER_REGISTRATION_RECEIVED": "Đăng ký môi giới",
    "BROKER_APPROVED": "Môi giới được duyệt",
    "BROKER_REJECTED": "Môi giới bị từ chối",
    "LISTING_EXPIRING": "Tin sắp hết hạn",
    "NEW_LISTING_FROM_FOLLOWED_USER": "Tin mới từ người bạn theo dõi",
    "MEMBERSHIP_EXPIRING": "Gói hội viên sắp hết hạn",
    "MEMBERSHIP_ACTIVATED": "Gói hội viên đã kích hoạt",
    "POSTING_BLOCKED": "Bị chặn đăng tin",
    "POSTING_UNBLOCKED": "Được mở lại quyền đăng tin",
    "PHONE_CLICK": "Lượt xem số điện thoại",
    "VIEW_MILESTONE": "Cột mốc lượt xem",
    "NEW_FOLLOWER": "Người theo dõi mới",
    "OTHER": "Khác",
}

# UserMembership.status — the subscription lifecycle, not a listing enum.
MEMBERSHIP_STATUS_LABELS: Dict[str, str] = {
    "ACTIVE": "Đang hoạt động",
    "PENDING": "Chờ kích hoạt",
    "QUEUED": "Đang xếp hàng chờ",
    "EXPIRED": "Đã hết hạn",
    "CANCELLED": "Đã hủy",
}

_FIELD_LABELS = (
    ("furnishing", FURNISHING_LABELS),
    ("direction", DIRECTION_LABELS),
    ("productType", PRODUCT_TYPE_LABELS),
    ("listingType", LISTING_TYPE_LABELS),
    ("listingStatus", LISTING_STATUS_LABELS),
    ("moderationStatus", MODERATION_STATUS_LABELS),
)


def localize_enum(value: Any, mapping: Dict[str, str]) -> Any:
    """Translate one raw enum value, keeping unknown / non-string input as-is."""
    if isinstance(value, str) and value:
        return mapping.get(value.upper(), value)
    return value


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
