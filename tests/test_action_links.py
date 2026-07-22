"""Tests for deep-link action chips and owner-facing enum localization.

Chat can only ever render a handful of rows, so tools that summarise a bigger
set register an action link (label + real app route) on ToolContext; the
orchestrator emits those ahead of the model's conversational chips. The same
tools used to hand the model raw enums (REVISION_REQUIRED, LISTING_APPROVED),
which it echoed verbatim into the answer.
"""

from unittest.mock import AsyncMock, patch

import pytest
from agents import RunContextWrapper

from app.agent.orchestrator import _MAX_FOLLOWUPS, _merge_action_links
from app.agent.tool_context import ToolContext
from app.agent.tools.my_listings_status import (
    _attention_listings,
    _dispatch_my_listings,
)
from app.agent.tools.notifications_inbox import _summarise


@pytest.fixture
def authed_ctx():
    return RunContextWrapper(context=ToolContext(auth_token="Bearer test-token"))


# ---------------------------------------------------------------------------
# ToolContext.action_links
# ---------------------------------------------------------------------------


def test_action_links_deduplicate_by_url():
    ctx = ToolContext()
    ctx.add_action_link("Xem tất cả tin đăng", "/seller/listings")
    ctx.add_action_link("Nhãn khác", "/seller/listings")
    ctx.add_action_link("Tin đã lưu", "/saved-listings")
    assert [link["url"] for link in ctx.action_links] == [
        "/seller/listings",
        "/saved-listings",
    ]
    assert ctx.action_links[0]["label"] == "Xem tất cả tin đăng"


# ---------------------------------------------------------------------------
# Orchestrator chip merge
# ---------------------------------------------------------------------------


def test_action_links_lead_and_total_is_capped():
    links = [{"label": "Xem tất cả tin đăng", "url": "/seller/listings"}]
    chips = [{"label": f"chip {i}", "query": f"q{i}"} for i in range(4)]
    merged = _merge_action_links(links, chips)
    assert merged[0] == links[0]
    assert len(merged) == _MAX_FOLLOWUPS


def test_merge_without_action_links_is_a_passthrough():
    chips = [{"label": "a", "query": "a"}]
    assert _merge_action_links([], chips) == chips


# ---------------------------------------------------------------------------
# my_listings_status — enum labels + focus-specific deep link
# ---------------------------------------------------------------------------


def test_attention_listings_localize_statuses():
    out = _attention_listings(
        [
            {
                "listingId": 756293,
                "title": "Cho thuê phòng sạch sẽ",
                "listingStatus": "REJECTED",
                "moderationStatus": "REVISION_REQUIRED",
                "address": {"districtName": "Huyện Nhà Bè"},
            }
        ]
    )
    assert out[0]["listingStatus"] == "Bị từ chối"
    assert out[0]["moderationStatus"] == "Cần chỉnh sửa"


def test_suspended_listing_is_flagged_for_attention():
    # SUSPENDED means a report is under review — it showed up in the chat
    # summary but was not in the attention set, so it only appeared by accident
    # when listingStatus happened to match.
    out = _attention_listings(
        [
            {
                "listingId": 753335,
                "title": "Cho thuê phòng mặt tiền",
                "listingStatus": "DISPLAYING",
                "moderationStatus": "SUSPENDED",
                "address": {},
            }
        ]
    )
    assert len(out) == 1
    assert out[0]["moderationStatus"] == "Bị đình chỉ"


@pytest.mark.parametrize(
    "focus,expected_label,expected_url",
    [
        ("all", "Xem tất cả tin đăng", "/seller/listings"),
        (
            "active",
            "Xem tất cả tin đang hoạt động",
            "/seller/listings?listingStatus=DISPLAYING",
        ),
        (
            "pending",
            "Xem tất cả tin chờ duyệt",
            "/seller/listings?listingStatus=IN_REVIEW",
        ),
        (
            "expiring",
            "Xem tất cả tin sắp hết hạn",
            "/seller/listings?listingStatus=EXPIRING_SOON",
        ),
        (
            "expired",
            "Xem tất cả tin hết hạn",
            "/seller/listings?listingStatus=EXPIRED",
        ),
        (
            "rejected",
            "Xem tất cả tin bị từ chối",
            "/seller/listings?listingStatus=REJECTED",
        ),
    ],
)
@pytest.mark.asyncio
async def test_focus_registers_matching_deep_link(
    authed_ctx, focus, expected_label, expected_url
):
    payload = {"listings": [], "statistics": {"active": 498}, "totalCount": 498}
    with patch(
        "app.agent.tools.my_listings_status.backend_client.get_my_listings",
        new=AsyncMock(return_value=payload),
    ):
        result = await _dispatch_my_listings(authed_ctx, focus)

    assert result["manageUrl"] == expected_url
    assert authed_ctx.context.action_links == [
        {"label": expected_label, "url": expected_url}
    ]
    # 498 listings, none rendered in chat → the model must mention there's more.
    assert result["moreAvailable"] is True


@pytest.mark.asyncio
async def test_focused_listings_are_returned_verbatim(authed_ctx):
    # "các tin của tôi chưa duyệt" → the backend already filtered to IN_REVIEW,
    # but the result was run through the attention filter a second time, which
    # drops everything that needs no owner action. The answer became a bare
    # "Bạn có 5 tin đang chờ duyệt." with nothing listed.
    pending = [
        {
            "listingId": 757000 + i,
            "title": f"Cho thuê phòng {i}",
            "listingStatus": "IN_REVIEW",
            "moderationStatus": "PENDING_REVIEW",
            "address": {"districtName": "Quận 5"},
        }
        for i in range(5)
    ]
    with patch(
        "app.agent.tools.my_listings_status.backend_client.get_my_listings",
        new=AsyncMock(return_value={"listings": pending, "totalCount": 5}),
    ):
        result = await _dispatch_my_listings(authed_ctx, "pending")

    assert result["listRole"] == "focus"
    assert len(result["listings"]) == 5
    assert result["listings"][0]["listingStatus"] == "Chờ duyệt"
    assert result["listings"][0]["moderationStatus"] == "Chờ duyệt"
    assert result["moreAvailable"] is False


@pytest.mark.asyncio
async def test_focused_query_asks_the_backend_for_that_status(authed_ctx):
    fetch = AsyncMock(return_value={"listings": [], "totalCount": 0})
    with patch(
        "app.agent.tools.my_listings_status.backend_client.get_my_listings", new=fetch
    ):
        await _dispatch_my_listings(authed_ctx, "pending")
    assert fetch.await_args.args[0]["listingStatus"] == "IN_REVIEW"


@pytest.mark.asyncio
async def test_overall_summary_still_shows_only_attention_rows(authed_ctx):
    # focus="all" is the mixed bag — a healthy DISPLAYING listing is noise
    # there, so it stays filtered out.
    mixed = [
        {"listingId": 1, "listingStatus": "DISPLAYING", "moderationStatus": "APPROVED"},
        {"listingId": 2, "listingStatus": "EXPIRED", "moderationStatus": "APPROVED"},
    ]
    with patch(
        "app.agent.tools.my_listings_status.backend_client.get_my_listings",
        new=AsyncMock(return_value={"listings": mixed, "totalCount": 2}),
    ):
        result = await _dispatch_my_listings(authed_ctx, None)

    assert result["listRole"] == "attention"
    assert [row["listingId"] for row in result["listings"]] == ["2"]


@pytest.mark.asyncio
async def test_guest_gets_no_deep_link():
    ctx = RunContextWrapper(context=ToolContext())
    result = await _dispatch_my_listings(ctx, None)
    assert result["status"] == "error"
    assert ctx.context.action_links == []


# ---------------------------------------------------------------------------
# notifications_inbox — type labels
# ---------------------------------------------------------------------------


def test_notification_types_are_localized_in_buckets_and_rows():
    summary = _summarise(
        [
            {"id": 1, "type": "LISTING_APPROVED", "title": "Tin được duyệt"},
            {"id": 2, "type": "LISTING_APPROVED", "title": "Tin được duyệt"},
            {"id": 3, "type": "MEMBERSHIP_EXPIRING", "title": "Gói sắp hết hạn"},
        ]
    )
    assert summary["byType"] == {"Tin được duyệt": 2, "Gói hội viên sắp hết hạn": 1}
    assert summary["recent"][0]["type"] == "Tin được duyệt"


def test_unknown_notification_type_degrades_to_raw_value():
    summary = _summarise([{"id": 1, "type": "SOME_FUTURE_TYPE"}])
    assert summary["byType"] == {"SOME_FUTURE_TYPE": 1}
