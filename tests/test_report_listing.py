"""Tests for report_listing — the submit step must (a) include the reporter
contact info + category the backend requires, and (b) send/echo the correct
reasonId (the catalog id, not the display position)."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.agent.tools.report_listing import _do_report

# The catalog id order != display order (LISTING ids 1-7, MAP ids 8-11, both
# restart display_order at 1 → interleaved). id 8 is "the 2nd shown reason".
_CATALOG = [
    {"reasonId": 1, "reasonText": "Các thông tin về: giá, diện tích, mô tả"},
    {"reasonId": 8, "reasonText": "Vị trí bất động sản chưa chính xác"},
    {"reasonId": 2, "reasonText": "Ảnh"},
]


def _patch_reasons():
    return patch(
        "app.core.backend_client.get_report_reasons",
        AsyncMock(return_value=_CATALOG),
    )


def test_confirm_without_login_asks_to_login():
    with _patch_reasons():
        result = asyncio.run(_do_report("561388", True, [1], "", None))
    assert result["status"] == "error"
    assert "đăng nhập" in result["error"].lower()


def test_confirm_fills_contact_and_category_from_profile():
    submit = AsyncMock(return_value={"submitted": True})
    with _patch_reasons(), patch(
        "app.core.backend_client.get_user_profile",
        AsyncMock(
            return_value={"email": "a@b.com", "contactPhoneNumber": "0900000000"}
        ),
    ), patch("app.core.backend_client.submit_listing_report", submit):
        result = asyncio.run(_do_report("561388", True, [1], "tin sai giá", "tok"))

    assert result["status"] == "success"
    body = submit.call_args.args[1]  # submit_listing_report(listing_id, body, token=)
    assert body["reasonIds"] == [1]
    assert body["reporterEmail"] == "a@b.com"
    assert body["reporterPhone"] == "0900000000"
    assert body["category"] == "LISTING"
    assert body["otherFeedback"] == "tin sai giá"


def test_confirm_profile_missing_contact_is_clear():
    with _patch_reasons(), patch(
        "app.core.backend_client.get_user_profile",
        AsyncMock(return_value={"email": "a@b.com", "contactPhoneNumber": ""}),
    ):
        result = asyncio.run(_do_report("561388", True, [1], "", "tok"))
    assert result["status"] == "error"
    assert "email" in result["error"].lower() or "điện thoại" in result["error"].lower()


def test_confirm_rejects_reason_id_not_in_catalog():
    # A mis-mapped/hallucinated id must be rejected (with the valid list), never
    # submitted — this is what silently reported the wrong reason before.
    submit = AsyncMock(return_value={"submitted": True})
    with _patch_reasons(), patch(
        "app.core.backend_client.submit_listing_report", submit
    ):
        result = asyncio.run(_do_report("561388", True, [99], "", "tok"))
    assert result["status"] == "error"
    assert "không hợp lệ" in result["error"].lower()
    submit.assert_not_called()


def test_confirm_echoes_reported_reason_text_for_the_id():
    # id 8 is "the 2nd shown reason"; the success payload must echo its real
    # text so a wrong id is caught immediately.
    with _patch_reasons(), patch(
        "app.core.backend_client.get_user_profile",
        AsyncMock(
            return_value={"email": "a@b.com", "contactPhoneNumber": "0900000000"}
        ),
    ), patch(
        "app.core.backend_client.submit_listing_report",
        AsyncMock(return_value={"submitted": True}),
    ):
        result = asyncio.run(_do_report("561388", True, [8], "", "tok"))
    assert result["status"] == "success"
    assert result["reportedReasons"] == ["Vị trí bất động sản chưa chính xác"]


def _patch_profile():
    return patch(
        "app.core.backend_client.get_user_profile",
        AsyncMock(
            return_value={"email": "a@b.com", "contactPhoneNumber": "0900000000"}
        ),
    )


def test_confirm_maps_reason_text_to_id():
    # id 8 = "the 2nd shown reason"; passing its text must map to id 8, not the
    # position 2 (= "Ảnh"). This is the deterministic fix for the wrong-reason bug.
    submit = AsyncMock(return_value={"submitted": True})
    with _patch_reasons(), _patch_profile(), patch(
        "app.core.backend_client.submit_listing_report", submit
    ):
        result = asyncio.run(
            _do_report(
                "561388",
                True,
                [],
                "",
                "tok",
                reason_texts=["Vị trí bất động sản chưa chính xác"],
            )
        )
    assert result["status"] == "success"
    assert submit.call_args.args[1]["reasonIds"] == [8]
    assert result["reportedReasons"] == ["Vị trí bất động sản chưa chính xác"]


def test_confirm_reason_text_match_is_case_and_space_insensitive():
    submit = AsyncMock(return_value={"submitted": True})
    with _patch_reasons(), _patch_profile(), patch(
        "app.core.backend_client.submit_listing_report", submit
    ):
        result = asyncio.run(
            _do_report("561388", True, [], "", "tok", reason_texts=["  ẢNH  "])
        )
    assert result["status"] == "success"
    assert submit.call_args.args[1]["reasonIds"] == [2]


def test_confirm_unmatched_reason_text_errors_without_submitting():
    submit = AsyncMock(return_value={"submitted": True})
    with _patch_reasons(), patch(
        "app.core.backend_client.submit_listing_report", submit
    ):
        result = asyncio.run(
            _do_report("561388", True, [], "", "tok", reason_texts=["Lý do bịa đặt"])
        )
    assert result["status"] == "error"
    assert "không khớp" in result["error"].lower()
    submit.assert_not_called()
