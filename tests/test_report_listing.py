"""Tests for report_listing — the submit step must include the reporter contact
info + category the backend requires (else it 400s and the bot says a misleading
"lỗi hệ thống")."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.agent.tools.report_listing import _do_report


def test_confirm_without_login_asks_to_login():
    # Backend requires reporterEmail/reporterPhone; we auto-fill from the
    # logged-in profile. No token → tell the user to log in, NOT a vague error.
    result = asyncio.run(_do_report("561388", True, [1], "", None))
    assert result["status"] == "error"
    assert "đăng nhập" in result["error"].lower()


def test_confirm_fills_contact_and_category_from_profile():
    submit = AsyncMock(return_value={"submitted": True})
    with patch(
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
    # Logged in but account has no phone → clear message, not "lỗi hệ thống".
    with patch(
        "app.core.backend_client.get_user_profile",
        AsyncMock(return_value={"email": "a@b.com", "contactPhoneNumber": ""}),
    ):
        result = asyncio.run(_do_report("561388", True, [1], "", "tok"))
    assert result["status"] == "error"
    assert "email" in result["error"].lower() or "điện thoại" in result["error"].lower()
