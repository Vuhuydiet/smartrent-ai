"""Saving something already saved is not a failure.

The backend rejects a duplicate save and an unsave-of-nothing, so the chatbot
kept reporting "Backend returned HTTP 500" and the model improvised an apology
for an operation that had, from the user's point of view, already happened.
These lock in the friendly outcome against both the new DomainCodes (24002 /
24003) and the old bare-RuntimeException backend still running in prod.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.agent.tools.bulk_save_listings import _do_bulk_save
from app.agent.tools.save_listing import _do_save_unsave, classify_saved_conflict
from app.core import backend_client


def _http_error(status: int, code: str = None, message: str = None):
    request = httpx.Request("POST", "http://backend/v1/saved-listings")
    body = {}
    if code is not None:
        body["code"] = code
    if message is not None:
        body["message"] = message
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


# ---------------------------------------------------------------------------
# backend_client.error_details
# ---------------------------------------------------------------------------


def test_error_details_unpacks_code_and_message():
    details = backend_client.error_details(_http_error(409, "24002", "Đã lưu rồi."))
    assert details == {"status": 409, "code": "24002", "message": "Đã lưu rồi."}


def test_error_details_falls_back_to_raw_body():
    request = httpx.Request("POST", "http://backend/v1/saved-listings")
    response = httpx.Response(502, text="<html>Bad Gateway</html>", request=request)
    details = backend_client.error_details(
        httpx.HTTPStatusError("err", request=request, response=response)
    )
    assert details["status"] == 502
    assert details["code"] is None
    assert "Bad Gateway" in details["message"]


# ---------------------------------------------------------------------------
# classify_saved_conflict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,status,code,message,expected",
    [
        # New backend: DomainCode carries the meaning.
        (
            "save",
            409,
            "24002",
            "Tin này đã có trong danh sách yêu thích.",
            "already_saved",
        ),
        ("unsave", 404, "24003", "Tin này không có trong danh sách.", "not_saved"),
        # Old backend: bare RuntimeException → 500 UNKNOWN_ERROR + raw message.
        ("save", 500, "9999", "Listing is already saved by this user", "already_saved"),
        ("unsave", 500, "9999", "Saved listing not found", "not_saved"),
        # Status alone is enough when the body is unreadable.
        ("save", 409, None, None, "already_saved"),
        ("unsave", 404, None, None, "not_saved"),
        # A 404 while SAVING means the listing itself is gone — a real error.
        ("save", 404, None, "Listing not found with ID: 42", None),
        # Genuine failures stay failures.
        ("save", 400, "24001", "Bạn chỉ có thể lưu tối đa 50 tin.", None),
        ("unsave", 401, None, "Unauthorized", None),
    ],
)
def test_classify_saved_conflict(action, status, code, message, expected):
    assert classify_saved_conflict(action, status, code, message) == expected


# ---------------------------------------------------------------------------
# save_listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_save_reports_already_saved():
    with patch(
        "app.agent.tools.save_listing.backend_client.save_listing",
        new=AsyncMock(side_effect=_http_error(409, "24002", "Đã lưu rồi.")),
    ):
        result = await _do_save_unsave("tok", "757549", "save")

    assert result["status"] == "already_saved"
    assert result["listingId"] == "757549"
    assert "đã có trong danh sách yêu thích" in result["message"]


@pytest.mark.asyncio
async def test_duplicate_save_on_legacy_backend_reports_already_saved():
    # Prod still answers 500 UNKNOWN_ERROR until the backend fix ships.
    with patch(
        "app.agent.tools.save_listing.backend_client.save_listing",
        new=AsyncMock(
            side_effect=_http_error(
                500, "9999", "Listing is already saved by this user"
            )
        ),
    ):
        result = await _do_save_unsave("tok", "757549", "save")

    assert result["status"] == "already_saved"


@pytest.mark.asyncio
async def test_unsave_of_never_saved_listing_is_not_an_error():
    with patch(
        "app.agent.tools.save_listing.backend_client.unsave_listing",
        new=AsyncMock(
            side_effect=_http_error(404, "24003", "Không có trong danh sách.")
        ),
    ):
        result = await _do_save_unsave("tok", "757549", "unsave")

    assert result["status"] == "not_saved"
    assert "vốn không có" in result["message"]


@pytest.mark.asyncio
async def test_save_limit_message_is_relayed_verbatim():
    # 24001 is a real constraint the user must act on — it must not be
    # flattened into "Backend returned HTTP 400".
    with patch(
        "app.agent.tools.save_listing.backend_client.save_listing",
        new=AsyncMock(
            side_effect=_http_error(
                400, "24001", "Bạn chỉ có thể lưu tối đa 50 tin. Hãy bỏ lưu một tin."
            )
        ),
    ):
        result = await _do_save_unsave("tok", "757549", "save")

    assert result["status"] == "error"
    assert "tối đa 50 tin" in result["error"]


@pytest.mark.asyncio
async def test_successful_save_still_reports_success():
    with patch(
        "app.agent.tools.save_listing.backend_client.save_listing",
        new=AsyncMock(return_value={"status": "saved"}),
    ):
        result = await _do_save_unsave("tok", "757549", "save")

    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# bulk_save_listings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_save_buckets_already_saved_apart_from_failures():
    async def fake_save(listing_id, token):
        if listing_id == "2":
            raise _http_error(409, "24002", "Đã lưu rồi.")
        if listing_id == "3":
            raise _http_error(500, "9999", "boom")
        return {"status": "saved"}

    with patch(
        "app.agent.tools.bulk_save_listings.backend_client.save_listing",
        new=AsyncMock(side_effect=fake_save),
    ):
        result = await _do_bulk_save("tok", ["1", "2", "3"], "save")

    assert result["succeeded"] == ["1"]
    assert result["alreadyDone"] == ["2"]
    assert [f["listingId"] for f in result["failed"]] == ["3"]
    assert "1 tin đã lưu sẵn từ trước nên bỏ qua" in result["message"]


@pytest.mark.asyncio
async def test_bulk_save_all_already_saved_is_still_success():
    with patch(
        "app.agent.tools.bulk_save_listings.backend_client.save_listing",
        new=AsyncMock(side_effect=_http_error(409, "24002", "Đã lưu rồi.")),
    ):
        result = await _do_bulk_save("tok", ["1", "2"], "save")

    assert result["status"] == "success"
    assert result["alreadyDone"] == ["1", "2"]
    assert result["failed"] == []
