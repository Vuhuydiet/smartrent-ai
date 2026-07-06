"""Tests for update_listing_price.

Backend #352 made the public GET /v1/listings/{id} return 404 for listings that
are not currently visible (pending review, rejected, expired, suspended). The
price-update *preview* fetches the current price from that endpoint, so for a
non-public owner listing it now comes back empty — the confirmation prompt must
degrade gracefully (never render a literal "None" as the current price). The
actual update (PUT .../price) is owner-scoped and unaffected.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from app.agent.tools.update_listing_price import _do_update_price


def _http_error(status: int, json_body: dict) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "http://backend/v1/listings/1")
    resp = httpx.Response(status, json=json_body, request=req)
    return httpx.HTTPStatusError(str(status), request=req, response=resp)


def test_preview_public_listing_shows_formatted_current_price():
    with patch(
        "app.core.backend_client.get_listing",
        AsyncMock(return_value={"price": 5000000}),
    ):
        result = asyncio.run(_do_update_price("tok", "561388", 4000000.0, False))
    assert result["status"] == "needs_confirmation"
    assert result["currentPrice"] == 5000000
    # both old and new price shown, thousands-separated
    assert "5,000,000" in result["message"]
    assert "4,000,000" in result["message"]


def test_preview_non_public_listing_never_shows_none():
    # get_listing 404s for non-public listings → old price unknown; the prompt
    # must not contain the string "None".
    with patch(
        "app.core.backend_client.get_listing",
        AsyncMock(side_effect=_http_error(404, {"code": "2005"})),
    ):
        result = asyncio.run(_do_update_price("tok", "561388", 4000000.0, False))
    assert result["status"] == "needs_confirmation"
    assert result["currentPrice"] is None
    assert "None" not in result["message"]
    assert "4,000,000" in result["message"]


def test_confirmed_update_unaffected_by_public_endpoint():
    # The real update uses the owner-scoped PUT endpoint, so it works even when
    # the public detail endpoint would 404.
    updater = AsyncMock(return_value={"updated": True})
    with patch("app.core.backend_client.update_listing_price", updater):
        result = asyncio.run(_do_update_price("tok", "561388", 4000000.0, True))
    assert result["status"] == "success"
    assert result["newPrice"] == 4000000.0
    updater.assert_awaited_once()
