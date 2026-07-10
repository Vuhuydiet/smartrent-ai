"""Tests for the search_listings tool schema and error surfacing."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.agent.tools.search_listings import _do_search, search_listings


def test_schema_uses_district_code_not_district_id():
    # The model knows districts by GSO code, not the internal surrogate PK. The
    # tool must expose districtCode (string) and NOT the old numeric districtId
    # (which silently matched nothing — Quận 1 code 760 != PK 541).
    props = search_listings.params_json_schema["properties"]
    assert "districtCode" in props
    assert "districtId" not in props


def test_schema_constrains_enums():
    # Enum params are Literals → JSON-schema enum, so the model can't invent
    # values the backend rejects (e.g. "CONDO", "cheapest").
    blob = json.dumps(search_listings.params_json_schema)
    for value in [
        "ROOM",
        "APARTMENT",
        "RENT",
        "SHARE",
        "PRICE_ASC",
        "FULLY_FURNISHED",
        "NORTHEAST",
    ]:
        assert value in blob


def test_do_search_surfaces_backend_error_body():
    # When the backend rejects a param (e.g. unknown districtCode → 400 with a
    # message), the tool must forward that message to the model so it can
    # self-correct — not collapse it into a bare "HTTP 400".
    request = httpx.Request("POST", "http://backend/v1/listings/search")
    response = httpx.Response(
        400,
        json={
            "code": "2011",
            "message": "Unknown district code '999'. Provide a valid GSO district code.",
        },
        request=request,
    )
    err = httpx.HTTPStatusError("400 Bad Request", request=request, response=response)
    with patch("app.core.backend_client.search_listings", AsyncMock(side_effect=err)):
        result = asyncio.run(
            _do_search(MagicMock(), {"size": 5, "districtCode": "999"})
        )
    assert result["status"] == "error"
    assert "Unknown district code" in result["error"]


def test_schema_hides_province_id():
    # provinceId (the legacy id) duplicates provinceCode and only invites the
    # model to fill it wrongly. It must not be model-facing; the tool syncs it
    # from provinceCode internally.
    props = search_listings.params_json_schema["properties"]
    assert "provinceCode" in props
    assert "provinceId" not in props


def test_rejects_search_without_location_or_keyword():
    # A search with neither a location nor a keyword is too broad to run — the
    # tool must reject it locally so the model asks the user for a location
    # instead of dumping the entire dataset.
    result = asyncio.run(_do_search(MagicMock(), {"size": 5}))
    assert result["status"] == "error"
    assert "keyword" in result["error"]


def test_zero_results_includes_structured_hint():
    # A genuine 0-result search must carry a hint so the model reports "not
    # found" and offers to relax filters, instead of silently re-searching
    # elsewhere.
    with patch(
        "app.core.backend_client.search_listings",
        AsyncMock(return_value={"listings": [], "totalCount": 0}),
    ):
        result = asyncio.run(_do_search(MagicMock(), {"size": 5, "provinceCode": "79"}))
    assert result["status"] == "success"
    assert result["count"] == 0
    assert "hint" in result


def test_search_result_carries_canonical_share_url():
    # The compact payload must include a canonical listing URL so the model
    # shares a real link instead of fabricating a domain/path (the "chia sẻ tin"
    # bug produced "smartrent.vn/listing/..").
    from app.core.config import settings

    with patch(
        "app.core.backend_client.search_listings",
        AsyncMock(
            return_value={
                "listings": [{"listingId": 704422, "title": "Phòng trọ Q1"}],
                "totalCount": 1,
            }
        ),
    ):
        result = asyncio.run(_do_search(MagicMock(), {"size": 5, "provinceCode": "79"}))

    assert result["status"] == "success"
    assert (
        result["listings"][0]["url"] == f"{settings.FRONTEND_URL}/listing-detail/704422"
    )
