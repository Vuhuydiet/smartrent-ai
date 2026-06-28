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
    for value in ["ROOM", "APARTMENT", "RENT", "SHARE",
                  "PRICE_ASC", "FULLY_FURNISHED", "NORTHEAST"]:
        assert value in blob


def test_do_search_surfaces_backend_error_body():
    # When the backend rejects a param (e.g. unknown districtCode → 400 with a
    # message), the tool must forward that message to the model so it can
    # self-correct — not collapse it into a bare "HTTP 400".
    request = httpx.Request("POST", "http://backend/v1/listings/search")
    response = httpx.Response(
        400,
        json={"code": "2011",
              "message": "Unknown district code '999'. Provide a valid GSO district code."},
        request=request,
    )
    err = httpx.HTTPStatusError("400 Bad Request", request=request, response=response)
    with patch("app.core.backend_client.search_listings", AsyncMock(side_effect=err)):
        result = asyncio.run(_do_search(MagicMock(), {"size": 5, "districtCode": "999"}))
    assert result["status"] == "error"
    assert "Unknown district code" in result["error"]
