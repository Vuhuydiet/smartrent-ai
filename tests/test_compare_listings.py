"""Tests for compare_listings — a whole-result-set operation with no arguments.

The model may decide *whether* to compare; it may never decide *which* listings
get compared. That keeps listing IDs out of the conversation entirely.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from agents import RunContextWrapper  # type: ignore[import]

from app.agent.tool_context import ToolContext
from app.agent.tools.compare_listings import (
    _MAX_LISTINGS,
    _do_compare,
    _resolve_target_ids,
    compare_listings,
)


def _ctx(collected=None, last_ids=None) -> RunContextWrapper[ToolContext]:
    return RunContextWrapper(
        ToolContext(
            collected_listings=[{"listingId": i} for i in (collected or [])],
            last_listing_ids=[str(i) for i in (last_ids or [])],
        )
    )


def _listing(listing_id, price=5_000_000, area=20.0):
    return {
        "listingId": listing_id,
        "title": f"Tin {listing_id}",
        "price": price,
        "area": area,
        "address": {"districtName": "Bình Thạnh"},
        "amenities": [{"name": "WiFi"}],
    }


# ---------------------------------------------------------------------------
# Schema — the model has no way to pick listings
# ---------------------------------------------------------------------------


def test_tool_takes_no_listing_ids():
    # The old signature accepted listingIds, which is what let the model
    # compare an arbitrary pair. Removing the parameter makes subset
    # comparison structurally impossible, not merely discouraged.
    props = compare_listings.params_json_schema.get("properties", {})
    assert "listingIds" not in props
    assert props == {} or not compare_listings.params_json_schema.get("required")


# ---------------------------------------------------------------------------
# Target-set resolution
# ---------------------------------------------------------------------------


def test_resolves_from_current_turn_when_search_just_ran():
    ids, source = _resolve_target_ids(_ctx(collected=[1, 2, 3], last_ids=[9, 8]))
    assert ids == ["1", "2", "3"]
    assert source == "current_turn"


def test_falls_back_to_previous_turn_result_set():
    ids, source = _resolve_target_ids(_ctx(collected=[], last_ids=[9, 8]))
    assert ids == ["9", "8"]
    assert source == "previous_turn"


def test_single_collected_listing_falls_back_to_previous_turn():
    # A lone get_listing_detail earlier in the turn must not shrink the
    # comparison to one row — the user asked to compare the set they saw.
    ids, source = _resolve_target_ids(_ctx(collected=[1], last_ids=[9, 8, 7]))
    assert ids == ["9", "8", "7"]
    assert source == "previous_turn"


def test_dedupes_and_normalises_ids():
    ctx = _ctx(collected=[35201, "35201", 35202.0, None, ""])
    ids, _ = _resolve_target_ids(ctx)
    assert ids == ["35201", "35202"]


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_errors_helpfully_when_nothing_to_compare():
    result = asyncio.run(_do_compare(_ctx()))
    assert result["status"] == "error"
    # The model must ask the user to search, never to supply an ID.
    assert "tìm kiếm" in result["error"].lower()
    assert "listingid" not in result["error"].lower()


def test_compares_every_listing_in_the_result_set():
    with patch(
        "app.core.backend_client.get_listing",
        AsyncMock(side_effect=lambda lid: _listing(lid)),
    ):
        result = asyncio.run(_do_compare(_ctx(last_ids=[1, 2, 3, 4])))
    assert result["status"] == "success"
    assert result["count"] == 4
    assert [row["listingId"] for row in result["listings"]] == ["1", "2", "3", "4"]
    assert not result.get("truncated")


def test_flags_truncation_rather_than_silently_dropping_listings():
    over_cap = list(range(1, _MAX_LISTINGS + 3))
    with patch(
        "app.core.backend_client.get_listing",
        AsyncMock(side_effect=lambda lid: _listing(lid)),
    ):
        result = asyncio.run(_do_compare(_ctx(last_ids=over_cap)))
    assert result["count"] == _MAX_LISTINGS
    assert result["truncated"] is True
    assert result["availableCount"] == len(over_cap)


def test_partial_fetch_failure_still_compares_the_rest():
    async def _fetch(lid):
        if lid == "2":
            raise RuntimeError("backend down")
        return _listing(lid)

    with patch("app.core.backend_client.get_listing", AsyncMock(side_effect=_fetch)):
        result = asyncio.run(_do_compare(_ctx(last_ids=[1, 2, 3])))
    assert result["status"] == "success"
    assert [row["listingId"] for row in result["listings"]] == ["1", "3"]
    assert [e["listingId"] for e in result["errors"]] == ["2"]


def test_callouts_are_grounded_in_the_fetched_rows():
    prices = {"1": 9_000_000, "2": 4_000_000, "3": 6_000_000}
    with patch(
        "app.core.backend_client.get_listing",
        AsyncMock(side_effect=lambda lid: _listing(lid, price=prices[lid])),
    ):
        result = asyncio.run(_do_compare(_ctx(last_ids=[1, 2, 3])))
    assert result["callouts"]["cheapest"] == "2"
    assert result["callouts"]["priceRangeVnd"] == [4_000_000, 9_000_000]
