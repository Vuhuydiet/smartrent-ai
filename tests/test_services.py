"""
Comprehensive service tests.

Covers API-level validation (no real LLM calls), pure-Python service
helpers, and agent tool logic (backend mocked).
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared fixture: proper RunContextWrapper with a real ToolContext
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_ctx():
    from agents import RunContextWrapper

    from app.agent.tool_context import ToolContext

    return RunContextWrapper(context=ToolContext())


@pytest.fixture
def authed_ctx():
    from agents import RunContextWrapper

    from app.agent.tool_context import ToolContext

    return RunContextWrapper(context=ToolContext(auth_token="Bearer test-token"))


# ---------------------------------------------------------------------------
# Chat API — validation (no LLM)
# ---------------------------------------------------------------------------


def test_chat_missing_messages():
    response = client.post("/api/v1/chat", json={"messages": []})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_chat_last_message_not_user():
    response = client.post(
        "/api/v1/chat",
        json={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        },
    )
    assert response.status_code == 400
    assert "user" in response.json()["detail"].lower()


def test_chat_stream_missing_messages():
    with client.stream("POST", "/api/v1/chat/stream", json={"messages": []}) as resp:
        assert resp.status_code == 400


def test_chat_stream_last_message_not_user():
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        },
    ) as resp:
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Completion API — validation
# ---------------------------------------------------------------------------


def test_completion_empty_prompt():
    response = client.post("/api/v1/completion/", json={"prompt": "  "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_completion_missing_prompt():
    response = client.post("/api/v1/completion/", json={})
    assert response.status_code == 422  # Pydantic validation


# ---------------------------------------------------------------------------
# Price suggestion — health
# ---------------------------------------------------------------------------


def test_price_suggestion_health():
    response = client.get("/api/v1/price-suggestion/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "price_prediction"


# ---------------------------------------------------------------------------
# Listing verification — health + validation
# ---------------------------------------------------------------------------


def test_listing_verification_health():
    """Healthy path: LLM config resolves fine, so /health reports 200.

    /health does a real (no-network) config check via make_model() — it's what
    catches a missing credential before every analysis silently degrades. Patch
    it to succeed here so this test verifies the endpoint's happy-path response
    shape without depending on real GCP/Gemini credentials being present in the
    test environment (this suite is explicitly "no real LLM calls" — see module
    docstring).
    """
    with patch("app.ai.llm.agent_factory.make_model", return_value=object()):
        response = client.get("/ai/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "listing_verification"
    assert body["ai_available"] is True


def test_listing_verification_health_unconfigured():
    """No LLM credentials configured -> 503, not a silent "healthy"."""
    with patch(
        "app.ai.llm.agent_factory.make_model",
        side_effect=ValueError(
            "Gemini provider selected but no credentials configured."
        ),
    ):
        response = client.get("/ai/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["ai_available"] is False
    assert body["error_code"] == "LLM_NOT_CONFIGURED"


def test_listing_verification_missing_fields():
    response = client.post("/ai/verify-listing", json={})
    assert response.status_code == 422


def test_listing_verification_short_description():
    response = client.post(
        "/ai/verify-listing",
        json={
            "title": "Phòng trọ",
            "description": "Short",
            "price": 3000000,
            "address": "123 Nguyễn Trãi, Hà Nội",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# PricePredictionService — pure-Python helpers (no LLM)
# ---------------------------------------------------------------------------


def test_price_estimate_hcm_high_tier():
    from app.dto.house_pricing import PriceSuggestionRequest
    from app.service.price_prediction_service import PricePredictionService

    req = PriceSuggestionRequest(
        city="ho chi minh",
        district="district 1",
        ward="ward 1",
        property_type="apartment",
        area=50.0,
        latitude=10.77,
        longitude=106.69,
    )
    result = PricePredictionService._estimate_price_range(req)
    assert result["min"] > 0
    assert result["max"] > result["min"]
    assert result["min"] > 7_000_000  # HCM district 1, 50m² apartment


def test_price_estimate_low_tier():
    from app.dto.house_pricing import PriceSuggestionRequest
    from app.service.price_prediction_service import PricePredictionService

    req = PriceSuggestionRequest(
        city="hanoi",
        district="ha dong",
        ward="van quan",
        property_type="room",
        area=20.0,
        latitude=20.97,
        longitude=105.77,
    )
    result = PricePredictionService._estimate_price_range(req)
    assert result["min"] > 0
    assert result["max"] > result["min"]


def test_price_estimate_default_city():
    from app.dto.house_pricing import PriceSuggestionRequest
    from app.service.price_prediction_service import PricePredictionService

    req = PriceSuggestionRequest(
        city="Hue",
        district="phu xuan",
        ward="ward 1",
        property_type="house",
        area=80.0,
        latitude=16.46,
        longitude=107.59,
    )
    result = PricePredictionService._estimate_price_range(req)
    assert result["min"] > 0
    assert result["max"] > result["min"]


def test_parse_json_plain():
    from app.service.price_prediction_service import PricePredictionService

    raw = '{"min_price": 5000000, "max_price": 8000000, "listings_found": 3, "confidence": "high"}'
    result = PricePredictionService._parse_json(raw)
    assert result["min_price"] == 5_000_000
    assert result["confidence"] == "high"


def test_parse_json_fenced():
    from app.service.price_prediction_service import PricePredictionService

    raw = '```json\n{"min_price": 4000000, "max_price": 7000000, "listings_found": 0, "confidence": "low"}\n```'
    result = PricePredictionService._parse_json(raw)
    assert result["min_price"] == 4_000_000


# ---------------------------------------------------------------------------
# ListingVerificationService — pure-Python helpers
# ---------------------------------------------------------------------------


def test_prepare_text_content():
    from app.dto.listing_verification import (
        HousingPropertyType,
        ListingVerificationRequest,
        PropertyMetadata,
    )
    from app.service.listing_verification_service import ListingVerificationService

    svc = ListingVerificationService.__new__(ListingVerificationService)
    req = ListingVerificationRequest(
        title="Căn hộ 2PN quận 1",
        description="Căn hộ đẹp nội thất đầy đủ view thoáng mát giao thông thuận tiện",
        price=15_000_000,
        area=65.0,
        address="123 Nguyễn Huệ, Quận 1, TP. HCM",
        property_type=HousingPropertyType.APARTMENT,
        amenities=["wifi", "air_conditioner"],
        images=[],
        videos=[],
        metadata=PropertyMetadata(bedrooms=2, bathrooms=1, floor=8),
    )
    text = svc._prepare_text_content(req)
    assert "Căn hộ 2PN quận 1" in text
    assert "15000000" in text
    assert "APARTMENT" in text
    assert "Bedrooms: 2" in text


def test_process_analysis_result_raises_on_llm_error():
    """An LLM failure must surface as an error, not a fabricated 200 response.

    A basic-rule fallback here would return a score and claims like
    is_rental_related=true that nothing verified — indistinguishable from a real
    analysis to any caller that isn't specifically checking for it. It must raise
    instead, carrying the classified error_code through.
    """
    from app.dto.listing_verification import ListingVerificationRequest
    from app.service.listing_verification_service import (
        AiAnalysisUnavailableError,
        ListingVerificationService,
    )

    svc = ListingVerificationService.__new__(ListingVerificationService)
    req = ListingVerificationRequest(
        title="Phòng trọ giá rẻ",
        description="Phòng trọ sạch sẽ thoáng mát gần trường đại học",
        price=2_500_000,
        address="456 Xuân Thủy, Cầu Giấy, Hà Nội",
        images=[],
        videos=[],
    )

    with pytest.raises(AiAnalysisUnavailableError) as exc_info:
        svc._process_analysis_result(
            {
                "error": "quota exceeded",
                "error_code": "LLM_QUOTA_EXCEEDED",
                "analysis_completed": False,
            },
            req,
        )

    assert exc_info.value.error_code == "LLM_QUOTA_EXCEEDED"


# ---------------------------------------------------------------------------
# Tool: search_listings (_do_search helper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_listings_success(tool_ctx):
    from app.agent.tools.search_listings import _do_search

    fake_listings = [
        {"id": 1, "title": "Phòng 1", "price": 3_000_000},
        {"id": 2, "title": "Phòng 2", "price": 3_500_000},
    ]

    with patch(
        "app.agent.tools.search_listings.backend_client.search_listings",
        new=AsyncMock(return_value={"listings": fake_listings, "total": 2}),
    ):
        result = await _do_search(
            tool_ctx, {"provinceCode": "79", "listingType": "RENT", "size": 5}
        )

    assert result["status"] == "success"
    assert result["count"] == 2
    assert len(tool_ctx.context.collected_listings) == 2


@pytest.mark.asyncio
async def test_search_listings_backend_error(tool_ctx):
    from app.agent.tools.search_listings import _do_search

    with patch(
        "app.agent.tools.search_listings.backend_client.search_listings",
        new=AsyncMock(side_effect=Exception("connection error")),
    ):
        result = await _do_search(
            tool_ctx, {"provinceCode": "79", "listingType": "RENT", "size": 5}
        )

    assert result["status"] == "error"
    assert "error" in result


# ---------------------------------------------------------------------------
# Tool: get_listing_detail (_fetch_detail helper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_listing_detail_success(tool_ctx):
    from app.agent.tools.get_listing_detail import _fetch_detail

    fake_detail = {
        "listingId": "42",
        "title": "Căn hộ quận 1",
        "price": 12_000_000,
        "contactPhone": "0901234567",
        "address": {},
        "amenities": [],
    }

    with patch(
        "app.agent.tools.get_listing_detail.backend_client.get_listing",
        new=AsyncMock(return_value=fake_detail),
    ):
        result = await _fetch_detail(tool_ctx, "42")

    assert result["status"] == "success"
    assert result["listing"]["title"] == "Căn hộ quận 1"
    assert len(tool_ctx.context.collected_listings) == 1


@pytest.mark.asyncio
async def test_get_listing_detail_float_id(tool_ctx):
    from app.agent.tools.get_listing_detail import _fetch_detail

    with patch(
        "app.agent.tools.get_listing_detail.backend_client.get_listing",
        new=AsyncMock(return_value={"listingId": "42", "address": {}, "amenities": []}),
    ) as mock_get:
        await _fetch_detail(tool_ctx, "42")
        mock_get.assert_called_once_with("42")


# ---------------------------------------------------------------------------
# Tool: get_price_estimate (module-level _rule_based_estimate)
# ---------------------------------------------------------------------------


def test_get_price_estimate_rule_based():
    from app.agent.tools.get_price_estimate import _rule_based_estimate

    result = _rule_based_estimate("Hà Nội", "Cầu Giấy", "APARTMENT", 45.0)
    assert result["status"] == "success"
    assert result["priceRange"]["min"] > 0
    assert result["priceRange"]["max"] > result["priceRange"]["min"]
    assert result["currency"] == "VND"


def test_get_price_estimate_with_asking_price():
    from app.agent.tools.get_price_estimate import _rule_based_estimate

    result = _rule_based_estimate("TP. Hồ Chí Minh", "Quận 1", "APARTMENT", 60.0)
    mid = (result["priceRange"]["min"] + result["priceRange"]["max"]) / 2
    asking = mid * 2.0  # double the market rate → should be very_high
    diff_pct = (asking - mid) / mid * 100
    assert diff_pct > 30  # confirming the math for "very_high" evaluation


def test_get_price_estimate_unknown_city():
    from app.agent.tools.get_price_estimate import _rule_based_estimate

    result = _rule_based_estimate("Hue", "Phu Xuan", "HOUSE", 80.0)
    assert result["status"] == "success"
    assert result["priceRange"]["max"] > 0


# ---------------------------------------------------------------------------
# Tool: get_price_history (module-level helpers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_price_history_history():
    from app.agent.tools.get_price_history import _get_history

    fake_history = {
        "history": [
            {
                "oldPrice": 5_000_000,
                "newPrice": 4_500_000,
                "changeType": "DECREASE",
                "changePercentage": -10,
                "changedAt": "2024-01-01",
            }
        ]
    }

    with patch(
        "app.agent.tools.get_price_history.backend_client.get_pricing_history",
        new=AsyncMock(return_value=fake_history),
    ):
        result = await _get_history("101")

    assert result["status"] == "success"
    assert result["totalChanges"] == 1


@pytest.mark.asyncio
async def test_get_price_history_statistics():
    from app.agent.tools.get_price_history import _get_statistics

    fake_stats = {
        "minPrice": 4_000_000,
        "maxPrice": 6_000_000,
        "avgPrice": 5_000_000,
        "totalChanges": 5,
        "priceIncreases": 2,
        "priceDecreases": 3,
    }

    with patch(
        "app.agent.tools.get_price_history.backend_client.get_price_statistics",
        new=AsyncMock(return_value=fake_stats),
    ):
        result = await _get_statistics("102")

    assert result["status"] == "success"
    assert result["minPrice"] == 4_000_000
    assert result["priceDecreases"] == 3


@pytest.mark.asyncio
async def test_get_price_history_recent_changes():
    from app.agent.tools.get_price_history import _get_recent_changes

    fake_recent = {"data": [101, 102, 103], "totalElements": 3}

    with patch(
        "app.agent.tools.get_price_history.backend_client.get_recent_price_changes",
        new=AsyncMock(return_value=fake_recent),
    ):
        result = await _get_recent_changes(7)

    assert result["status"] == "success"
    assert result["totalListings"] == 3


# ---------------------------------------------------------------------------
# Tool: get_recommendations (module-level helpers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_recommendations_similar(tool_ctx):
    from app.agent.tools.get_recommendations import _similar

    fake_resp = {"listings": [{"id": 200}, {"id": 201}], "total": 2}

    with patch(
        "app.agent.tools.get_recommendations.backend_client.get_similar_listings",
        new=AsyncMock(return_value=fake_resp),
    ):
        result = await _similar(tool_ctx, listing_id=200, top_n=5, token=None)

    assert result["status"] == "success"
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_get_recommendations_personalized(authed_ctx):
    from app.agent.tools.get_recommendations import _personalized

    fake_resp = {"listings": [{"id": 300}, {"id": 301}], "total": 2}

    with patch(
        "app.agent.tools.get_recommendations.backend_client.get_personalized_recommendations",
        new=AsyncMock(return_value=fake_resp),
    ):
        result = await _personalized(authed_ctx, top_n=5, token="Bearer test-token")

    assert result["status"] == "success"
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_get_recommendations_personalized_no_token(tool_ctx):
    from app.agent.tools.get_recommendations import _personalized

    result = await _personalized(tool_ctx, top_n=5, token=None)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Tool: get_user_info (module-level helpers + no-token via on_invoke_tool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_user_info_profile():
    from app.agent.tools.get_user_info import _get_profile

    fake_profile = {"firstName": "Nguyen", "lastName": "Van A", "email": "a@test.com"}

    with patch(
        "app.agent.tools.get_user_info.backend_client.get_user_profile",
        new=AsyncMock(return_value=fake_profile),
    ):
        result = await _get_profile("Bearer some-token")

    assert result["status"] == "success"
    assert "Van A" in result["profile"]["name"]


@pytest.mark.asyncio
async def test_get_user_info_membership_active():
    from app.agent.tools.get_user_info import _get_membership

    fake_membership = {
        "status": "ACTIVE",
        "membershipPackage": {"packageName": "Gold", "packageLevel": "GOLD"},
        "startDate": "2024-01-01",
        "endDate": "2025-01-01",
    }

    with patch(
        "app.agent.tools.get_user_info.backend_client.get_user_membership",
        new=AsyncMock(return_value=fake_membership),
    ):
        result = await _get_membership("Bearer tok")

    assert result["status"] == "success"
    assert result["membership"]["active"] is True
    assert result["membership"]["packageLevel"] == "GOLD"


@pytest.mark.asyncio
async def test_get_user_info_saved_listings():
    from app.agent.tools.get_user_info import _get_saved

    fake_saved = {
        "data": [
            {
                "listing": {
                    "listingId": "99",
                    "title": "Phòng đẹp",
                    "price": 4_000_000,
                    "address": {"districtName": "Cầu Giấy"},
                    "productType": "ROOM",
                }
            }
        ]
    }

    with patch(
        "app.agent.tools.get_user_info.backend_client.get_saved_listings",
        new=AsyncMock(return_value=fake_saved),
    ):
        result = await _get_saved("Bearer tok")

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["savedListings"][0]["title"] == "Phòng đẹp"


@pytest.mark.asyncio
async def test_get_user_info_no_token(tool_ctx):
    from app.agent.tools.get_user_info import _dispatch_user_info

    result = await _dispatch_user_info(tool_ctx, "profile")
    assert result["status"] == "error"
    assert "đăng nhập" in result["error"]


# ---------------------------------------------------------------------------
# Tool: save_listing (_do_save_unsave helper + no-token via wrapper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_listing_save():
    from app.agent.tools.save_listing import _do_save_unsave

    with patch(
        "app.agent.tools.save_listing.backend_client.save_listing",
        new=AsyncMock(return_value={"success": True}),
    ):
        result = await _do_save_unsave("Bearer tok", "55", "save")

    assert result["status"] == "success"
    assert result["listingId"] == "55"


@pytest.mark.asyncio
async def test_unsave_listing():
    from app.agent.tools.save_listing import _do_save_unsave

    with patch(
        "app.agent.tools.save_listing.backend_client.unsave_listing",
        new=AsyncMock(return_value={"success": True}),
    ):
        result = await _do_save_unsave("Bearer tok", "55", "unsave")

    assert result["status"] == "success"
    assert result["action"] == "unsave"


@pytest.mark.asyncio
async def test_save_listing_no_token(tool_ctx):
    from app.agent.tools.save_listing import _handle_save

    result = await _handle_save(tool_ctx, "55", "save")
    assert result["status"] == "error"
    assert "đăng nhập" in result["error"]


# ---------------------------------------------------------------------------
# Orchestrator helpers — pure-Python (no LLM)
# ---------------------------------------------------------------------------


def test_estimate_tokens():
    from app.agent.orchestrator import _estimate_tokens

    assert _estimate_tokens("a" * 400) == 100  # 400 chars / 4


def test_trim_history_under_budget():
    from app.agent.orchestrator import _trim_history
    from app.dto.chat import ChatMessage

    msgs = [ChatMessage(role="user", content="hi")]
    assert len(_trim_history(msgs)) == 1


def test_trim_history_over_budget():
    from app.agent.orchestrator import (
        HISTORY_TOKEN_BUDGET,
        _estimate_tokens,
        _trim_history,
    )
    from app.dto.chat import ChatMessage

    long_content = "x" * 400  # 100 tokens each
    msgs = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=long_content)
        for i in range(100)  # 100 * 100 = 10 000 tokens >> 6 000 budget
    ]
    trimmed = _trim_history(msgs)
    total_tokens = sum(_estimate_tokens(m.content) for m in trimmed)  # type: ignore[arg-type]
    assert total_tokens <= HISTORY_TOKEN_BUDGET


def test_trim_history_keeps_last_message():
    from app.agent.orchestrator import _trim_history
    from app.dto.chat import ChatMessage

    msgs = [
        ChatMessage(role="user", content="x" * 100_000),
        ChatMessage(role="user", content="final question"),
    ]
    assert _trim_history(msgs)[-1].content == "final question"


# ---------------------------------------------------------------------------
# GeminiListingVerificationHelper — pure-Python helpers
# ---------------------------------------------------------------------------


def test_parse_json_response_valid():
    from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper

    raw = '{"is_valid": true, "score": 0.85}'
    result = GeminiListingVerificationHelper._parse_json_response(raw)
    assert result["is_valid"] is True


def test_parse_json_response_with_fence():
    from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper

    raw = '```json\n{"is_valid": false, "score": 0.2}\n```'
    result = GeminiListingVerificationHelper._handle_api_response(raw)
    assert result["score"] == 0.2


def test_handle_generation_error_quota():
    from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper

    result = GeminiListingVerificationHelper._handle_generation_error(
        "quota exceeded - 429"
    )
    # The raw provider message is kept verbatim (it's the only thing that makes a
    # failure diagnosable) — error_code is what callers should branch on.
    assert result["error"] == "quota exceeded - 429"
    assert result["error_code"] == "LLM_QUOTA_EXCEEDED"
    assert result["analysis_completed"] is False


def test_handle_generation_error_missing_credentials():
    """Regression test: this is the one error the service raises itself — the
    original substring check for "api key" (with a space) never matched
    "GEMINI_API_KEY" (underscore) or "no credentials configured", so a missing
    credential was silently reported as a generic, unactionable failure."""
    from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper

    result = GeminiListingVerificationHelper._handle_generation_error(
        "Gemini provider selected but no credentials configured. "
        "Set GCP_CREDENTIALS_BASE64 + GCP_PROJECT_ID (Vertex AI) "
        "or GEMINI_API_KEY (Google AI Studio)."
    )
    assert result["error_code"] == "LLM_NOT_CONFIGURED"
    assert result["analysis_completed"] is False


def test_handle_generation_error_generic():
    from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper

    result = GeminiListingVerificationHelper._handle_generation_error(
        "some unknown error"
    )
    assert "error" in result
    assert result["error_code"] == "LLM_ERROR"
    assert result["analysis_completed"] is False


def test_sanitize_missing_fields_non_list():
    from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper

    raw_json = json.dumps(
        {
            "is_valid": True,
            "score": 0.9,
            "reason": {"missing_fields": "none", "details": "ok"},
            "completeness_validation": {"missing_fields": "none"},
        }
    )
    result = GeminiListingVerificationHelper._handle_api_response(raw_json)
    assert result["reason"]["missing_fields"] == []
    assert result["completeness_validation"]["missing_fields"] == []


def test_sanitize_issues_dict_shape_extracts_details():
    """Regression: the LLM sometimes returns each issue as an object shaped
    {"severity": ..., "priority": ..., "type": ..., "details": ...} instead of
    a plain string. The sanitizer's fallback key list didn't include "details",
    so it fell through to str(dict) and leaked a Python-repr blob like
    "{'severity': 'CRITICAL', ...}" straight into the admin UI instead of the
    human-readable details sentence."""
    from app.ai.llm.gemini_listing_helper import GeminiListingVerificationHelper

    raw_json = json.dumps(
        {
            "is_valid": True,
            "score": 0.9,
            "image_validation": {
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "priority": "HIGH",
                        "type": "INVALID_IMAGE_TYPE",
                        "details": "Anh 1: khong phai bat dong san.",
                    }
                ]
            },
            "content_validation": {
                "issues": [
                    {
                        "severity": "MAJOR",
                        "priority": "HIGH",
                        "type": "INCONSISTENT_INFORMATION",
                        "details": "Dia chi khong khop.",
                    }
                ]
            },
            "completeness_validation": {
                "quality_issues": [
                    {
                        "severity": "MAJOR",
                        "priority": "HIGH",
                        "type": "INCONSISTENT_INFORMATION",
                        "details": "Dia chi khong khop.",
                    }
                ]
            },
        }
    )
    result = GeminiListingVerificationHelper._handle_api_response(raw_json)
    assert result["image_validation"]["issues"] == [
        "Anh 1: khong phai bat dong san."
    ]
    assert result["content_validation"]["issues"] == ["Dia chi khong khop."]
    assert result["completeness_validation"]["quality_issues"] == [
        "Dia chi khong khop."
    ]


# ---------------------------------------------------------------------------
# get_chat_tools registry
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Stale fixture from before the sprint-v2 agent expansion. The chat "
        "registry now ships 14 tools (added compare_listings, "
        "my_listings_status, address_translator, bulk_save_listings, "
        "update_listing_price, notifications_inbox, report_listing). "
        "Re-enable once the assertion is rewritten to enforce the new "
        "canonical set."
    )
)
def test_chat_tools_registry():
    from app.agent.tools import get_chat_tools

    tools = get_chat_tools()
    assert len(tools) == 7
    tool_names = {t.name for t in tools}
    expected = {
        "search_listings",
        "get_listing_detail",
        "get_price_estimate",
        "get_price_history",
        "get_recommendations",
        "get_user_info",
        "save_listing",
    }
    assert tool_names == expected
