"""Tests for the auth guard and response cache on POST /price-suggestion.

`app.api.v1.price_suggestion` owns a process-global cache (single uvicorn
worker, no Redis) — every test resets it via the autouse fixture below so
tests don't leak cache entries into each other.
"""

import time

import pytest

from fastapi.testclient import TestClient

import app.api.v1.price_suggestion as price_suggestion
from app.core.config import settings
from app.dto.house_pricing import PriceSuggestionRequest, PriceSuggestionResponse
from app.main import app

client = TestClient(app)
URL = "/api/v1/price-suggestion/get-price-suggestion"


def _request(**overrides) -> PriceSuggestionRequest:
    payload = {
        "city": "Hà Nội",
        "district": "Hoàn Kiếm",
        "ward": "Phường Hàng Bạc",
        "property_type": "APARTMENT",
        "area": 50.0,
        "latitude": 21.0333,
        "longitude": 105.85,
    }
    payload.update(overrides)
    return PriceSuggestionRequest(**payload)


def _payload(**overrides) -> dict:
    payload = {
        "city": "Hà Nội",
        "district": "Hoàn Kiếm",
        "ward": "Phường Hàng Bạc",
        "property_type": "APARTMENT",
        "area": 50.0,
        "latitude": 21.0333,
        "longitude": 105.85,
    }
    payload.update(overrides)
    return payload


def _response(source="ai_comparables", **overrides) -> PriceSuggestionResponse:
    payload = {
        "price_range": {"min": 5_000_000, "max": 8_000_000},
        "location": "Hoàn Kiếm, Hà Nội",
        "property_type": "APARTMENT",
        "currency": "VND",
        "source": source,
        "listings_found": 4,
        "confidence": "medium",
    }
    payload.update(overrides)
    return PriceSuggestionResponse(**payload)


class _FakeService:
    """Stand-in for PricePredictionService that counts calls instead of
    actually running the OpenAI-Agents-SDK agent."""

    def __init__(self, response: PriceSuggestionResponse):
        self.response = response
        self.call_count = 0

    async def predict_price(
        self, request: PriceSuggestionRequest
    ) -> PriceSuggestionResponse:
        self.call_count += 1
        return self.response


def _use_fake_service(response: PriceSuggestionResponse) -> _FakeService:
    fake = _FakeService(response)
    app.dependency_overrides[
        price_suggestion.get_price_prediction_service
    ] = lambda: fake
    return fake


@pytest.fixture(autouse=True)
def _isolate_cache_and_overrides():
    price_suggestion._cache = None
    price_suggestion._cache_config = None
    yield
    app.dependency_overrides.pop(price_suggestion.get_price_prediction_service, None)
    price_suggestion._cache = None
    price_suggestion._cache_config = None


class TestAuthGuard:
    def test_rejects_when_internal_key_configured_and_missing(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "s3cret")
        _use_fake_service(_response())
        resp = client.post(URL, json=_payload())
        assert resp.status_code == 401

    def test_accepts_matching_internal_key(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "s3cret")
        _use_fake_service(_response())
        resp = client.post(
            URL, json=_payload(), headers={"X-Internal-Api-Key": "s3cret"}
        )
        assert resp.status_code == 200

    def test_open_when_internal_key_unset(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "")
        _use_fake_service(_response())
        resp = client.post(URL, json=_payload())
        assert resp.status_code == 200


class TestCacheBehavior:
    def test_second_identical_request_does_not_rerun_the_agent(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "")
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_TTL_SECONDS", 3600)
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_MAXSIZE", 100)
        fake = _use_fake_service(_response())

        first = client.post(URL, json=_payload())
        second = client.post(URL, json=_payload())

        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert fake.call_count == 1

    def test_rule_based_fallback_is_not_cached(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "")
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_TTL_SECONDS", 3600)
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_MAXSIZE", 100)
        fake = _use_fake_service(_response(source="rule_based_fallback"))

        client.post(URL, json=_payload())
        client.post(URL, json=_payload())

        assert fake.call_count == 2

    def test_ttl_expiry_forces_a_fresh_agent_run(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "")
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_TTL_SECONDS", 0.1)
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_MAXSIZE", 100)
        fake = _use_fake_service(_response())

        client.post(URL, json=_payload())
        time.sleep(0.25)
        client.post(URL, json=_payload())

        assert fake.call_count == 2

    def test_maxsize_setting_is_actually_wired(self, monkeypatch):
        """maxsize=1 forces eviction on the second distinct key, so the
        original request must miss (and re-run the agent) the third time."""
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "")
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_TTL_SECONDS", 3600)
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_MAXSIZE", 1)
        fake = _use_fake_service(_response())

        client.post(URL, json=_payload())
        client.post(URL, json=_payload(district="Hà Đông"))
        client.post(URL, json=_payload())

        assert fake.call_count == 3

    def test_cache_rebuilds_when_settings_change_at_request_time(self, monkeypatch):
        """Settings must be read per-request (not captured at import) so a
        runtime config change actually takes effect."""
        monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "")
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_TTL_SECONDS", 3600)
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_MAXSIZE", 100)
        fake = _use_fake_service(_response())

        client.post(URL, json=_payload())
        assert fake.call_count == 1

        # Changing maxsize rebuilds the underlying TTLCache, dropping the
        # entry cached under the old config.
        monkeypatch.setattr(settings, "PRICE_SUGGESTION_CACHE_MAXSIZE", 200)
        client.post(URL, json=_payload())
        assert fake.call_count == 2


class TestCacheKeyNormalization:
    def test_diacritics_whitespace_and_case_collide(self):
        a = price_suggestion._cache_key(_request())
        b = price_suggestion._cache_key(
            _request(
                city=" hà nội ",
                district=" HOÀN KIẾM ",
                ward=" phường hàng bạc ",
                property_type=" apartment ",
            )
        )
        assert a == b

    def test_ascii_spelling_does_not_collide_with_diacritics(self):
        """Normalization only folds case/whitespace/diacritic-form, not
        transliteration — 'Hanoi' and 'Hà Nội' are different strings."""
        a = price_suggestion._cache_key(_request())
        b = price_suggestion._cache_key(_request(city="Hanoi", district="Hoan Kiem"))
        assert a != b

    def test_coordinates_round_to_the_same_bucket(self):
        a = price_suggestion._cache_key(
            _request(latitude=21.03331, longitude=105.85002)
        )
        b = price_suggestion._cache_key(
            _request(latitude=21.03334, longitude=105.84999)
        )
        assert a == b

    def test_coordinates_far_enough_apart_do_not_collide(self):
        a = price_suggestion._cache_key(_request())
        b = price_suggestion._cache_key(_request(latitude=21.05, longitude=105.90))
        assert a != b

    def test_area_buckets_nearby_values_together(self):
        a = price_suggestion._cache_key(_request(area=50.0))
        b = price_suggestion._cache_key(_request(area=52.0))
        assert a == b

    def test_area_far_apart_values_do_not_collide(self):
        a = price_suggestion._cache_key(_request(area=30.0))
        b = price_suggestion._cache_key(_request(area=90.0))
        assert a != b

    def test_area_none_is_distinct_from_any_numeric_bucket(self):
        with_area = price_suggestion._cache_key(_request(area=0.1))
        without_area = price_suggestion._cache_key(_request(area=None))
        assert with_area != without_area
        assert with_area[4] == 0.0
        assert without_area[4] is None
