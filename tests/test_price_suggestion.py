"""Tests for the price-suggestion response contract and rule-based fallback."""

import pytest

from app.dto.house_pricing import PriceSuggestionRequest
from app.service.price_prediction_service import PricePredictionService


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


class TestValidatedRange:
    def test_accepts_a_plausible_range(self):
        assert PricePredictionService._validated_range(
            {"min_price": 5_000_000, "max_price": 8_000_000}
        ) == {"min": 5_000_000, "max": 8_000_000}

    def test_reorders_an_inverted_range(self):
        assert PricePredictionService._validated_range(
            {"min_price": 8_000_000, "max_price": 5_000_000}
        ) == {"min": 5_000_000, "max": 8_000_000}

    @pytest.mark.parametrize(
        "result",
        [
            {},
            {"min_price": 5_000_000},
            {"min_price": None, "max_price": 8_000_000},
            {"min_price": "abc", "max_price": "def"},
            # Answered in millions instead of VND.
            {"min_price": 5, "max_price": 8},
            # Absurdly large.
            {"min_price": 1, "max_price": 9_000_000_000_000},
            {"min_price": -1000, "max_price": 8_000_000},
        ],
    )
    def test_rejects_unusable_output(self, result):
        with pytest.raises(ValueError):
            PricePredictionService._validated_range(result)


class TestCountUniqueListings:
    def test_deduplicates_across_repeated_searches(self):
        listings = [{"id": 1}, {"id": 2}, {"id": 1}]
        assert PricePredictionService._count_unique_listings(listings) == 2

    def test_counts_entries_without_an_id(self):
        listings = [{"id": 1}, {"title": "no id"}, {"title": "also no id"}]
        assert PricePredictionService._count_unique_listings(listings) == 3

    def test_empty(self):
        assert PricePredictionService._count_unique_listings([]) == 0


class TestResolveConfidence:
    def test_no_evidence_forces_low(self):
        assert PricePredictionService._resolve_confidence("high", 0) == "low"

    def test_high_requires_enough_comparables(self):
        assert PricePredictionService._resolve_confidence("high", 3) == "medium"
        assert PricePredictionService._resolve_confidence("high", 5) == "high"

    def test_garbage_defaults_to_medium(self):
        assert PricePredictionService._resolve_confidence("very sure", 10) == "medium"
        assert PricePredictionService._resolve_confidence(None, 10) == "medium"

    def test_passes_through_valid_values(self):
        assert PricePredictionService._resolve_confidence("low", 10) == "low"
        assert PricePredictionService._resolve_confidence("MEDIUM", 10) == "medium"


class TestRuleBasedFallback:
    def test_matches_vietnamese_city_names_with_diacritics(self):
        """The frontend sends 'Hà Nội', not 'hanoi' — the old ASCII-only table
        silently fell through to the generic default for every real request."""
        hanoi = PricePredictionService._estimate_price_range(_request())
        generic = PricePredictionService._estimate_price_range(
            _request(city="Nowhere Province", district="Nowhere District")
        )
        assert hanoi["min"] != generic["min"]

    def test_matches_district_tier_with_diacritics(self):
        high = PricePredictionService._estimate_price_range(
            _request(district="Hoàn Kiếm")
        )
        low = PricePredictionService._estimate_price_range(_request(district="Hà Đông"))
        assert high["min"] > low["min"]

    def test_ascii_spelling_still_matches(self):
        with_diacritics = PricePredictionService._estimate_price_range(_request())
        ascii_spelling = PricePredictionService._estimate_price_range(
            _request(city="Hanoi", district="Hoan Kiem")
        )
        assert with_diacritics == ascii_spelling

    def test_hcmc_is_priced_above_hanoi(self):
        hanoi = PricePredictionService._estimate_price_range(_request())
        hcmc = PricePredictionService._estimate_price_range(
            _request(city="Thành phố Hồ Chí Minh", district="Quận 1")
        )
        assert hcmc["min"] > hanoi["min"]

    def test_range_is_ordered_and_scales_with_area(self):
        small = PricePredictionService._estimate_price_range(_request(area=30))
        large = PricePredictionService._estimate_price_range(_request(area=90))
        assert small["min"] < small["max"]
        assert large["min"] > small["min"]

    def test_missing_area_uses_a_default(self):
        result = PricePredictionService._estimate_price_range(_request(area=None))
        assert result["min"] > 0
