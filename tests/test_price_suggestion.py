"""Tests for the price-suggestion response contract and rule-based fallback."""

from typing import Any

import pytest

from app.dto.house_pricing import PriceSuggestionRequest
from app.service.price_prediction_service import PricePredictionService


def _request(**overrides: Any) -> PriceSuggestionRequest:
    payload: dict[str, Any] = {
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


class TestRangeFromStats:
    def test_uses_interquartile_band(self):
        stats = {"min": 3_000_000, "p25": 4_000_000, "p75": 6_000_000, "max": 9_000_000}
        assert PricePredictionService._range_from_stats(stats) == {
            "min": 4_000_000,
            "max": 6_000_000,
        }

    def test_falls_back_to_min_max_when_quartiles_absent(self):
        stats = {"min": 3_000_000, "max": 9_000_000}
        assert PricePredictionService._range_from_stats(stats) == {
            "min": 3_000_000,
            "max": 9_000_000,
        }

    def test_spreads_a_degenerate_band_so_min_never_equals_max(self):
        stats = {"p25": 5_000_000, "p75": 5_000_000}
        result = PricePredictionService._range_from_stats(stats)
        assert result["min"] < result["max"]

    @pytest.mark.parametrize(
        "stats",
        [
            {},
            {"p25": None, "p75": None},
            # Aggregates in millions instead of VND.
            {"p25": 5, "p75": 8},
            # Absurdly large.
            {"p25": 1, "p75": 9_000_000_000_000},
        ],
    )
    def test_rejects_unusable_stats(self, stats):
        with pytest.raises(ValueError):
            PricePredictionService._range_from_stats(stats)


class TestBestStats:
    def test_picks_the_query_with_the_largest_sample(self):
        calls = [
            {"sampleSize": 3, "median": 4_000_000},
            {"sampleSize": 25, "median": 5_000_000},
            {"sampleSize": 0},
        ]
        assert PricePredictionService._best_stats(calls)["sampleSize"] == 25

    def test_raises_when_no_query_found_comparables(self):
        with pytest.raises(ValueError):
            PricePredictionService._best_stats([{"sampleSize": 0}, {"error": "x"}])


class TestConfidenceFromStats:
    def test_no_evidence_is_low(self):
        assert PricePredictionService._confidence_from_stats({"sampleSize": 0}) == "low"

    def test_large_tight_sample_is_high(self):
        stats = {
            "sampleSize": 30,
            "p25": 4_500_000,
            "median": 5_000_000,
            "p75": 5_500_000,
        }
        assert PricePredictionService._confidence_from_stats(stats) == "high"

    def test_moderate_sample_is_medium(self):
        stats = {
            "sampleSize": 10,
            "p25": 4_000_000,
            "median": 5_000_000,
            "p75": 6_500_000,
        }
        assert PricePredictionService._confidence_from_stats(stats) == "medium"

    def test_wide_dispersion_drops_to_low(self):
        stats = {
            "sampleSize": 30,
            "p25": 2_000_000,
            "median": 5_000_000,
            "p75": 9_000_000,
        }
        assert PricePredictionService._confidence_from_stats(stats) == "low"


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
