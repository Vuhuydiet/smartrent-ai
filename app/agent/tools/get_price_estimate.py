"""
Tool: get_price_estimate

Estimates the fair market rental price for a property using the XGBoost
price predictor model. Falls back to a rule-based market estimate when the
trained model file is not available.

Model loading strategy
----------------------
The XGBoost model (RealEstatePricePredictorModel) is expensive to train (~minutes).
On startup this tool attempts to load a pre-saved pickle from the path configured
in PRICE_MODEL_PATH. If the file does not exist the tool transparently uses a
lightweight rule-based fallback — the agent still works, just with lower accuracy.

To train and save the model run (one-time):
    python -m app.ai.house_pricing.train_and_save
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from google.genai import types  # type: ignore[import]

from app.agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton for the XGBoost model
# ---------------------------------------------------------------------------
_predictor: Optional[Any] = None
_predictor_loaded = False  # True once we have attempted to load (even if it failed)


def _get_predictor() -> Optional[Any]:
    """
    Return the singleton RealEstatePricePredictorModel, or None if unavailable.
    Load attempt happens once; subsequent calls return the cached result.
    """
    global _predictor, _predictor_loaded
    if _predictor_loaded:
        return _predictor

    _predictor_loaded = True
    model_path = os.environ.get("PRICE_MODEL_PATH", "")
    if not model_path or not os.path.exists(model_path):
        logger.info(
            "PRICE_MODEL_PATH not set or file not found — using rule-based price fallback."
        )
        return None

    try:
        from app.ai.house_pricing.price_predictor import RealEstatePricePredictorModel

        m = RealEstatePricePredictorModel()
        m.load_model(model_path)
        _predictor = m
        logger.info("Price predictor model loaded from %s", model_path)
    except Exception as e:
        logger.warning("Failed to load price predictor model: %s", e)

    return _predictor


# ---------------------------------------------------------------------------
# Rule-based fallback (market estimates — same logic as PricePredictionService)
# ---------------------------------------------------------------------------

# Monthly rent per m² in VND, keyed by city keyword → tier
_CITY_RENT: Dict[str, Dict[str, int]] = {
    "hà nội": {"high": 220_000, "medium": 160_000, "low": 110_000},
    "hanoi": {"high": 220_000, "medium": 160_000, "low": 110_000},
    "hồ chí minh": {"high": 270_000, "medium": 190_000, "low": 130_000},
    "ho chi minh": {"high": 270_000, "medium": 190_000, "low": 130_000},
    "đà nẵng": {"high": 190_000, "medium": 140_000, "low": 90_000},
    "da nang": {"high": 190_000, "medium": 140_000, "low": 90_000},
}
_DEFAULT_RENT = {"high": 180_000, "medium": 130_000, "low": 90_000}

_HIGH_TIER_DISTRICTS = {
    "hoàn kiếm",
    "ba đình",
    "tây hồ",  # Hanoi premium
    "quận 1",
    "quận 3",
    "bình thạnh",  # HCM premium
    "hải châu",  # Da Nang premium
}
_LOW_TIER_DISTRICTS = {
    "hà đông",
    "thanh trì",
    "gia lâm",  # Hanoi fringe
    "thủ đức",
    "bình tân",
    "gò vấp",  # HCM fringe
}

_PROPERTY_MULTIPLIERS: Dict[str, float] = {
    "room": 0.75,
    "apartment": 1.0,
    "studio": 0.9,
    "house": 1.15,
    "office": 1.25,
}


def _rule_based_estimate(
    city: str,
    district: str,
    property_type: str,
    area: float,
) -> Dict[str, Any]:
    city_lower = city.lower()
    district_lower = district.lower()
    ptype_lower = property_type.lower()

    city_rents = _DEFAULT_RENT
    for key, rents in _CITY_RENT.items():
        if key in city_lower:
            city_rents = rents
            break

    tier = "medium"
    for d in _HIGH_TIER_DISTRICTS:
        if d in district_lower:
            tier = "high"
            break
    else:
        for d in _LOW_TIER_DISTRICTS:
            if d in district_lower:
                tier = "low"
                break

    rent_per_m2 = city_rents[tier]
    multiplier = next(
        (v for k, v in _PROPERTY_MULTIPLIERS.items() if k in ptype_lower), 1.0
    )
    monthly_rent = rent_per_m2 * multiplier * area

    return {
        "status": "success",
        "source": "rule_based_estimate",
        "estimatedMonthlyRent": int(monthly_rent),
        "priceRange": {
            "min": int(monthly_rent * 0.80),
            "max": int(monthly_rent * 1.25),
        },
        "pricePerM2": int(rent_per_m2 * multiplier),
        "currency": "VND",
        "confidence": "low",
        "note": (
            "Estimate based on market averages. "
            "Actual price depends on condition, floor, and amenities."
        ),
    }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class GetPriceEstimateTool(BaseTool):
    name = "get_price_estimate"
    description = (
        "Estimate the fair market monthly rental price for a property based on "
        "its location, type, and size. Use this when the user asks whether a price "
        "is reasonable, wants to know the going rate for a property, or wants to "
        "compare a listing price against market expectations."
    )

    def to_function_declaration(self) -> Any:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters={  # type: ignore[arg-type]
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City/province name in Vietnamese or English "
                            "(e.g. 'Hà Nội', 'TP. Hồ Chí Minh', 'Đà Nẵng')."
                        ),
                    },
                    "district": {
                        "type": "string",
                        "description": "District name (e.g. 'Cầu Giấy', 'Quận 1').",
                    },
                    "ward": {
                        "type": "string",
                        "description": "Ward name (optional).",
                    },
                    "propertyType": {
                        "type": "string",
                        "description": "ROOM, APARTMENT, HOUSE, STUDIO, or OFFICE.",
                    },
                    "area": {
                        "type": "number",
                        "description": "Property area in m².",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude (improves ML model accuracy).",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude (improves ML model accuracy).",
                    },
                    "askingPrice": {
                        "type": "number",
                        "description": (
                            "Optional: the price being asked (VND/month). "
                            "When provided, the tool also evaluates whether the price is fair."
                        ),
                    },
                },
                "required": ["city", "district", "propertyType", "area"],
            },
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:  # noqa: C901
        city: str = kwargs["city"]
        district: str = kwargs["district"]
        propertyType: str = kwargs["propertyType"]  # noqa: N806
        area: float = float(kwargs["area"])
        ward: str = kwargs.get("ward", "")
        latitude: Optional[float] = kwargs.get("latitude")
        longitude: Optional[float] = kwargs.get("longitude")
        askingPrice: Optional[float] = kwargs.get("askingPrice")  # noqa: N806
        predictor = _get_predictor()

        # --- ML model path ---------------------------------------------------
        if predictor is not None and latitude is not None and longitude is not None:
            try:
                now = datetime.now()
                property_data = {
                    "city": city,
                    "district": district,
                    "ward": ward,
                    "property_type": propertyType,
                    "latitude": latitude,
                    "longitude": longitude,
                    "post_date": now.strftime("%Y-%m-%d"),
                }
                prediction = predictor.predict_price_range(property_data)
                result: Dict[str, Any] = {
                    "status": "success",
                    "source": "ml_model",
                    "estimatedMonthlyRent": int(
                        prediction["predicted_price"] * 1_000_000
                    ),
                    "priceRange": {
                        "min": int(prediction["price_range"]["min"] * 1_000_000),
                        "max": int(prediction["price_range"]["max"] * 1_000_000),
                    },
                    "currency": "VND",
                    "confidence": "high",
                }

                if askingPrice is not None:
                    evaluation = predictor.evaluate_price_vs_market(
                        property_data, askingPrice / 1_000_000
                    )
                    result["marketEvaluation"] = evaluation["market_evaluation"]
                    result["priceDifferencePercent"] = round(
                        evaluation["price_difference_percentage"], 1
                    )

                return result

            except Exception as e:
                logger.warning(
                    "ML price prediction failed, falling back to rule-based: %s", e
                )

        # --- Rule-based fallback ---------------------------------------------
        result = _rule_based_estimate(city, district, propertyType, area)

        if askingPrice is not None:
            mid = (result["priceRange"]["min"] + result["priceRange"]["max"]) / 2
            diff_pct = (askingPrice - mid) / mid * 100
            if diff_pct < -25:
                category = "very_low"
            elif diff_pct < -10:
                category = "low"
            elif diff_pct <= 15:
                category = "reasonable"
            elif diff_pct <= 30:
                category = "high"
            else:
                category = "very_high"

            result["marketEvaluation"] = {"category": category}
            result["priceDifferencePercent"] = round(diff_pct, 1)

        return result
