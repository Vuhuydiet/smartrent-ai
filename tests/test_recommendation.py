# import pytest (removed unused)

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_similar_listings():
    """Test getting similar listings from the recommendation engine."""
    payload = {
        "target": {
            "listing_id": 100,
            "product_type": "APARTMENT",
            "listing_type": "RENT",
            "price": 10000000.0,
            "area": 50.0,
            "bedrooms": 2,
            "province_code": "01",
            "district_id": 1,
            "vip_type": "NORMAL",
            "post_date_days_ago": 2,
        },
        "candidates": [
            {
                "listing_id": 101,
                "product_type": "APARTMENT",
                "listing_type": "RENT",
                "price": 11000000.0,
                "area": 55.0,
                "bedrooms": 2,
                "province_code": "01",
                "district_id": 1,
                "vip_type": "NORMAL",
                "post_date_days_ago": 3,
            },
            {
                "listing_id": 102,
                "product_type": "ROOM",
                "listing_type": "RENT",
                "price": 3000000.0,
                "area": 20.0,
                "bedrooms": 1,
                "province_code": "02",
                "district_id": 2,
                "vip_type": "NORMAL",
                "post_date_days_ago": 10,
            },
        ],
        "top_n": 2,
        "alpha": 0.4,
    }

    response = client.post("/api/v1/recommendations/similar", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    # We provided 2 candidates, so we expect exactly 2 results back
    assert len(data) == 2

    # Verify the structure of the recommended items
    first_item = data[0]
    assert "listing_id" in first_item
    assert "score" in first_item
    assert "cf_score" in first_item
    assert "cbf_score" in first_item

    # Item 101 should be closer to 100 than 102 (apartments in province 01 vs room in province 02)
    # The list should ideally be sorted descending by "score"
    scores = [item["score"] for item in data]
    assert scores[0] >= scores[1]


def test_personalized_feed():
    """Test getting personalized listings based on user interactions."""
    payload = {
        "user_id": "test_user_789",
        "user_interactions": [
            {"user_id": "test_user_789", "listing_id": 101, "weight": 3.0},  # saved
            {"user_id": "test_user_789", "listing_id": 102, "weight": 1.0},  # viewed
        ],
        "all_interactions": [
            {"user_id": "test_user_789", "listing_id": 101, "weight": 3.0},
            {"user_id": "other_user_1", "listing_id": 101, "weight": 2.5},
            {
                "user_id": "other_user_1",
                "listing_id": 103,
                "weight": 3.0,
            },  # this implies 101 and 103 might be similar via CF
        ],
        "candidates": [
            {
                "listing_id": 101,
                "product_type": "APARTMENT",
                "listing_type": "RENT",
                "price": 8000000.0,
                "area": 45.0,
                "bedrooms": 1,
                "province_code": "01",
                "district_id": 1,
                "vip_type": "SILVER",
                "post_date_days_ago": 1,
            },
            {
                "listing_id": 102,
                "product_type": "ROOM",
                "listing_type": "RENT",
                "price": 3000000.0,
                "area": 20.0,
                "bedrooms": 1,
                "province_code": "01",
                "district_id": 2,
                "vip_type": "NORMAL",
                "post_date_days_ago": 5,
            },
            {
                "listing_id": 103,
                "product_type": "APARTMENT",
                "listing_type": "RENT",
                "price": 8500000.0,
                "area": 48.0,
                "bedrooms": 1,
                "province_code": "01",
                "district_id": 1,
                "vip_type": "GOLD",
                "post_date_days_ago": 0,
            },
        ],
        "top_n": 3,
        "alpha": 0.5,
    }

    response = client.post("/api/v1/recommendations/personalized", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3

    first_item = data[0]
    assert "listing_id" in first_item
    assert "score" in first_item

    # Check if they are sorted by score descending
    scores = [item["score"] for item in data]
    assert scores == sorted(scores, reverse=True)
