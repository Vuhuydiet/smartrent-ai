from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_house_pricing_api_structure():
    """Test house pricing API structure."""
    # Test with valid data
    valid_data = {
        "latitude": 21.0285,
        "longitude": 105.8542,
        "property_type": "Apartment",
        "city": "Hanoi",
        "district": "Ba Dinh",
        "ward": "Dien Bien Ward",
    }

    response = client.post("/api/v1/house-pricing/get-price-range", json=valid_data)
    assert response.status_code == 200

    data = response.json()
    assert "address" in data
    assert "property_type" in data
    assert "predicted_price" in data
    assert "price_range" in data
    assert "confidence" in data
    assert "currency" in data

    # Validate response structure
    assert isinstance(data["predicted_price"], (int, float))
    assert data["predicted_price"] > 0
    assert isinstance(data["price_range"], dict)
    assert "min_price" in data["price_range"]
    assert "max_price" in data["price_range"]
    assert data["confidence"] >= 0.0
    assert data["confidence"] <= 1.0
    assert data["currency"] == "VND_millions"


def test_house_pricing_api_validation():
    """Test house pricing API validation."""
    # Test with invalid latitude
    invalid_data = {
        "latitude": 50.0,  # Outside Vietnam range
        "longitude": 105.8542,
        "property_type": "Apartment",
        "city": "Hanoi",
        "district": "Ba Dinh",
        "ward": "Dien Bien Ward",
    }

    response = client.post("/api/v1/house-pricing/get-price-range", json=invalid_data)
    assert response.status_code == 422  # Validation error

    # Test with missing required fields
    incomplete_data = {
        "latitude": 21.0285,
        "longitude": 105.8542
        # Missing other required fields
    }

    response = client.post(
        "/api/v1/house-pricing/get-price-range", json=incomplete_data
    )
    assert response.status_code == 422
