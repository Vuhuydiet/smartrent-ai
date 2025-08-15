from fastapi.testclient import TestClient

# Import the actual app from main.py
from app.main import app

client = TestClient(app)


def test_health_check_endpoint() -> None:
    """Test the health check endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint() -> None:
    """Test the root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to SmartRent AI API"}
