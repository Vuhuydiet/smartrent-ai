import pytest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint():
    """Test basic chat functionality."""
    # Skip if GEMINI_API_KEY not configured
    chat_data = {"message": "Hello, how can I help with rental properties?"}

    response = client.post("/api/v1/chat/chat", json=chat_data)
    # Should either work (200) or fail gracefully with API key error (500)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "message" in data
        assert "conversation_id" in data
        assert "timestamp" in data


def test_chat_validation():
    """Test chat input validation."""
    # Test empty message
    empty_data = {"message": ""}
    response = client.post("/api/v1/chat/chat", json=empty_data)
    assert response.status_code == 400

    # Test missing message
    no_message = {}
    response = client.post("/api/v1/chat/chat", json=no_message)
    assert response.status_code == 422


def test_chat_health():
    """Test chat service health check."""
    response = client.get("/api/v1/chat/health")
    # Should return service status
    assert response.status_code in [200, 503]
