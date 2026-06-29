from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core import security
from app.core.config import settings


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/guarded", dependencies=[Depends(security.require_internal_key)])
    def guarded():
        return {"ok": True}

    return TestClient(app)


def test_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "s3cret")
    assert _client().get("/guarded").status_code == 401


def test_accepts_matching_key(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "s3cret")
    resp = _client().get("/guarded", headers={"X-Internal-Api-Key": "s3cret"})
    assert resp.status_code == 200


def test_allows_all_when_secret_unset(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_AI_API_KEY", "")
    assert _client().get("/guarded").status_code == 200
