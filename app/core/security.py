from fastapi import Header, HTTPException, status

from app.core.config import settings


async def require_internal_key(
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    """Gate internal-only endpoints behind a shared secret.

    No-op when ``INTERNAL_AI_API_KEY`` is unset (local dev); otherwise the
    ``X-Internal-Api-Key`` header must match exactly. The backend proxy sends
    this header so the AI service can reject public callers.
    """
    expected = settings.INTERNAL_AI_API_KEY
    if not expected:
        return
    if x_internal_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key",
        )
