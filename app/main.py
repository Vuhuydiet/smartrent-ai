from fastapi import FastAPI

from app.api.v1.api import api_router as apiv1_router
from app.core.config import settings

app = FastAPI(
    title="SmartRent AI",
    version=settings.VERSION,
    description="SmartRent AI API",
    openapi_url="/openapi.json",
)

app.include_router(apiv1_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to SmartRent AI API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
