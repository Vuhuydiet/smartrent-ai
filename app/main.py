from fastapi import FastAPI

from app.api.v1.api import api_router as apiv1_router
from app.core.config import settings

app = FastAPI(
    title="SmartRent AI - House Pricing API",
    version=settings.VERSION,
    description="AI-powered real estate price prediction for Vietnamese market",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(apiv1_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to SmartRent AI API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
