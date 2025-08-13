from fastapi import FastAPI

from app.core.config import settings
from app.api.v1.api import api_router as apiv1_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SmartRent AI API",
    openapi_url=f"/openapi.json"
)

app.include_router(apiv1_router)


@app.get("/")
async def root():
    return {"message": "Welcome to SmartRent AI API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
