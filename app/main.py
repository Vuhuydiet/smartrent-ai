from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
