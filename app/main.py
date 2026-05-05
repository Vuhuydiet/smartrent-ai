import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.api import api_router as apiv1_router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Init Vertex AI / Langfuse on startup, flush traces on shutdown."""
    # Eagerly construct the LLM gateway so vertexai.init() runs before any
    # request is served. Without this, the first endpoint to construct a
    # GenerativeModel directly (e.g. /api/v1/completion/) crashes with
    # GoogleAuthError because Vertex AI has no project configured yet.
    try:
        from app.ai.llm.gateway import get_gateway

        get_gateway()
    except Exception as e:
        logger.error("Failed to initialise LLMGateway on startup: %s", e, exc_info=True)

    yield

    try:
        from app.ai.llm.gateway import _gateway_instance

        if _gateway_instance is not None:
            _gateway_instance.flush()
    except Exception as e:
        logger.warning("Failed to flush Langfuse on shutdown: %s", e)


app = FastAPI(
    title="SmartRent AI - House Pricing API",
    version=settings.VERSION,
    description="AI-powered real estate price prediction for Vietnamese market",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {type(exc).__name__}: {str(exc)}"},
    )


# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
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
