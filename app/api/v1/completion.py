import logging
from typing import Optional

from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, status

from app.ai.llm.gateway import get_gateway
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class CompletionResponse(BaseModel):
    text: str
    model_used: str


@router.post("/", response_model=CompletionResponse, status_code=status.HTTP_200_OK)
async def completion(request: CompletionRequest) -> CompletionResponse:
    """
    Raw completion endpoint: sends a prompt directly to Gemini via the shared
    LLMGateway (so all calls are traced through Langfuse and benefit from
    quota retry). Used by backend to generate listing descriptions.
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty"
        )

    gateway = get_gateway()
    model_name = request.model or settings.GEMINI_CHAT_MODEL

    generation_config: dict = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.max_tokens is not None:
        generation_config["max_output_tokens"] = request.max_tokens

    trace = gateway.create_trace(
        name="completion",
        input=request.prompt[:500],
        metadata={"model": model_name},
    )

    try:
        response = await gateway.generate(
            prompt=request.prompt,
            model_name=model_name,
            generation_config=generation_config or None,
            trace=trace,
            span_name="completion-generate",
        )

        # Safe text extraction — response.text raises ValueError on non-text parts
        try:
            text = response.text
        except (ValueError, AttributeError):
            text = ""
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    text = part.text
                    break

        trace.update(output={"text": text[:500]})

        return CompletionResponse(text=text, model_used=model_name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error calling completion endpoint: %s", e, exc_info=True)
        trace.update(output={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating completion",
        )
