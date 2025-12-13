import logging
from typing import Optional

from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, status

from app.ai import get_llm_instance
from app.ai.llm.base_llm import BaseLLM

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


def get_llm() -> BaseLLM:
    try:
        return get_llm_instance()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service not available: {str(e)}",
        )


@router.post("/", response_model=CompletionResponse, status_code=status.HTTP_200_OK)
async def completion(
    request: CompletionRequest,
    llm: BaseLLM = Depends(get_llm),
) -> CompletionResponse:
    """
    Raw completion endpoint: sends a prompt directly to the configured LLM and returns the generated text and token usage.

    - **prompt**: The text prompt to send to the model
    - **model**: Optional model override (currently informational only)
    - **max_tokens**, **temperature**: Optional tuning params (not all LLM implementations use these yet)
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty"
        )

    try:
        # Currently the BaseLLM interface expects a single string and returns (text, token_usage)
        text = await llm.generate_response(request.prompt)

        return CompletionResponse(
            text=text, model_used=getattr(llm, "model_name", request.model or "unknown")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calling completion endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating completion",
        )
