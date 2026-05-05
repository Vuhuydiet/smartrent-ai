import logging
from typing import Optional

from agents import Agent, ModelSettings, Runner  # type: ignore[import]
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, status

from app.ai.llm.agent_factory import make_model
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
    Raw completion endpoint — sends a prompt to the configured LLM provider
    via the OpenAI Agents SDK. Used by the backend to generate listing
    descriptions. Calls are traced through Langfuse via the gateway.
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty"
        )

    gateway = get_gateway()
    model_name = request.model or settings.LLM_CHAT_MODEL

    settings_kwargs: dict = {}
    if request.temperature is not None:
        settings_kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        settings_kwargs["max_tokens"] = request.max_tokens

    trace = gateway.create_trace(
        name="completion",
        input=request.prompt[:500],
        metadata={"model": model_name, "provider": settings.LLM_PROVIDER},
    )

    agent = Agent(
        name="Raw Completion",
        instructions="Respond with exactly what the user asks for. Do not add commentary.",
        model=make_model(model_name),
        model_settings=ModelSettings(**settings_kwargs),
    )

    span = trace.generation(
        name="completion-generate",
        model=model_name,
        input=request.prompt[:2000],
    )
    try:
        result = await Runner.run(
            starting_agent=agent,
            input=request.prompt,
            max_turns=2,
        )
        text = str(result.final_output or "")
        span.end(output=text[:2000])
        trace.update(output={"text": text[:500]})

        return CompletionResponse(text=text, model_used=model_name)

    except HTTPException:
        raise
    except Exception as e:
        span.end(level="ERROR", status_message=str(e))
        logger.error("Error calling completion endpoint: %s", e, exc_info=True)
        trace.update(output={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating completion",
        )
