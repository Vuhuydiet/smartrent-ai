import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dto.chat import ChatRequest, ChatResponse
from app.service.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service() -> ChatService:
    """Dependency to get chat service instance."""
    try:
        return ChatService()
    except ValueError as e:
        logger.error(f"Chat service initialization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat service not available: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error initializing chat service: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat service initialization error: {str(e)}",
        )


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Chat with AI assistant to find property listings.

    Send conversation messages and get AI-powered responses.
    The AI can search listings using natural language queries.

    - **messages**: Array of conversation messages with role (user/assistant) and content
    """
    try:
        if not chat_request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Messages cannot be empty",
            )

        # Validate last message is from user
        if chat_request.messages[-1].role != "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from user",
            )

        response = await chat_service.process_chat(chat_request.messages)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in chat endpoint: {type(e).__name__}: {str(e)}",
            exc_info=True,
            extra={"request": chat_request.model_dump()}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {type(e).__name__}: {str(e)}",
        )


@router.get("/health")
async def chat_health() -> dict[str, str]:
    """
    Check if the chat service is healthy.
    """
    try:
        # Try to initialize the service to check if API key is configured
        ChatService()
        return {"status": "healthy", "service": "chat"}
    except ValueError as e:
        logger.error(f"Chat service unavailable: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chat service unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error checking chat health: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking chat service health: {type(e).__name__}: {str(e)}",
        )
