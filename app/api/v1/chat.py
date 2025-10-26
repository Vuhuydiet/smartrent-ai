from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional
import logging

from app.dto.chat import ChatRequest, ChatResponse, ConversationHistory, AccountTokenInfo
from app.service.chatbot_service import ChatbotService


logger = logging.getLogger(__name__)

router = APIRouter()


def get_chatbot_service() -> ChatbotService:
    """Dependency to get chatbot service instance."""
    try:
        return ChatbotService()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot service not available: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    chat_request: ChatRequest,
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> ChatResponse:
    """
    Send a message to the AI chatbot and get a response.
    
    - **message**: The user's message to send to the chatbot
    - **conversation_id**: Optional conversation ID to continue existing conversation
    - **context**: Optional additional context for the conversation
    """
    try:
        if not chat_request.message or not chat_request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        response = await chatbot_service.process_chat(chat_request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message"
        )


@router.get("/conversation/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str,
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> ConversationHistory:
    """
    Get conversation history by conversation ID.
    
    - **conversation_id**: The ID of the conversation to retrieve
    """
    try:
        conversation = await chatbot_service.get_conversation_history(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving conversation"
        )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> dict[str, str]:
    """
    Clear conversation history by conversation ID.
    
    - **conversation_id**: The ID of the conversation to clear
    """
    try:
        success = await chatbot_service.clear_conversation(conversation_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return {"message": "Conversation cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while clearing conversation"
        )


@router.get("/conversations", response_model=List[str])
async def list_conversations(
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> List[str]:
    """
    List all conversation IDs.
    """
    try:
        conversations = await chatbot_service.list_conversations()
        return conversations
        
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while listing conversations"
        )


@router.get("/health")
async def chatbot_health() -> dict[str, str]:
    """
    Check if the chatbot service is healthy.
    """
    try:
        # Try to initialize the service to check if API key is configured
        chatbot_service = ChatbotService()
        return {"status": "healthy", "service": "chatbot"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chatbot service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error checking chatbot health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking chatbot service health"
        )


@router.get("/token-usage", response_model=AccountTokenInfo)
async def get_token_usage(
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> AccountTokenInfo:
    """
    Get current token usage statistics for the session.
    
    Returns information about API usage including:
    - Number of requests made today (session-based)
    - Estimated token usage
    - Last updated timestamp
    
    Note: Gemini API doesn't expose real-time quota limits,
    so some fields may be null.
    """
    try:
        return await chatbot_service.get_token_usage_stats()
    except Exception as e:
        logger.error(f"Error getting token usage stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving token usage statistics"
        )


@router.post("/reset-counters")
async def reset_token_counters(
    chatbot_service: ChatbotService = Depends(get_chatbot_service)
) -> dict[str, str]:
    """
    Reset daily token usage counters.
    
    This is useful for testing or manual reset of session-based counters.
    """
    try:
        await chatbot_service.reset_daily_counters()
        return {"message": "Token usage counters have been reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting token counters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while resetting token counters"
        )
