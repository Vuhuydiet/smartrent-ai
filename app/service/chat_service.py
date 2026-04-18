import logging
from typing import List, Optional

from app.agent.orchestrator import get_orchestrator
from app.dto.chat import ChatMessage, ChatResponse, LastListingRef

logger = logging.getLogger(__name__)


class ChatService:
    """
    Thin HTTP service layer — delegates all chat logic to AgentOrchestrator.

    Kept as a class so the FastAPI dependency injection in chat.py remains
    unchanged (it still calls ChatService() and chat_service.process_chat()).
    """

    def __init__(self) -> None:
        # Initialises the singleton orchestrator (and all its dependencies)
        # on the first request. Raises ValueError if GCP_PROJECT_ID is missing.
        self._orchestrator = get_orchestrator()

    async def process_chat(
        self,
        messages: List[ChatMessage],
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        last_listings: Optional[List[LastListingRef]] = None,
    ) -> ChatResponse:
        """Delegate to AgentOrchestrator and map AgentResult → ChatResponse."""
        result = await self._orchestrator.run(
            messages,
            user_id=user_id,
            auth_token=auth_token,
            last_listings=last_listings,
        )
        return ChatResponse(
            message=ChatMessage(role="assistant", content=result.message),
            metadata=result.metadata,
            listings=result.listings,
        )
