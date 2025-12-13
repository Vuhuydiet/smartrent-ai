from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Single message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request to chat endpoint with conversation history."""

    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    message: ChatMessage
    metadata: Optional[Dict[str, Any]] = None
