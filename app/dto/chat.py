from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Single message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


class LastListingRef(BaseModel):
    """Reference to a listing shown in the previous response."""

    position: int
    listingId: str
    title: str = ""


class ChatRequest(BaseModel):
    """Request to chat endpoint with conversation history."""

    messages: List[ChatMessage]
    user_id: Optional[str] = None  # Authenticated user ID
    auth_token: Optional[str] = None  # JWT for backend API calls
    last_listings: Optional[
        List[LastListingRef]
    ] = None  # Listings from previous response


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    message: ChatMessage
    metadata: Optional[Dict[str, Any]] = None
    listings: Optional[Dict[str, Any]] = None  # Raw listing data from search results
