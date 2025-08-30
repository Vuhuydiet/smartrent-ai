from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[str] = None


class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    timestamp: datetime
    model_used: str = "gemini-2.5-pro"
    token_usage: Optional[TokenUsage] = None


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


class AccountTokenInfo(BaseModel):
    """Information about API account token usage and limits."""
    requests_made_today: Optional[int] = None
    requests_remaining_today: Optional[int] = None
    daily_request_limit: Optional[int] = None
    rate_limit_per_minute: Optional[int] = None
    last_updated: datetime
