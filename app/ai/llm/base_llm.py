from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

from app.dto.chat import ChatMessage, TokenUsage


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_requests_today = 0
        self.total_tokens_used_today = 0
    
    @abstractmethod
    async def generate_response(self, conversation_context: str) -> Tuple[str, TokenUsage]:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass
    
    def log_token_usage(self, token_usage: TokenUsage) -> None:
        """Log token usage information."""
        if token_usage:
            self.total_requests_today += 1
            self.total_tokens_used_today += token_usage.total_tokens or 0
    
    def reset_daily_counters(self) -> None:
        """Reset daily counters."""
        self.total_requests_today = 0
        self.total_tokens_used_today = 0
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "requests_made_today": self.total_requests_today,
            "total_tokens_used_today": self.total_tokens_used_today,
            "model_name": self.model_name,
            "last_updated": datetime.now()
        }
