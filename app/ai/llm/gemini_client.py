import google.generativeai as genai
from typing import Tuple
import logging

from app.core.config import settings
from app.dto.chat import TokenUsage
from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLM):
    """Gemini AI client implementation."""
    
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        """Initialize the Gemini client."""
        super().__init__(model_name)
        
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        
        # System prompt for SmartRent context
        self.system_prompt = """
You are an AI assistant for SmartRent, a smart rental property management platform. 
You help users with questions about:
- Property management
- Rental inquiries
- Smart home features
- Tenant services
- Maintenance requests
- Payment and billing questions

Be helpful, professional, and provide accurate information about rental properties and smart home technologies.
If you don't know something specific about SmartRent, acknowledge it and provide general helpful guidance.
"""
    
    async def generate_response(self, conversation_context: str) -> Tuple[str, TokenUsage]:
        """Generate response using Gemini API and return response with token usage."""
        try:
            # Count input tokens (approximate)
            input_tokens = self.estimate_tokens(conversation_context)
            
            # Generate response
            response = self.model.generate_content(conversation_context)
            
            # Count output tokens (approximate)
            output_tokens = self.estimate_tokens(response.text)
            
            # Create token usage object
            token_usage = TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            )
            
            # Log token usage
            self.log_token_usage(token_usage)
            
            return response.text, token_usage
            
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            raise Exception(f"Failed to generate AI response: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (approximate calculation)."""
        # Simple approximation: ~4 characters per token for English text
        # This is a rough estimate as Gemini uses its own tokenization
        return max(1, len(text) // 4)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for SmartRent context."""
        return self.system_prompt
