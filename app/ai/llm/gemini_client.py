import logging

import google.generativeai as genai  # type: ignore

from app.core.config import settings

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLM):
    """Gemini AI client implementation."""

    def __init__(self, model_name: str = ""):
        """Initialize the Gemini client."""
        model_name = model_name or settings.GEMINI_CHAT_MODEL
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

    async def generate_response(self, conversation_context: str) -> str:
        """Generate response using Gemini API."""
        try:
            # Generate response
            response = self.model.generate_content(conversation_context)
            return response.text

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            raise Exception(f"Failed to generate AI response: {str(e)}")

    def get_system_prompt(self) -> str:
        """Get the system prompt for SmartRent context."""
        return self.system_prompt
