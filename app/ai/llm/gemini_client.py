import logging

from vertexai.generative_models import GenerativeModel  # type: ignore[import]

from app.core.config import settings

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLM):
    """Gemini AI client implementation via Vertex AI."""

    def __init__(self, model_name: str = ""):
        """Initialize the Gemini client."""
        model_name = model_name or settings.GEMINI_CHAT_MODEL
        super().__init__(model_name)

        # Vertex AI must already be initialised by LLMGateway singleton
        self.model = GenerativeModel(model_name)

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
        """Generate response using Vertex AI Gemini API."""
        try:
            response = await self.model.generate_content_async(conversation_context)
            try:
                return response.text
            except ValueError:
                # Vertex AI raises ValueError when response contains non-text parts
                for part in response.candidates[0].content.parts:
                    if part.text:
                        return part.text
                return ""

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            raise Exception(f"Failed to generate AI response: {str(e)}")

    def get_system_prompt(self) -> str:
        """Get the system prompt for SmartRent context."""
        return self.system_prompt
