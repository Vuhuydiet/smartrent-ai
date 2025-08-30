import google.generativeai as genai
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from app.core.config import settings
from app.dto.chat import ChatRequest, ChatResponse, ChatMessage, ConversationHistory, TokenUsage, AccountTokenInfo


logger = logging.getLogger(__name__)


class ChatbotService:
    def __init__(self):
        """Initialize the Gemini chatbot service."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        # In-memory conversation storage (in production, use database)
        self.conversations: Dict[str, ConversationHistory] = {}
        
        # Token usage tracking
        self.total_requests_today = 0
        self.total_tokens_used_today = 0
        
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

    async def process_chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Process a chat request and return a response."""
        try:
            # Get or create conversation
            conversation_id = chat_request.conversation_id or str(uuid.uuid4())
            conversation = self._get_or_create_conversation(conversation_id)
            
            # Add user message to conversation
            user_message = ChatMessage(
                role="user",
                content=chat_request.message,
                timestamp=datetime.now()
            )
            conversation.messages.append(user_message)
            
            # Prepare conversation context for Gemini
            conversation_context = self._build_conversation_context(
                conversation.messages, 
                chat_request.context
            )
            
            # Generate response using Gemini
            response, token_usage = await self._generate_response(conversation_context)
            
            # Log token usage
            self._log_token_usage(token_usage)
            
            # Add assistant message to conversation
            assistant_message = ChatMessage(
                role="assistant",
                content=response,
                timestamp=datetime.now()
            )
            conversation.messages.append(assistant_message)
            conversation.updated_at = datetime.now()
            
            # Store updated conversation
            self.conversations[conversation_id] = conversation
            
            return ChatResponse(
                message=response,
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                model_used="gemini-2.5-pro",
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            raise Exception(f"Failed to process chat request: {str(e)}")

    async def _generate_response(self, conversation_context: str) -> tuple[str, TokenUsage]:
        """Generate response using Gemini API and return response with token usage."""
        try:
            # Count input tokens (approximate)
            input_tokens = self._estimate_tokens(conversation_context)
            
            # Generate response
            response = self.model.generate_content(conversation_context)
            
            # Count output tokens (approximate)
            output_tokens = self._estimate_tokens(response.text)
            
            # Create token usage object
            token_usage = TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            )
            
            return response.text, token_usage
            
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            raise Exception(f"Failed to generate AI response: {str(e)}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (approximate calculation)."""
        # Simple approximation: ~4 characters per token for English text
        # This is a rough estimate as Gemini uses its own tokenization
        return max(1, len(text) // 4)

    def _log_token_usage(self, token_usage: TokenUsage) -> None:
        """Log token usage information."""
        if token_usage:
            self.total_requests_today += 1
            self.total_tokens_used_today += token_usage.total_tokens or 0
            
            logger.info(
                f"Token Usage - Prompt: {token_usage.prompt_tokens}, "
                f"Completion: {token_usage.completion_tokens}, "
                f"Total: {token_usage.total_tokens}"
            )
            logger.info(
                f"Session Stats - Total requests today: {self.total_requests_today}, "
                f"Total tokens used today: {self.total_tokens_used_today}"
            )

    def _get_or_create_conversation(self, conversation_id: str) -> ConversationHistory:
        """Get existing conversation or create a new one."""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    def _build_conversation_context(self, messages: List[ChatMessage], additional_context: Optional[str] = None) -> str:
        """Build conversation context for Gemini."""
        context_parts = [self.system_prompt]
        
        if additional_context:
            context_parts.append(f"Additional context: {additional_context}")
        
        # Add conversation history (limit to last 10 messages to avoid token limits)
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        if recent_messages:
            context_parts.append("Conversation history:")
            for msg in recent_messages:
                role_label = "Human" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")
        
        # Add current user message prompt
        if messages and messages[-1].role == "user":
            context_parts.append(f"\nPlease respond to the latest message: {messages[-1].content}")
        
        return "\n\n".join(context_parts)

    async def get_conversation_history(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get conversation history by ID."""
        return self.conversations.get(conversation_id)

    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    async def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self.conversations.keys())

    async def get_token_usage_stats(self) -> AccountTokenInfo:
        """Get current token usage statistics for the session."""
        # Note: Gemini API doesn't provide real-time quota information
        # These are session-based estimates
        return AccountTokenInfo(
            requests_made_today=self.total_requests_today,
            requests_remaining_today=None,  # Not available from Gemini API
            daily_request_limit=None,  # Varies by plan, not exposed by API
            rate_limit_per_minute=None,  # Not exposed by API
            last_updated=datetime.now()
        )

    async def reset_daily_counters(self) -> None:
        """Reset daily counters (useful for testing or manual reset)."""
        self.total_requests_today = 0
        self.total_tokens_used_today = 0
        logger.info("Daily token counters have been reset")
