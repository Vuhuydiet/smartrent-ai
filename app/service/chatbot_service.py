import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from app.ai import get_llm_instance
from app.ai.llm.gemini_client import GeminiClient
from app.dto.chat import ChatRequest, ChatResponse, ChatMessage, ConversationHistory, TokenUsage, AccountTokenInfo


logger = logging.getLogger(__name__)


class ChatbotService:
    def __init__(self):
        """Initialize the chatbot service with LLM instance."""
        self.llm = get_llm_instance()
        
        # In-memory conversation storage (in production, use database)
        self.conversations: Dict[str, ConversationHistory] = {}

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
            
            # Generate response using LLM
            response, token_usage = await self.llm.generate_response(conversation_context)
            
            # Log token usage
            self.llm.log_token_usage(token_usage)
            
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
                model_used=self.llm.model_name,
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            raise Exception(f"Failed to process chat request: {str(e)}")

    def _build_conversation_context(self, messages: List[ChatMessage], additional_context: Optional[str] = None) -> str:
        """Build conversation context for LLM."""
        # Get system prompt from LLM (if it's a GeminiClient)
        if isinstance(self.llm, GeminiClient):
            system_prompt = self.llm.get_system_prompt()
        else:
            # Default system prompt for other LLM types
            system_prompt = """
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
        
        context_parts = [system_prompt]
        
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
        usage_stats = self.llm.get_usage_stats()
        
        # Note: Gemini API doesn't provide real-time quota information
        # These are session-based estimates
        return AccountTokenInfo(
            requests_made_today=usage_stats["requests_made_today"],
            requests_remaining_today=None,  # Not available from Gemini API
            daily_request_limit=None,  # Varies by plan, not exposed by API
            rate_limit_per_minute=None,  # Not exposed by API
            last_updated=usage_stats["last_updated"]
        )

    async def reset_daily_counters(self) -> None:
        """Reset daily counters (useful for testing or manual reset)."""
        self.llm.reset_daily_counters()
        logger.info("Daily token counters have been reset")
