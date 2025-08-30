import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.dto.chat import ChatRequest, ChatResponse, TokenUsage
from app.service.chatbot_service import ChatbotService


class TestChatbotService:
    @patch('app.ai.get_llm_instance')
    def test_chatbot_service_initialization(self, mock_get_llm):
        """Test that ChatbotService initializes correctly."""
        # Mock the LLM instance
        mock_llm = Mock()
        mock_llm.model_name = "gemini-2.5-pro"
        mock_get_llm.return_value = mock_llm
        
        # Test initialization
        service = ChatbotService()
        
        # Verify LLM instance is retrieved
        mock_get_llm.assert_called_once()
        assert service.llm == mock_llm

    @patch('app.ai.get_llm_instance')
    async def test_process_chat_new_conversation(self, mock_get_llm):
        """Test processing a chat request with a new conversation."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm.model_name = "gemini-2.5-pro"
        mock_llm.generate_response = AsyncMock(return_value=(
            "Hello! How can I help you with SmartRent today?",
            TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25)
        ))
        mock_llm.log_token_usage = Mock()
        mock_get_llm.return_value = mock_llm
        
        service = ChatbotService()
        
        # Test chat request
        request = ChatRequest(message="Hello, I need help with my rental")
        response = await service.process_chat(request)
        
        # Verify response
        assert isinstance(response, ChatResponse)
        assert response.message == "Hello! How can I help you with SmartRent today?"
        assert response.conversation_id is not None
        assert response.model_used == "gemini-2.5-pro"
        assert isinstance(response.timestamp, datetime)
        assert response.token_usage.total_tokens == 25

    @patch('app.ai.get_llm_instance')
    async def test_process_chat_existing_conversation(self, mock_get_llm):
        """Test processing a chat request with an existing conversation."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm.model_name = "gemini-2.5-pro"
        mock_llm.generate_response = AsyncMock(side_effect=[
            ("Hello! How can I help you?", TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)),
            ("I can help you with maintenance requests.", TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30))
        ])
        mock_llm.log_token_usage = Mock()
        mock_get_llm.return_value = mock_llm
        
        service = ChatbotService()
        
        # First request to create conversation
        request1 = ChatRequest(message="Hello")
        response1 = await service.process_chat(request1)
        conversation_id = response1.conversation_id
        
        # Second request with existing conversation
        request2 = ChatRequest(
            message="How do I submit a maintenance request?",
            conversation_id=conversation_id
        )
        response2 = await service.process_chat(request2)
        
        # Verify response uses same conversation
        assert response2.conversation_id == conversation_id
        assert response2.message == "I can help you with maintenance requests."

    @patch('app.ai.get_llm_instance')
    async def test_get_conversation_history(self, mock_get_llm):
        """Test retrieving conversation history."""
        mock_llm = Mock()
        mock_llm.model_name = "gemini-2.5-pro"
        mock_llm.generate_response = AsyncMock(return_value=(
            "Test response",
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        ))
        mock_llm.log_token_usage = Mock()
        mock_get_llm.return_value = mock_llm
        
        service = ChatbotService()
        
        # Create a conversation
        request = ChatRequest(message="Test message")
        response = await service.process_chat(request)
        conversation_id = response.conversation_id
        
        # Get conversation history
        history = await service.get_conversation_history(conversation_id)
        
        # Verify history
        assert history is not None
        assert history.conversation_id == conversation_id
        assert len(history.messages) == 2  # User message + AI response
        assert history.messages[0].role == "user"
        assert history.messages[0].content == "Test message"
        assert history.messages[1].role == "assistant"
        assert history.messages[1].content == "Test response"

    @patch('app.ai.get_llm_instance')
    async def test_clear_conversation(self, mock_get_llm):
        """Test clearing conversation history."""
        mock_llm = Mock()
        mock_llm.model_name = "gemini-2.5-pro"
        mock_llm.generate_response = AsyncMock(return_value=(
            "Test response",
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        ))
        mock_llm.log_token_usage = Mock()
        mock_get_llm.return_value = mock_llm
        
        service = ChatbotService()
        
        # Create a conversation
        request = ChatRequest(message="Test message")
        response = await service.process_chat(request)
        conversation_id = response.conversation_id
        
        # Verify conversation exists
        history = await service.get_conversation_history(conversation_id)
        assert history is not None
        
        # Clear conversation
        success = await service.clear_conversation(conversation_id)
        assert success is True
        
        # Verify conversation is cleared
        history = await service.get_conversation_history(conversation_id)
        assert history is None
        
        # Test clearing non-existent conversation
        success = await service.clear_conversation("non-existent-id")
        assert success is False
