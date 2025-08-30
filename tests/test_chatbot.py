import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.dto.chat import ChatRequest, ChatResponse
from app.service.chatbot_service import ChatbotService


class TestChatbotService:
    @patch('app.ai.chatbot_service.genai.configure')
    @patch('app.ai.chatbot_service.genai.GenerativeModel')
    def test_chatbot_service_initialization(self, mock_model, mock_configure):
        """Test that ChatbotService initializes correctly."""
        # Mock the model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        # Test initialization
        with patch('app.core.config.settings.GEMINI_API_KEY', 'test-api-key'):
            service = ChatbotService()
            
            # Verify API key configuration
            mock_configure.assert_called_once_with(api_key='test-api-key')
            mock_model.assert_called_once_with('gemini-2.5-pro')
            assert service.model == mock_model_instance

    def test_chatbot_service_no_api_key(self):
        """Test that ChatbotService raises error when no API key is provided."""
        with patch('app.core.config.settings.GEMINI_API_KEY', ''):
            with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
                ChatbotService()

    @patch('app.ai.chatbot_service.genai.configure')
    @patch('app.ai.chatbot_service.genai.GenerativeModel')
    async def test_process_chat_new_conversation(self, mock_model, mock_configure):
        """Test processing a chat request with a new conversation."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Hello! How can I help you with SmartRent today?"
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        with patch('app.core.config.settings.GEMINI_API_KEY', 'test-api-key'):
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

    @patch('app.ai.chatbot_service.genai.configure')
    @patch('app.ai.chatbot_service.genai.GenerativeModel')
    async def test_process_chat_existing_conversation(self, mock_model, mock_configure):
        """Test processing a chat request with an existing conversation."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "I can help you with maintenance requests."
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        with patch('app.core.config.settings.GEMINI_API_KEY', 'test-api-key'):
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

    @patch('app.ai.chatbot_service.genai.configure')
    @patch('app.ai.chatbot_service.genai.GenerativeModel')
    async def test_get_conversation_history(self, mock_model, mock_configure):
        """Test retrieving conversation history."""
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        with patch('app.core.config.settings.GEMINI_API_KEY', 'test-api-key'):
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

    @patch('app.ai.chatbot_service.genai.configure')
    @patch('app.ai.chatbot_service.genai.GenerativeModel')
    async def test_clear_conversation(self, mock_model, mock_configure):
        """Test clearing conversation history."""
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        with patch('app.core.config.settings.GEMINI_API_KEY', 'test-api-key'):
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
