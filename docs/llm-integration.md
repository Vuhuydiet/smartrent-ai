# LLM Integration Documentation

## Overview

The project has been restructured to provide a clean separation between LLM configuration, instances, and the chatbot service. This new structure makes it easier to:

- Switch between different LLM providers (currently Gemini, but extensible)
- Manage LLM instances globally
- Test and mock LLM functionality
- Track usage statistics centrally

## Directory Structure

```
app/
├── ai/
│   ├── __init__.py              # Exports main LLM functionality
│   └── llm/
│       ├── __init__.py          # LLM module exports
│       ├── base_llm.py          # Abstract base class for all LLMs
│       ├── gemini_client.py     # Gemini-specific implementation
│       └── llm_manager.py       # Factory and instance management
└── service/
    └── chatbot_service.py       # Refactored to use LLM instances
```

## Key Components

### 1. BaseLLM (Abstract Base Class)

Located in `app/ai/llm/base_llm.py`

Defines the interface that all LLM implementations must follow:

- `generate_response()` - Generate AI responses
- `estimate_tokens()` - Estimate token usage
- `log_token_usage()` - Track usage statistics
- `reset_daily_counters()` - Reset usage counters
- `get_usage_stats()` - Get current usage statistics

### 2. GeminiClient

Located in `app/ai/llm/gemini_client.py`

Concrete implementation for Google's Gemini API:

- Handles Gemini-specific configuration
- Implements token estimation
- Manages SmartRent-specific system prompts
- Handles error logging and response generation

### 3. LLMManager

Located in `app/ai/llm/llm_manager.py`

Provides factory pattern and global instance management:

- `LLMFactory.create_llm()` - Create LLM instances
- `get_llm_instance()` - Get global singleton instance
- `reset_llm_instance()` - Reset global instance (useful for testing)

### 4. ChatbotService (Refactored)

Located in `app/service/chatbot_service.py`

Now uses the LLM instance instead of directly managing Gemini:

- Automatically gets LLM instance on initialization
- Delegates AI generation to the LLM instance
- Maintains conversation management logic
- Provides simplified token usage statistics

## Usage Examples

### Using LLM Directly

```python
from app.ai import get_llm_instance

# Get the global LLM instance
llm = get_llm_instance()

# Generate a response
response, token_usage = await llm.generate_response("Hello!")

# Get usage statistics
stats = llm.get_usage_stats()
```

### Using ChatbotService

```python
from app.service.chatbot_service import ChatbotService
from app.dto.chat import ChatRequest

# Create service (automatically uses LLM instance)
chatbot = ChatbotService()

# Process a chat request
request = ChatRequest(message="Hello!", conversation_id=None)
response = await chatbot.process_chat(request)
```

### Switching LLM Types (Future)

```python
from app.ai.llm import LLMFactory, reset_llm_instance

# Reset current instance
reset_llm_instance()

# Create a specific LLM type (when more are added)
llm = LLMFactory.create_llm("gemini", "gemini-2.5-pro")
```

## Configuration

The LLM configuration still uses the existing settings in `app/core/config.py`:

```python
class Settings(BaseSettings):
    # AI Configuration
    GEMINI_API_KEY: str = ""
```

## Benefits of New Structure

1. **Separation of Concerns**: LLM logic is separated from business logic
2. **Testability**: Easy to mock LLM instances for testing
3. **Extensibility**: Simple to add new LLM providers
4. **Global Instance Management**: Avoid recreating expensive LLM connections
5. **Consistent Interface**: All LLMs implement the same interface
6. **Centralized Logging**: Token usage and statistics tracked in one place

## Migration Notes

- Existing API endpoints remain unchanged
- `ChatbotService` interface is the same
- Configuration requirements are the same
- All LLM-specific logic is now contained in the `ai/llm` module

## Testing

Run the example usage file to test the new structure:

```bash
python example_usage.py
```

This will demonstrate both direct LLM usage and ChatbotService usage with the new architecture.
