# SmartRent AI - Gemini Chatbot Integration

This document describes the Gemini AI chatbot integration for the SmartRent AI platform.

## Overview

The SmartRent AI platform now includes an intelligent chatbot powered by Google's Gemini AI model. The chatbot can assist users with:

- Property management questions
- Rental inquiries
- Smart home features
- Tenant services
- Maintenance requests
- Payment and billing questions

## Setup

### 1. Install Dependencies

The required dependency is already added to `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Update the `.env` file with your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 3. Start the Application

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### POST `/api/v1/chat/chat`

Send a message to the AI chatbot.

**Request Body:**

```json
{
  "message": "How do I submit a maintenance request?",
  "conversation_id": "optional-conversation-id",
  "context": "optional-additional-context"
}
```

**Response:**

```json
{
  "message": "To submit a maintenance request, you can...",
  "conversation_id": "uuid-conversation-id",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_used": "gemini-2.5-pro"
}
```

### GET `/api/v1/chat/conversation/{conversation_id}`

Retrieve conversation history.

**Response:**

```json
{
  "conversation_id": "uuid-conversation-id",
  "messages": [
    {
      "role": "user",
      "content": "How do I submit a maintenance request?",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "To submit a maintenance request, you can...",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:05Z"
}
```

### DELETE `/api/v1/chat/conversation/{conversation_id}`

Clear conversation history.

**Response:**

```json
{
  "message": "Conversation cleared successfully"
}
```

### GET `/api/v1/chat/conversations`

List all conversation IDs.

**Response:**

```json
["uuid-conversation-id-1", "uuid-conversation-id-2"]
```

### GET `/api/v1/chat/health`

Check chatbot service health.

**Response:**

```json
{
  "status": "healthy",
  "service": "chatbot"
}
```

## Usage Examples

### Basic Chat

```bash
curl -X POST "http://localhost:8000/api/v1/chat/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I need help with my smart lock"
  }'
```

### Continue Conversation

```bash
curl -X POST "http://localhost:8000/api/v1/chat/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "The lock is not responding to my app",
    "conversation_id": "your-conversation-id-here"
  }'
```

### With Additional Context

```bash
curl -X POST "http://localhost:8000/api/v1/chat/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need help with my lease",
    "context": "User is a tenant in Building A, Unit 205, lease expires in 3 months"
  }'
```

## Architecture

The chatbot integration follows the existing project architecture:

- **DTOs**: `app/dto/chat.py` - Data transfer objects for chat requests/responses
- **Service**: `app/ai/chatbot_service.py` - Core chatbot logic and Gemini integration
- **API**: `app/api/v1/chat.py` - FastAPI endpoints for chat functionality
- **Config**: `app/core/config.py` - Configuration management including API key

## Features

### Conversation Management

- Maintains conversation history in memory
- Supports multiple concurrent conversations
- Conversation IDs for session management

### Smart Context Handling

- System prompt optimized for SmartRent domain
- Conversation history included in context
- Optional additional context per request
- Token limit management (last 10 messages)

### Error Handling

- Comprehensive error handling and logging
- Graceful fallbacks for API failures
- Input validation and sanitization

### Health Monitoring

- Health check endpoint for service monitoring
- API key validation on service initialization

## Testing

Run the chatbot tests:

```bash
pytest tests/test_chatbot.py -v
```

## Security Considerations

1. **API Key Security**: Store the Gemini API key securely in environment variables
2. **Input Validation**: All user inputs are validated and sanitized
3. **Rate Limiting**: Consider implementing rate limiting for production use
4. **Conversation Storage**: Current implementation stores conversations in memory - consider database storage for production

## Production Considerations

1. **Database Storage**: Replace in-memory conversation storage with database persistence
2. **Caching**: Implement Redis or similar for conversation caching
3. **Rate Limiting**: Add rate limiting to prevent abuse
4. **Monitoring**: Add comprehensive logging and monitoring
5. **Scaling**: Consider implementing conversation partitioning for horizontal scaling

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY is not configured"**

   - Ensure your `.env` file contains a valid API key
   - Check that the environment variable is loaded correctly

2. **"Import 'google.generativeai' could not be resolved"**

   - Run `pip install google-generativeai==0.3.2`
   - Verify the package is installed in your virtual environment

3. **API Rate Limits**
   - Google has rate limits on the Gemini API
   - Implement exponential backoff for retries
   - Consider upgrading your API quota if needed
