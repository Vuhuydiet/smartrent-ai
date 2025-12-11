"""
Example script to test the Chat API

This script demonstrates how to use the chat endpoint to interact with
the AI assistant for finding property listings.
"""

import asyncio
from typing import Any, Dict, List

import httpx


async def chat_with_ai(
    messages: List[Dict[str, str]], base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Send chat messages to the AI assistant

    Args:
        messages: List of messages with 'role' and 'content'
        base_url: The base URL of the API

    Returns:
        The chat response
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/api/v1/chat",
                json={"messages": messages},
                headers={"Content-Type": "application/json"},
                timeout=60.0,  # Allow time for AI processing
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}", "detail": response.text}

        except Exception as e:
            print(f"Request failed: {str(e)}")
            return {"error": str(e)}


async def test_health_check(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test the chat health check endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/api/v1/chat/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def print_chat_result(result: Dict[str, Any], title: str = "Chat Response"):
    """Pretty print chat result"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "detail" in result:
            print(f"   Detail: {result['detail']}")
        return

    message = result.get("message", {})
    metadata = result.get("metadata", {})

    print(f"🤖 Assistant: {message.get('content', '')}")
    
    if metadata:
        print(f"\n📊 Metadata:")
        if "function_calls" in metadata:
            print(f"   Function Calls: {metadata['function_calls']}")
        if "model" in metadata:
            print(f"   Model: {metadata['model']}")
        if "timestamp" in metadata:
            print(f"   Timestamp: {metadata['timestamp']}")


async def test_simple_query() -> None:
    """Test a simple property search query"""
    print("\n🧪 Test 1: Simple Search Query")
    print("-" * 60)
    
    messages = [
        {
            "role": "user",
            "content": "I'm looking for a 2-bedroom apartment in Hanoi with a budget around $1000/month"
        }
    ]
    
    print(f"👤 User: {messages[0]['content']}")
    result = await chat_with_ai(messages)
    print_chat_result(result, "Response")


async def test_multi_turn_conversation() -> None:
    """Test a multi-turn conversation"""
    print("\n🧪 Test 2: Multi-Turn Conversation")
    print("-" * 60)
    
    messages = [
        {
            "role": "user",
            "content": "I need an apartment in Ho Chi Minh City"
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you find an apartment in Ho Chi Minh City. To narrow down the search, could you tell me more about your preferences? For example:\n- How many bedrooms do you need?\n- What's your budget?\n- Any specific district or area?"
        },
        {
            "role": "user",
            "content": "I need 1 bedroom, budget is $600-800, and I prefer District 1 or District 3"
        }
    ]
    
    for msg in messages:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        print(f"{role_icon} {msg['role'].capitalize()}: {msg['content'][:100]}...")
    
    result = await chat_with_ai(messages)
    print_chat_result(result, "Response")


async def test_specific_requirements() -> None:
    """Test search with specific requirements"""
    print("\n🧪 Test 3: Search with Specific Requirements")
    print("-" * 60)
    
    messages = [
        {
            "role": "user",
            "content": "Find me a furnished studio apartment near Hoan Kiem Lake, pet-friendly, with gym and parking, max $700"
        }
    ]
    
    print(f"👤 User: {messages[0]['content']}")
    result = await chat_with_ai(messages)
    print_chat_result(result, "Response")


async def test_invalid_request() -> None:
    """Test error handling with invalid request"""
    print("\n🧪 Test 4: Invalid Request (Empty Messages)")
    print("-" * 60)
    
    messages = []
    
    print("👤 User: [sending empty messages array]")
    result = await chat_with_ai(messages)
    print_chat_result(result, "Response")


async def test_assistant_last_message() -> None:
    """Test error handling when last message is from assistant"""
    print("\n🧪 Test 5: Invalid Request (Last Message from Assistant)")
    print("-" * 60)
    
    messages = [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hi! How can I help you?"
        }
    ]
    
    for msg in messages:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        print(f"{role_icon} {msg['role'].capitalize()}: {msg['content']}")
    
    result = await chat_with_ai(messages)
    print_chat_result(result, "Response")


async def main() -> None:
    """Main function to run tests"""
    print("💬 Chat API Test")
    print("=" * 60)

    # Test health check
    print("\n🏥 Testing health check...")
    health_result = await test_health_check()
    if "error" in health_result:
        print(f"❌ Health check failed: {health_result['error']}")
        print("Make sure the server is running on localhost:8000")
        print("Also ensure GEMINI_API_KEY is set in your .env file")
        return
    else:
        print(f"✅ Service is healthy: {health_result.get('status', 'unknown')}")

    # Run test scenarios
    await test_simple_query()
    await test_multi_turn_conversation()
    await test_specific_requirements()
    await test_invalid_request()
    await test_assistant_last_message()

    print(f"\n{'='*60}")
    print("🎉 Testing completed!")
    print("💡 The AI assistant uses Gemini to understand your queries and")
    print("   search for listings using the backend API.")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Install required packages first:
    # uv add httpx (if using uv)
    # or: pip install httpx

    asyncio.run(main())
