#!/usr/bin/env python3
"""
Simple demo script to test the SmartRent AI Chatbot integration.

Before running this script:
1. Make sure you have set GEMINI_API_KEY in your .env file
2. Install dependencies: pip install -r requirements.txt
3. Run: python demo_chatbot.py
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.dto.chat import ChatRequest
from app.service.chatbot_service import ChatbotService


async def demo_chat():
    """Demonstrate the chatbot functionality."""
    print("🏠 SmartRent AI Chatbot Demo")
    print("=" * 40)
    
    try:
        # Initialize the chatbot service
        print("Initializing chatbot service...")
        chatbot = ChatbotService()
        print("✅ Chatbot service initialized successfully!")
        print()
        
        # Test conversation
        conversation_id = None
        
        # Demo questions about SmartRent
        demo_questions = [
            "Hello! I'm new to SmartRent. Can you help me?",
            "How do I submit a maintenance request for my apartment?",
            "What smart home features are available?",
            "I'm having trouble with my smart lock. What should I do?",
        ]
        
        print("Starting demo conversation...")
        print("-" * 40)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"👤 User: {question}")
            
            # Create chat request
            request = ChatRequest(
                message=question,
                conversation_id=conversation_id,
                context="Demo user testing the SmartRent AI chatbot system"
            )
            
            # Get response from chatbot
            try:
                response = await chatbot.process_chat(request)
                conversation_id = response.conversation_id
                
                print(f"🤖 Assistant: {response.message}")
                print(f"   (Conversation ID: {conversation_id[:8]}...)")
                print(f"   (Response time: {response.timestamp.strftime('%H:%M:%S')})")
                
                # Show token usage if available
                if response.token_usage:
                    print(f"   📊 Tokens - Prompt: {response.token_usage.prompt_tokens}, "
                          f"Completion: {response.token_usage.completion_tokens}, "
                          f"Total: {response.token_usage.total_tokens}")
                print()
                
                # Small delay between messages
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Error getting response: {str(e)}")
                break
        
        # Show conversation history
        if conversation_id:
            print("📜 Retrieving conversation history...")
            history = await chatbot.get_conversation_history(conversation_id)
            if history:
                print(f"   Total messages: {len(history.messages)}")
                print(f"   Conversation started: {history.created_at.strftime('%H:%M:%S')}")
                print(f"   Last updated: {history.updated_at.strftime('%H:%M:%S')}")
            
            # List all conversations
            conversations = await chatbot.list_conversations()
            print(f"   Total active conversations: {len(conversations)}")
            
            # Show token usage statistics
            print("📊 Token Usage Statistics:")
            token_stats = await chatbot.get_token_usage_stats()
            print(f"   Requests made this session: {token_stats.requests_made_today}")
            print(f"   Last updated: {token_stats.last_updated.strftime('%H:%M:%S')}")
            print()
        
        print("✅ Demo completed successfully!")
        
    except ValueError as e:
        if "GEMINI_API_KEY is not configured" in str(e):
            print("❌ Configuration Error:")
            print("   Please set your GEMINI_API_KEY in the .env file")
            print("   1. Copy .env.example to .env")
            print("   2. Get your API key from: https://makersuite.google.com/app/apikey")
            print("   3. Set GEMINI_API_KEY=your_actual_api_key_here in .env")
        else:
            print(f"❌ Error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")


def check_environment():
    """Check if the environment is properly configured."""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_file):
        print("⚠️  .env file not found. Copy .env.example to .env and configure it.")
        return False
    
    # Check if google-generativeai is installed
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package is installed")
    except ImportError:
        print("❌ google-generativeai package not found")
        print("   Please install it: pip install google-generativeai==0.3.2")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ GEMINI_API_KEY not configured in .env file")
        return False
    
    print("✅ Environment looks good!")
    return True


if __name__ == "__main__":
    print(f"Demo started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if check_environment():
        print()
        asyncio.run(demo_chat())
    else:
        print("\n❌ Please fix the environment issues above and try again.")
    
    print(f"\nDemo ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
