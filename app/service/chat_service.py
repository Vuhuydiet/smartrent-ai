import logging
from typing import Any, Dict, List

import google.generativeai as genai  # type: ignore
from google.ai.generativelanguage import Content, Part, FunctionResponse  # type: ignore
import httpx

from app.core.config import settings
from app.dto.chat import ChatMessage, ChatResponse

# Type checking ignored for genai.protos - library lacks complete type stubs
# mypy: disable-error-code="attr-defined"

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat conversations with tool calling."""

    def __init__(self) -> None:
        """Initialize the chat service with Gemini and tools."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")

        genai.configure(api_key=settings.GEMINI_API_KEY)  # type: ignore

        # System instruction
        system_instruction = """You are a helpful real estate assistant for SmartRent, a property rental platform in Vietnam.

Your role is to help users find suitable rental properties based on their requirements. When users ask about finding properties:
1. Extract their requirements (location, price range, property type, size, etc.)
2. Use the search_listings tool to find matching properties
3. Present results in a friendly, conversational way
4. Highlight key details like price, area, location, and number of bedrooms/bathrooms
5. Ask clarifying questions if requirements are unclear

Important notes:
- Prices are in VND (Vietnamese Dong)
- Common locations: Hanoi (province_code: '01'), Ho Chi Minh City (province_code: '79')
- Default listing_type is RENT unless user specifies otherwise
- Be conversational and helpful, not robotic"""

        # Define search_listings tool
        search_listings_tool = {
            "function_declarations": [
                {
                    "name": "search_listings",
                    "description": "Search property listings from SmartRent backend. Use this when users want to find rental properties, apartments, houses, or other real estate.",
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {
                            "province_code": {
                                "type_": "STRING",
                                "description": "Province code (e.g., '01' for Hanoi, '79' for Ho Chi Minh City)",
                            },
                            "district_id": {
                                "type_": "INTEGER",
                                "description": "District ID within province",
                            },
                            "ward_id": {
                                "type_": "STRING",
                                "description": "Ward ID within district",
                            },
                            "listing_type": {
                                "type_": "STRING",
                                "description": "Type of listing: RENT, SALE, or SHARE",
                                "enum": ["RENT", "SALE", "SHARE"],
                            },
                            "product_type": {
                                "type_": "STRING",
                                "description": "Type of property: ROOM, APARTMENT, HOUSE, OFFICE, or STUDIO",
                                "enum": ["ROOM", "APARTMENT", "HOUSE", "OFFICE", "STUDIO"],
                            },
                            "min_price": {
                                "type_": "NUMBER",
                                "description": "Minimum price in VND",
                            },
                            "max_price": {
                                "type_": "NUMBER",
                                "description": "Maximum price in VND",
                            },
                            "price_unit": {
                                "type_": "STRING",
                                "description": "Price unit: MONTH, DAY, or YEAR",
                                "enum": ["MONTH", "DAY", "YEAR"],
                            },
                            "min_area": {
                                "type_": "NUMBER",
                                "description": "Minimum area in square meters",
                            },
                            "max_area": {
                                "type_": "NUMBER",
                                "description": "Maximum area in square meters",
                            },
                            "min_bedrooms": {
                                "type_": "INTEGER",
                                "description": "Minimum number of bedrooms",
                            },
                            "max_bedrooms": {
                                "type_": "INTEGER",
                                "description": "Maximum number of bedrooms",
                            },
                            "min_bathrooms": {
                                "type_": "INTEGER",
                                "description": "Minimum number of bathrooms",
                            },
                            "max_bathrooms": {
                                "type_": "INTEGER",
                                "description": "Maximum number of bathrooms",
                            },
                            "furnishing": {
                                "type_": "STRING",
                                "description": "Furnishing status",
                                "enum": ["FULLY_FURNISHED", "SEMI_FURNISHED", "UNFURNISHED"],
                            },
                            "keyword": {
                                "type_": "STRING",
                                "description": "Keyword to search in title and description",
                            },
                            "page": {
                                "type_": "INTEGER",
                                "description": "Page number (default 1)",
                            },
                            "size": {
                                "type_": "INTEGER",
                                "description": "Results per page (default 20, max 100)",
                            },
                            "sort_by": {
                                "type_": "STRING",
                                "description": "Sort field",
                                "enum": ["DEFAULT", "PRICE_ASC", "PRICE_DESC", "NEWEST", "OLDEST"],
                            },
                        },
                    },
                }
            ]
        }

        # Initialize model with tools
        self.model = genai.GenerativeModel(  # type: ignore[call-arg]
            model_name="gemini-2.0-flash",
            tools=[search_listings_tool],  # type: ignore[arg-type]
        )
        self.system_instruction = system_instruction

    async def process_chat(self, messages: List[ChatMessage]) -> ChatResponse:
        """Process chat messages and return response."""
        try:
            # Convert messages to Gemini format
            chat_history = []
            for msg in messages[:-1]:  # All except last message
                chat_history.append(  # type: ignore
                    {
                        "role": "user" if msg.role == "user" else "model",
                        "parts": [msg.content],
                    }
                )

            # Start chat with history
            chat = self.model.start_chat(history=chat_history)  # type: ignore

            # Send last message
            last_message = messages[-1].content
            response = chat.send_message(last_message)  # type: ignore

            # Check if Gemini wants to call a function
            tools_used: List[str] = []

            # Handle function calls
            while response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]

                # Check if it's a function call
                if hasattr(part, "function_call") and part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    tools_used.append(function_name)

                    logger.info(f"Gemini called function: {function_name}")

                    if function_name == "search_listings":
                        # Extract parameters
                        params = dict(function_call.args)

                        # Call backend API
                        search_results = await self._call_search_listings(params)

                        # Send results back to Gemini using proper types
                        response = chat.send_message(  # type: ignore
                            Content(
                                parts=[
                                    Part(
                                        function_response=FunctionResponse(
                                            name=function_name,
                                            response={"result": search_results},
                                        )
                                    )
                                ]
                            )
                        )
                    else:
                        break
                else:
                    # No more function calls, get final text response
                    break

            # Extract final text response
            final_response = response.text

            return ChatResponse(
                message=ChatMessage(role="assistant", content=final_response),
                metadata={
                    "tools_used": tools_used,
                    "model": "gemini-2.0-flash",
                },
            )

        except Exception as e:
            logger.error(
                f"Error in chat service: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise Exception(f"Failed to process chat: {type(e).__name__}: {str(e)}") from e

    async def _call_search_listings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the backend search listings API."""
        try:
            # Add default excludeExpired if not specified
            if "excludeExpired" not in params:
                params["excludeExpired"] = True

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.SMARTRENT_BACKEND_URL}/v1/listings/search",
                    json=params,
                    timeout=30.0,
                )
                response.raise_for_status()

                result = response.json()

                # Extract data from API response
                if result.get("code") == "999999" and "data" in result:
                    return result["data"]
                else:
                    return {
                        "error": result.get("message", "Unknown error"),
                        "code": result.get("code"),
                    }

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling backend: {type(e).__name__}: {str(e)}", exc_info=True)
            return {"error": f"Failed to search listings: {str(e)}"}
        except Exception as e:
            logger.error(f"Error calling backend: {type(e).__name__}: {str(e)}", exc_info=True)
            return {"error": f"Error searching listings: {str(e)}"}
