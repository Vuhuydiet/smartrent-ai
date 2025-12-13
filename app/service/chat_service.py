import logging
from typing import Any, Dict, List

import google.generativeai as genai  # type: ignore
import httpx
from google.ai.generativelanguage import (  # type: ignore
    Content,
    FunctionDeclaration,
    FunctionResponse,
    Part,
    Schema,
    Tool,
    Type,
)

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
        system_instruction = """Bạn là trợ lý tìm BĐS cho SmartRent tại Việt Nam.

Khi người dùng tìm BĐS, gọi search_listings với tiêu chí phù hợp.

Lưu ý: HN='01', HCM='79', giá VND, mặc định RENT"""

        # Define tools using genai types
        search_listings_func = FunctionDeclaration(
            name="search_listings",
            description="Search for real estate listings based on criteria",
            parameters=Schema(
                type=Type.OBJECT,
                properties={
                    "cityCode": Schema(
                        type=Type.STRING, description="City code (01=Hanoi, 79=HCM)"
                    ),
                    "districtCode": Schema(
                        type=Type.STRING, description="District code"
                    ),
                    "wardCode": Schema(type=Type.STRING, description="Ward code"),
                    "minPrice": Schema(
                        type=Type.NUMBER, description="Minimum price in VND"
                    ),
                    "maxPrice": Schema(
                        type=Type.NUMBER, description="Minimum price in VND"
                    ),
                    "minArea": Schema(
                        type=Type.NUMBER, description="Minimum area in sqm"
                    ),
                    "maxArea": Schema(
                        type=Type.NUMBER, description="Maximum area in sqm"
                    ),
                    "bedrooms": Schema(
                        type=Type.INTEGER, description="Number of bedrooms"
                    ),
                    "propertyType": Schema(
                        type=Type.STRING, description="ROOM, APARTMENT, HOUSE, LAND"
                    ),
                    "listingType": Schema(type=Type.STRING, description="RENT or SALE"),
                },
            ),
        )

        tools = Tool(function_declarations=[search_listings_func])

        # Initialize model with function calling
        logger.info("Initializing Gemini model with function declarations")
        self.model = genai.GenerativeModel(  # type: ignore[call-arg]
            model_name="gemini-2.0-flash",
            tools=[tools],  # Pass Tool object directly
        )
        self.system_instruction = system_instruction

    async def process_chat(self, messages: List[ChatMessage]) -> ChatResponse:
        """Process chat messages and return response."""
        try:
            logger.info("=== Starting process_chat ===")
            logger.info(f"Number of messages: {len(messages)}")
            logger.info(f"Last message: {messages[-1].content[:100]}...")

            # Convert messages to Gemini format
            chat_history = []

            # Add system instruction as first message if no history
            if len(messages) == 1:
                chat_history.append(
                    {
                        "role": "user",
                        "parts": [self.system_instruction],
                    }
                )
                chat_history.append(
                    {
                        "role": "model",
                        "parts": [
                            "Understood. I will help users find real estate in Vietnam and respond in Vietnamese."
                        ],
                    }
                )

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
            logger.info("=== Sending message to Gemini ===")
            response = chat.send_message(last_message)  # type: ignore

            logger.info("=== Received response from Gemini ===")
            logger.info(
                f"Response has {len(response.candidates[0].content.parts)} parts"
            )

            # Check if Gemini wants to call a function
            tools_used: List[str] = []
            listing_data = None  # Store listing data to return
            all_listings: List[Dict[str, Any]] = []  # Store all search results
            ai_rankings: List[Dict[str, Any]] = []  # Store AI rankings

            # Handle function calls
            while response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]

                # Check if it's a function call
                if hasattr(part, "function_call") and part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    tools_used.append(function_name)

                    logger.info(f"AI CALLED FUNCTION: {function_name}")
                    logger.info(f"Function call args: {dict(function_call.args)}")

                    if function_name == "search_listings":
                        logger.info(">>> Handling search_listings function call")
                        # Extract parameters
                        params = dict(function_call.args)

                        # Call backend API
                        search_results = await self._call_search_listings(params)
                        all_listings = search_results.get("listings", [])

                        logger.info(f"Search returned {len(all_listings)} listings")

                        # Send simplified acknowledgment - we'll rank separately
                        response = chat.send_message(  # type: ignore
                            Content(
                                parts=[
                                    Part(
                                        function_response=FunctionResponse(
                                            name=function_name,
                                            response={
                                                "status": "success",
                                                "message": f"Found {len(all_listings)} listings matching criteria.",
                                            },
                                        )
                                    )
                                ]
                            )
                        )
                        logger.info(">>> Search complete, will rank separately...")
                    else:
                        logger.warning(f">>> Unknown function name: {function_name}")
                        break
                else:
                    # No more function calls, get final text response
                    logger.info(
                        ">>> No more function calls detected, extracting final response"
                    )
                    break

            # Extract final text response safely
            final_response = ""
            try:
                # Try simple text accessor first
                final_response = response.text
            except ValueError:
                # Response has multiple parts or function calls, extract text manually
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        final_response = part.text
                        break

                # If still no text found, it might be a function call response
                if not final_response:
                    logger.warning("No text response found after function calls")
                    final_response = "Tôi đã tìm thấy một số bất động sản phù hợp."

            # If we have listings, call AI separately to rank them
            if all_listings:
                logger.info(f">>> Calling AI to rank {len(all_listings)} listings...")
                ai_rankings = await self._rank_listings_with_ai(
                    all_listings, messages[-1].content
                )
                logger.info(f">>> AI provided {len(ai_rankings)} rankings")

                # After ranking, call AI again to generate final message about TOP listings
                if ai_rankings:
                    sorted_rankings = sorted(
                        ai_rankings, key=lambda x: x.get("score", 0), reverse=True
                    )
                    top_rankings = sorted_rankings[: settings.MAX_LISTINGS_RETURN]

                    logger.info(
                        f">>> Calling AI to generate final message about top {len(top_rankings)} listings..."
                    )
                    final_response = await self._generate_final_message(
                        top_rankings, all_listings, messages[-1].content
                    )
                    logger.info(">>> Generated final message successfully")

            # Build listing_data from AI rankings
            if ai_rankings and all_listings:
                logger.info(
                    f">>> Building response from AI rankings: {len(ai_rankings)} items ranked"
                )
                # Sort by score (highest first) and take top N
                sorted_rankings = sorted(
                    ai_rankings, key=lambda x: x.get("score", 0), reverse=True
                )
                top_ranking_ids = [
                    r["listingId"]
                    for r in sorted_rankings[: settings.MAX_LISTINGS_RETURN]
                ]

                # Filter listings by top ranked IDs, preserve ranking order
                listings_map = {
                    listing["listingId"]: listing for listing in all_listings
                }
                selected_listings = [
                    listings_map[lid] for lid in top_ranking_ids if lid in listings_map
                ]

                listing_data = {
                    "listings": selected_listings,
                    "totalCount": len(selected_listings),
                    "currentPage": 1,
                    "pageSize": len(selected_listings),
                    "totalPages": 1,
                    "selectedFromTotal": len(all_listings),
                    "aiRankings": sorted_rankings[
                        : settings.MAX_LISTINGS_RETURN
                    ],  # Include AI's reasoning
                }
                logger.info(
                    f">>> SUCCESS: Returning {len(selected_listings)} top-ranked listings from {len(all_listings)} total"
                )
                logger.info(f">>> Top ranked IDs: {top_ranking_ids}")
            elif all_listings:
                # No rankings - shouldn't happen
                logger.error(">>> ERROR: Failed to get rankings from AI!")
                logger.error(f">>> all_listings count: {len(all_listings)}")
                # Return first N as emergency fallback
                selected_listings = all_listings[: settings.MAX_LISTINGS_RETURN]
                listing_data = {
                    "listings": selected_listings,
                    "totalCount": len(selected_listings),
                    "currentPage": 1,
                    "pageSize": len(selected_listings),
                    "totalPages": 1,
                    "selectedFromTotal": len(all_listings),
                }
                logger.error(
                    f">>> Emergency fallback: returning first {len(selected_listings)} listings"
                )
            else:
                logger.warning(">>> No listings available at all!")

            return ChatResponse(
                message=ChatMessage(role="assistant", content=final_response),
                metadata={
                    "tools_used": tools_used,
                    "model": "gemini-2.0-flash",
                },
                listings=listing_data,
            )

        except Exception as e:
            logger.error(
                f"Error in chat service: {type(e).__name__}: {str(e)}", exc_info=True
            )
            raise Exception(
                f"Failed to process chat: {type(e).__name__}: {str(e)}"
            ) from e

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
            logger.error(
                f"HTTP error calling backend: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            return {"error": f"Failed to search listings: {str(e)}"}
        except Exception as e:
            logger.error(
                f"Error calling backend: {type(e).__name__}: {str(e)}", exc_info=True
            )
            return {"error": f"Error searching listings: {str(e)}"}

    async def _rank_listings_with_ai(
        self, all_listings: List[Dict[str, Any]], user_query: str
    ) -> List[Dict[str, Any]]:
        """
        Call AI separately to rank listings.

        Args:
            all_listings: All listings from search
            user_query: Original user query

        Returns:
            List of rankings: [{listingId, score, reason}, ...]
        """
        try:
            # Simplify listings for AI
            simplified_listings = [
                {
                    "listingId": listing.get("listingId"),
                    "price": listing.get("price"),
                    "area": listing.get("area"),
                    "bedrooms": listing.get("bedrooms"),
                    "bathrooms": listing.get("bathrooms"),
                    "district": listing.get("districtName"),
                    "ward": listing.get("wardName"),
                    "propertyType": listing.get("propertyType"),
                }
                for listing in all_listings
            ]

            # Create ranking prompt
            ranking_prompt = f"""Yêu cầu của người dùng: {user_query}

Danh sách {len(simplified_listings)} BĐS tìm được:
{simplified_listings}

Hãy phân tích và đánh giá từng BĐS theo tiêu chí:
- Độ phù hợp với yêu cầu người dùng
- Giá trị đồng tiền (price/area ratio)
- Vị trí
- Tiện nghi (số phòng, diện tích)

Trả về JSON array với format:
[
  {{"listingId": <id>, "score": <0-100>, "reason": "<lý do ngắn gọn>"}},
  ...
]

Chỉ trả về JSON, không thêm text nào khác."""

            # Call AI without JSON mode (not supported in this version)
            response = self.model.generate_content(ranking_prompt)

            # Parse JSON response - handle markdown code blocks if present
            import json
            import re

            response_text = response.text.strip()

            # Try to extract JSON from markdown code block if present
            json_match = re.search(
                r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL
            )
            if json_match:
                response_text = json_match.group(1)

            # Remove any leading/trailing non-JSON text
            if response_text.startswith("["):
                # Find the end of JSON array
                try:
                    rankings = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to find just the JSON array portion
                    json_end = response_text.rfind("]")
                    if json_end > 0:
                        rankings = json.loads(response_text[: json_end + 1])
                    else:
                        raise
            else:
                # Try to find JSON array in response
                json_start = response_text.find("[")
                json_end = response_text.rfind("]")
                if json_start >= 0 and json_end > json_start:
                    rankings = json.loads(response_text[json_start : json_end + 1])
                else:
                    raise ValueError("No JSON array found in response")

            logger.info(f"AI ranked {len(rankings)} listings successfully")
            return rankings

        except Exception as e:
            logger.error(f"Error ranking with AI: {e}")
            # Return empty to trigger emergency fallback
            return []

    async def _generate_final_message(
        self,
        top_rankings: List[Dict[str, Any]],
        all_listings: List[Dict[str, Any]],
        user_query: str,
    ) -> str:
        """
        Generate final Vietnamese message presenting the top listings.

        Args:
            top_rankings: Top N rankings from AI
            all_listings: All original listings
            user_query: Original user query

        Returns:
            Vietnamese message introducing the listings
        """
        try:
            # Get full details of top listings
            listings_map = {listing["listingId"]: listing for listing in all_listings}
            top_listings = [
                listings_map[r["listingId"]]
                for r in top_rankings
                if r["listingId"] in listings_map
            ]

            # Create presentation prompt
            presentation_prompt = f"""Yêu cầu của người dùng: {user_query}

Bạn đã phân tích và chọn được {len(top_listings)} BĐS tốt nhất từ {len(all_listings)} kết quả tìm kiếm.

Danh sách {len(top_listings)} BĐS đã chọn (theo thứ tự từ tốt nhất):
{top_listings}

Lý do chọn mỗi căn:
{top_rankings}

Hãy viết một đoạn giới thiệu bằng TIẾNG VIỆT tự nhiên, thân thiện:
1. Mở đầu: "Tôi đã tìm được {len(top_listings)} phòng trọ phù hợp với yêu cầu của bạn"
2. Giới thiệu ngắn gọn từng căn (2-3 câu mỗi căn):
   - Giá, diện tích, vị trí
   - Điểm nổi bật (tại sao chọn căn này)
3. Kết: Gợi ý người dùng xem chi tiết

Chỉ trả về đoạn text tiếng Việt, KHÔNG thêm JSON hay format khác."""

            response = self.model.generate_content(presentation_prompt)
            return response.text.strip()

        except Exception as e:
            logger.error(f"Error generating final message: {e}")
            # Fallback message
            return f"Tôi đã tìm được {len(top_rankings)} bất động sản phù hợp với yêu cầu của bạn. Hãy xem chi tiết bên dưới."
