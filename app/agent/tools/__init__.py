"""
Public registry for all chat-agent tools.

Tools are plain `@function_tool` decorated coroutines — pass them straight
into `Agent(tools=[...])`. No per-tool class hierarchy is needed.
"""

from typing import Any, List

from app.agent.tools.get_listing_detail import get_listing_detail
from app.agent.tools.get_price_estimate import get_price_estimate
from app.agent.tools.get_price_history import get_price_history
from app.agent.tools.get_recommendations import get_recommendations
from app.agent.tools.get_user_info import get_user_info
from app.agent.tools.save_listing import save_listing
from app.agent.tools.search_listings import search_listings


def get_chat_tools() -> List[Any]:
    """Return the list of @function_tool callables registered with the chat agent."""
    return [
        search_listings,
        get_listing_detail,
        get_price_estimate,
        get_price_history,
        get_recommendations,
        get_user_info,
        save_listing,
    ]


__all__ = [
    "get_chat_tools",
    "search_listings",
    "get_listing_detail",
    "get_price_estimate",
    "get_price_history",
    "get_recommendations",
    "get_user_info",
    "save_listing",
]
