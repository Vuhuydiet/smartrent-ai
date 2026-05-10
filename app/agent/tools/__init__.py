"""
Public registry for all chat-agent tools.

Tools are plain `@function_tool` decorated coroutines — pass them straight
into `Agent(tools=[...])`. No per-tool class hierarchy is needed.
"""

from typing import Any, List

from app.agent.tools.address_translator import address_translator
from app.agent.tools.bulk_save_listings import bulk_save_listings
from app.agent.tools.compare_listings import compare_listings
from app.agent.tools.get_listing_detail import get_listing_detail
from app.agent.tools.get_price_estimate import get_price_estimate
from app.agent.tools.get_price_history import get_price_history
from app.agent.tools.get_recommendations import get_recommendations
from app.agent.tools.get_user_info import get_user_info
from app.agent.tools.my_listings_status import my_listings_status
from app.agent.tools.notifications_inbox import notifications_inbox
from app.agent.tools.report_listing import report_listing
from app.agent.tools.save_listing import save_listing
from app.agent.tools.search_listings import search_listings
from app.agent.tools.update_listing_price import update_listing_price


def get_chat_tools() -> List[Any]:
    """Return the list of @function_tool callables registered with the chat agent."""
    return [
        # Existing
        search_listings,
        get_listing_detail,
        get_price_estimate,
        get_price_history,
        get_recommendations,
        get_user_info,
        save_listing,
        # Sprint v2 — Tier 1 (flagship)
        compare_listings,
        my_listings_status,
        address_translator,
        # Sprint v2 — Tier 2 (micro-actions)
        bulk_save_listings,
        update_listing_price,
        notifications_inbox,
        report_listing,
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
    "compare_listings",
    "my_listings_status",
    "address_translator",
    "bulk_save_listings",
    "update_listing_price",
    "notifications_inbox",
    "report_listing",
]
