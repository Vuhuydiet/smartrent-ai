from app.agent.tools.base_tool import BaseTool
from app.agent.tools.get_listing_detail import GetListingDetailTool
from app.agent.tools.get_nearby_places import GetNearbyPlacesTool
from app.agent.tools.get_price_estimate import GetPriceEstimateTool
from app.agent.tools.get_price_history import GetPriceHistoryTool
from app.agent.tools.get_recommendations import GetRecommendationsTool
from app.agent.tools.get_user_info import GetUserInfoTool
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.save_listing import SaveListingTool
from app.agent.tools.search_listings import SearchListingsTool


def build_default_registry() -> ToolRegistry:
    """Build and return a ToolRegistry pre-loaded with all agent tools."""
    registry = ToolRegistry()
    registry.register(SearchListingsTool())
    registry.register(GetListingDetailTool())
    registry.register(GetPriceEstimateTool())
    registry.register(GetPriceHistoryTool())
    registry.register(GetNearbyPlacesTool())
    registry.register(GetRecommendationsTool())
    registry.register(GetUserInfoTool())
    registry.register(SaveListingTool())
    return registry


__all__ = [
    "BaseTool",
    "ToolRegistry",
    "SearchListingsTool",
    "GetListingDetailTool",
    "GetNearbyPlacesTool",
    "GetPriceEstimateTool",
    "GetPriceHistoryTool",
    "GetRecommendationsTool",
    "GetUserInfoTool",
    "SaveListingTool",
    "build_default_registry",
]
