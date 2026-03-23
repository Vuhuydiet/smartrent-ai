from app.agent.tools.base_tool import BaseTool
from app.agent.tools.get_listing_detail import GetListingDetailTool
from app.agent.tools.get_price_estimate import GetPriceEstimateTool
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.search_listings import SearchListingsTool


def build_default_registry() -> ToolRegistry:
    """Build and return a ToolRegistry pre-loaded with all agent tools."""
    registry = ToolRegistry()
    registry.register(SearchListingsTool())
    registry.register(GetListingDetailTool())
    registry.register(GetPriceEstimateTool())
    return registry


__all__ = [
    "BaseTool",
    "ToolRegistry",
    "SearchListingsTool",
    "GetListingDetailTool",
    "GetPriceEstimateTool",
    "build_default_registry",
]
