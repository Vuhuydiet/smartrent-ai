"""MCP Server for SmartRent - Provides listing access and house pricing via MCP protocol."""
import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from app.core.config import settings
from app.mcp.house_pricing_client import HousePricingClient
from app.mcp.listing_client import ListingClient


class SmartRentMCPServer:
    """SmartRent MCP Server for listing operations and house pricing predictions."""

    def __init__(self) -> None:
        """Initialize the MCP server."""
        self.server = Server("smartrent-server")
        self.listing_client = ListingClient(base_url=settings.SMARTRENT_BACKEND_URL)
        self.pricing_client = HousePricingClient(
            base_url="http://localhost:8000"  # SmartRent AI service
        )
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup MCP request handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available MCP tools."""
            return [
                Tool(
                    name="get_listing",
                    description="Get a single listing by ID from SmartRent backend",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "listing_id": {
                                "type": "integer",
                                "description": "The ID of the listing to retrieve",
                            }
                        },
                        "required": ["listing_id"],
                    },
                ),
                Tool(
                    name="predict_house_price",
                    description="Predict price range for a property using AI. Returns estimated min and max price in VND based on location, property type, and coordinates.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City or province name (e.g., 'Hanoi', 'Ho Chi Minh City', 'Tien Giang')",
                            },
                            "district": {
                                "type": "string",
                                "description": "District or county name (e.g., 'Ba Dinh', 'District 1', 'My Tho')",
                            },
                            "ward": {
                                "type": "string",
                                "description": "Ward or commune name (e.g., 'Dien Bien Ward', 'Ward 1')",
                            },
                            "property_type": {
                                "type": "string",
                                "enum": ["APARTMENT", "HOUSE", "ROOM", "STUDIO"],
                                "description": "Type of property",
                            },
                            "latitude": {
                                "type": "number",
                                "description": "Property latitude coordinate (decimal degrees)",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Property longitude coordinate (decimal degrees)",
                            },
                            "area": {
                                "type": "number",
                                "description": "Property area in square meters (m²) - optional",
                            },
                        },
                        "required": [
                            "city",
                            "district",
                            "ward",
                            "property_type",
                            "latitude",
                            "longitude",
                        ],
                    },
                ),
                Tool(
                    name="search_listings",
                    description="Search and filter listings from SmartRent backend. Supports filtering by location, price, area, amenities, and more.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "Search keyword for title or description",
                            },
                            "listing_type": {
                                "type": "string",
                                "enum": ["RENT", "SALE", "SHARE"],
                                "description": "Type of listing",
                            },
                            "province_id": {
                                "type": "string",
                                "description": "Province ID for location filter",
                            },
                            "district_id": {
                                "type": "string",
                                "description": "District ID for location filter",
                            },
                            "ward_id": {
                                "type": "string",
                                "description": "Ward ID for location filter",
                            },
                            "min_price": {
                                "type": "number",
                                "description": "Minimum price",
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Maximum price",
                            },
                            "min_area": {
                                "type": "number",
                                "description": "Minimum area in m²",
                            },
                            "max_area": {
                                "type": "number",
                                "description": "Maximum area in m²",
                            },
                            "product_type": {
                                "type": "string",
                                "enum": [
                                    "APARTMENT",
                                    "HOUSE",
                                    "ROOM",
                                    "STUDIO",
                                    "OFFICE",
                                ],
                                "description": "Type of property",
                            },
                            "min_bedrooms": {
                                "type": "integer",
                                "description": "Minimum number of bedrooms",
                            },
                            "max_bedrooms": {
                                "type": "integer",
                                "description": "Maximum number of bedrooms",
                            },
                            "page": {
                                "type": "integer",
                                "description": "Page number (0-based)",
                                "default": 0,
                            },
                            "size": {
                                "type": "integer",
                                "description": "Page size",
                                "default": 20,
                            },
                        },
                    },
                ),
                Tool(
                    name="list_listings",
                    description="List listings with pagination or get specific listings by IDs",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Optional list of listing IDs to fetch",
                            },
                            "page": {
                                "type": "integer",
                                "description": "Page number (0-based)",
                                "default": 0,
                            },
                            "size": {
                                "type": "integer",
                                "description": "Page size (max 100)",
                                "default": 20,
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle tool calls."""
            if name == "get_listing":
                listing_id = arguments.get("listing_id")
                result = await self.listing_client.get_listing(listing_id)
                return [TextContent(type="text", text=str(result))]

            elif name == "predict_house_price":
                city = arguments.get("city")
                district = arguments.get("district")
                ward = arguments.get("ward")
                property_type = arguments.get("property_type")
                latitude = arguments.get("latitude")
                longitude = arguments.get("longitude")
                area = arguments.get("area")

                result = await self.pricing_client.get_price_range(
                    city=city,
                    district=district,
                    ward=ward,
                    property_type=property_type,
                    latitude=latitude,
                    longitude=longitude,
                    area=area,
                )
                return [TextContent(type="text", text=str(result))]

            elif name == "search_listings":
                result = await self.listing_client.search_listings(arguments)
                return [TextContent(type="text", text=str(result))]

            elif name == "list_listings":
                ids = arguments.get("ids")
                page = arguments.get("page", 0)
                size = arguments.get("size", 20)
                result = await self.listing_client.list_listings(
                    ids=ids, page=page, size=size
                )
                return [TextContent(type="text", text=str(result))]

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    """Main entry point for the MCP server."""
    server = SmartRentMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
