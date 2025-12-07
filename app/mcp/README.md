# SmartRent MCP Server

This directory contains the Model Context Protocol (MCP) server for SmartRent, which provides access to listing data from the SmartRent backend and AI-powered house pricing predictions.

## Overview

The MCP server exposes both listing operations and house pricing predictions as MCP tools that can be called by MCP clients (like Claude Desktop, IDEs, or other AI applications).

## Features

### Listing Operations
- **Get Listing**: Retrieve a single listing by ID
- **Search Listings**: Comprehensive search with multiple filters (location, price, area, amenities, etc.)
- **List Listings**: Get paginated listings or specific listings by IDs

### House Pricing Predictions
- **Predict House Price**: AI-powered price range prediction based on location, property type, and coordinates

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the backend URL in `.env`:
```
SMARTRENT_BACKEND_URL=http://localhost:8080
SMARTRENT_AI_URL=http://localhost:8000
```

## Running the MCP Server

### Prerequisites
Make sure both services are running:

1. **SmartRent Backend** (for listing APIs):
```bash
cd smartrent-backend/smart-rent
./gradlew bootRun
```

2. **SmartRent AI Service** (for house pricing):
```bash
cd smartrent-ai
uvicorn app.main:app --reload
```

### Start the MCP Server

### Using Python directly:
```bash
python -m app.mcp.server
```

### Using uv (recommended):
```bash
uv run app/mcp/server.py
```

## Configuration for MCP Clients

### Claude Desktop Configuration

Add to your Claude Desktop config file (`claude_desktop_config.json`):

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "smartrent-listings": {
      "command": "uv",
      "args": [
        "--directory",
        "d:\\DATN\\smartrent-ai",
        "run",
        "app/mcp/server.py"
      ]
    }
  }
}
```

Or using Python directly:
```json
{
  "mcpServers": {
    "smartrent": {
      "command": "python",
      "args": [
        "-m",
        "app.mcp.server"
      ],
      "cwd": "d:\\DATN\\smartrent-ai"
    }
  }
}
```

## Available Tools

### 1. get_listing

Get a single listing by ID.

**Parameters:**
- `listing_id` (integer, required): The ID of the listing to retrieve

**Example:**
```json
{
  "listing_id": 123
}
```

### 2. predict_house_price

Predict price range for a property using AI.

**Parameters:**
- `city` (string, required): City or province name (e.g., "Hanoi", "Ho Chi Minh City")
- `district` (string, required): District or county name
- `ward` (string, required): Ward or commune name
- `property_type` (string, required): Type of property - "APARTMENT", "HOUSE", "ROOM", or "STUDIO"
- `latitude` (number, required): Property latitude coordinate
- `longitude` (number, required): Property longitude coordinate
- `area` (number, optional): Property area in square meters

**Example:**
```json
{
  "city": "Hanoi",
  "district": "Ba Dinh",
  "ward": "Dien Bien",
  "property_type": "APARTMENT",
  "latitude": 21.0285,
  "longitude": 105.8342,
  "area": 75.5
}
```

**Response:**
```json
{
  "price_range": {
    "min": 15000000,
    "max": 50000000
  },
  "location": "Ba Dinh, Hanoi",
  "property_type": "APARTMENT",
  "currency": "VND"
}
```

### 3. search_listings

Search and filter listings with comprehensive options.

**Parameters:**
- `keyword` (string, optional): Search keyword for title or description
- `listing_type` (string, optional): Type of listing - "RENT", "SALE", or "SHARE"
- `province_id` (string, optional): Province ID for location filter
- `district_id` (string, optional): District ID for location filter
- `ward_id` (string, optional): Ward ID for location filter
- `min_price` (number, optional): Minimum price
- `max_price` (number, optional): Maximum price
- `min_area` (number, optional): Minimum area in m²
- `max_area` (number, optional): Maximum area in m²
- `product_type` (string, optional): "APARTMENT", "HOUSE", "ROOM", "STUDIO", or "OFFICE"
- `min_bedrooms` (integer, optional): Minimum number of bedrooms
- `max_bedrooms` (integer, optional): Maximum number of bedrooms
- `page` (integer, optional): Page number (0-based), default: 0
- `size` (integer, optional): Page size, default: 20

**Example:**
```json
{
  "listing_type": "RENT",
  "product_type": "APARTMENT",
  "min_price": 5000000,
  "max_price": 15000000,
  "min_bedrooms": 2,
  "page": 0,
  "size": 10
}
```

### 4. list_listings

List listings with pagination or get specific listings by IDs.

**Parameters:**
- `ids` (array of integers, optional): List of listing IDs to fetch
- `page` (integer, optional): Page number (0-based), default: 0
- `size` (integer, optional): Page size (max 100), default: 20

**Example (by IDs):**
```json
{
  "ids": [123, 456, 789]
}
```

**Example (paginated):**
```json
{
  "page": 0,
  "size": 20
}
```

## Architecture

```
MCP Client (Claude Desktop, IDEs)
    ↓ MCP Protocol
SmartRent MCP Server
    ↓ HTTP (Listing APIs)
SmartRent Backend (Spring Boot) → MySQL
    ↓ HTTP (House Pricing)
SmartRent AI Service (FastAPI) → ML Model
```

The MCP server acts as a bridge, exposing both SmartRent backend APIs and AI services through the MCP protocol for AI assistants and other MCP clients.

## File Structure

```
app/mcp/
├── __init__.py                  # Package initialization
├── server.py                    # MCP server implementation
├── listing_client.py            # HTTP client for backend listing API
├── house_pricing_client.py      # HTTP client for AI pricing API
└── README.md                    # This file
```

## Error Handling

The server handles common HTTP errors:
- 404: Listing not found
- 500: Backend server error
- Network errors: Connection timeout or failure

Errors are returned as text content with error details.

## Development

### Testing the MCP Server

1. Start the SmartRent backend:
```bash
cd smartrent-backend/smart-rent
./gradlew bootRun
```

2. Start the SmartRent AI service:
```bash
cd smartrent-ai
uvicorn app.main:app --reload
```

3. Run the MCP server:
```bash
cd smartrent-ai
python run_mcp_server.py
```

4. Test with an MCP client or use the MCP Inspector tool

### Adding New Tools

1. Add the tool definition in `_setup_handlers()` method in `server.py`
2. Implement the handler in `call_tool()` method
3. Add corresponding method in the appropriate client (`ListingClient` or `HousePricingClient`) if needed
4. Update this README with the new tool documentation

## Use Cases

### For Real Estate Agents
- Search listings by location and criteria
- Get detailed listing information
- Predict market prices for new properties
- Validate asking prices against market rates

### For Property Owners
- Get AI-powered price suggestions for their property
- Compare their property with similar listings
- Understand price ranges in different locations

### For AI Assistants
- Answer user questions about available properties
- Provide price estimates for properties
- Compare multiple listings
- Suggest properties based on user preferences
