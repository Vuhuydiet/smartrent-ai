# SmartRent AI - MCP Server Integration

## New MCP Server for Listing Access

An MCP (Model Context Protocol) server has been added to provide programmatic access to SmartRent backend listing APIs.

### Quick Start

1. **Install MCP dependency**:
```bash
pip install mcp
```

2. **Configure backend URL** in `.env`:
```bash
SMARTRENT_BACKEND_URL=http://localhost:8080
```

3. **Run the MCP server**:
```bash
python run_mcp_server.py
```

### Usage with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "smartrent-listings": {
      "command": "python",
      "args": ["run_mcp_server.py"],
      "cwd": "d:\\DATN\\smartrent-ai"
    }
  }
}
```

### Available MCP Tools

1. **get_listing** - Get a single listing by ID
2. **search_listings** - Search with filters (location, price, area, etc.)
3. **list_listings** - List paginated listings or get by IDs

See `app/mcp/README.md` for detailed documentation.

### Files Added

- `app/mcp/server.py` - MCP server implementation
- `app/mcp/listing_client.py` - HTTP client for backend API
- `app/mcp/README.md` - Detailed MCP documentation
- `run_mcp_server.py` - Standalone server runner

### Architecture

```
smartrent-ai (MCP Server)
    ↓ HTTP
smartrent-backend (Spring Boot API)
    ↓
MySQL Database
```

The MCP server acts as a bridge, exposing SmartRent backend APIs through the MCP protocol for AI assistants and other MCP clients.
