#!/usr/bin/env python
"""Standalone MCP server runner for SmartRent listings."""
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if __name__ == "__main__":
    import asyncio

    from app.mcp.server import main

    asyncio.run(main())
