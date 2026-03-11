#!/usr/bin/env python3
"""FastMCP server with an add tool using HTTP transport."""
from fastmcp import FastMCP

# Create the FastMCP server instance
mcp = FastMCP("Add Server")

@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers together.

    Args:
        a: First number to add
        b: Second number to add

    Returns:
        The sum of a and b as a string
    """
    result = a + b
    return f"{a} + {b} = {result}"

if __name__ == "__main__":
    # Run the MCP server with HTTP transport
    import asyncio

    async def run():
        await mcp.run_http_async(
            transport="streamable-http",
            host="0.0.0.0",
            port=8002,
            path="/mcp"
        )

    asyncio.run(run())
