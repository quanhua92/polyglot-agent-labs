"""
Polyglot Agent Labs — Use Case 02: MCP Client
Connects to the Weather MCP server (server.py) via stdio subprocess,
calls the get_weather tool, and prints results.
"""

import asyncio
import sys

from fastmcp import Client


async def main():
    print("=== Python — MCP Client → Weather Server ===")
    print()

    # FastMCP Client auto-launches server.py as a subprocess over stdio
    client = Client("server.py")

    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")
        print()

        # Call with known cities
        for city in ["Tokyo", "London", "Sydney"]:
            result = await client.call_tool("get_weather", {"city": city})
            print(f"  {result.content[0].text}")

        # Call with unknown city
        print()
        result = await client.call_tool("get_weather", {"city": "Mars"})
        print(f"  {result.content[0].text}")

    print()
    print("✓ MCP client-server communication successful!")


if __name__ == "__main__":
    asyncio.run(main())
