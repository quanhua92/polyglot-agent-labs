"""
Polyglot Agent Labs — Use Case 02: Simple MCP Server
A minimal MCP server exposing a `get_weather` tool via FastMCP.
Run directly for stdio transport, or launched as subprocess by client.py.
"""

from fastmcp import FastMCP

mcp = FastMCP("Weather Server")

# Mock weather data
WEATHER_DATA = {
    "tokyo": "☀️ Sunny, 22°C — clear skies with light breeze",
    "london": "🌧️ Rainy, 14°C — overcast with intermittent showers",
    "new york": "⛅ Partly cloudy, 18°C — mild with occasional sun",
    "paris": "🌤️ Mostly sunny, 20°C — warm with gentle winds",
    "sydney": "🌡️ Hot, 30°C — bright sunshine, UV index high",
}


@mcp.tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: Name of the city to get weather for.
    """
    key = city.strip().lower()
    if key in WEATHER_DATA:
        return f"Weather in {city}: {WEATHER_DATA[key]}"
    return f"Weather in {city}: 🌍 No data available — try Tokyo, London, New York, Paris, or Sydney."


if __name__ == "__main__":
    mcp.run()
