"""
Polyglot Agent Labs — Use Case 02: Full-Featured MCP Server
Demonstrates tools, prompts, and resources via FastMCP — mirroring
the counter.rs example from the official rust-sdk.

Run directly for stdio transport, or launched as subprocess by main.py.
"""

from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP("Weather & Counter Server")

# ─── State ───────────────────────────────────────────────────────────────────

_counter = 0

WEATHER_DATA = {
    "tokyo": "☀️ Sunny, 22°C — clear skies with light breeze",
    "london": "🌧️ Rainy, 14°C — overcast with intermittent showers",
    "new york": "⛅ Partly cloudy, 18°C — mild with occasional sun",
    "paris": "🌤️ Mostly sunny, 20°C — warm with gentle winds",
    "sydney": "🌡️ Hot, 30°C — bright sunshine, UV index high",
}

# ─── Tools ───────────────────────────────────────────────────────────────────


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


@mcp.tool
def increment() -> str:
    """Increment the counter by 1."""
    global _counter
    _counter += 1
    return str(_counter)


@mcp.tool
def decrement() -> str:
    """Decrement the counter by 1."""
    global _counter
    _counter -= 1
    return str(_counter)


@mcp.tool
def get_value() -> str:
    """Get the current counter value."""
    return str(_counter)


@mcp.tool
def say_hello() -> str:
    """Say hello to the client."""
    return "hello"


@mcp.tool
def echo(object: dict) -> str:
    """Repeat what you say.

    Args:
        object: An arbitrary JSON object to echo back.
    """
    import json
    return json.dumps(object, separators=(',', ':'))


@mcp.tool
def sum(a: int, b: int) -> str:
    """Calculate the sum of two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


# ─── Prompts ─────────────────────────────────────────────────────────────────


@mcp.prompt
def example_prompt(message: str) -> str:
    """A simple example prompt that takes one required argument.

    Args:
        message: A message to put in the prompt.
    """
    return f"This is an example prompt with your message here: '{message}'"


@mcp.prompt
def weather_analysis(city: str, style: str = "brief") -> list["Message"]:
    """Analyze the weather for a city, combining with counter state.

    Args:
        city: City to analyze weather for.
        style: Preferred style — 'brief' or 'detailed'.
    """
    key = city.strip().lower()
    weather_text = WEATHER_DATA.get(key, "No weather data available for this city")
    return [
        Message(
            {
                "role": "assistant",
                "content": "I'll analyze the weather situation and suggest the best approach.",
            }
        ),
        Message(
            {
                "role": "user",
                "content": (
                    f"City: {city}\n"
                    f"Weather: {weather_text}\n"
                    f"Counter value: {_counter}\n"
                    f"Style preference: {style}\n\n"
                    "Please analyze the weather and suggest activities."
                ),
            }
        ),
    ]


# ─── Resources ───────────────────────────────────────────────────────────────


@mcp.resource("weather://cities", name="Available Cities")
def weather_cities() -> str:
    """List of available cities with weather data."""
    return "Available cities: Tokyo, London, New York, Paris, Sydney"


@mcp.resource("counter://value", name="Counter Value")
def counter_value() -> str:
    """Current counter value."""
    return str(_counter)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
