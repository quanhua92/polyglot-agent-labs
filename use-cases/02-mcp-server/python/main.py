"""
Polyglot Agent Labs — Use Case 02: MCP Client
Connects to the Weather & Counter MCP server (server.py) via stdio subprocess,
exercises all tools, resources, and prompts, then prints results.
"""

import asyncio

from fastmcp import Client


async def main():
    print("=== Python -- MCP Client -> Weather & Counter Server ===")
    print()

    # FastMCP Client auto-launches server.py as a subprocess over stdio
    client = Client("server.py")

    async with client:
        # ── 1. List available tools ──────────────────────────────────────
        tools = await client.list_tools()
        import json
        tool_names = sorted([t.name for t in tools])
        print(f"📦 Available tools: {json.dumps(tool_names)}")
        print()

        # ── 2. Call get_weather ──────────────────────────────────────────
        print("── Weather Tools ──")
        for city in ["Tokyo", "London", "Sydney", "Mars"]:
            result = await client.call_tool("get_weather", {"city": city})
            print(f"  {result.content[0].text}")

        # ── 3. Counter tools ────────────────────────────────────────────
        print()
        print("── Counter Tools ──")

        # Increment 3 times
        for _ in range(3):
            await client.call_tool("increment", {})

        result = await client.call_tool("get_value", {})
        print(f"  After 3 increments: {result.content[0].text}")

        # Decrement once
        await client.call_tool("decrement", {})
        result = await client.call_tool("get_value", {})
        print(f"  After 1 decrement:  {result.content[0].text}")

        # ── 4. Misc tools ───────────────────────────────────────────────
        print()
        print("── Misc Tools ──")

        result = await client.call_tool("say_hello", {})
        print(f"  say_hello → {result.content[0].text}")

        result = await client.call_tool("echo", {"object": {"msg": "Hello MCP!", "n": 42}})
        print(f"  echo      → {result.content[0].text}")

        result = await client.call_tool("sum", {"a": 17, "b": 25})
        print(f"  sum(17,25) → {result.content[0].text}")

        # ── 5. Resources ────────────────────────────────────────────────
        print()
        print("── Resources ──")

        resources = await client.list_resources()
        for r in resources:
            print(f"  📄 {r.name} ({r.uri})")

        cities_content = await client.read_resource("weather://cities")
        if type(cities_content) is list and len(cities_content) > 0:
            print(f"  → {cities_content[0].text}")
        elif hasattr(cities_content, 'text'):
            print(f"  → {cities_content.text}")
        else:
            print(f"  → {cities_content}")

        counter_content = await client.read_resource("counter://value")
        if type(counter_content) is list and len(counter_content) > 0:
            print(f"  → Counter value: {counter_content[0].text}")
        elif hasattr(counter_content, 'text'):
            print(f"  → Counter value: {counter_content.text}")
        else:
            print(f"  → Counter value: {counter_content}")

        # ── 6. Prompts ──────────────────────────────────────────────────
        print()
        print("── Prompts ──")

        prompts = await client.list_prompts()
        for p in prompts:
            arg_names = [a.name for a in (p.arguments or [])]
            print(f"  💬 {p.name} (args: {json.dumps(arg_names)})")

        prompt_result = await client.get_prompt(
            "example_prompt", arguments={"message": "Hello from client!"}
        )
        if prompt_result.messages:
            msg = prompt_result.messages[0]
            print(f"  → example_prompt: {msg.content.text}")

        prompt_result = await client.get_prompt(
            "weather_analysis", arguments={"city": "Tokyo", "style": "detailed"}
        )
        if prompt_result.messages:
            msg = prompt_result.messages[-1]
            text_content = msg.content.text
            if isinstance(text_content, str) and text_content.startswith('{"role"'):
                import json
                try:
                    parsed = json.loads(text_content)
                    text_content = parsed.get("content", text_content)
                except Exception:
                    pass
            elif isinstance(text_content, dict):
                 text_content = text_content.get("content", str(text_content))
            print(f"  → weather_analysis: {text_content}")

    print()
    print("✓ Full MCP client-server communication successful!")


if __name__ == "__main__":
    asyncio.run(main())
