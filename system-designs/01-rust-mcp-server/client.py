#!/usr/bin/env python3
"""Simple test script to verify the MCP server is working."""
import json
import subprocess
import sys

def test_server():
    """Test the MCP server by sending JSON-RPC requests via stdin/stdout."""
    # Prepare the messages
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "add",
            "arguments": {"a": 5, "b": 3}
        }}
    ]

    # Start the server
    server = subprocess.Popen(
        ["./target/release/mcp-add-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Send each message
        for msg in messages:
            print(f"> Sending: {json.dumps(msg)}", file=sys.stderr)
            server.stdin.write(json.dumps(msg) + "\n")
            server.stdin.flush()

        # Read responses
        server.stdin.close()
        output, errors = server.communicate()

        print("\n=== Server Output ===")
        if output:
            print(output)
        if errors:
            print("=== Server Errors ===")
            print(errors)

    except Exception as e:
        print(f"Error: {e}")
        server.kill()
    finally:
        server.terminate()

if __name__ == "__main__":
    test_server()
