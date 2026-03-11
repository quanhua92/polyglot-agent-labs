#!/usr/bin/env python3
"""Test script for the FastMCP server via HTTP - using requests."""
import json
import requests

def test_http_server():
    """Test the FastMCP server via HTTP using requests library."""
    base_url = "http://localhost:8002/mcp"
    session_id = None

    print("Testing FastMCP Server via HTTP")
    print("=" * 50)

    try:
        # Step 1: Initialize
        print("\n1. Initializing...")
        response = requests.post(
            base_url,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream',
            },
            json={
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {},
                    'clientInfo': {'name': 'test-client', 'version': '1.0'}
                }
            },
            timeout=10
        )
        print(f"Status: {response.status_code}")
        session_id = response.headers.get('mcp-session-id')
        print(f"Session ID: {session_id}")

        # Parse SSE response
        print(f"\nResponse:")
        for line in response.text.strip().split('\n'):
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data.strip():
                    print(f"  {data}")

        # Step 2: Send initialized notification
        print("\n2. Sending initialized notification...")
        response = requests.post(
            base_url,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream',
                'mcp-session-id': session_id,
            },
            json={
                'jsonrpc': '2.0',
                'method': 'notifications/initialized'
            },
            timeout=10
        )
        print(f"Status: {response.status_code}")

        # Step 3: List tools
        print("\n3. Listing tools...")
        response = requests.post(
            base_url,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream',
                'mcp-session-id': session_id,
            },
            json={
                'jsonrpc': '2.0',
                'id': 2,
                'method': 'tools/list',
                'params': {}
            },
            timeout=10
        )
        print(f"Status: {response.status_code}")

        # Parse SSE response for tools/list
        for line in response.text.strip().split('\n'):
            if line.startswith('data: '):
                data = line[6:]
                if data.strip():
                    try:
                        parsed = json.loads(data)
                        print(f"  Tools: {json.dumps(parsed, indent=2)}")
                    except:
                        print(f"  Raw: {data}")

        # Step 4: Call add tool
        print("\n4. Calling add tool with a=5, b=3...")
        response = requests.post(
            base_url,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream',
                'mcp-session-id': session_id,
            },
            json={
                'jsonrpc': '2.0',
                'id': 3,
                'method': 'tools/call',
                'params': {
                    'name': 'add',
                    'arguments': {'a': 5, 'b': 3}
                }
            },
            timeout=10
        )
        print(f"Status: {response.status_code}")

        # Parse SSE response for tools/call
        for line in response.text.strip().split('\n'):
            if line.startswith('data: '):
                data = line[6:]
                if data.strip():
                    try:
                        parsed = json.loads(data)
                        print(f"  Result: {json.dumps(parsed, indent=2)}")
                    except:
                        print(f"  Raw: {data}")

        print("\n" + "=" * 50)
        print("✅ Server is working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_http_server()
