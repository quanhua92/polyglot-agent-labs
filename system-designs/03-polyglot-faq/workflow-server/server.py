#!/usr/bin/env python3
"""FAQ Workflow Server using FastMCP with streamable-http transport.

This server provides a tool that runs a 3-node LangGraph workflow:
1. Expand query into variants
2. Search documents using Rust MCP server
3. Generate final response
"""

import asyncio
import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Create the FastMCP server instance
mcp = FastMCP("FAQ Workflow Server")

DOC_SEARCH_URL = os.getenv("DOC_SEARCH_URL", "http://doc-search-server:8004/mcp")

# Import workflow after MCP is created to avoid circular imports
from workflow import create_faq_graph, run_faq_workflow


@mcp.tool()
async def faq_workflow(question: str, stream: bool = False) -> str:
    """Run FAQ workflow to answer customer questions.

    This workflow:
    1. Expands the query into 3 search variants
    2. Searches FAQ documents using the Rust MCP server
    3. Generates a comprehensive answer based on retrieved documents

    Args:
        question: The customer's question
        stream: Whether to stream intermediate results (default: False)

    Returns:
        JSON string containing the workflow results with steps and final answer
    """
    result = await run_faq_workflow(question, DOC_SEARCH_URL, stream)
    return json.dumps(result)


@mcp.tool()
async def search_faq_documents(query: str, top_n: int = 5) -> str:
    """Directly search FAQ documents without the full workflow.

    Args:
        query: The search query
        top_n: Maximum number of results to return (default: 5)

    Returns:
        JSON string containing search results
    """
    import httpx

    # Make MCP call to document search server
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_documents",
            "arguments": {
                "query": query,
                "top_n": top_n
            }
        }
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            DOC_SEARCH_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.text


async def main():
    """Run the MCP server with HTTP transport."""
    print("=== Python FastMCP — FAQ Workflow Server ===")
    print(f"Doc Search URL: {DOC_SEARCH_URL}")
    print()

    await mcp.run_http_async(
        transport="streamable-http",
        host="0.0.0.0",
        port=8003,
        path="/mcp"
    )


if __name__ == "__main__":
    asyncio.run(main())
