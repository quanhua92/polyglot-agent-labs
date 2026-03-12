# System Design 02: FastMCP Server

Python FastMCP 3.1.0 server demonstrating the **streamable-http transport**.

## Overview

- **FastMCP 3.1.0**: Latest FastMCP with HTTP transport
- **streamable-http**: HTTP-based MCP protocol
- **Simple Tools**: Basic add and greet tools

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Test Client    │────────▶│  Python MCP      │
│  (requests)     │  HTTP   │  Server          │
│                 │  /mcp    │  (FastMCP)       │
└─────────────────┘         │  Port 8001       │
                            └──────────────────┘
```

## Quick Start

### 1. Start Services

```bash
just sys-up 02-fastmcp-server
```

### 2. Run Tests

```bash
just sys-client 02-fastmcp-server
```

### 3. View Logs

```bash
just sys-logs 02-fastmcp-server
```

### 4. Stop Services

```bash
just sys-down 02-fastmcp-server
```

## Manual Testing

### Initialize MCP Session

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":1,"method":"initialize",
    "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}
  }'
```

### List Available Tools

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

### Call a Tool

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":3,"method":"tools/call",
    "params":{"name":"add","arguments":{"a":5,"b":3}}
  }'
```

### Call Greet Tool

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":4,"method":"tools/call",
    "params":{"name":"greet","arguments":{"name":"World"}}
  }'
```

## Project Structure

```
system-designs/02-fastmcp-server/
├── docker-compose.yml
├── Dockerfile
├── README.md
├── test_client.py       # Test client
├── pyproject.toml       # Python deps
└── server.py            # FastMCP server
```

## Key Technologies

- **fastmcp >=3.1.0**: MCP server framework for Python
- **uvicorn**: ASGI server
- **streamable-http**: HTTP-based MCP transport
- **pydantic**: Tool parameter validation

## Available Tools

- **add(a: int, b: int) → int**: Add two numbers
- **greet(name: str) → str**: Greet someone by name
