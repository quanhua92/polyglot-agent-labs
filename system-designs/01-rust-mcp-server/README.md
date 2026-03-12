# System Design 01: Rust MCP Server

Rust MCP server demonstrating the **rmcp 1.2 SDK** with Axum 0.8.

## Overview

- **rmcp 1.2**: Rust MCP SDK with streamable-http transport
- **Axum 0.8**: HTTP server framework
- **JSON Schema**: Tool parameter validation

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Test Client    │────────▶│  Rust MCP        │
│  (Python)       │  HTTP   │  Server          │
│                 │  /mcp    │  (Axum 0.8)      │
└─────────────────┘         │  Port 8000       │
                            └──────────────────┘
```

## Quick Start

### 1. Start Services

```bash
just sys-up 01-rust-mcp-server
```

### 2. Run Tests

```bash
just sys-client 01-rust-mcp-server
```

### 3. View Logs

```bash
just sys-logs 01-rust-mcp-server
```

### 4. Stop Services

```bash
just sys-down 01-rust-mcp-server
```

## Manual Testing

### Initialize MCP Session

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":1,"method":"initialize",
    "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}
  }'
```

### List Available Tools

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

### Call a Tool

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":3,"method":"tools/call",
    "params":{"name":"add","arguments":{"a":5,"b":3}}
  }'
```

## Project Structure

```
system-designs/01-rust-mcp-server/
├── docker-compose.yml
├── README.md
├── test_client.py       # Python test client
├── pyproject.toml       # Python deps
├── server/              # Rust MCP server
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
└── client.py            # Simple client example
```

## Key Technologies

- **rmcp 1.2**: Rust MCP SDK
- **axum 0.8**: HTTP server framework
- **streamable-http**: HTTP-based MCP transport
