# System Design 03: Polyglot FAQ System

A polyglot architecture demonstrating **Rust + Python MCP integration**:

- **Rust Axum** API Gateway (port 3030)
- **Python FastMCP + LangGraph** Workflow Server (port 8003)
- **Rust MCP** Document Search Server (port 8004)
- **MCP Protocol** over HTTP/SSE for service communication

## Architecture

```
+---------------+         +---------------+         +---------------+         +---------------+
| Client        |-------->| Rust API      |-------->| Python MCP    |-------->| Rust MCP      |
| (test_client) |  HTTP   | Gateway       |   SSE   | Workflow      |  MCP    | Doc Search    |
|               |  /ask   | (Axum)        |         | Server        |         | Server        |
+---------------+         | Port 3030     |         | Port 8003     |         | Port 8004     |
                          +---------------+         | (FastMCP)     |         +---------------+
                                                   | (LangGraph)   |
                                                   +---------------+
```

### Request Flow

1. Client → API Gateway: `POST /ask` with question
2. API Gateway → Workflow Server: MCP call via `rmcp` HTTP client
3. Workflow Server (LangGraph 3-node workflow):
   - Expand query into variants
   - Call Rust MCP `search_documents` tool
   - Generate final response
4. Results flow back through the chain

## Quick Start

### 1. Setup Environment

Root `.env` file (already configured if use-cases work):

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
```

### 2. Start Services

```bash
just sys-up 03-polyglot-faq
```

### 3. Run Tests

```bash
just sys-client 03-polyglot-faq
```

### 4. View Logs

```bash
just sys-logs 03-polyglot-faq
```

### 5. Stop Services

```bash
just sys-down 03-polyglot-faq
```

## Manual Testing

### Test API Gateway

```bash
curl -X POST http://localhost:3030/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I reset my password?","stream":false}'
```

### Test Document Search Server (MCP)

```bash
curl -X POST http://localhost:8004/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":1,"method":"tools/call",
    "params":{"name":"search_documents","arguments":{"query":"password","top_n":3}}
  }'
```

### Test Workflow Server (MCP)

```bash
curl -X POST http://localhost:8003/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":1,"method":"tools/call",
    "params":{"name":"faq_workflow","arguments":{"question":"How do I reset my password?","stream":false}}
  }'
```

## Project Structure

```
system-designs/03-polyglot-faq/
├── docker-compose.yml
├── README.md
├── test_client.py       # Test client
├── pyproject.toml       # Python deps for test client
├── api-gateway/         # Rust Axum API Gateway
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       ├── main.rs
│       └── mcp_client.rs
├── workflow-server/     # Python FastMCP + LangGraph
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── server.py
│   └── workflow.py
└── doc-search-server/   # Rust MCP Server
    ├── Cargo.toml
    ├── Dockerfile
    └── src/
        └── main.rs
```

## Key Technologies

- **rmcp 1.2**: Rust MCP SDK with `transport-streamable-http-client-reqwest`
- **FastMCP 3.1.0**: Python MCP server with streamable-http transport
- **Axum 0.8**: Rust web framework
- **LangGraph**: Python workflow orchestration
- **OpenRouter**: LLM provider (stepfun/step-3.5-flash:free)

## Available Documents

12 FAQ documents covering:
- Password Reset, Account Creation, Order Tracking
- Return Policy, Payment Methods, Shipping Options
- Profile Updates, Order Cancellation, Technical Support
- Loyalty Program, Gift Cards, Account Management

## Troubleshooting

```bash
# Check service status
docker compose ps

# View logs for specific service
just sys-logs 03-polyglot-faq

# Restart services
just sys-down 03-polyglot-faq && just sys-up 03-polyglot-faq
```
