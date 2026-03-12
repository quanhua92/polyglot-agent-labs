# Polyglot Agent Labs

A hands-on laboratory for building AI agents using **Python (LangGraph)** and **Rust (Rig)** with **MCP (Model Context Protocol)** integration.

## Quick Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│                    polyglot-agent-labs                       │
├─────────────────────────────────────────────────────────────┤
│  use-cases/          # Single-file agent examples           │
│  ├── python/         # LangGraph implementations             │
│  └── rust/           # Rig implementations                   │
│                                                              │
│  system-designs/     # Multi-service architectures          │
│  ├── 01-rust-mcp-server    # Rust MCP server (rmcp 1.2)     │
│  ├── 02-fastmcp-server     # Python FastMCP server          │
│  └── 03-polyglot-faq       # Full FAQ system (Rust+Python)  │
│                                                              │
│  justfile            # Task runner (all commands)           │
└─────────────────────────────────────────────────────────────┘
```

## Essential Commands

### Use Cases (Single Agents)
```bash
just compare 00-hello-world    # Run both Python & Rust implementations
just py 00-hello-world         # Run Python (LangGraph) only
just rs 00-hello-world         # Run Rust (Rig) only
```

### System Designs (Multi-Service)
```bash
just sys-up 03-polyglot-faq    # Start all services via Docker Compose
just sys-client 03-polyglot-faq # Run test client
just sys-logs 03-polyglot-faq  # View service logs
just sys-down 03-polyglot-faq  # Stop services
```

## Environment Setup

Root `.env` file is loaded automatically by `just`:
```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-xxxx...
#   ANTHROPIC_API_KEY=sk-ant-xxxx...
#   OPENROUTER_API_KEY=sk-or-xxxx...
#   LLM_PROVIDER=openrouter
```

## Project Structure Reference

### Use Cases (14 examples)
Each numbered `00-14` demonstrates a specific agent pattern:

| ID | Pattern | Description |
|----|--------|-------------|
| 00 | Hello World | Basic agent setup |
| 01 | Simple Completion | Single LLM call |
| 02 | MCP Server | Tool definition & exposure |
| 03 | Conversational | Memory & context |
| 04 | Tool Use | Function calling |
| 05 | RAG | Local document retrieval |
| 06 | Structured Output | JSON schema validation |
| 08 | Multi-Agent | Agent collaboration |
| 10 | Customer Support | Real-world application |
| 11 | Code Review | Code analysis |
| 13 | Workflow Automation | Task orchestration |
| 14 | Content Writing | Creative generation |

### System Designs (3 architectures)

#### 01-rust-mcp-server
- **Stack**: Rust + rmcp 1.2 + Axum 0.8
- **Transport**: streamable-http
- **Purpose**: Minimal MCP server with tool definitions

#### 02-fastmcp-server
- **Stack**: Python + FastMCP 3.1.0
- **Transport**: HTTP-based MCP
- **Purpose**: Simple Python MCP server

#### 03-polyglot-faq (Latest)
- **Stack**: Rust API Gateway + Python LangGraph + Rust MCP Server
- **Architecture**: 3-service polyglot system
- **Features**:
  - 5 MCP tools: search, list, get_document, find_related, get_current_date
  - 4-node workflow: expand → search → agent_with_tools → generate
  - Agentic tool selection via `bind_tools()`

## Key Technologies

| Component | Python | Rust |
|-----------|--------|------|
| **Agent Framework** | LangGraph | Rig |
| **MCP SDK** | FastMCP 3.1+ | rmcp 1.2 |
| **HTTP Server** | FastAPI/Uvicorn | Axum 0.8 |
| **HTTP Transport** | streamable-http | streamable-http |
| **Tool Schema** | Python typing + schemars | Rust serde + schemars |

## Working with This Repo

### For Learning
1. Start with `just compare 00-hello-world`
2. Progress through use cases `01-14`
3. Study system designs to see multi-service patterns

### For Development
1. Each use-case is self-contained
2. Python uses `uv` for dependency management
3. Rust uses `cargo` for builds
4. System designs use Docker Compose

### For MCP Integration
- Study `system-designs/03-polyglot-faq/` for full polyglot MCP pattern
- Rust servers: use `rmcp` crate with `#[tool]` macros
- Python servers: use `FastMCP` with `@mcp.tool()` decorators
- HTTP transport: both support `/mcp` endpoint with SSE

## Common Patterns

### Defining MCP Tools (Rust)
```rust
#[tool(description = "Tool description")]
async fn my_tool(&self, Parameters(input): Parameters<ToolInput>)
    -> Result<CallToolResult, ErrorData> {
    // Tool implementation
}
```

### Defining MCP Tools (Python)
```python
@mcp.tool()
async def my_tool(arg: str) -> str:
    # Tool implementation
```

### LangGraph Node Pattern
```python
async def my_node(state: AgentState) -> dict:
    # Node logic
    return {"key": value}
```

## Testing System Designs

Each system design has a `test_client.py`:
```bash
just sys-up <name>       # Start services
just sys-client <name>   # Run tests (health + API calls)
just sys-down <name>     # Cleanup
```

## Notes

- Root `.env` is auto-loaded by `just` via `set dotenv-load := true`
- All system designs expose MCP over HTTP at `/mcp` endpoint
- Python uses `uv run` for consistent dependency management
- Rust builds are optimized for production (`cargo build --release`)
