# polyglot-agent-labs

A polyglot playground for building AI agents in Python (LangGraph) and Rust (Rig).

## Quick Start

```bash
# 1. Copy the example env file and add your API keys
cp .env.example .env

# 2. Edit .env with your keys
# OPENAI_API_KEY=sk-xxxx...
# ANTHROPIC_API_KEY=sk-ant-xxxx...
# OPENROUTER_API_KEY=sk-or-xxxx...

# 3. Run the hello-world example
just compare 00-hello-world
```

## Project Structure

```
polyglot-agent-labs/
├── .env              # Your API keys (gitignored)
├── .env.example      # Template for env vars
├── justfile          # Task runner commands
├── use-cases/        # Individual agent implementations
│   └── 00-hello-world/
│       ├── python/   # LangGraph implementation
│       └── rust/     # Rig implementation
└── system-designs/   # Multi-service architectures
    ├── 01-rust-mcp-server/
    ├── 02-fastmcp-server/
    └── 03-polyglot-faq/
```

## Justfile Commands

### Use Cases

| Command | Description |
|---------|-------------|
| `just py <id>` | Run Python agent for a use-case |
| `just rs <id>` | Run Rust agent for a use-case |
| `just compare <id>` | Run both back-to-back |

### System Designs

| Command | Description |
|---------|-------------|
| `just sys-list` | List available system designs |
| `just sys-up <name>` | Start system design services |
| `just sys-down <name>` | Stop system design services |
| `just sys-client <name>` | Run test client |
| `just sys-logs <name>` | View service logs |

## Use Cases

| ID | Use Case | Description | Commands |
|----|----------|-------------|----------|
| 00 | Hello World | Basic agent setup and initialization | `just compare 00-hello-world` |
| 01 | Simple Completion | Single LLM call for direct responses | `just compare 01-simple-completion` |
| 02 | MCP Server | Tool definition and exposure via MCP | `just compare 02-mcp-server` |
| 03 | Conversational Agent | Memory and context management | `just compare 03-conversational-agent` |
| 04 | Agent Tool Use | Function calling and tool execution | `just compare 04-agent-tool-use` |
| 05 | RAG Local Docs | Local document retrieval and Q&A | `just compare 05-rag-local-docs` |
| 06 | Structured Output | JSON schema validation and formatting | `just compare 06-structured-output` |
| 08 | Multi-Agent | Agent collaboration and coordination | `just compare 08-multi-agent-collaboration` |
| 10 | Customer Support | Real-world support automation | `just compare 10-customer-support-agent` |
| 11 | Code Review | Code analysis and review automation | `just compare 11-code-review-agent` |
| 13 | Workflow Automation | Task orchestration and automation | `just compare 13-workflow-automation-agent` |
| 14 | Content Writing | Creative content generation | `just compare 14-content-writing-agent` |

## System Designs

### 01-rust-mcp-server

Rust MCP server demonstrating:
- rmcp 1.2 with Axum 0.8
- streamable-http transport
- Tool definitions with JSON schema

```bash
just sys-up 01-rust-mcp-server
just sys-client 01-rust-mcp-server
just sys-down 01-rust-mcp-server
```

### 02-fastmcp-server

Python FastMCP 3.1.0 server demonstrating:
- FastMCP with streamable-http transport
- Simple tool implementations
- HTTP-based MCP protocol

```bash
just sys-up 02-fastmcp-server
just sys-client 02-fastmcp-server
just sys-down 02-fastmcp-server
```

### 03-polyglot-faq

Polyglot FAQ system demonstrating:
- Rust Axum API Gateway
- Python FastMCP + LangGraph workflow
- Rust MCP document search server
- MCP protocol over HTTP/SSE

```bash
just sys-up 03-polyglot-faq
just sys-client 03-polyglot-faq
just sys-down 03-polyglot-faq
```

## Environment Variables

The root `.env` file is loaded by `just` and exported to all subprocesses:

- `OPENAI_API_KEY` - For GPT-4o, etc.
- `ANTHROPIC_API_KEY` - For Claude
- `OPENROUTER_API_KEY` - For OpenRouter models
- `LLM_PROVIDER` - Default LLM provider (openrouter/openai/anthropic)
- `AGENT_LOG_LEVEL` - debug/info/warn/error
