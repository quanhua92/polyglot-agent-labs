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
└── use-cases/
    └── 00-hello-world/
        ├── python/   # LangGraph implementation
        └── rust/     # Rig implementation
```

## Justfile Commands

| Command | Description |
|---------|-------------|
| `just py <id>` | Run Python agent for a use-case |
| `just rs <id>` | Run Rust agent for a use-case |
| `just compare <id>` | Run both back-to-back |
| `just new-case <name>` | Create new use-case structure |

## Environment Variables

The root `.env` file is loaded by `just` and exported to all subprocesses. Both Python and Rust read from the same source:

- `OPENAI_API_KEY` - For GPT-4o, etc.
- `ANTHROPIC_API_KEY` - For Claude
- `OPENROUTER_API_KEY` - For OpenRouter models
- `AGENT_LOG_LEVEL` - debug/info/warn/error