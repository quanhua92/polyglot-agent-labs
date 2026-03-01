# Polyglot Agent Labs - Task Runner
# Usage: just <command> [args]

# Load environment variables from root .env
set dotenv-load := true

# Default: list available commands
default:
    @just --list

# =============================================================================
# USE CASE RUNNERS
# =============================================================================

# Run Python implementation for a specific use-case
# Example: just py 00-hello-world
py id:
    cd use-cases/{{id}}/python && uv run main.py

# Run Rust implementation for a specific use-case
# Example: just rs 00-hello-world
rs id:
    cd use-cases/{{id}}/rust && cargo run

# Run both implementations back-to-back for comparison
# Example: just compare 00-hello-world
compare id:
    @echo "=========================================="
    @echo "  RUNNING PYTHON (LANGGRAPH) - Use Case {{id}}"
    @echo "=========================================="
    just py {{id}}
    @echo ""
    @echo "=========================================="
    @echo "  RUNNING RUST (RIG) - Use Case {{id}}"
    @echo "=========================================="
    just rs {{id}}

# =============================================================================
# SETUP & UTILITIES
# =============================================================================

# Create a new use-case directory structure
# Example: just new-case 02-custom-agent
new-case name:
    mkdir -p use-cases/{{name}}/python
    mkdir -p use-cases/{{name}}/rust/src
    @echo "Created use-cases/{{name}}/"

# Install Python dependencies for a use-case
py-install id:
    cd use-cases/{{id}}/python && uv sync

# Install Rust dependencies for a use-case
rs-install id:
    cd use-cases/{{id}}/rust && cargo build

# =============================================================================
# QUALITY CHECKS
# =============================================================================

# Run Python linter for a use-case
py-lint id:
    cd use-cases/{{id}}/python && uv run ruff check .

# Run Rust clippy for a use-case
rs-lint id:
    cd use-cases/{{id}}/rust && cargo clippy -- -D warnings
