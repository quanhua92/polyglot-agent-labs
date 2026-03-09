"""
Polyglot Agent Labs — Use Case 05: RAG over Local Documents
Answer questions by retrieving relevant context from a knowledge base of documents.
Switch provider with env var LLM_PROVIDER (default: openrouter).
Switch embedding provider with env var EMBEDDING_PROVIDER (default: openrouter).

Usage:
  python main.py
"""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# Hard-coded Documents
# ============================================================================

DOCUMENTS = [
    {
        "id": "faq",
        "title": "Polyglot Agent Labs FAQ",
        "content": """# Polyglot Agent Labs FAQ

## What is Polyglot Agent Labs?

Polyglot Agent Labs is a progressive roadmap of 16 use cases for building AI agents,
with each use case implemented in both Python and Rust side-by-side. The project
demonstrates how to build practical AI systems using modern frameworks and patterns.

## What are the main learning goals?

The project covers:
- Simple LLM completion and multi-provider support
- MCP (Model Context Protocol) servers
- Conversational agents with memory
- Tool use and function calling
- RAG (Retrieval-Augmented Generation)
- Structured output and data extraction
- Streaming responses and human-in-the-loop workflows
- Multi-agent collaboration and research agents
- Customer support agents with knowledge bases
- Code review and analysis agents
- Data analysis with CSV/SQL
- Workflow automation and content generation

## How do I get started?

Start with use case 00 (environment check), then progress through each use case
sequentially. Each use case builds upon concepts from previous ones.

## Why Python and Rust?

Python excels at rapid development and has the richest AI/ML ecosystem. Rust provides
memory safety, performance, and modern async capabilities. Learning both gives you
versatility in AI engineering.
"""
    },
    {
        "id": "rust-guidelines",
        "title": "Rust Coding Guidelines",
        "content": """# Rust Coding Guidelines

## Naming Conventions

- **Structs**: Use `PascalCase` (e.g., `ChatMessage`, `ToolDefinition`)
- **Functions**: Use `snake_case` (e.g., `chat_prompt`, `tool_call`)
- **Constants**: Use `SCREAMING_SNAKE_CASE` (e.g., `MAX_TOKENS`, `DEFAULT_MODEL`)
- **Modules**: Use `snake_case` for file and module names

## Error Handling

- Use `anyhow::Result<T>` for application errors
- Use `thiserror` to define custom error types for libraries
- Prefer `?` operator over explicit match for error propagation
- Provide context with `.context()` or `.with_context()`

## Async Patterns

- Use `tokio` as the async runtime
- Mark async functions with `async fn`
- Use `.await` to call async functions
- Prefer async I/O over blocking operations

## Code Organization

- Keep main.rs focused on setup and orchestration
- Extract business logic into separate modules
- Use builder patterns for complex configuration (e.g., `AgentBuilder`)

## Style Tips

- Prefer `&str` over `String` for function arguments when ownership isn't needed
- Use `format!` for string building over concatenation
- Leverage rustfmt for consistent formatting
- Use clippy to catch common mistakes
"""
    },
    {
        "id": "python-practices",
        "title": "Python Best Practices",
        "content": """# Python Best Practices for LangChain

## Error Handling

- Always handle API errors gracefully with try/except blocks
- Use specific exception types when possible
- Provide helpful error messages for API key issues
- Log errors for debugging while showing user-friendly messages

## LangChain Patterns

- Use `@tool` decorator for function calling tools
- Use `TypedDict` for state in LangGraph workflows
- Prefer `HumanMessage`, `AIMessage`, `SystemMessage` over raw strings
- Use `bind_tools()` to attach tools to models

## Memory Management

- Keep conversation history in memory for simple applications
- Use `ConversationBufferMemory` for persistent history
- Be mindful of token limits - summarize old messages when needed

## Provider Configuration

- Use environment variables for API keys
- Support multiple providers (OpenAI, Anthropic, OpenRouter)
- Provide clear error messages when API keys are missing
- Default to a free/low-cost option for development

## Type Hints

- Always add type hints to function signatures
- Use `from typing import ...` for generic types
- Return specific types, not `Any` when possible
- Use `|` syntax for unions in Python 3.12+
"""
    }
]

# Demo questions
DEMO_QUESTIONS = [
    "What are the main learning goals of Polyglot Agent Labs?",
    "What naming conventions should I follow in Rust?",
    "How should I handle errors in LangChain?",
    "What's the weather in Tokyo?",  # Edge case: not in docs
]

# Provider configurations
PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}

EMBEDDING_PROVIDERS = {
    "openrouter": {
        "model": "text-embedding-3-small",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY"
    },
    "openai": {
        "model": "text-embedding-3-small",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY"
    },
}


# ============================================================================
# LLM Model Creation
# ============================================================================

def create_chat_model(provider: str, model_id: str):
    """Create the appropriate chat model based on provider."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("✗ OPENAI_API_KEY not set")
            sys.exit(1)
        return ChatOpenAI(model=model_id, api_key=api_key)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("✗ ANTHROPIC_API_KEY not set")
            sys.exit(1)
        return ChatAnthropic(model=model_id, api_key=api_key)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("✗ OPENROUTER_API_KEY not set")
            sys.exit(1)
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        print(f"✗ Unknown provider type: '{provider}'")
        sys.exit(1)


# ============================================================================
# Embeddings Creation
# ============================================================================

def create_embeddings(provider: str = "openrouter") -> OpenAIEmbeddings:
    """Create embeddings with provider support."""
    config = EMBEDDING_PROVIDERS.get(provider, EMBEDDING_PROVIDERS["openrouter"])
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        print(f"✗ {config['api_key_env']} not set")
        sys.exit(1)

    return OpenAIEmbeddings(
        model=config["model"],
        openai_api_key=api_key,
        openai_api_base=config.get("api_base")
    )


# ============================================================================
# Document Processing
# ============================================================================

def process_documents(embedding_provider: str = "openrouter"):
    """Process documents into chunks and create vector store."""
    # Convert hard-coded documents to LangChain Document format
    docs = [
        Document(page_content=doc["content"], metadata={"source": doc["id"], "title": doc["title"]})
        for doc in DOCUMENTS
    ]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings with provider support
    embeddings = create_embeddings(embedding_provider)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


# ============================================================================
# RAG Pipeline Construction
# ============================================================================

def build_rag_prompt(context: str, question: str) -> str:
    """Build a RAG prompt with context and question."""
    return f"""Answer the question based only on the following context:

{context}

Question: {question}

If the answer cannot be found in the context, say "I don't have enough information to answer this question."
"""


# ============================================================================
# Demo Execution
# ============================================================================

def run_demo(model, vectorstore, provider_key: str, model_id: str, chunk_count: int):
    print("=== Python — RAG over Local Documents ===")
    print(f"Provider: {provider_key}")
    print(f"Model: {model_id}")
    print(f"Documents indexed: {len(DOCUMENTS)}")
    print(f"Total chunks: {chunk_count}")
    print()

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"[{i}/{len(DEMO_QUESTIONS)}] Question: {question}")
        print("-" * 50)

        # Retrieve relevant chunks for display
        retrieved_docs = vectorstore.similarity_search(question, k=3)
        print(f"\nRetrieved {len(retrieved_docs)} chunks:")
        for j, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "unknown")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  [{j}] {title} ({source})")
            print(f"      {content_preview}")

        # Format context from retrieved documents
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('title', 'unknown')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        # Build prompt and get RAG response
        prompt = build_rag_prompt(context, question)
        response = model.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        print(f"\nAnswer: {answer}")
        print("=" * 50)
        print()

    # Print session summary
    print("Session Summary")
    print(f"  Questions processed: {len(DEMO_QUESTIONS)}")


# ============================================================================
# Main
# ============================================================================

def main():
    load_dotenv()

    # Get provider settings
    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openrouter").lower()

    if provider_key not in PROVIDERS:
        print(f"✗ Unknown LLM provider: '{provider_key}'")
        print(f"  Supported: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    if embedding_provider not in EMBEDDING_PROVIDERS:
        print(f"✗ Unknown embedding provider: '{embedding_provider}'")
        print(f"  Supported: {', '.join(EMBEDDING_PROVIDERS.keys())}")
        sys.exit(1)

    # Get model configuration
    model_id, provider_type = PROVIDERS[provider_key]

    # Create LLM and process documents
    chat = create_chat_model(provider_type, model_id)
    vectorstore, chunk_count = process_documents(embedding_provider)

    # Run demo
    run_demo(chat, vectorstore, provider_key, model_id, chunk_count)


if __name__ == "__main__":
    main()
