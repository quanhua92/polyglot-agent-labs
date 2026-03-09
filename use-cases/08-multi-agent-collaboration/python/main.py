"""
Polyglot Agent Labs — Use Case 08: Multi-Agent Collaboration (Researcher + Writer)
Demonstrates multi-agent collaboration where two specialized agents work together:
- Researcher: Uses ReAct loop to autonomously search and gather information
- Writer: Transforms research findings into a polished article

Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

Key Learning Goals:
- Agent composition and orchestration
- Role specialization with system prompts
- Inter-agent communication and data passing
- Multi-step workflows with LangGraph
- Tool use for knowledge base access
"""

import os
import sys
from typing import Annotated, Literal, Sequence

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing import TypedDict

# Provider -> (model_id, provider_type)
PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}

# Demo topic
DEMO_TOPIC = "Explain the benefits of Rust for systems programming"
MAX_RESEARCH_ITERATIONS = 5

# ============================================================================
# Knowledge Base (Hard-coded Documents about Rust)
# ============================================================================

KNOWLEDGE_BASE = [
    {
        "id": "rust-performance",
        "title": "Rust Performance Characteristics",
        "content": """Rust provides zero-cost abstractions, memory safety without garbage collection, and predictable performance. This makes it ideal for systems programming where performance is critical.

Key performance benefits:
- No runtime garbage collector means consistent latency
- LLVM-based compiler generates optimized machine code
- Zero-cost abstractions allow high-level programming without performance penalty
- Efficient memory layout control with explicit lifetime management
- Compile-time optimizations eliminate runtime overhead

Rust's performance profile is comparable to C++ while providing stronger safety guarantees. The borrow checker enables memory safety without a garbage collector, and ownership semantics allow for aggressive compiler optimizations."""
    },
    {
        "id": "rust-memory-safety",
        "title": "Rust Memory Safety Guarantees",
        "content": """Rust's ownership model ensures memory safety at compile time, eliminating entire classes of bugs that plague systems programming in C and C++.

Memory safety features:
- Ownership rules prevent data races at compile time
- Borrow checker ensures references are always valid
- No null pointers (Option<T> instead of null)
- No buffer overflows with bounds-checked slices
- No use-after-free errors due to ownership semantics
- No dangling pointers through lifetime annotations

The compiler enforces these rules, meaning memory safety issues are caught before code ever runs. This "fight the borrow checker" experience initially challenges developers but leads to more reliable software."""
    },
    {
        "id": "rust-concurrency",
        "title": "Rust Concurrency Benefits",
        "content": """Rust makes concurrent programming safer and easier through its ownership system. The same rules that prevent memory errors also prevent data races.

Concurrency advantages:
- "Fearless concurrency" - compiler prevents data races
- Send and Sync traits mark thread-safe types
- Message passing with channels (mpsc, rpc)
- Async/await with tokio runtime for efficient I/O
- No data race guarantees at compile time

The ownership system ensures that either:
1. Only one thread can access data (mutable reference)
2. Multiple threads can read data (immutable references)

This eliminates the most common concurrency bugs while maintaining high performance."""
    },
    {
        "id": "rust-ecosystem",
        "title": "Rust Ecosystem and Tooling",
        "content": """Rust has a modern, developer-friendly ecosystem that enhances productivity.

Key ecosystem features:
- Cargo: Integrated package manager and build system
- Crates.io: Central package registry with 100k+ packages
- rustfmt: Consistent code formatting
- clippy: Linter for catching common mistakes
- rustdoc: Documentation generator from code comments
- Excellent IDE support via rust-analyzer

The tooling "just works" out of the box, eliminating configuration headaches common in C++ projects. Dependency management, building, testing, and documentation are all unified under Cargo."""
    },
    {
        "id": "rust-use-cases",
        "title": "Rust Use Cases in Systems Programming",
        "content": """Rust is increasingly being adopted for systems programming across diverse domains.

Major use cases:
- Operating systems: Redox OS, components of Windows and Linux
- Embedded systems: Firmware, microcontrollers, IoT devices
- Network services: High-performance servers and proxies
- Blockchain: Solana, Polkadot, and many others
- WebAssembly: Compile to Wasm for browser and serverless
- CLI tools: ripgrep, bat, exa, and many replacements for GNU tools
- Database engines: TiKV, Detox, and database drivers

Companies using Rust include Microsoft, Amazon, Google, Mozilla, Dropbox, Cloudflare, and many more. It's particularly valued for infrastructure software where reliability and performance are paramount."""
    },
    {
        "id": "rust-learning-curve",
        "title": "Rust Learning Curve Considerations",
        "content": """Rust has a steeper learning curve than many languages, but the investment pays off in code quality.

Learning challenges:
- Ownership, borrowing, and lifetimes are unique concepts
- No automatic garbage collection requires thinking about memory
- Pattern matching and algebraic data types (enums)
- Error handling with Result<T, E> instead of exceptions

However, once these concepts click:
- Debugging time decreases dramatically
- Refactoring becomes safer
- Code reviews focus on logic, not memory safety
- Production incidents related to memory vanish

Most developers report 2-3 months to become productive in Rust, with continued improvement over years."""
    }
]

# ============================================================================
# Tool Definitions
# ============================================================================

@tool
def search_notes(query: str) -> str:
    """Search the knowledge base for information about a given query.

    Args:
        query: The search query string

    Returns:
        Relevant information from the knowledge base (single document)
    """
    query_lower = query.lower()

    for doc in KNOWLEDGE_BASE:
        # Simple keyword matching - check if query terms appear in title or content
        title_lower = doc["title"].lower()
        content_lower = doc["content"].lower()

        # Check for keyword matches
        query_words = query_lower.split()
        if any(word in title_lower for word in query_words if len(word) > 3):
            # Return only the first matching document to force ReAct loop iteration
            return f"## {doc['title']}\n\n{doc['content']}"
        elif any(word in content_lower for word in query_words if len(word) > 3):
            # Return only the first matching document to force ReAct loop iteration
            return f"## {doc['title']}\n\n{doc['content']}"

    return "No relevant information found in the knowledge base. Try a different search query with different keywords."

# All tools list
TOOLS = [search_notes]

# ============================================================================
# LangGraph State and Nodes
# ============================================================================

class CollaborationState(TypedDict):
    """State passed between agent nodes."""
    topic: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iteration_count: int
    max_iterations: int
    research_complete: bool
    article: str | None


def researcher_node(state: CollaborationState, config):
    """ReAct Researcher agent - reasons and searches iteratively."""
    messages = state["messages"]
    iteration = state["iteration_count"]
    max_iter = state["max_iterations"]

    # Get model from config
    model = config["configurable"]["model"]

    # Build prompt with iteration context
    system_prompt = f"""You are a research assistant. Your task is to thoroughly research the topic using the search tool.

Topic: {state["topic"]}

Iteration {iteration + 1} of {max_iter}

Instructions:
1. Reason about what information you need to gather
2. Use the search_notes tool to find relevant information
3. Evaluate if you have enough information (need 3-5 key facts)
4. If you need more information, search again with a different query
5. When satisfied, respond with "RESEARCH_COMPLETE" followed by a summary

Important: Be thorough. Don't stop until you have comprehensive information covering:
- Performance characteristics
- Memory safety features
- Concurrency benefits
- Ecosystem and tooling
- Real-world use cases

Use diverse search queries to gather comprehensive information."""

    # Bind tools to model
    model_with_tools = model.bind_tools(TOOLS)

    # Build messages with system prompt
    full_messages = [SystemMessage(content=system_prompt)] + list(messages)

    response = model_with_tools.invoke(full_messages)

    return {"messages": [response]}


def should_continue_research(state: CollaborationState) -> Literal["continue_research", "done_research", "max_reached"]:
    """Decide whether to continue research or move to writing."""
    iteration = state["iteration_count"]
    max_iter = state["max_iterations"]
    messages = state["messages"]

    # Check max iterations
    if iteration >= max_iter:
        return "max_reached"

    # Check last message for completion signal or tool calls
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            content = last_msg.content.lower() if last_msg.content else ""
            # Check for completion signals
            if "research_complete" in content or "done researching" in content:
                return "done_research"

            # Check if there are tool calls (agent wants to search more)
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "continue_research"

    # Default: continue if we haven't reached max
    if iteration < max_iter:
        return "continue_research"

    return "done_research"


def has_tool_calls(state: CollaborationState) -> Literal["tools", "increment"]:
    """Check if the last message has tool calls."""
    messages = state["messages"]

    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tools"

    return "increment"


def increment_iteration(state: CollaborationState):
    """Increment iteration counter after research."""
    return {"iteration_count": state["iteration_count"] + 1}


def writer_node(state: CollaborationState, config):
    """Writer agent - transforms research into article (static workflow)."""
    topic = state["topic"]
    messages = state["messages"]

    # Compile research findings from messages
    research_parts = []
    for msg in messages:
        if isinstance(msg, (AIMessage, HumanMessage)):
            if msg.content and not msg.content.startswith("Tool"):
                research_parts.append(msg.content)

    research_content = "\n\n".join(research_parts)

    model = config["configurable"]["model"]

    writer_prompt = f"""You are a technical writer. Your task is to turn research findings into a clear, well-structured article.

Topic: {topic}

Research Findings:
{research_content}

Write a comprehensive article that:
1. Has a compelling title (use # heading)
2. Starts with an engaging introduction
3. Covers 3-5 main sections with proper ## headings
4. Each section is based on the research findings above
5. Has a conclusion that summarizes key takeaways
6. Is formatted in markdown with proper headings and structure

Return ONLY the markdown article, no additional commentary."""

    response = model.invoke([HumanMessage(content=writer_prompt)])
    return {"article": response.content}


# ============================================================================
# LangGraph Construction
# ============================================================================

def build_collaboration_graph(model):
    """Build the multi-agent collaboration graph with ReAct researcher."""
    workflow = StateGraph(CollaborationState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("increment", increment_iteration)
    workflow.add_node("writer", writer_node)

    # Set entry point
    workflow.set_entry_point("researcher")

    # Research loop: researcher -> (has tools? -> tools : increment)
    # After tools, always increment
    workflow.add_conditional_edges(
        "researcher",
        has_tool_calls,
        {"tools": "tools", "increment": "increment"}
    )

    # After tools, always increment
    workflow.add_edge("tools", "increment")

    # After increment, check if we should continue
    workflow.add_conditional_edges(
        "increment",
        should_continue_research,
        {
            "continue_research": "researcher",
            "done_research": "writer",
            "max_reached": "writer"
        }
    )

    # Writer -> end
    workflow.add_edge("writer", END)

    return workflow.compile()


# ============================================================================
# Provider Setup
# ============================================================================

def create_chat_model(provider: str, model_id: str):
    """Create the appropriate chat model based on provider.

    Args:
        provider: The provider type (openai, anthropic, openrouter)
        model_id: The model ID to use

    Returns:
        A chat model instance
    """
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
# Demo Execution
# ============================================================================

def run_demo(model, provider_key: str, model_id: str) -> tuple[int, int]:
    """Run the multi-agent collaboration demo.

    Args:
        model: The chat model
        provider_key: The provider name for display
        model_id: The model ID for display

    Returns:
        Tuple of (research_iterations, word_count)
    """
    print("=== Python — Multi-Agent Collaboration (ReAct Researcher + Writer) ===")
    print(f"Provider: {provider_key}")
    print(f"Model: {model_id}")
    print()

    # Build graph
    graph = build_collaboration_graph(model)

    # Run with demo topic
    print(f"Topic: {DEMO_TOPIC}")
    print("-" * 50)

    # Initial state
    initial_state = {
        "topic": DEMO_TOPIC,
        "messages": [HumanMessage(content=f"Research: {DEMO_TOPIC}")],
        "iteration_count": 0,
        "max_iterations": MAX_RESEARCH_ITERATIONS,
        "research_complete": False,
        "article": None,
    }

    config = {"configurable": {"model": model}}

    # Stream to show progress
    print("\n[Research Phase - ReAct Loop]")
    for step in graph.stream(initial_state, config):
        for node_name, node_output in step.items():
            if node_name != "__end__":
                if node_name == "researcher":
                    iteration = node_output.get("iteration_count", 0) + 1
                    print(f"\n  [Iteration {iteration}/{MAX_RESEARCH_ITERATIONS}] Researcher thinking...")

                    # Check for tool calls
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    print(f"    → Searching for: {tool_call['args'].get('query', tool_call['name'])}")
                            elif msg.content and "research_complete" in msg.content.lower():
                                print(f"    ✓ Research complete")

                elif node_name == "tools":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            # Show a preview of the search result
                            content = msg.content
                            preview = content[:100].replace("\n", " ") + "..." if len(content) > 100 else content
                            print(f"    ← Found: {preview}")

    # Get final result
    result = graph.invoke(initial_state, config)

    # Print research summary
    iterations_used = result["iteration_count"]
    print(f"\nResearch iterations used: {iterations_used}/{MAX_RESEARCH_ITERATIONS}")

    # Extract and print final research summary
    print("\n--- Research Findings Summary ---")
    research_messages = result["messages"]

    # Show only substantive messages (not tool calls)
    for msg in research_messages:
        if isinstance(msg, AIMessage):
            content = msg.content
            if content and not content.startswith("Tool"):
                # Only show the final summary message
                if "research_complete" in content.lower() or "summary" in content.lower():
                    print(content)

    # Print final article
    print("\n[Writing Phase]")
    print("-" * 50)
    article = result["article"]
    print(article)

    # Session summary
    word_count = len(article.split())
    print(f"\n--- Session Summary ---")
    print(f"Research iterations: {iterations_used}/{MAX_RESEARCH_ITERATIONS}")
    print(f"Article word count: {word_count}")

    return iterations_used, word_count


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    load_dotenv()

    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider_key not in PROVIDERS:
        print(f"✗ Unknown provider: '{provider_key}'")
        print(f"  Supported: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    model_id, provider_type = PROVIDERS[provider_key]
    model = create_chat_model(provider_type, model_id)

    # Run demo
    run_demo(model, provider_key, model_id)


if __name__ == "__main__":
    main()
