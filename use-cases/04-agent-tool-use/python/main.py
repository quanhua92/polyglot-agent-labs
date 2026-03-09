"""
Polyglot Agent Labs — Use Case 04: Agent with Tool Use (Function Calling)
An agent that can decide when to call external tools to answer user questions.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

The agent demonstrates function calling with 3 tools:
- calculator: Evaluate mathematical expressions
- get_current_time: Get current date/time
- string_length: Count characters in a string
"""

import os
import sys
from datetime import datetime
from typing import Annotated, Sequence

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Provider → (model_id, provider_type)
PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}

SYSTEM_PROMPT = "You are a helpful assistant with access to tools. Use tools when needed to answer questions accurately."

# Demo prompts that should trigger tool calls
DEMO_PROMPTS = [
    "What is 42 * 137?",
    "What time is it right now?",
    "How many characters in 'Polyglot Agent Labs'?",
]


# ============================================================================
# Tool Definitions
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like '42 * 137' or '2 + 2'

    Returns:
        The result of the evaluation as a string, or an error message.
    """
    try:
        # For demo purposes, using eval with restricted globals
        # In production, use a proper math parser
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        The current date and time formatted as 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def string_length(text: str) -> int:
    """Get the length of a string.

    Args:
        text: The text to measure

    Returns:
        The number of characters in the string.
    """
    return len(text)


# All tools list
TOOLS = [calculator, get_current_time, string_length]


# ============================================================================
# LangGraph Construction
# ============================================================================

from typing import TypedDict


class AgentState(TypedDict):
    """The state of the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


def should_continue(state: AgentState) -> str:
    """Check if the last message has tool calls.

    Args:
        state: The current agent state

    Returns:
        "tools" if there are tool calls to execute, "end" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def call_model(state: AgentState, model, tools):
    """Call the LLM with tools bound.

    Args:
        state: The current agent state
        model: The chat model
        tools: List of tools to bind

    Returns:
        Updated state with the model response
    """
    messages = state["messages"]
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def build_graph(model, tools):
    """Build the LangGraph for tool calling.

    Args:
        model: The chat model
        tools: List of available tools

    Returns:
        A compiled LangGraph
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", lambda state: call_model(state, model, tools))
    workflow.add_node("tools", ToolNode(tools))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

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

def run_demo(app, provider_key: str, model_id: str) -> int:
    """Run the demo prompts and display tool calls and responses.

    Args:
        app: The compiled LangGraph
        provider_key: The provider name for display
        model_id: The model ID for display

    Returns:
        The number of prompts processed
    """
    print("=== Python — Agent with Tool Use ===")
    print(f"Provider: {provider_key}")
    print(f"Model: {model_id}")
    print()

    tool_call_count = 0

    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        print(f"[{i}/{len(DEMO_PROMPTS)}] Question: {prompt}")
        print("-" * 50)

        inputs = {"messages": [HumanMessage(content=prompt)]}

        # Stream to show reasoning chain
        for step in app.stream(inputs):
            for node_name, node_output in step.items():
                if node_name != "__end__":
                    print(f"\n[{node_name}]")

                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    print(f"  Tool Call: {tool_call['name']}({tool_call['args']})")
                                    tool_call_count += 1
                            else:
                                print(f"  Response: {msg.content}")
                        elif isinstance(msg, ToolMessage):
                            print(f"  Tool Result: {msg.content}")

        # Get final answer
        final_state = app.invoke(inputs)
        final_message = final_state["messages"][-1]

        print(f"\nFinal Answer: {final_message.content}")
        print("=" * 50)
        print()

    # Print session summary
    print("Session Summary")
    print(f"  Prompts processed: {len(DEMO_PROMPTS)}")
    print(f"  Tool calls made: {tool_call_count}")

    return len(DEMO_PROMPTS)


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

    # Build the agent graph with tools
    app = build_graph(model, TOOLS)

    # Run demo
    run_demo(app, provider_key, model_id)


if __name__ == "__main__":
    main()
