"""
Polyglot Agent Labs — Use Case 13: Workflow Automation Agent
An agent that uses tool calls to plan and execute workflows using LangGraph functional API.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

This demo demonstrates:
- LangGraph functional API with @graph_node decorators
- Tool API for structured output - tools accept typed parameters from LLM
- Multi-turn tool calling - agent decides which tools to call
- Sequential execution - tools are called in the order the LLM decides

Key Learning Goals:
- LangGraph functional API with decorator-based node definition
- Tool API with bind_tools() for reliable tool calling
- State management with TypedDict
- Multi-turn conversations with tool execution
"""

import asyncio
import os
import sys
from typing import Annotated, Sequence, TypedDict

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing import Callable, Literal, cast


# ============================================================================
# Mock Contact Database
# ============================================================================

CONTACTS = [
    {"name": "Alice Johnson", "email": "alice.johnson@company.com"},
    {"name": "Bob Smith", "email": "bob.smith@company.com"},
    {"name": "Carol Williams", "email": "carol.williams@company.com"},
    {"name": "David Brown", "email": "david.brown@company.com"},
]


# ============================================================================
# Tool API - Workflow Tools
# ============================================================================

def search_contacts(query: str) -> str:
    """Search for people by name or email in the contact directory."""
    query_lower = query.lower()
    found = [
        c for c in CONTACTS
        if query_lower in c["name"].lower() or query_lower in c["email"].lower()
    ]

    if not found:
        return f"No contacts found matching '{query}'"
    else:
        emails = [c["email"] for c in found]
        return f"Found contacts: {', '.join(emails)}"


def create_calendar_event(title: str, date: str = "") -> str:
    """Schedule a calendar event/meeting."""
    date_str = date if date else "TBD"
    return f"✓ Calendar event created: '{title}' on {date_str}"


def send_email(subject: str, recipients: str = "") -> str:
    """Send an email to recipients."""
    recipients_str = recipients if recipients else "recipients"
    return f"✓ Email sent: '{subject}' to {recipients_str}"


def create_task(title: str, assignee: str = "", due_date: str = "") -> str:
    """Create a task for someone."""
    assignee_str = assignee if assignee else "assignee"
    due_str = due_date if due_date else "due date TBD"
    return f"✓ Task created: '{title}' assigned to {assignee_str}, due {due_str}"


# Create tool definitions
search_contacts_tool = StructuredTool.from_function(
    func=search_contacts,
    name="search_contacts",
    description="Search for people by name or email in the contact directory",
)

create_calendar_event_tool = StructuredTool.from_function(
    func=create_calendar_event,
    name="create_calendar_event",
    description="Schedule a calendar event/meeting",
)

send_email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="Send an email to recipients",
)

create_task_tool = StructuredTool.from_function(
    func=create_task,
    name="create_task",
    description="Create a task for someone",
)

WORKFLOW_TOOLS = [
    search_contacts_tool,
    create_calendar_event_tool,
    send_email_tool,
    create_task_tool,
]


# ============================================================================
# LangGraph State (TypedDict)
# ============================================================================

class WorkflowState(TypedDict):
    """State for workflow automation."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    instruction: str
    tool_calls_made: int
    max_turns: int


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are a workflow automation assistant. Use the available tools to complete the user's requests step by step.

Available tools:
- search_contacts(query): Find people by name or email
- create_calendar_event(title, date): Schedule a meeting
- send_email(subject, recipients): Send an email (recipients is comma-separated emails)
- create_task(title, assignee, due_date): Create a task

When the user gives you an instruction:
1. Break it down into the steps needed
2. Call the appropriate tools in order
3. Provide a summary of what was done

For dates like "next Tuesday" or "Friday", use that exact phrase - don't calculate specific dates."""


# ============================================================================
# Provider Configuration
# ============================================================================

PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}


def create_chat_model(provider: str, model_id: str) -> BaseChatModel:
    """Create the appropriate chat model based on provider."""
    # Common timeout settings for all providers
    timeout = 30.0  # 30 seconds for fast failure

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("✗ OPENAI_API_KEY not set")
            sys.exit(1)
        return ChatOpenAI(model=model_id, api_key=api_key, timeout=timeout)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("✗ ANTHROPIC_API_KEY not set")
            sys.exit(1)
        return ChatAnthropic(model=model_id, api_key=api_key, timeout=timeout)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("✗ OPENROUTER_API_KEY not set")
            sys.exit(1)
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
        )
    else:
        print(f"✗ Unknown provider type: '{provider}'")
        sys.exit(1)


# ============================================================================
# Demo Scenarios
# ============================================================================

DEMO_SCENARIOS = [
    "Schedule a meeting with Alice and Bob next Tuesday about Q2 planning",
    "Create a task for Alice to review the design doc by Friday",
    "Send an email to Bob about the project update",
    "Schedule a meeting with Alice for Q2 planning and email her the agenda",
]


# ============================================================================
# LangGraph Functional API
# ============================================================================

from langgraph.constants import END
from langgraph.graph import StateGraph


def create_workflow_graph(model: BaseChatModel):
    """Create workflow graph using LangGraph functional API."""

    # Bind tools to model
    model_with_tools = model.bind_tools(WORKFLOW_TOOLS)

    async def agent_node(state: WorkflowState) -> dict:
        """Agent node that decides what to do next."""
        messages = state["messages"]
        instruction = state.get("instruction", "")

        # If this is the first turn, add the instruction
        if not messages:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=instruction),
            ]

        # Get response from model
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(state: WorkflowState) -> dict:
        """Tool node that executes tool calls."""
        messages = state["messages"]
        tool_calls_made = state.get("tool_calls_made", 0)

        # Find the last AI message with tool calls
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                last_ai_message = msg
                break

        if not last_ai_message:
            return {"messages": []}

        # Execute all tool calls
        tool_messages = []
        for tool_call in last_ai_message.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")

            # Find and execute the tool
            tool = next((t for t in WORKFLOW_TOOLS if t.name == tool_name), None)
            if tool:
                result = tool.func(**tool_args)
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                tool_calls_made += 1
            else:
                tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found", tool_call_id=tool_id))

        return {
            "messages": tool_messages,
            "tool_calls_made": tool_calls_made,
        }

    def should_continue(state: WorkflowState) -> Literal["agent", "tools", END]:
        """Decide whether to continue calling the agent or tools."""
        messages = state["messages"]

        if not messages:
            return "agent"

        last_message = messages[-1]

        # If the last message has tool calls, execute them
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"

        # If the last message is a ToolMessage, go back to agent
        if isinstance(last_message, ToolMessage):
            return "agent"

        # Otherwise we're done
        return END

    # Build the graph using functional API
    workflow = StateGraph(WorkflowState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# ============================================================================
# Demo Execution
# ============================================================================

async def run_demo_async(model: BaseChatModel, provider_name: str, model_id: str):
    """Run workflow automation demo asynchronously."""
    print("=== Python — Workflow Automation Agent (LangGraph Functional API) ===")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()

    graph = create_workflow_graph(model)

    for i, instruction in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n[{i}/{len(DEMO_SCENARIOS)}] {instruction}")
        print("-" * 60)

        # Initialize state
        initial_state: WorkflowState = {
            "messages": [],
            "instruction": instruction,
            "tool_calls_made": 0,
            "max_turns": 10,
        }

        try:
            # Track all messages from streaming
            all_messages = []
            agent_responses = []

            # Stream the graph execution
            async for event in graph.astream(initial_state):
                for node_name, node_output in event.items():
                    new_messages = node_output.get("messages", [])
                    all_messages.extend(new_messages)

                    if node_name == "agent" and new_messages:
                        for msg in new_messages:
                            if isinstance(msg, AIMessage):
                                agent_responses.append(msg)
                                # Show tool call intent
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    print(f"Agent: {msg.content if msg.content else '[Calling tools...]'}")
                                    for tc in msg.tool_calls:
                                        print(f"  → {tc.get('name', 'unknown')}")
                                # Show final response
                                elif msg.content and not any(
                                    m.content == msg.content for m in agent_responses[:-1] if isinstance(m, AIMessage) and m.content
                                ):
                                    print(f"\nAgent: {msg.content}\n")

                    elif node_name == "tools" and new_messages:
                        for msg in new_messages:
                            if isinstance(msg, ToolMessage):
                                print(f"  ✓ {msg.content}")

        except Exception as e:
            import traceback
            print(f"\n✗ Error: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

        print("=" * 60)

    # Print session summary
    print("\nSession Summary")
    print(f"  Scenarios processed: {len(DEMO_SCENARIOS)}")


def run_demo(model: BaseChatModel, provider_name: str, model_id: str):
    """Synchronous wrapper for demo."""
    asyncio.run(run_demo_async(model, provider_name, model_id))


# ============================================================================
# Main
# ============================================================================

def main():
    load_dotenv()

    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider_key not in PROVIDERS:
        print(f"✗ Unknown provider: '{provider_key}'")
        print(f"  Supported: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    model_id, provider_type = PROVIDERS[provider_key]
    model = create_chat_model(provider_type, model_id)

    run_demo(model, provider_key, model_id)


if __name__ == "__main__":
    main()
