"""
Polyglot Agent Labs — Use Case 13: Workflow Automation Agent
An agent that uses tool calls to plan and execute workflows using Tool API.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

This demo demonstrates:
- Tool API for structured output - tools accept typed parameters from LLM
- Multi-turn tool calling - agent decides which tools to call
- Sequential execution - tools are called in the order the LLM decides
- Real-world API simulation - Mocking external service integrations

Key Learning Goals:
- Tool API with bind_tools() for reliable tool calling
- Multi-turn conversations with tool execution
- Agent-driven workflow planning

The agent uses 4 mock tools:
- search_contacts: Find people in the contact database
- create_calendar_event: Schedule meetings
- send_email: Send emails to recipients
- create_task: Assign tasks to team members
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


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
    """Search for people by name or email in the contact directory.

    Args:
        query: The person's name or email to search for

    Returns:
        Found contacts as JSON string
    """
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
    """Schedule a calendar event/meeting.

    Args:
        title: Meeting title
        date: Meeting date (e.g., 'next Tuesday', '2024-03-15')

    Returns:
        Confirmation message
    """
    date_str = date if date else "TBD"
    return f"✓ Calendar event created: '{title}' on {date_str}"


def send_email(subject: str, recipients: str = "") -> str:
    """Send an email to recipients.

    Args:
        subject: Email subject
        recipients: Email addresses (comma-separated)

    Returns:
        Confirmation message
    """
    recipients_str = recipients if recipients else "recipients"
    return f"✓ Email sent: '{subject}' to {recipients_str}"


def create_task(title: str, assignee: str = "", due_date: str = "") -> str:
    """Create a task for someone.

    Args:
        title: Task title
        assignee: Person assigned to the task
        due_date: Due date (e.g., 'Friday', '2024-03-15')

    Returns:
        Confirmation message
    """
    assignee_str = assignee if assignee else "assignee"
    due_str = due_date if due_date else "due date TBD"
    return f"✓ Task created: '{title}' assigned to {assignee_str}, due {due_str}"


# Create tool definitions
search_contacts_tool = StructuredTool.from_function(
    func=search_contacts,
    name="search_contacts",
    description="Search for people by name or email in the contact directory",
    args_schema=type("Args", (BaseModel,), {"__annotations__": {"query": str}}),
)

create_calendar_event_tool = StructuredTool.from_function(
    func=create_calendar_event,
    name="create_calendar_event",
    description="Schedule a calendar event/meeting",
    args_schema=type("Args", (BaseModel,), {
        "__annotations__": {
            "title": str,
            "date": str,
        }
    }),
)

send_email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="Send an email to recipients",
    args_schema=type("Args", (BaseModel,), {
        "__annotations__": {
            "subject": str,
            "recipients": str,
        }
    }),
)

create_task_tool = StructuredTool.from_function(
    func=create_task,
    name="create_task",
    description="Create a task for someone",
    args_schema=type("Args", (BaseModel,), {
        "__annotations__": {
            "title": str,
            "assignee": str,
            "due_date": str,
        }
    }),
)

WORKFLOW_TOOLS = [
    search_contacts_tool,
    create_calendar_event_tool,
    send_email_tool,
    create_task_tool,
]


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
# Demo Scenarios
# ============================================================================

DEMO_SCENARIOS = [
    "Schedule a meeting with Alice and Bob next Tuesday about Q2 planning",
    "Create a task for Alice to review the design doc by Friday",
    "Send an email to Bob about the project update",
    "Schedule a meeting with Alice for Q2 planning and email her the agenda",
]


# ============================================================================
# Demo Execution
# ============================================================================

async def run_demo_async(model: BaseChatModel, provider_name: str, model_id: str):
    """Run workflow automation demo asynchronously."""
    print("=== Python — Workflow Automation Agent (Tool API) ===")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()

    # Bind tools to model
    model_with_tools = model.bind_tools(WORKFLOW_TOOLS)

    for i, instruction in enumerate(DEMO_SCENARIOS, 1):
        print(f"[{i}/{len(DEMO_SCENARIOS)}] {instruction}")
        print("-" * 60)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=instruction),
        ]

        try:
            # Multi-turn conversation with tool calling
            for turn in range(5):
                response = await model_with_tools.ainvoke(messages)
                messages.append(response)

                # Check if the model wants to call tools
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Execute all tool calls
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_id = tool_call.get("id", "")

                        # Find and execute the tool
                        tool = next((t for t in WORKFLOW_TOOLS if t.name == tool_name), None)
                        if tool:
                            result = tool.func(**tool_args)
                            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                            print(f"  [{tool_name}]")
                            print(f"    {result}")
                        else:
                            messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found", tool_call_id=tool_id))
                else:
                    # No tool calls, agent is done
                    break

            # Print final response
            if messages and hasattr(messages[-1], 'content'):
                print(f"\nAgent: {messages[-1].content}")

        except Exception as e:
            print(f"\n✗ Error: {e}")

        print("=" * 60)
        print()

    # Print session summary
    print("Session Summary")
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
