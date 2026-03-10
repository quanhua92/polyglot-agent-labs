"""
Polyglot Agent Labs — Use Case 13: Workflow Automation Agent
An agent that decomposes high-level instructions into executable steps using mock tools.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

This demo demonstrates:
- Task decomposition: Breaking complex instructions into atomic steps
- Sequential tool execution: Executing tools in dependency order
- Error recovery: Handling partial failures gracefully
- Real-world API simulation: Mocking external service integrations

The agent uses 4 mock tools:
- send_email: Send emails to recipients
- create_calendar_event: Schedule meetings
- create_task: Assign tasks to team members
- search_contacts: Find people in the contact database
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Sequence

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


# ============================================================================
# Data Models
# ============================================================================


class StepType(str, Enum):
    """Types of workflow steps."""
    SEARCH_CONTACTS = "search_contacts"
    CREATE_EVENT = "create_calendar_event"
    SEND_EMAIL = "send_email"
    CREATE_TASK = "create_task"


class Contact(BaseModel):
    """Contact information."""
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    department: str | None = Field(default=None, description="Department")


class WorkflowStep(BaseModel):
    """Individual step in decomposed workflow."""
    step_type: StepType
    description: str
    parameters: dict[str, Any]
    optional: bool = False


class ExecutionResult(BaseModel):
    """Result of tool execution."""
    step_id: str
    tool_name: str
    success: bool
    result: str | None = None
    error: str | None = None
    timestamp: str


# ============================================================================
# Mock Contact Database
# ============================================================================


CONTACTS = [
    Contact(name="Alice Johnson", email="alice.johnson@company.com", phone="555-0101", department="Engineering"),
    Contact(name="Bob Smith", email="bob.smith@company.com", phone="555-0102", department="Product"),
    Contact(name="Carol Williams", email="carol.williams@company.com", phone="555-0103", department="Marketing"),
    Contact(name="David Brown", email="david.brown@company.com", phone="555-0104", department="Engineering"),
    Contact(name="Eva Martinez", email="eva.martinez@company.com", phone="555-0105", department="Sales"),
]


# ============================================================================
# Mock Tools
# ============================================================================


@tool
def send_email(to: list[str], subject: str, body: str) -> str:
    """Send an email to specified recipients.

    Args:
        to: List of email addresses
        subject: Email subject line
        body: Email body content

    Returns:
        Confirmation message with email ID
    """
    email_hash = hashlib.sha256(''.join(to + [subject, body]).encode()).hexdigest()[:8]
    email_id = f"EMAIL-{int(email_hash, 16) % 10000:04d}"
    return f"✓ Email sent (ID: {email_id}) to {len(to)} recipient(s)"


@tool
def create_calendar_event(title: str, date: str, attendees: list[str]) -> str:
    """Create a calendar event.

    Args:
        title: Event title
        date: Event date (YYYY-MM-DD format)
        attendees: List of attendee names or emails

    Returns:
        Confirmation message with event ID
    """
    event_hash = hashlib.sha256((title + date + ','.join(attendees)).encode()).hexdigest()[:8]
    event_id = f"EVT-{int(event_hash, 16) % 10000:04d}"
    return f"✓ Calendar event created (ID: {event_id}): '{title}' on {date} with {len(attendees)} attendee(s)"


@tool
def create_task(title: str, assignee: str, context: str, due_date: str) -> str:
    """Create a task for a team member.

    Args:
        title: Task title
        assignee: Name or email of the assignee
        context: Additional context or description
        due_date: Due date (YYYY-MM-DD format)

    Returns:
        Confirmation message with task ID
    """
    task_hash = hashlib.sha256((title + assignee + context + due_date).encode()).hexdigest()[:8]
    task_id = f"TSK-{int(task_hash, 16) % 10000:04d}"
    return f"✓ Task created (ID: {task_id}): '{title}' assigned to {assignee}, due {due_date}"


@tool
def search_contacts(query: str) -> list[dict]:
    """Search for contacts by name or email.

    Args:
        query: Search query (name or email)

    Returns:
        List of matching contacts as dictionaries
    """
    query_lower = query.lower()
    matches = [
        contact.model_dump()
        for contact in CONTACTS
        if query_lower in contact.name.lower() or query_lower in contact.email.lower()
    ]
    return matches


WORKFLOW_TOOLS = [send_email, create_calendar_event, create_task, search_contacts]


# ============================================================================
# LangGraph State
# ============================================================================


from typing import TypedDict


class WorkflowState(TypedDict):
    """State for workflow automation."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_instruction: str
    decomposition: list[dict]
    execution_plan: list[dict]
    execution_results: list[dict]
    current_step_index: int
    errors: list[str]


# ============================================================================
# Task Decomposer Prompt
# ============================================================================


TASK_DECOMPOSER_PROMPT = """You are a workflow planning assistant. Break down the user's instruction into specific steps.

Available tools:
- search_contacts(query): Find people by name or email
- create_calendar_event(title, date, attendees): Schedule a meeting
- send_email(to, subject, body): Send an email
- create_task(title, assignee, context, due_date): Create a task

Analyze the instruction and return a structured plan with steps in execution order.

IMPORTANT: Return ONLY valid JSON. No extra text.

Response format (valid JSON):
{
  "steps": [
    {
      "step_type": "search_contacts|create_calendar_event|send_email|create_task",
      "description": "What this step does",
      "parameters": {"param": "value"}
    }
  ]
}

Notes:
- For dates like "next Tuesday", calculate the actual date (today is {TODAY})
- For person names, you MUST search for them first using search_contacts
- After finding contacts, use their email addresses for subsequent operations
- For attendees in calendar events, use email addresses from contact search
- For task assignee, use email addresses from contact search
"""


# ============================================================================
# JSON Extraction Helper
# ============================================================================


def extract_json_from_response(response: str) -> str:
    """Extract JSON from a response that may contain extra text."""
    response = response.strip()

    # Look for JSON block between ```json and ```
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # Look for JSON block between ``` and ```
    json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
        if candidate.startswith('{'):
            return candidate

    # Look for { and } as JSON boundaries
    start = response.find('{')
    if start != -1:
        brace_count = 0
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response[start:i+1]

    return response


# ============================================================================
# Workflow Nodes
# ============================================================================


def task_decomposer_node(state: WorkflowState, config):
    """Decompose user instruction into workflow steps."""
    model = config["configurable"]["model"]
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = TASK_DECOMPOSER_PROMPT.replace("{TODAY}", today)
    full_prompt = f"{prompt}\n\nUser instruction: {state['user_instruction']}"

    response = model.invoke([HumanMessage(content=full_prompt)])

    # Extract JSON from response
    decomposition = extract_json_from_response(response.content)
    try:
        plan = json.loads(decomposition)
        steps = plan.get("steps", [])
    except json.JSONDecodeError:
        # Fallback: create a simple plan
        steps = []

    return {
        "decomposition": steps,
        "messages": [AIMessage(content=f"Decomposed into {len(steps)} steps")]
    }


def execution_planner_node(state: WorkflowState):
    """Create execution plan from decomposition."""
    decomposition = state.get("decomposition", [])

    execution_plan = []
    for i, step in enumerate(decomposition):
        execution_plan.append({
            "index": i,
            "step_type": step.get("step_type"),
            "description": step.get("description"),
            "parameters": step.get("parameters", {}),
        })

    return {
        "execution_plan": execution_plan,
        "current_step_index": 0
    }


def tool_executor_node(state: WorkflowState, config):
    """Execute current step in the plan."""
    current_index = state.get("current_step_index", 0)
    execution_plan = state.get("execution_plan", [])

    if current_index >= len(execution_plan):
        return {"current_step_index": current_index}

    current_step = execution_plan[current_index]
    tool_name = current_step.get("step_type")
    parameters = current_step.get("parameters", {})

    # Find the tool
    tool_map = {t.name: t for t in WORKFLOW_TOOLS}
    tool = tool_map.get(tool_name)

    if not tool:
        return {
            "errors": state.get("errors", []) + [f"Unknown tool: {tool_name}"],
            "current_step_index": current_index + 1
        }

    try:
        # Execute tool
        result = tool.invoke(parameters)

        # Handle contact search results - update parameters for next steps
        if tool_name == "search_contacts":
            contacts = result if isinstance(result, list) else []
            # Store found contacts for use in subsequent steps
            execution_results = state.get("execution_results", [])
            execution_results.append({
                "step_id": f"step-{current_index}",
                "tool_name": tool_name,
                "success": True,
                "result": f"Found {len(contacts)} contact(s)",
                "contacts": contacts,
                "timestamp": datetime.now().isoformat()
            })
            return {
                "execution_results": execution_results,
                "current_step_index": current_index + 1
            }

        execution_result = {
            "step_id": f"step-{current_index}",
            "tool_name": tool_name,
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        execution_results = state.get("execution_results", [])
        return {
            "execution_results": execution_results + [execution_result],
            "current_step_index": current_index + 1
        }
    except Exception as e:
        # Try recovery for missing contacts
        if tool_name == "search_contacts":
            query = parameters.get("query", "")
            placeholder = {
                "name": query,
                "email": f"{query.lower().replace(' ', '.')}@company.com",
                "phone": None,
                "department": "Unknown"
            }
            print(f"  ⚠ Contact '{query}' not found, using placeholder")

            execution_result = {
                "step_id": f"step-{current_index}",
                "tool_name": tool_name,
                "success": True,
                "result": f"[PLACEHOLDER] Created placeholder for '{query}'",
                "contacts": [placeholder],
                "timestamp": datetime.now().isoformat()
            }

            execution_results = state.get("execution_results", [])
            return {
                "execution_results": execution_results + [execution_result],
                "current_step_index": current_index + 1
            }

        return {
            "errors": state.get("errors", []) + [str(e)],
            "current_step_index": current_index + 1
        }


def should_continue_execution(state: WorkflowState) -> str:
    """Check if there are more steps to execute."""
    current_index = state.get("current_step_index", 0)
    execution_plan = state.get("execution_plan", [])

    if current_index >= len(execution_plan):
        return "end"
    return "continue"


def results_aggregator_node(state: WorkflowState):
    """Aggregate and format results."""
    execution_results = state.get("execution_results", [])
    errors = state.get("errors", [])

    summary = {
        "total_steps": len(state.get("execution_plan", [])),
        "completed_steps": len(execution_results),
        "errors": len(errors),
        "success": len(errors) == 0
    }

    return {"messages": [AIMessage(content=f"Execution complete: {summary}")]}


def build_workflow_graph(model, tools):
    """Build the workflow automation graph."""
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("decomposer", lambda state: task_decomposer_node(state, {"configurable": {"model": model}}))
    workflow.add_node("planner", execution_planner_node)
    workflow.add_node("executor", lambda state: tool_executor_node(state, {"configurable": {"model": model}}))
    workflow.add_node("aggregator", results_aggregator_node)

    # Set entry point
    workflow.set_entry_point("decomposer")

    # Add edges
    workflow.add_edge("decomposer", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "executor",
        should_continue_execution,
        {"continue": "executor", "end": "aggregator"}
    )
    workflow.add_edge("aggregator", END)

    return workflow.compile()


# ============================================================================
# Provider Configuration
# ============================================================================


PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}


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
# Demo Scenarios
# ============================================================================


DEMO_SCENARIOS = [
    {
        "name": "Meeting Scheduling with Email",
        "instruction": "Schedule a meeting with Alice and Bob next Tuesday about Q2 planning, then email them the agenda",
    },
    {
        "name": "Bulk Task Assignment",
        "instruction": "Create a task for each team member to review the design doc by Friday. Team members: Alice, Carol, and David",
    },
    {
        "name": "Partial Failure Recovery",
        "instruction": "Schedule a meeting with UnknownPerson and Alice about project kickoff",
    },
]


def run_demo(model, provider_name: str, model_id: str):
    """Run workflow automation demo."""
    print("=== Python — Workflow Automation Agent ===")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()

    graph = build_workflow_graph(model, WORKFLOW_TOOLS)

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"[{i}/{len(DEMO_SCENARIOS)}] {scenario['name']}")
        print(f"Instruction: {scenario['instruction']}")
        print("-" * 60)

        initial_state = {
            "messages": [],
            "user_instruction": scenario['instruction'],
            "decomposition": [],
            "execution_plan": [],
            "execution_results": [],
            "current_step_index": 0,
            "errors": [],
        }

        try:
            result = graph.invoke(initial_state)

            print("\n--- Execution Summary ---")
            print(f"Steps completed: {len(result.get('execution_results', []))}")
            print(f"Errors: {len(result.get('errors', []))}")

            print("\n--- Tool Call Log ---")
            for exec_result in result.get("execution_results", []):
                print(f"  [{exec_result['tool_name']}]")
                print(f"    {exec_result['result']}")

            if result.get("errors"):
                print("\n--- Errors ---")
                for error in result.get("errors", []):
                    print(f"  ✗ {error}")
        except Exception as e:
            print(f"\n✗ Error executing workflow: {e}")

        print("=" * 60)
        print()

    # Print session summary
    print("Session Summary")
    print(f"  Scenarios processed: {len(DEMO_SCENARIOS)}")


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
