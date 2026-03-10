#!/usr/bin/env python3
"""
Use Case 11: Code Review Agent

Demonstrates an agent that reads source code files, analyzes them,
and provides structured code review feedback including bugs, style issues,
security concerns, and improvement suggestions using Tool API.

Key Learning Goals:
- Tool API for structured output - tools accept typed parameters from LLM
- File I/O tools for reading source files
- Large context handling for code analysis
- Code analysis prompting techniques
"""

import asyncio
import json
import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# =============================================================================
# Pydantic Models for Tool Arguments (Input Schema)
# =============================================================================


class CodeFinding(BaseModel):
    """A single code review finding."""
    severity: str = Field(description="Severity level: critical, high, medium, low")
    category: str = Field(description="Category: bugs, security, style, best_practices")
    message: str = Field(description="Clear description of the issue")


class SubmitCodeReviewArgs(BaseModel):
    """Arguments for submitting code review."""
    summary: str = Field(description="Brief summary of the code quality")
    overall_score: int = Field(description="Overall score from 0-100", ge=0, le=100)
    issues: list[str] = Field(description="List of issues found (each as a brief description)")
    file_count: int = Field(description="Number of files reviewed")


# =============================================================================
# Pydantic Models for Structured Output (Result Types)
# =============================================================================


class CodeReview(BaseModel):
    """Structured code review output."""
    summary: str
    overall_score: int
    issues: list[str]
    file_count: int


# =============================================================================
# Shared State for Tool Outputs
# =============================================================================

class CodeReviewResults:
    """Shared state to store code review result."""
    def __init__(self):
        self.review: CodeReview | None = None

    def set_review(self, result: CodeReview):
        self.review = result


# Global results container
review_results = CodeReviewResults()


# =============================================================================
# SAMPLE CODE FILES (with intentional issues)
# =============================================================================

SAMPLE_FILES = {
    "insecure_login.py": """# SQL Injection vulnerability
def login(username, password):
    query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
    return db.execute(query)

# Hardcoded credentials
API_KEY = "sk-1234567890abcdef"

# Missing input validation
def reset_email(email):
    send_email(email)""",
    "poor_error_handling.py": """# Poor error handling
def divide(a, b):
    return a / b  # No zero division check

# Unused import
import os
import sys
import json

# Magic number
def calculate_discount(price):
    return price * 0.15

# Inconsistent naming
def getUserData():
    pass

def process_item(item):
    x = item['value']  # Unreadable variable name
    return x""",
    "resource_leak.py": """# Resource leak (file not closed)
def read_config(path):
    f = open(path, 'r')
    return f.read()

# Missing docstring
def process(data):
    x = []
    for i in range(len(data)):
        x.append(data[i] * 2)
    return x

# Global variable mutable default
cache = {}

def add_to_cache(key, value, cache=cache):
    cache[key] = value""",
}


# =============================================================================
# FINDING CATEGORIES AND SEVERITY LEVELS
# =============================================================================

FINDING_CATEGORIES = {
    "bugs": "Runtime errors, logic errors, incorrect behavior",
    "security": "Security vulnerabilities, injection risks, exposure issues",
    "style": "Code style, formatting, naming conventions",
    "best_practices": "Best practice violations, maintainability issues",
}

SEVERITY_LEVELS = {
    "critical": "Must fix - poses immediate risk",
    "high": "Should fix - important issue",
    "medium": "Consider fixing - improvement opportunity",
    "low": "Optional - minor issue or nitpick",
}


# =============================================================================
# Tool API - Code Review Tool
# =============================================================================

def submit_code_review_tool(
    summary: str,
    overall_score: int,
    issues: list[str],
    file_count: int,
) -> str:
    """Submit structured code review feedback.

    Args:
        summary: Brief summary of the code quality
        overall_score: Overall score from 0-100
        issues: List of issues found (each as a brief description)
        file_count: Number of files reviewed

    Returns:
        Confirmation message with review summary
    """
    review = CodeReview(
        summary=summary,
        overall_score=overall_score,
        issues=issues,
        file_count=file_count,
    )
    review_results.set_review(review)
    return f"Code review submitted: score {overall_score}/100, {len(issues)} issues found"


# Create tool definition
code_review_tool = StructuredTool.from_function(
    func=submit_code_review_tool,
    name="submit_code_review",
    description="Submit structured code review feedback with score and issues",
    args_schema=SubmitCodeReviewArgs,
)


# =============================================================================
# MULTI-PROVIDER SETUP
# =============================================================================

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


# =============================================================================
# CODE REVIEWER PROMPT
# =============================================================================

CODE_REVIEWER_PROMPT = """You are an expert code reviewer. Analyze the provided code and give structured feedback.

Provide:
- A summary of the code quality
- An overall score from 0-100
- A list of issues found (each as a brief description)

Focus on: bugs, security issues, style problems, and best practices.

Scoring guidelines:
- Start at 100
- Critical: -20 points each
- High: -10 points each
- Medium: -5 points each
- Low: -2 points each

IMPORTANT: You MUST use the submit_code_review tool to submit your assessment."""


# =============================================================================
# Tool Calling Helper
# =============================================================================

async def review_code_with_tools(
    code_text: str,
    model: BaseChatModel,
    max_turns: int = 5,
) -> CodeReview | None:
    """Review code using Tool API with multi-turn conversation.

    Args:
        code_text: Code to review
        model: Chat model to use
        max_turns: Maximum number of conversation turns

    Returns:
        CodeReview if successful, None otherwise
    """
    # Bind tools to model for tool calling support
    model_with_tools = model.bind_tools([code_review_tool])

    messages = [
        SystemMessage(content=CODE_REVIEWER_PROMPT),
        HumanMessage(content=f"Analyze this code:\n\n{code_text}\n\nUse the submit_code_review tool to submit your review."),
    ]

    for turn in range(max_turns):
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        # Check if the model wants to call a tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute all tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "")

                if tool_name == "submit_code_review":
                    try:
                        result = code_review_tool.func(**tool_args)
                        messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                    except Exception as e:
                        messages.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_id))
                else:
                    messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found", tool_call_id=tool_id))
        else:
            # No tool calls, review complete
            break

    return review_results.review


# =============================================================================
# REPORT FORMATTING
# =============================================================================

def print_review_report(review: CodeReview, provider_name: str, model_id: str):
    """Print formatted code review report."""
    print(f"\n{'='*60}")
    print("CODE REVIEW REPORT")
    print(f"{'='*60}\n")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()
    print(f"Summary: {review.summary}")
    print(f"Files Reviewed: {review.file_count}")
    print(f"Overall Score: {review.overall_score}/100")
    print(f"\n{'─'*60}")
    print("ISSUES FOUND")
    print(f"{'─'*60}")

    if not review.issues:
        print("\n✓ No issues found!")
    else:
        for i, issue in enumerate(review.issues, 1):
            print(f"\n[{i}] {issue}")

    print(f"\n{'='*60}")
    print(f"Total Issues: {len(review.issues)}")
    print(f"{'='*60}")


# =============================================================================
# DEMO EXECUTION
# =============================================================================

async def run_demo_async(model: BaseChatModel, provider_name: str, model_id: str):
    """Run code review demo asynchronously."""
    print("=== Python — Code Review Agent (Tool API) ===")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()

    # Format code for review
    code_text = ""
    for path, content in SAMPLE_FILES.items():
        code_text += f"\n{'='*60}\n"
        code_text += f"FILE: {path}\n"
        code_text += f"{'='*60}\n"
        # Add line numbers
        lines = content.strip().split("\n")
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        code_text += "\n".join(numbered)
        code_text += "\n"

    # Reset global result
    review_results.review = None

    # Perform code review using Tool API
    print("Analyzing code...")
    review = await review_code_with_tools(code_text, model)

    if review:
        # Print the report
        print_review_report(review, provider_name, model_id)
    else:
        print("\n✗ Error: Code review tool was not called")


def run_demo(model: BaseChatModel, provider_name: str, model_id: str):
    """Synchronous wrapper for demo."""
    asyncio.run(run_demo_async(model, provider_name, model_id))


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    load_dotenv()

    # Get provider from environment or use default
    provider_key = os.getenv("LLM_PROVIDER", "openrouter")
    model_id, provider_name = PROVIDERS.get(provider_key, PROVIDERS["openrouter"])

    model = create_chat_model(provider_name, model_id)
    run_demo(model, provider_name, model_id)


if __name__ == "__main__":
    main()
