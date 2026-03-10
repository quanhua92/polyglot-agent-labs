#!/usr/bin/env python3
"""
Use Case 11: Code Review Agent

Demonstrates an agent that reads source code files, analyzes them,
and provides structured code review feedback including bugs, style issues,
security concerns, and improvement suggestions.

Key Learning Goals:
- File I/O tools for reading source files
- Structured output for code review findings
- Large context handling for code analysis
- Code analysis prompting techniques
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()

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
# FILE I/O TOOLS
# =============================================================================


def list_files(directory: str = ".") -> list[str]:
    """List all files in a directory.

    Args:
        directory: Path to the directory to list files from

    Returns:
        List of file names in the directory
    """
    try:
        if not os.path.exists(directory):
            return [f"Error: Directory '{directory}' does not exist"]

        files = []
        for entry in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, entry)):
                files.append(entry)
        return sorted(files)
    except Exception as e:
        return [f"Error listing files: {str(e)}"]


def read_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        The contents of the file as a string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Add line numbers for easier reference
            lines = content.split("\n")
            numbered_lines = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered_lines)
    except FileNotFoundError:
        return f"Error: File not found at '{file_path}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def get_sample_file(file_name: str) -> str:
    """Get a sample code file with intentional issues for review.

    Args:
        file_name: Name of the sample file (e.g., 'insecure_login.py')

    Returns:
        The file contents with line numbers
    """
    if file_name not in SAMPLE_FILES:
        available = ", ".join(SAMPLE_FILES.keys())
        return f"Error: Sample file '{file_name}' not found. Available: {available}"

    content = SAMPLE_FILES[file_name]
    lines = content.strip().split("\n")
    numbered_lines = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


# =============================================================================
# MULTI-PROVIDER SETUP
# =============================================================================

PROVIDERS = {
    "openai": ("gpt-4.1-mini", "openai"),
    "anthropic": ("claude-3-7-haiku-20250221", "anthropic"),
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


# =============================================================================
# CODE REVIEWER PROMPT
# =============================================================================

CODE_REVIEWER_PROMPT = """You are an expert code reviewer. Analyze the provided code and give structured feedback.

For each issue found, provide:
- severity: low, medium, high, or critical
- category: bugs, security, style, or best_practices
- line_number: The line where the issue occurs
- code_snippet: The relevant code snippet
- message: Clear description of the issue
- suggestion: Specific recommendation for fixing it

Scoring guidelines:
- Start at 100
- Critical: -20 points each
- High: -10 points each
- Medium: -5 points each
- Low: -2 points each

Provide your response as valid JSON matching this schema:
{
  "summary": "Overall assessment",
  "findings": [
    {
      "severity": "level",
      "category": "type",
      "line_number": number,
      "code_snippet": "code",
      "message": "description",
      "suggestion": "fix"
    }
  ],
  "overall_score": number,
  "file_count": number,
  "lines_reviewed": number
}"""


# =============================================================================
# JSON EXTRACTION HELPER
# =============================================================================


def extract_json_from_response(response: str) -> str:
    """Extract JSON from a response that may contain extra text."""
    response = response.strip()

    # Look for JSON block between ```json and ```
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # Look for JSON block between ``` and ```
    json_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
        if candidate.startswith("{"):
            return candidate

    # Look for { and } as JSON boundaries
    start = response.find("{")
    if start != -1:
        brace_count = 0
        for i in range(start, len(response)):
            if response[i] == "{":
                brace_count += 1
            elif response[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return response[start : i + 1]

    return response


# =============================================================================
# REPORT FORMATTING
# =============================================================================


def print_review_report(review: dict, provider_name: str, model_id: str):
    """Print formatted code review report."""
    print(f"\n{'='*60}")
    print("CODE REVIEW REPORT")
    print(f"{'='*60}\n")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()
    print(f"Summary: {review.get('summary', 'No summary')}")
    print(f"Files Reviewed: {review.get('file_count', 0)}")
    print(f"Lines Reviewed: {review.get('lines_reviewed', 0)}")
    print(f"Overall Score: {review.get('overall_score', 0)}/100")
    print(f"\n{'─'*60}")
    print("FINDINGS")
    print(f"{'─'*60}")

    findings = review.get("findings", [])
    if not findings:
        print("\n✓ No issues found!")
    else:
        # Group by severity
        severity_order = ["critical", "high", "medium", "low"]
        grouped = {s: [] for s in severity_order}
        for f in findings:
            sev = f.get("severity", "low").lower()
            if sev in grouped:
                grouped[sev].append(f)

        for severity in severity_order:
            if grouped[severity]:
                for i, finding in enumerate(grouped[severity], 1):
                    category = finding.get("category", "unknown")
                    line = finding.get("line_number", "?")
                    msg = finding.get("message", "")

                    print(f"\n[{i}] {severity.upper()} | {category} | Line {line}")
                    print(f"    {msg}")

                    snippet = finding.get("code_snippet", "")
                    if snippet:
                        truncated = snippet[:60] + "..." if len(snippet) > 60 else snippet
                        print(f"    Code: {truncated}")

                    suggestion = finding.get("suggestion", "")
                    if suggestion:
                        print(f"    → {suggestion}")

    print(f"\n{'='*60}")
    print(f"Total Issues: {len(findings)}")
    print(f"{'='*60}")


# =============================================================================
# CODE REVIEW FUNCTION
# =============================================================================


def review_code(model, code_contents: dict[str, str]) -> dict:
    """Review code files and return structured findings."""
    # Format code for review
    code_text = ""
    for path, content in code_contents.items():
        code_text += f"\n{'='*60}\n"
        code_text += f"FILE: {path}\n"
        code_text += f"{'='*60}\n"
        # Add line numbers
        lines = content.split("\n")
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        code_text += "\n".join(numbered)
        code_text += "\n"

    prompt = f"""{CODE_REVIEWER_PROMPT}

CODE TO REVIEW:
{code_text}

Analyze this code and return valid JSON with the review findings."""

    response = model.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    # Extract and parse JSON
    json_str = extract_json_from_response(response_text)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing review: {e}")
        print(f"Raw response: {json_str[:500]}...")
        return {
            "summary": "Failed to parse review",
            "findings": [],
            "overall_score": 0,
            "file_count": len(code_contents),
            "lines_reviewed": sum(len(c.split("\n")) for c in code_contents.values()),
        }


# =============================================================================
# DEMO EXECUTION
# =============================================================================


def run_demo(model, provider_name: str, model_id: str):
    """Run code review demo."""
    print("=== Python — Code Review Agent ===")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()

    # Prepare sample code
    code_contents = {
        name: content.strip() for name, content in SAMPLE_FILES.items()
    }

    # Perform code review
    review = review_code(model, code_contents)

    # Print the report
    print_review_report(review, provider_name, model_id)


def main():
    """Main entry point."""
    # Get provider from environment or use default
    provider_key = os.getenv("LLM_PROVIDER", "openrouter")
    model_id, provider_name = PROVIDERS.get(provider_key, PROVIDERS["openrouter"])

    print("=== Use Case 11: Code Review Agent ===")
    print(f"Using provider: {provider_name} (model: {model_id})")
    print()

    model = create_chat_model(provider_name, model_id)
    run_demo(model, provider_name, model_id)


if __name__ == "__main__":
    main()
