"""
Polyglot Agent Labs - Hello World (Python)
Demonstrates environment variable loading for LangGraph/Rig agents.
"""

import os
from dotenv import load_dotenv


def main():
    # Load from root .env (via justfile's dotenv-load)
    # When running via `just py 00`, the env vars are already exported
    # This is just a fallback for direct python execution
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    log_level = os.getenv("AGENT_LOG_LEVEL", "info")

    print("=== Python Environment Check ===")
    print(f"OPENAI_API_KEY:     {'✓ Set' if openai_key else '✗ Missing'}")
    print(f"ANTHROPIC_API_KEY:  {'✓ Set' if anthropic_key else '✗ Missing'}")
    print(f"OPENROUTER_API_KEY: {'✓ Set' if openrouter_key else '✗ Missing'}")
    print(f"AGENT_LOG_LEVEL:    {log_level}")

    # Example: Initialize a LangGraph agent
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o-mini")  # Uses OPENAI_API_KEY from env


if __name__ == "__main__":
    main()
