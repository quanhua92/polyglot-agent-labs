"""
Polyglot Agent Labs — Use Case 03: Conversational Agent with Memory
A chatbot that maintains conversation history across multiple turns.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py                           # Non-interactive mode (predefined conversation)
  python main.py --interactive             # Enable interactive REPL mode

Commands:
  /quit, /exit, /q    Exit the session (interactive mode only)
"""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Provider → (model_id, provider_type)
PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}

SYSTEM_PROMPT = "You are a helpful assistant. Be concise and friendly."

# Predefined conversation for non-interactive mode
PREDEFINED_CONVERSATION = [
    "hello! my name is Alice.",
    "What's the weather in Tokyo?",
    "What's the weather like in London?",
    "Thanks! Goodbye!",
]


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


def run_interactive(chat, provider_key: str, model_id: str) -> int:
    """Run the interactive REPL mode."""
    print("=== Python — Conversational Agent with Memory ===")
    print(f"Provider:  {provider_key}")
    print(f"Model:     {model_id}")
    print("Mode:      interactive")
    print()
    print("Commands: /quit, /exit, /q to end session")
    print()

    # Initialize message history with system prompt
    history: list = [SystemMessage(content=SYSTEM_PROMPT)]
    turn_count = 0

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            # Check for exit commands
            if user_input.lower() in ("/quit", "/exit", "/q"):
                break

            # Skip empty input
            if not user_input:
                continue

            # Add user message to history
            history.append(HumanMessage(content=user_input))

            try:
                # Call LLM with full history
                response = chat.invoke(history)
                assistant_message = response.content

                # Add assistant response to history
                history.append(AIMessage(content=assistant_message))

                # Print response
                print(f"Assistant: {assistant_message}")
                print()

                turn_count += 1

            except Exception as e:
                print(f"✗ Error: {e}")
                print()
                # Remove the failed user message from history
                history.pop()

    finally:
        # Print session summary
        print("=" * 50)
        print("Session ended")
        print(f"Total turns: {turn_count}")

    return turn_count


def run_non_interactive(chat, provider_key: str, model_id: str) -> int:
    """Run the non-interactive mode with predefined conversation."""
    print("=== Python — Conversational Agent with Memory ===")
    print(f"Provider:  {provider_key}")
    print(f"Model:     {model_id}")
    print("Mode:      non-interactive (predefined conversation)")
    print()

    # Initialize message history with system prompt
    history: list = [SystemMessage(content=SYSTEM_PROMPT)]
    turn_count = 0

    for user_input in PREDEFINED_CONVERSATION:
        print(f"You: {user_input}")

        # Add user message to history
        history.append(HumanMessage(content=user_input))

        try:
            # Call LLM with full history
            response = chat.invoke(history)
            assistant_message = response.content

            # Add assistant response to history
            history.append(AIMessage(content=assistant_message))

            # Print response
            print(f"Assistant: {assistant_message}")
            print()

            turn_count += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            print()
            # Remove the failed user message from history
            history.pop()
            break

    # Print session summary
    print("=" * 50)
    print("Session ended")
    print(f"Total turns: {turn_count}")

    return turn_count


def main():
    load_dotenv()

    # Check for --interactive flag
    interactive = "--interactive" in sys.argv

    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider_key not in PROVIDERS:
        print(f"✗ Unknown provider: '{provider_key}'")
        print(f"  Supported: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    model_id, provider_type = PROVIDERS[provider_key]
    chat = create_chat_model(provider_type, model_id)

    if interactive:
        run_interactive(chat, provider_key, model_id)
    else:
        run_non_interactive(chat, provider_key, model_id)


if __name__ == "__main__":
    main()
