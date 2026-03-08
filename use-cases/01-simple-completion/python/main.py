"""
Polyglot Agent Labs — Use Case 01: Simple LLM Completion (Multi-Provider)
Sends a prompt to an LLM via litellm, supporting OpenAI, Anthropic, and OpenRouter.
Switch provider with env var LLM_PROVIDER (default: openrouter).
"""

import os
import sys

from dotenv import load_dotenv
from litellm import completion


# Provider → model mapping
PROVIDERS = {
    "openai": "openai/gpt-4.1-nano",
    "anthropic": "anthropic/claude-3-haiku-20240307",
    "openrouter": "openrouter/stepfun/step-3.5-flash:free",
}

PROMPT = "Hello! Tell me a fun fact about programming."


def main():
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider not in PROVIDERS:
        print(f"✗ Unknown provider: '{provider}'")
        print(f"  Supported: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    model = PROVIDERS[provider]

    print("=== Python — Simple LLM Completion ===")
    print(f"Provider:  {provider}")
    print(f"Model:     {model}")
    print()

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
        )

        text = response.choices[0].message.content
        usage = response.usage

        print(f"Prompt:    {PROMPT}")
        print(f"Response:  {text}")
        print()
        if usage:
            print(f"Tokens:    {usage.prompt_tokens} in / {usage.completion_tokens} out / {usage.total_tokens} total")

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
