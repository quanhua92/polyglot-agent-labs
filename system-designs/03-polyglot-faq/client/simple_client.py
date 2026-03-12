#!/usr/bin/env python3
"""Simple FAQ client - minimal example of how to use the FAQ API."""

import json
import sys
from typing import Optional

import requests


def ask_question(question: str, stream: bool = False, base_url: str = "http://localhost:3030") -> dict:
    """Ask a question to the FAQ system.

    Args:
        question: The question to ask
        stream: Whether to use streaming mode (not implemented in this simple client)
        base_url: Base URL of the API gateway

    Returns:
        The response dictionary
    """
    url = f"{base_url}/ask"
    payload = {"question": question, "stream": stream}

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)


def main():
    """Run the simple FAQ client."""
    print("=== Simple FAQ Client ===")
    print()

    # Example questions
    questions = [
        "How do I reset my password?",
        "What is your return policy?",
        "How can I track my order?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        print("-" * 60)

        result = ask_question(question)

        if "answer" in result:
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result}")

        if "metadata" in result:
            metadata = result["metadata"]
            print(f"Metadata: {metadata['steps_taken']} steps, {metadata['documents_found']} documents, {metadata['processing_time_ms']}ms")

        print()

    print("✓ All questions processed!")


if __name__ == "__main__":
    main()
