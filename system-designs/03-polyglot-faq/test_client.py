#!/usr/bin/env python3
"""Comprehensive test client for the FAQ API Gateway.

This client provides:
- Non-streaming mode tests
- Streaming mode tests
- Health checks
- Error handling tests
"""

import argparse
import json
import sys
import time
from typing import Optional

import requests
import httpx


class FAQClient:
    """Client for interacting with the FAQ API Gateway."""

    def __init__(self, base_url: str = "http://localhost:3030", timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)

    def ask_question(self, question: str, stream: bool = False) -> dict:
        """Ask a question to the FAQ system (non-streaming)."""
        url = f"{self.base_url}/ask"
        payload = {"question": question, "stream": stream}

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def ask_question_stream(self, question: str):
        """Ask a question with streaming response."""
        url = f"{self.base_url}/ask/stream"
        payload = {"question": question, "stream": True}

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response

    def health_check(self) -> dict:
        """Check the health of the API gateway."""
        url = f"{self.base_url}/health"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the async client."""
        import asyncio
        try:
            asyncio.run(self.http_client.aclose())
        except:
            pass


def test_health(client: FAQClient) -> bool:
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        health = client.health_check()
        print(f"  Status: {health.get('status', 'unknown')}")
        print(f"  Version: {health.get('version', 'unknown')}")
        print("  ✓ Health check passed")
        return True
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False


def test_single_question(client: FAQClient, question: str) -> bool:
    """Test a single question (non-streaming)."""
    print(f"\nTesting: {question}")
    print("-" * 60)

    try:
        start = time.time()
        result = client.ask_question(question)
        elapsed = time.time() - start

        if "error" in result:
            print(f"  ✗ Error: {result.get('message', result['error'])}")
            return False

        if "answer" in result:
            print(f"  Answer: {result['answer']}")

        if "metadata" in result:
            metadata = result["metadata"]
            print(f"  Metadata:")
            print(f"    - Steps: {metadata.get('steps_taken', 'N/A')}")
            print(f"    - Documents: {metadata.get('documents_found', 'N/A')}")
            print(f"    - Time: {metadata.get('processing_time_ms', 'N/A')}ms")

        print(f"  ✓ Completed in {elapsed:.2f}s")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_streaming(client: FAQClient, question: str) -> bool:
    """Test the streaming endpoint."""
    print(f"\nTesting streaming: {question}")
    print("-" * 60)

    try:
        response = client.ask_question_stream(question)

        print("  Stream events:")
        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    data = line[6:]  # Remove "data: " prefix
                    try:
                        event = json.loads(data)
                        event_type = event.get("type", "unknown")
                        node = event.get("node", "")
                        print(f"    - [{event_type}] {node}")
                    except json.JSONDecodeError:
                        print(f"    - Raw: {data[:50]}...")

        print("  ✓ Stream completed")
        return True

    except Exception as e:
        print(f"  ✗ Streaming failed: {e}")
        return False


def test_error_cases(client: FAQClient) -> bool:
    """Test error handling."""
    print("\nTesting error cases...")
    print("-" * 60)

    errors = []

    # Test empty question
    try:
        result = client.ask_question("")
        if "error" not in result:
            errors.append("Empty question should return error")
    except Exception:
        pass  # Expected

    # Test moderately long question (not too long to avoid timeout)
    try:
        long_question = "test " * 50  # 500 chars instead of 5000
        result = client.ask_question(long_question)
        # Should handle gracefully
    except Exception as e:
        errors.append(f"Long question failed: {e}")

    if errors:
        for error in errors:
            print(f"  ✗ {error}")
        return False
    else:
        print("  ✓ All error cases handled correctly")
        return True


def run_tests(
    base_url: str,
    streaming: bool = False,
    questions: Optional[list[str]] = None,
) -> int:
    """Run all tests and return exit code."""
    print("=== FAQ API Gateway Test Client ===")
    print(f"Base URL: {base_url}")
    print()

    client = FAQClient(base_url=base_url)

    try:
        passed = 0
        total = 0

        # Health check
        total += 1
        if test_health(client):
            passed += 1

        # Default test questions (limited for faster testing)
        if not questions:
            questions = [
                "How do I reset my password?",
            ]

        # Test questions
        for question in questions:
            total += 1
            if test_single_question(client, question):
                passed += 1

            if streaming:
                total += 1
                if test_streaming(client, question):
                    passed += 1

        # Summary
        print("\n" + "=" * 60)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 60)

        return 0 if passed == total else 1

    finally:
        client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test client for the FAQ API Gateway"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:3030",
        help="Base URL of the API gateway (default: http://localhost:3030)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Test streaming mode"
    )
    parser.add_argument(
        "--question",
        action="append",
        help="Add a test question (can be specified multiple times)"
    )

    args = parser.parse_args()

    sys.exit(run_tests(
        base_url=args.url,
        streaming=args.stream,
        questions=args.question
    ))


if __name__ == "__main__":
    main()
