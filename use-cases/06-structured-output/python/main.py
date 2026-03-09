"""
Polyglot Agent Labs — Use Case 06: Structured Output & Data Extraction
Extract typed, validated data from unstructured text using structured LLM output.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

This demo extracts structured data from 3 types of unstructured text:
- Job listings: title, company, location, salary_range, required_skills, employment_type, description
- Product reviews: product_name, rating, pros, cons, summary, would_recommend
- Emails: sender, recipients, subject, action_items, urgency, key_points, deadline
"""

import json
import os
import sys
import re

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

# ============================================================================
# Pydantic Models for Structured Extraction
# ============================================================================


class JobListing(BaseModel):
    """Structured data extracted from a job listing."""
    title: str = Field(description="The job title")
    company: str = Field(description="The company name")
    location: str = Field(description="The job location (city, state/country, or remote)")
    salary_range: str | None = Field(description="Salary range if specified", default=None)
    required_skills: list[str] = Field(description="List of required technical skills")
    employment_type: str | None = Field(description="Employment type (full-time, part-time, contract, etc.)", default=None)
    description: str | None = Field(description="Brief description of the role", default=None)


class ProductReview(BaseModel):
    """Structured data extracted from a product review."""
    product_name: str = Field(description="The name of the product")
    rating: int = Field(description="Numeric rating (1-5 stars)", ge=1, le=5)
    pros: list[str] = Field(description="List of positive aspects mentioned")
    cons: list[str] = Field(description="List of negative aspects mentioned")
    summary: str = Field(description="Brief summary of the reviewer's sentiment")
    would_recommend: bool | None = Field(description="Whether the reviewer would recommend this product", default=None)


class EmailInfo(BaseModel):
    """Structured data extracted from an email."""
    sender: str = Field(description="Email sender name or address")
    recipients: list[str] = Field(description="List of email recipients")
    subject: str = Field(description="Email subject line")
    action_items: list[str] = Field(description="Action items or tasks mentioned in the email")
    urgency: str = Field(description="Urgency level (high, medium, low)")
    key_points: list[str] = Field(description="Key points or information conveyed")
    deadline: str | None = Field(description="Any deadline mentioned", default=None)


# ============================================================================
# Hard-coded Sample Inputs
# ============================================================================

SAMPLE_JOB_LISTING = """
SENIOR RUST ENGINEER
CloudScale Technologies - Remote / San Francisco, CA

We're looking for a Senior Rust Engineer to join our growing cloud infrastructure team.

Requirements:
- 3+ years of experience with Rust
- Strong knowledge of async programming (tokio, async-std)
- Experience with distributed systems
- Familiarity with WebAssembly (Wasm)
- Understanding of container technologies (Docker, Kubernetes)

What you'll do:
- Build and maintain high-performance cloud services
- Design scalable microservices architectures
- Mentor junior engineers
- Contribute to open-source projects

We offer competitive salary ($150k-$200k), equity, comprehensive health benefits, and flexible work arrangements. This is a full-time position with opportunities for growth.

Apply at careers@cloudscale.io
"""

SAMPLE_PRODUCT_REVIEW = """
Sony WH-1000XM5 Wireless Noise Canceling Headphones

I've been using these headphones for 6 months now and I'm impressed! The noise cancellation is genuinely the best I've experienced - it completely blocks out my noisy commute.

Pros:
- Exceptional ANC (Active Noise Cancellation)
- Comfortable for long sessions (I wear them 4+ hours daily)
- 30-hour battery life
- Great sound quality with deep bass
- Multipoint connection works flawlessly

Cons:
- Expensive at $399
- No folding design (bulky in bag)
- Case feels cheap for the price
- Touch controls can be finicky

Overall: 4.5/5 stars
I'd definitely recommend these if budget isn't a concern. The ANC alone makes them worth it for frequent travelers. Sound quality is premium but audiophiles might prefer wired options.
"""

SAMPLE_EMAIL = """
From: sarah.chen@techcorp.com
To: dev-team@techcorp.com, john.smith@techcorp.com
Subject: Q2 API Migration - Action Required

Hi team,

We need to complete the API migration by April 15th. This is critical for the client launch scheduled for May 1st.

ACTION ITEMS:
- Maria: Complete auth service migration (due: March 25th)
- John: Update payment gateway integration (due: April 1st)
- Team: Review and test all endpoints (due: April 10th)
- Sarah: Document API changes for clients (due: April 12th)

URGENT: We have a blocker with the payment gateway - please prioritize fixing the transaction timeout issue.

Key points:
- Use the new API documentation in Confluence
- All migrations must be tested in staging first
- Daily standup will focus on migration status
- Client demo is April 20th

Let me know if you need any resources allocated.

Thanks,
Sarah
"""

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
        # Use JSON mode for OpenAI
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("✗ ANTHROPIC_API_KEY not set")
            sys.exit(1)
        # Anthropic doesn't have native JSON mode, use prompt engineering
        return ChatAnthropic(model=model_id, api_key=api_key)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("✗ OPENROUTER_API_KEY not set")
            sys.exit(1)
        # Use JSON mode for OpenRouter
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    else:
        print(f"✗ Unknown provider type: '{provider}'")
        sys.exit(1)


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
        # Verify it looks like JSON
        if candidate.startswith('{'):
            return candidate

    # Look for { and } as JSON boundaries
    start = response.find('{')
    if start != -1:
        # Find the matching closing brace
        brace_count = 0
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response[start:i+1]

    # Return as-is if no JSON markers found
    return response


# ============================================================================
# Extraction Functions
# ============================================================================

def extract_job_listing(text: str, model) -> JobListing:
    """Extract structured job listing data."""
    schema = """
{
  "title": "string - The job title",
  "company": "string - The company name",
  "location": "string - The job location (city, state/country, or remote)",
  "salary_range": "string or null - Salary range if specified",
  "required_skills": ["string"] - List of required technical skills",
  "employment_type": "string or null - Employment type (full-time, part-time, contract, etc.)",
  "description": "string or null - Brief description of the role"
}"""

    prompt = f"""Extract structured data from the following job listing. Return ONLY valid JSON matching this schema:

{schema}

Job listing:
{text}"""

    response = model.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    json_str = extract_json_from_response(content)
    return JobListing.model_validate_json(json_str)


def extract_product_review(text: str, model) -> ProductReview:
    """Extract structured product review data."""
    schema = """
{
  "product_name": "string - The name of the product",
  "rating": integer (1-5) - Numeric rating",
  "pros": ["string"] - List of positive aspects mentioned",
  "cons": ["string"] - List of negative aspects mentioned",
  "summary": "string - Brief summary of the reviewer's sentiment",
  "would_recommend": boolean or null - Whether the reviewer would recommend this product
}"""

    prompt = f"""Extract structured data from the following product review. Return ONLY valid JSON matching this schema:

{schema}

Product review:
{text}"""

    response = model.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    json_str = extract_json_from_response(content)
    return ProductReview.model_validate_json(json_str)


def extract_email_info(text: str, model) -> EmailInfo:
    """Extract structured email information."""
    schema = """
{
  "sender": "string - Email sender name or address",
  "recipients": ["string"] - List of email recipients",
  "subject": "string - Email subject line",
  "action_items": ["string"] - Action items or tasks mentioned in the email",
  "urgency": "string - Urgency level (high, medium, low)",
  "key_points": ["string"] - Key points or information conveyed",
  "deadline": "string or null - Any deadline mentioned"
}"""

    prompt = f"""Extract structured data from the following email. Return ONLY valid JSON matching this schema:

{schema}

Email:
{text}"""

    response = model.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    json_str = extract_json_from_response(content)
    return EmailInfo.model_validate_json(json_str)


# ============================================================================
# Validation and Retry Logic
# ============================================================================

def validate_extraction(result: BaseModel, model_class: type[BaseModel]) -> bool:
    """Validate that required fields are populated."""
    # Check that required string fields are not empty
    for field_name, field_info in model_class.model_fields.items():
        if field_info.is_required():
            value = getattr(result, field_name)
            if isinstance(value, str) and not value.strip():
                print(f"  ✗ Validation failed: {field_name} is empty")
                return False
            if isinstance(value, list) and len(value) == 0:
                print(f"  ✗ Validation failed: {field_name} list is empty")
                return False
    return True


def extract_with_retry(text: str, model, ModelClass, extract_fn, max_retries: int = 2):
    """Extract with retry logic for malformed output."""
    for attempt in range(max_retries + 1):
        try:
            result = extract_fn(text, model)
            if validate_extraction(result, ModelClass):
                return result
            else:
                if attempt < max_retries:
                    print(f"  ⚠ Validation failed, retrying... (attempt {attempt + 1}/{max_retries})")
                else:
                    print(f"  ✗ Validation failed after {max_retries} retries")
                    return result
        except ValidationError as e:
            if attempt < max_retries:
                print(f"  ⚠ Validation error: {e}")
                print(f"  ⚠ Retrying... (attempt {attempt + 1}/{max_retries})")
            else:
                print(f"  ✗ Validation error after {max_retries} retries: {e}")
                raise
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⚠ Extraction error: {e}")
                print(f"  ⚠ Retrying... (attempt {attempt + 1}/{max_retries})")
            else:
                print(f"  ✗ Extraction failed after {max_retries} retries: {e}")
                raise
    return None


# ============================================================================
# Demo Execution
# ============================================================================

def run_demo(model, provider_key: str, model_id: str) -> tuple[int, int]:
    """Run the structured extraction demo."""
    print("=== Python — Structured Output & Data Extraction ===")
    print(f"Provider: {provider_key}")
    print(f"Model: {model_id}")
    print()

    success_count = 0
    failure_count = 0

    # ========================================================================
    # Job Listing Extraction
    # ========================================================================
    print("[1/3] Job Listing Extraction")
    print("-" * 50)
    print(f"Input: {SAMPLE_JOB_LISTING[:100]}...")
    print()

    try:
        result = extract_with_retry(SAMPLE_JOB_LISTING, model, JobListing, extract_job_listing)
        if result:
            print("\nExtracted Data:")
            print(result.model_dump_json(indent=2))
            print(f"\nValidation: ✓ All required fields present")
            success_count += 1
        else:
            failure_count += 1
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        failure_count += 1

    print("=" * 50)
    print()

    # ========================================================================
    # Product Review Extraction
    # ========================================================================
    print("[2/3] Product Review Extraction")
    print("-" * 50)
    print(f"Input: {SAMPLE_PRODUCT_REVIEW[:100]}...")
    print()

    try:
        result = extract_with_retry(SAMPLE_PRODUCT_REVIEW, model, ProductReview, extract_product_review)
        if result:
            print("\nExtracted Data:")
            print(result.model_dump_json(indent=2))
            print(f"\nValidation: ✓ All required fields present")
            success_count += 1
        else:
            failure_count += 1
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        failure_count += 1

    print("=" * 50)
    print()

    # ========================================================================
    # Email Info Extraction
    # ========================================================================
    print("[3/3] Email Information Extraction")
    print("-" * 50)
    print(f"Input: {SAMPLE_EMAIL[:100]}...")
    print()

    try:
        result = extract_with_retry(SAMPLE_EMAIL, model, EmailInfo, extract_email_info)
        if result:
            print("\nExtracted Data:")
            print(result.model_dump_json(indent=2))
            print(f"\nValidation: ✓ All required fields present")
            success_count += 1
        else:
            failure_count += 1
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        failure_count += 1

    print("=" * 50)
    print()

    # Session Summary
    print("Session Summary")
    print(f"  Extractions completed: 3/3")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")

    return success_count, failure_count


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
