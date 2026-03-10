"""
Polyglot Agent Labs — Use Case 06: Structured Output & Data Extraction
Extract typed, validated data from unstructured text using Tool API.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

This demo extracts structured data from 3 types of unstructured text:
- Job listings: title, company, location, salary_range, required_skills, employment_type, description
- Product reviews: product_name, rating, pros, cons, summary, would_recommend
- Emails: sender, recipients, subject, action_items, urgency, key_points, deadline

Key Learning Goals:
- Tool API for structured output - tools accept typed parameters from LLM
- Multi-turn tool calling for reliable extraction
- Shared state pattern for storing tool results
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models for Tool Arguments (Input Schema)
# ============================================================================


class ExtractJobListingArgs(BaseModel):
    """Arguments for extracting job listing data."""
    title: str = Field(description="The job title")
    company: str = Field(description="The company name")
    location: str = Field(description="The job location (city, state/country, or remote)")
    salary_range: Optional[str] = Field(default=None, description="Salary range if specified")
    required_skills: list[str] = Field(description="List of required technical skills")
    employment_type: Optional[str] = Field(default=None, description="Employment type (full-time, part-time, contract, etc.)")
    description: Optional[str] = Field(default=None, description="Brief description of the role")


class ExtractProductReviewArgs(BaseModel):
    """Arguments for extracting product review data."""
    product_name: str = Field(description="The name of the product")
    rating: int = Field(description="Numeric rating (1-5 stars)", ge=1, le=5)
    pros: list[str] = Field(description="List of positive aspects mentioned")
    cons: list[str] = Field(description="List of negative aspects mentioned")
    summary: str = Field(description="Brief summary of the reviewer's sentiment")
    would_recommend: Optional[bool] = Field(default=None, description="Whether the reviewer would recommend this product")


class ExtractEmailInfoArgs(BaseModel):
    """Arguments for extracting email information."""
    sender: str = Field(description="Email sender name or address")
    recipients: list[str] = Field(description="List of email recipients")
    subject: str = Field(description="Email subject line")
    action_items: list[str] = Field(description="Action items or tasks mentioned in the email")
    urgency: str = Field(description="Urgency level (high, medium, low)")
    key_points: list[str] = Field(description="Key points or information conveyed")
    deadline: Optional[str] = Field(default=None, description="Any deadline mentioned")


# ============================================================================
# Pydantic Models for Structured Output (Result Types)
# ============================================================================


class JobListing(BaseModel):
    """Structured data extracted from a job listing."""
    title: str
    company: str
    location: str
    salary_range: Optional[str] = None
    required_skills: list[str]
    employment_type: Optional[str] = None
    description: Optional[str] = None


class ProductReview(BaseModel):
    """Structured data extracted from a product review."""
    product_name: str
    rating: int
    pros: list[str]
    cons: list[str]
    summary: str
    would_recommend: Optional[bool] = None


class EmailInfo(BaseModel):
    """Structured data extracted from an email."""
    sender: str
    recipients: list[str]
    subject: str
    action_items: list[str]
    urgency: str
    key_points: list[str]
    deadline: Optional[str] = None


# ============================================================================
# Shared State for Tool Outputs
# ============================================================================

class ExtractionResults:
    """Shared state to store tool results (equivalent to Arc<Mutex<Option<T>>> in Rust)."""
    def __init__(self):
        self.job_listing: Optional[JobListing] = None
        self.product_review: Optional[ProductReview] = None
        self.email_info: Optional[EmailInfo] = None

    def set_job_listing(self, result: JobListing):
        self.job_listing = result

    def set_product_review(self, result: ProductReview):
        self.product_review = result

    def set_email_info(self, result: EmailInfo):
        self.email_info = result


# Global results container
results = ExtractionResults()


# ============================================================================
# Tool API - Structured Output Tools
# ============================================================================

def extract_job_listing_tool(
    title: str,
    company: str,
    location: str,
    salary_range: Optional[str] = None,
    required_skills: list[str] = None,
    employment_type: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Extract structured job listing data from unstructured text.

    Args:
        title: The job title
        company: The company name
        location: The job location (city, state/country, or remote)
        salary_range: Salary range if specified
        required_skills: List of required technical skills
        employment_type: Employment type (full-time, part-time, contract, etc.)
        description: Brief description of the role

    Returns:
        Confirmation message with extracted data summary
    """
    if required_skills is None:
        required_skills = []

    listing = JobListing(
        title=title,
        company=company,
        location=location,
        salary_range=salary_range,
        required_skills=required_skills,
        employment_type=employment_type,
        description=description,
    )
    results.set_job_listing(listing)
    return f"Extracted: {title} at {company}"


def extract_product_review_tool(
    product_name: str,
    rating: int,
    pros: list[str] = None,
    cons: list[str] = None,
    summary: str = "",
    would_recommend: Optional[bool] = None,
) -> str:
    """Extract structured product review data from unstructured text.

    Args:
        product_name: The name of the product
        rating: Numeric rating (1-5 stars)
        pros: List of positive aspects mentioned
        cons: List of negative aspects mentioned
        summary: Brief summary of the reviewer's sentiment
        would_recommend: Whether the reviewer would recommend this product

    Returns:
        Confirmation message with extracted data summary
    """
    if pros is None:
        pros = []
    if cons is None:
        cons = []

    review = ProductReview(
        product_name=product_name,
        rating=rating,
        pros=pros,
        cons=cons,
        summary=summary,
        would_recommend=would_recommend,
    )
    results.set_product_review(review)
    return f"Extracted review for {product_name} - {rating}/5"


def extract_email_info_tool(
    sender: str,
    recipients: list[str] = None,
    subject: str = "",
    action_items: list[str] = None,
    urgency: str = "",
    key_points: list[str] = None,
    deadline: Optional[str] = None,
) -> str:
    """Extract structured email information from unstructured text.

    Args:
        sender: Email sender name or address
        recipients: List of email recipients
        subject: Email subject line
        action_items: Action items or tasks mentioned in the email
        urgency: Urgency level (high, medium, low)
        key_points: Key points or information conveyed
        deadline: Any deadline mentioned

    Returns:
        Confirmation message with extracted data summary
    """
    if recipients is None:
        recipients = []
    if action_items is None:
        action_items = []
    if key_points is None:
        key_points = []

    info = EmailInfo(
        sender=sender,
        recipients=recipients,
        subject=subject,
        action_items=action_items,
        urgency=urgency,
        key_points=key_points,
        deadline=deadline,
    )
    results.set_email_info(info)
    return f"Extracted email from {sender} - urgency: {urgency}"


# Create tool definitions
job_listing_tool = StructuredTool.from_function(
    func=extract_job_listing_tool,
    name="extract_job_listing",
    description="Extract structured job listing data from unstructured text",
    args_schema=ExtractJobListingArgs,
)

product_review_tool = StructuredTool.from_function(
    func=extract_product_review_tool,
    name="extract_product_review",
    description="Extract structured product review data from unstructured text",
    args_schema=ExtractProductReviewArgs,
)

email_info_tool = StructuredTool.from_function(
    func=extract_email_info_tool,
    name="extract_email_info",
    description="Extract structured email information from unstructured text",
    args_schema=ExtractEmailInfoArgs,
)


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
# Agent Execution with Tool Calling
# ============================================================================

async def extract_with_tools(
    text: str,
    model: BaseChatModel,
    tools: list,
    system_prompt: str,
    max_turns: int = 5,
) -> bool:
    """Run extraction using tools with multi-turn conversation.

    Args:
        text: Input text to extract from
        model: Chat model to use
        tools: List of tools available to the agent
        system_prompt: System prompt for the agent
        max_turns: Maximum number of conversation turns

    Returns:
        True if extraction was successful, False otherwise
    """
    # Bind tools to model for tool calling support
    model_with_tools = model.bind_tools(tools)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract structured data from this text:\n\n{text}"),
    ]

    for turn in range(max_turns):
        response = await model_with_tools.ainvoke(messages)

        # Check if the model wants to call a tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages.append(response)

            # Execute all tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "")

                # Find and execute the tool
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    try:
                        result = tool.func(**tool_args)
                        messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                    except Exception as e:
                        messages.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_id))
                else:
                    messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found", tool_call_id=tool_id))

            # Continue conversation to get final response
            continue
        else:
            # No tool calls, extraction complete
            return True

    return True


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
        return ChatOpenAI(model=model_id, api_key=api_key, temperature=0)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("✗ ANTHROPIC_API_KEY not set")
            sys.exit(1)
        return ChatAnthropic(model=model_id, api_key=api_key, temperature=0)

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("✗ OPENROUTER_API_KEY not set")
            sys.exit(1)
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )

    else:
        print(f"✗ Unknown provider type: '{provider}'")
        sys.exit(1)


# ============================================================================
# Demo Execution
# ============================================================================

async def run_demo(model: BaseChatModel, provider_key: str, model_id: str) -> tuple[int, int]:
    """Run the structured extraction demo using Tool API."""
    print("=== Python — Structured Output & Data Extraction (Tool API) ===")
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

    results.job_listing = None  # Reset
    job_system_prompt = """You are an expert data extractor. Extract structured job listing data from the provided text.

IMPORTANT: You MUST use the extract_job_listing tool to submit your extraction.

Fields to extract:
- title: The job title
- company: The company name
- location: The job location
- salary_range: Salary range if specified
- required_skills: List of required skills
- employment_type: Employment type if specified
- description: Brief description of the role"""

    try:
        success = await extract_with_tools(
            SAMPLE_JOB_LISTING,
            model,
            [job_listing_tool],
            job_system_prompt,
            max_turns=5,
        )
        if success and results.job_listing:
            print("\nExtracted Data:")
            print(results.job_listing.model_dump_json(indent=2))
            print(f"\nValidation: ✓ All required fields present")
            success_count += 1
            time.sleep(2)  # Rate limiting
        else:
            print("\n✗ Error: Tool was not called")
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

    results.product_review = None  # Reset
    review_system_prompt = """You are an expert data extractor. Extract structured product review data from the provided text.

IMPORTANT: You MUST use the extract_product_review tool to submit your extraction.

Fields to extract:
- product_name: The name of the product
- rating: Numeric rating (1-5 stars)
- pros: List of positive aspects mentioned
- cons: List of negative aspects mentioned
- summary: Brief summary of the reviewer's sentiment
- would_recommend: Whether reviewer recommends the product (if mentioned)"""

    try:
        success = await extract_with_tools(
            SAMPLE_PRODUCT_REVIEW,
            model,
            [product_review_tool],
            review_system_prompt,
            max_turns=5,
        )
        if success and results.product_review:
            print("\nExtracted Data:")
            print(results.product_review.model_dump_json(indent=2))
            print(f"\nValidation: ✓ All required fields present")
            success_count += 1
            time.sleep(2)  # Rate limiting
        else:
            print("\n✗ Error: Tool was not called")
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

    results.email_info = None  # Reset
    email_system_prompt = """You are an expert data extractor. Extract structured email information from the provided text.

IMPORTANT: You MUST use the extract_email_info tool to submit your extraction.

Fields to extract:
- sender: Email sender
- recipients: List of email recipients
- subject: Email subject line
- action_items: Action items or tasks mentioned (as strings)
- urgency: Urgency level (high, medium, low)
- key_points: Key points from the email
- deadline: Deadline if mentioned"""

    try:
        success = await extract_with_tools(
            SAMPLE_EMAIL,
            model,
            [email_info_tool],
            email_system_prompt,
            max_turns=5,
        )
        if success and results.email_info:
            print("\nExtracted Data:")
            print(results.email_info.model_dump_json(indent=2))
            print(f"\nValidation: ✓ All required fields present")
            success_count += 1
        else:
            print("\n✗ Error: Tool was not called")
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

async def main_async():
    load_dotenv()

    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider_key not in PROVIDERS:
        print(f"✗ Unknown provider: '{provider_key}'")
        print(f"  Supported: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    model_id, provider_type = PROVIDERS[provider_key]
    model = create_chat_model(provider_type, model_id)

    await run_demo(model, provider_key, model_id)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
