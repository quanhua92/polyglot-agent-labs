"""
Polyglot Agent Labs — Use Case 10: Customer Support Agent
A complete customer support pipeline with intent classification, KB retrieval,
response generation, and escalation logic with multi-turn conversation support.
Switch provider with env var LLM_PROVIDER (default: openrouter).
Switch embedding provider with env var EMBEDDING_PROVIDER (default: openrouter).

Usage:
  python main.py --interactive     # Interactive REPL mode
  python main.py --demo           # Demo mode with predefined scenarios

Commands:
  /quit, /exit, /q    Exit the session (interactive mode only)
"""

import json
import os
import re
import sys
from typing import Annotated, Literal, Sequence

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class IntentClassification(BaseModel):
    """Structured output from intent classification."""
    intent: str = Field(
        description="Classified intent: billing, shipping, returns, account, general, escalate"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities like order_id, email, product_name"
    )
    urgency: str = Field(
        description="Urgency level: low, medium, high",
        default="medium"
    )


class SupportResponse(BaseModel):
    """Structured output from response generator."""
    response: str = Field(description="The helpful response to the customer")
    sources: list[str] = Field(default_factory=list, description="KB articles cited")
    escalation_reason: str | None = Field(default=None, description="Reason if escalating")
    should_escalate: bool = Field(default=False, description="Whether to escalate to human")


# ============================================================================
# Hard-coded Knowledge Base
# ============================================================================

KNOWLEDGE_BASE = [
    {
        "id": "return-policy",
        "title": "Return Policy",
        "content": """We offer a 30-day return policy for all unused items in original packaging.

To return an item:
1. Log into your account and go to Order History
2. Select the item you wish to return
3. Print the prepaid return shipping label
4. Pack the item securely in original packaging
5. Drop off at any authorized shipping location

Refunds are processed within 5-7 business days after we receive the return. The refund will be credited to the original payment method.

Items that cannot be returned:
- Personalized or custom items
- Perishable goods
- Gift cards
- Items marked as "Final Sale"

Exchanges are available for different sizes of the same product."""
    },
    {
        "id": "shipping-info",
        "title": "Shipping Information",
        "content": """We offer several shipping options:

Standard Shipping (5-7 business days): $4.99
Express Shipping (2-3 business days): $9.99
Overnight Shipping (1 business day): $19.99

FREE standard shipping on orders over $50!

Orders placed before 2 PM EST ship the same day. Orders placed after 2 PM EST ship the next business day.

We ship to all 50 US states. International shipping is not currently available.

Track your order:
- Use the tracking number in your shipping confirmation email
- Or log into your account and view Order Status

Delivery delays may occur during holidays and severe weather. We'll notify you of any significant delays."""
    },
    {
        "id": "billing-payments",
        "title": "Billing and Payments",
        "content": """We accept the following payment methods:
- Credit/Debit Cards (Visa, MasterCard, American Express, Discover)
- PayPal
- Apple Pay
- Google Pay

Payment is charged at the time of order confirmation. A detailed invoice is emailed to you immediately.

To request a duplicate invoice:
- Log into your account
- Go to Order History
- Click "Download Invoice" next to the order

For billing questions:
- Email: billing@example.com
- Phone: 1-800-BILLING
- Hours: Mon-Fri 9 AM - 6 PM EST

If a payment fails:
- Check your card details are correct
- Ensure sufficient funds
- Try an alternative payment method
- Contact your bank if the issue persists"""
    },
    {
        "id": "account-management",
        "title": "Account Management",
        "content": """Managing Your Account:

Password Reset:
1. Click "Forgot Password" on the login page
2. Enter your email address
3. Check your email for reset link (expires in 24 hours)
4. Create a new password

Profile Updates:
- Go to Account Settings
- Update your email, phone, shipping address
- Changes save automatically

Order History:
- View all past orders
- Track current orders
- Reorder previous purchases
- Download invoices

Account Deletion:
- Go to Account Settings
- Scroll to "Delete Account"
- Confirm deletion
- All data will be permanently removed within 30 days

For account issues not resolved here, contact customer support."""
    },
    {
        "id": "order-status",
        "title": "Order Status and Tracking",
        "content": """Track Your Order:

1. Log into your account
2. Go to Order History
3. Click on the order number
4. View real-time tracking information

Order Status Meanings:
- Processing: We're preparing your order
- Shipped: Order is on its way
- Delivered: Order has been delivered
- Delayed: Shipping delay has occurred
- Cancelled: Order was cancelled

Delivery Times:
- Standard: 5-7 business days
- Express: 2-3 business days
- Overnight: 1 business day

Cancellations:
- Cancel within 1 hour of placing the order
- Go to Order History and click "Cancel"
- After 1 hour, contact customer support

Missing Orders:
- Check tracking for delivery status
- Verify shipping address is correct
- Allow extra time for standard shipping
- Contact support if not received within 10 business days"""
    },
    {
        "id": "contact-info",
        "title": "Contact Customer Support",
        "content": """How to Reach Us:

Phone Support:
- 1-800-SUPPORT
- Mon-Fri: 8 AM - 8 PM EST
- Sat-Sun: 10 AM - 6 PM EST

Email Support:
- support@example.com
- Response within 24 hours

Live Chat:
- Available on our website
- Mon-Fri: 9 AM - 9 PM EST

Social Media:
- Twitter: @SupportTeam
- Facebook: /SupportTeam

For fastest response, use live chat during business hours.

When contacting us, please have:
- Your order number (if applicable)
- Your account email
- A brief description of your issue

Urgent issues (cancellation, wrong item): Call us directly.

General inquiries: Email or live chat is preferred."""
    },
]


# ============================================================================
# Provider Configuration
# ============================================================================

PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}

EMBEDDING_PROVIDERS = {
    "openrouter": {
        "model": "text-embedding-3-small",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY"
    },
    "openai": {
        "model": "text-embedding-3-small",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY"
    },
}


# ============================================================================
# System Prompts
# ============================================================================

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a customer support system.

Analyze the customer's message and classify it into one of these intents:
- billing: Questions about charges, invoices, payments, refunds
- shipping: Questions about delivery, tracking, shipping options
- returns: Questions about return policy, exchanges, refunds
- account: Questions about login, password, profile, settings
- general: General inquiries, product info, company info
- escalate: Explicit request to speak to a human/manager

Extract relevant entities (order_id, email, product_name, etc.)

Return structured output with intent, confidence (0.0-1.0), entities, and urgency (low/medium/high)."""

RESPONSE_GENERATOR_PROMPT = """You are a helpful customer support agent.

Your task is to provide a friendly, helpful response using the knowledge base context provided.

Guidelines:
- Be warm and empathetic
- Provide specific, actionable information
- Cite the source articles you're using
- If you're unsure, say so and offer to connect them with a human
- Keep responses concise but complete
- Use formatting (bullet points, numbered lists) for readability"""

ESCALATION_CONFIDENCE_THRESHOLD = 0.6


# ============================================================================
# LLM Model Creation
# ============================================================================

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
# Embeddings Creation
# ============================================================================

def create_embeddings(provider: str = "openrouter") -> OpenAIEmbeddings:
    """Create embeddings with provider support."""
    config = EMBEDDING_PROVIDERS.get(provider, EMBEDDING_PROVIDERS["openrouter"])
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        print(f"✗ {config['api_key_env']} not set")
        sys.exit(1)

    return OpenAIEmbeddings(
        model=config["model"],
        openai_api_key=api_key,
        openai_api_base=config.get("api_base")
    )


# ============================================================================
# Vector Store Creation
# ============================================================================

def create_vector_store(embeddings):
    """Create and populate FAISS vector store from knowledge base."""
    docs = [
        Document(
            page_content=doc["content"],
            metadata={"source": doc["id"], "title": doc["title"]}
        )
        for doc in KNOWLEDGE_BASE
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)


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
# LangGraph State and Nodes
# ============================================================================

class SupportState(dict):
    """State passed between support agent nodes."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
    intent: str | None
    confidence: float | None
    entities: dict
    kb_context: str
    kb_sources: list[str]
    response: str | None
    should_escalate: bool
    escalation_reason: str | None
    turn_count: int


def intent_classifier_node(state: SupportState, config):
    """Classify user intent using structured output."""
    user_input = state["user_input"]
    model = config["configurable"]["model"]

    prompt = f"""{INTENT_CLASSIFIER_PROMPT}

Customer message: {user_input}

Return valid JSON with this exact format:
{{
  "intent": "category",
  "confidence": 0.95,
  "entities": {{"key": "value"}},
  "urgency": "medium"
}}

Where intent is one of: billing, shipping, returns, account, general, escalate"""

    response = model.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    json_str = extract_json_from_response(content)

    try:
        parsed = json.loads(json_str)
        return {
            "intent": parsed.get("intent", "general"),
            "confidence": float(parsed.get("confidence", 0.8)),
            "entities": parsed.get("entities", {}),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback if JSON parsing fails
        return {
            "intent": "general",
            "confidence": 0.5,
            "entities": {},
        }


def kb_retriever_node(state: SupportState, config):
    """Retrieve relevant KB articles using RAG."""
    user_input = state["user_input"]
    vector_store = config["configurable"]["vector_store"]

    # Retrieve top 2 relevant chunks
    retrieved_docs = vector_store.similarity_search(user_input, k=2)

    # Format context with sources
    kb_context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('title', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    kb_sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]

    return {
        "kb_context": kb_context,
        "kb_sources": kb_sources,
    }


def response_generator_node(state: SupportState, config):
    """Generate helpful response using KB context."""
    user_input = state["user_input"]
    kb_context = state["kb_context"]
    model = config["configurable"]["model"]

    prompt = f"""{RESPONSE_GENERATOR_PROMPT}

Customer question: {user_input}

Relevant information from our knowledge base:
{kb_context}

Provide a helpful response. Include citations to the source articles."""

    response = model.invoke(prompt)

    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def escalation_check_node(state: SupportState):
    """Check if escalation is needed based on confidence or explicit request."""
    intent = state.get("intent", "")
    confidence = state.get("confidence", 1.0)

    # Escalate if low confidence or explicit request
    if confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        return {
            "should_escalate": True,
            "escalation_reason": f"Low confidence score ({confidence:.2f} < {ESCALATION_CONFIDENCE_THRESHOLD})",
        }
    elif intent == "escalate":
        return {
            "should_escalate": True,
            "escalation_reason": "Customer explicitly requested human assistance",
        }

    return {
        "should_escalate": False,
        "escalation_reason": None,
    }


def escalate_node(state: SupportState):
    """Handle escalation to human agent."""
    escalation_reason = state.get("escalation_reason", "Unknown")

    escalation_message = f"""🚨 Transferring to human agent...

Reason for escalation: {escalation_reason}

A human agent will be with you shortly. Your conversation has been logged and the agent will have full context."""

    return {
        "response": escalation_message,
        "messages": [AIMessage(content=escalation_message)],
    }


def should_escalate(state: SupportState) -> Literal["escalate", "retrieve_kb"]:
    """Decide whether to escalate or continue with KB retrieval."""
    if state.get("should_escalate", False):
        return "escalate"
    return "retrieve_kb"


def build_support_graph(model, vector_store):
    """Build the customer support agent graph."""
    workflow = StateGraph(SupportState)

    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("escalation_check", escalation_check_node)
    workflow.add_node("kb_retriever", kb_retriever_node)
    workflow.add_node("response_generator", response_generator_node)
    workflow.add_node("escalate", escalate_node)

    # Set entry point
    workflow.set_entry_point("intent_classifier")

    # Intent classifier → escalation check
    workflow.add_edge("intent_classifier", "escalation_check")

    # Escalation check → (escalate OR retrieve KB)
    workflow.add_conditional_edges(
        "escalation_check",
        should_escalate,
        {"escalate": "escalate", "retrieve_kb": "kb_retriever"}
    )

    # KB retriever → response generator
    workflow.add_edge("kb_retriever", "response_generator")

    # Both paths → end
    workflow.add_edge("response_generator", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


# ============================================================================
# Demo Scenarios
# ============================================================================

DEMO_SCENARIOS = [
    {
        "name": "Return Policy Inquiry",
        "input": "How do I return a product I bought last week?",
        "expected_intent": "returns",
    },
    {
        "name": "Explicit Escalation",
        "input": "I want to speak to a manager!",
        "expected_intent": "escalate",
    },
    {
        "name": "Order Status",
        "input": "What's the status of order #12345?",
        "expected_intent": "general",
    },
]


# ============================================================================
# Interactive REPL
# ============================================================================

def run_interactive_repl(model, vector_store, provider_key: str, model_id: str):
    """Run interactive REPL for customer support."""
    print("=== Python — Customer Support Agent ===")
    print(f"Provider: {provider_key}")
    print(f"Model: {model_id}")
    print("Mode: interactive")
    print()
    print("Type your message to chat with the support agent.")
    print("Commands: /quit, /exit, /q to quit")
    print("-" * 50)
    print()

    graph = build_support_graph(model, vector_store)
    config = {"configurable": {"model": model, "vector_store": vector_store}}

    # Initialize state
    state: dict = {
        "messages": [],
        "turn_count": 0,
    }

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

        # Update state with user input
        state["user_input"] = user_input
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]

        # Initialize default values
        state.setdefault("should_escalate", False)
        state.setdefault("escalation_reason", None)
        state.setdefault("kb_context", "")
        state.setdefault("kb_sources", [])
        state.setdefault("response", None)
        state.setdefault("intent", None)
        state.setdefault("confidence", None)
        state.setdefault("entities", {})

        # Run the graph
        try:
            result = graph.invoke(state, config)

            # Update state from result
            state.update(result)
            state["turn_count"] += 1

            # Print results
            print(f"\n[Intent: {result.get('intent', 'unknown').upper()} | Confidence: {result.get('confidence', 0):.2f}]")
            print(f"\nAgent: {result.get('response', 'I apologize, but I encountered an error.')}")

            # Show KB sources if used
            if result.get("kb_sources"):
                print(f"\n[Sources: {', '.join(result['kb_sources'])}]")

            print()
        except Exception as e:
            print(f"\n✗ Error: {e}\n")
            # Remove the failed user message from history
            state["messages"] = state["messages"][:-1]

    # Session summary
    print("-" * 50)
    print("Session Summary")
    print(f"Total turns: {state['turn_count']}")


# ============================================================================
# Demo Mode
# ============================================================================

def run_demo_scenarios(model, vector_store, provider_key: str, model_id: str):
    """Run pre-defined demo scenarios."""
    print("=== Python — Customer Support Agent ===")
    print(f"Provider: {provider_key}")
    print(f"Model: {model_id}")
    print("Mode: demo")
    print("-" * 50)
    print()

    graph = build_support_graph(model, vector_store)
    config = {"configurable": {"model": model, "vector_store": vector_store}}

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"[{i}/{len(DEMO_SCENARIOS)}] Scenario: {scenario['name']}")
        print(f"Input: {scenario['input']}")
        print("-" * 50)

        # Initialize state
        state = {
            "messages": [],
            "user_input": scenario['input'],
            "turn_count": 0,
            "should_escalate": False,
            "escalation_reason": None,
            "kb_context": "",
            "kb_sources": [],
            "response": None,
            "intent": None,
            "confidence": None,
            "entities": {},
        }

        # Run the graph
        try:
            result = graph.invoke(state, config)

            # Print results
            print(f"Intent: {result.get('intent', 'unknown').upper()}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"\nResponse:\n{result.get('response', 'No response')}")

            if result.get("escalation_reason"):
                print(f"\n⚠️ Escalated: {result['escalation_reason']}")

            if result.get("kb_sources"):
                print(f"\nKB Sources: {', '.join(result['kb_sources'])}")
        except Exception as e:
            print(f"✗ Error: {e}")

        print("=" * 50)
        print()

    # Session summary
    print("Session Summary")
    print(f"Scenarios processed: {len(DEMO_SCENARIOS)}")


# ============================================================================
# Main
# ============================================================================

def main():
    load_dotenv()

    # Check for mode flags
    interactive_mode = "--interactive" in sys.argv
    demo_mode = "--demo" in sys.argv

    if not interactive_mode and not demo_mode:
        print("Usage:")
        print("  python main.py --interactive     # Interactive REPL mode")
        print("  python main.py --demo           # Demo mode with predefined scenarios")
        sys.exit(1)

    # Get provider settings
    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openrouter").lower()

    if provider_key not in PROVIDERS:
        print(f"✗ Unknown LLM provider: '{provider_key}'")
        print(f"  Supported: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    if embedding_provider not in EMBEDDING_PROVIDERS:
        print(f"✗ Unknown embedding provider: '{embedding_provider}'")
        print(f"  Supported: {', '.join(EMBEDDING_PROVIDERS.keys())}")
        sys.exit(1)

    # Get model configuration
    model_id, provider_type = PROVIDERS[provider_key]

    # Create LLM and embeddings
    model = create_chat_model(provider_type, model_id)
    embeddings = create_embeddings(embedding_provider)
    vector_store = create_vector_store(embeddings)

    # Run in requested mode
    if interactive_mode:
        run_interactive_repl(model, vector_store, provider_key, model_id)
    else:
        run_demo_scenarios(model, vector_store, provider_key, model_id)


if __name__ == "__main__":
    main()
