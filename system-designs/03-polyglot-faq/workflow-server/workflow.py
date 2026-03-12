"""LangGraph workflow implementation for FAQ processing.

This module implements a 4-node workflow:
1. expand_query: Generate 3 search query variants
2. search_documents: Query the Rust MCP document search server
3. agent_with_tools: Agent with all 5 tools bound (search, list, get, related, date)
4. generate_response: Create a comprehensive answer based on results

Uses langchain-mcp-adapters for proper MCP client communication.
Uses bind_tools() pattern for agentic tool selection.
"""

import asyncio
import json
import os
from typing import Annotated, Any, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph
from operator import add

load_dotenv()

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
    timeout = 30.0

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(model=model_id, api_key=api_key, timeout=timeout)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return ChatAnthropic(model=model_id, api_key=api_key, timeout=timeout)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_llm() -> BaseChatModel:
    """Get LLM instance based on environment configuration."""
    provider_key = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider_key not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_key}")
    model_id, provider_type = PROVIDERS[provider_key]
    return create_chat_model(provider_type, model_id)


# ============================================================================
# State Definition
# ============================================================================

class FAQState(TypedDict):
    """State for FAQ workflow."""

    messages: Annotated[Sequence[BaseMessage], add]
    question: str
    query_variants: list[str]
    search_results: list[dict]
    final_answer: str | None
    steps: list[dict]
    llm: BaseChatModel
    search_tools: list[StructuredTool]  # All MCP tools (5 total)
    agent_tools: list[StructuredTool]  # Filtered: search, get_document, find_related, get_current_date
    tool_results: list[dict]  # Track agent tool call results


# ============================================================================
# Workflow Nodes
# ============================================================================

EXPAND_QUERY_PROMPT = """You are a query expansion specialist. Given a user's question, generate 3 different search query variants that will help find relevant FAQ documents.

Generate 3 query variants that:
- Use different wording and phrasing
- Focus on different aspects of the question
- Include relevant synonyms and related terms

User question: {question}

Return ONLY a JSON array of 3 query strings, like:
["query variant 1", "query variant 2", "query variant 3"]"""

RESPONSE_GENERATION_PROMPT = """You are a helpful customer support assistant. Based on the user's question and the retrieved FAQ documents, provide a comprehensive and helpful answer.

User Question: {question}

Retrieved Documents:
{documents}

Guidelines:
- Answer the question directly and clearly
- Use information from the retrieved documents
- Cite specific document titles you're referencing
- If documents don't fully answer, acknowledge limitations
- Be friendly and professional
- Use bullet points for multi-step instructions
- Keep responses concise but complete

Provide your answer:"""


async def expand_query_node(state: FAQState) -> dict:
    """Expand input query into 3 search variants."""
    llm = state.get("llm")

    prompt = EXPAND_QUERY_PROMPT.format(question=state["question"])
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # Parse the response to extract JSON array
    content = response.content
    try:
        # Try to parse JSON directly
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()

        variants = json.loads(content)
        if isinstance(variants, list):
            query_variants = [str(v) for v in variants[:3]]
        else:
            query_variants = [content]
    except json.JSONDecodeError:
        # Fallback: split by newlines and clean up
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        query_variants = lines[:3]

    # If all else fails, use original query
    if not query_variants:
        query_variants = [state["question"]]

    return {
        "query_variants": query_variants,
        "steps": state.get("steps", []) + [{
            "node": "expand_query",
            "result": f"Generated {len(query_variants)} query variants"
        }]
    }


def parse_mcp_search_result(result: Any) -> list[dict]:
    """Parse MCP search result into list of documents.

    Handles multiple result formats from langchain-mcp-adapters:
    - List of content blocks with JSON text
    - Raw JSON string
    - LangChain Document objects
    - Direct dict items
    """
    docs = []

    if isinstance(result, list) and len(result) > 0:
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                # Parse the JSON string from the 'text' field
                text_content = item.get("text", "")
                if text_content:
                    try:
                        parsed = json.loads(text_content)
                        docs.extend(parsed if isinstance(parsed, list) else [parsed])
                    except json.JSONDecodeError:
                        docs.append({"content": text_content})
            elif isinstance(item, dict):
                # Direct dict with document fields
                docs.append(item)
    elif isinstance(result, str):
        # Raw JSON string
        docs = json.loads(result)
    elif hasattr(result, "page_content"):
        # LangChain Document object
        content = result.page_content
        if content.startswith(("[", "{")):
            docs = json.loads(content)
        else:
            docs = [{"content": content}]

    return docs


async def search_documents_node(state: FAQState) -> dict:
    """Search FAQ documents using parallel queries.

    OPTIMIZATION: All query variants searched in parallel using asyncio.gather
    to reduce total search time from 3-6s to 1-2s.
    """
    search_tools = state.get("search_tools", [])
    all_results = []
    seen_ids = set()

    # Get the search_documents tool
    search_tool = next((t for t in search_tools if t.name == "search_documents"), None)

    if not search_tool:
        print("Warning: search_documents tool not found")
        return {
            "search_results": [],
            "steps": state.get("steps", []) + [{
                "node": "search_documents",
                "result": "No search tool available"
            }]
        }

    # OPTIMIZATION: Create search tasks for all query variants and execute in parallel
    search_tasks = [
        search_tool.ainvoke({"query": query, "top_n": 3})
        for query in state["query_variants"]
    ]

    # Execute all searches in parallel
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Process results with deduplication using helper function
    for result in results:
        if isinstance(result, Exception):
            print(f"Search error: {result}")
            continue

        # Parse result using helper function
        docs = parse_mcp_search_result(result)

        # Deduplicate documents by ID
        for doc in docs:
            if isinstance(doc, dict):
                doc_id = doc.get("id")
                if not doc_id:
                    # Generate an ID if none exists
                    doc_id = f"doc_{len(all_results)}_{hash(str(doc))}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)
                    print(f"Found document: {doc.get('title', doc_id)}")

    return {
        "search_results": all_results,
        "steps": state.get("steps", []) + [{
            "node": "search_documents",
            "result": f"Found {len(all_results)} relevant documents"
        }]
    }


async def generate_response_node(state: FAQState) -> dict:
    """Generate final response based on search results.

    OPTIMIZATION: Reduced document context from 5 docs to 3 docs, 500 chars to 250 chars
    to reduce LLM processing time.
    """
    llm = state.get("llm")

    if not state["search_results"] and not state.get("tool_results"):
        final_answer = ("I couldn't find specific information about your question in our FAQ. "
                       "Please try rephrasing your question or contact our support team at help@example.com "
                       "for personalized assistance.")
    else:
        # Format documents for the prompt - OPTIMIZED: reduced from 5 to 3 docs, 500 to 250 chars
        docs_text = "\n\n".join([
            f"Document: {doc.get('title', doc.get('id', 'Unknown'))}\n{doc.get('content', '')[:250]}"
            for doc in state["search_results"][:3]  # Changed from [:5]
        ])

        prompt = RESPONSE_GENERATION_PROMPT.format(
            question=state["question"],
            documents=docs_text
        )

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        final_answer = response.content

    return {
        "final_answer": final_answer,
        "steps": state.get("steps", []) + [{
            "node": "generate_response",
            "result": "Generated final response"
        }]
    }


def build_search_context(search_results: list[dict]) -> str:
    """Build concise context for LLM - only top 3 docs, 150 chars each.

    OPTIMIZATION: Reduced from 5 docs to 3 docs, 300 chars to 150 chars
    to reduce LLM processing time by 30-50%.
    """
    if not search_results:
        return "No initial documents found."

    context = []
    for doc in search_results[:3]:  # Reduced from 5 to 3
        title = doc.get('title', doc.get('id', 'Unknown'))
        # Just title + brief preview, not full content
        content = doc.get('content', '')[:150]  # Reduced from 300
        context.append(f"- {title}: {content}...")

    return "Top documents:\n" + "\n".join(context)


async def agent_with_tools_node(state: FAQState) -> dict:
    """Agent node with all tools - LLM chooses what to use (can search again if needed).

    This node receives search results as context and has all 5 tools bound:
    - search_documents: Can search again with different queries
    - list_documents: Browse all available documents
    - get_document: Get full content by ID
    - find_related_documents: Find related for "see also"
    - get_current_date: Time awareness
    """
    llm = state["llm"]
    agent_tools = state.get("agent_tools", [])

    if not agent_tools:
        print("Warning: No agent tools available")
        return {
            "messages": [],
            "tool_results": [],
            "steps": state.get("steps", []) + [{
                "node": "agent_with_tools",
                "result": "No tools available"
            }]
        }

    # Build context message with search results
    search_context = build_search_context(state.get("search_results", []))

    # Bind all agent tools to LLM
    llm_with_tools = llm.bind_tools(agent_tools)

    # Get previous messages if any
    messages = list(state.get("messages", []))

    # Add our query message if no messages yet
    if not messages:
        messages.append(HumanMessage(content=f"""User question: {state['question']}

{search_context}

You have access to these tools:
- search_documents: Search for FAQ documents
- list_documents: Browse all available documents
- get_document: Get full content of a specific document
- find_related_documents: Find documents related to given IDs
- get_current_date: Get current date and time

Use the tools to provide a comprehensive answer. You can:
1. Search again with different queries if needed
2. Get full content of specific documents
3. Find related documents for additional context
4. Get current date for time-sensitive questions"""))

    response = await llm_with_tools.ainvoke(messages)

    # Log tool choices for debugging
    if response.tool_calls:
        tool_names = [tc.get("name", "unknown") for tc in response.tool_calls]
        print(f"LLM chose tools: {tool_names}")

    return {
        "messages": [response],
        "tool_results": [],
        "steps": state.get("steps", []) + [{
            "node": "agent_with_tools",
            "result": f"Agent response with {len(response.tool_calls) if response.tool_calls else 0} tool calls"
        }]
    }


async def tool_executor_node(state: FAQState) -> dict:
    """Execute tool calls from the last AI message."""
    agent_tools = state.get("agent_tools", [])
    messages = state.get("messages", [])

    if not messages:
        return {
            "messages": [],
            "tool_results": []
        }

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {
            "messages": [],
            "tool_results": []
        }

    # Execute tool calls
    tool_results = []
    tool_messages = []

    # Create a tool map for quick lookup
    tool_map = {tool.name: tool for tool in agent_tools}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")

        if tool_name not in tool_map:
            print(f"Warning: Tool {tool_name} not found")
            tool_messages.append(ToolMessage(
                content=f"Error: Tool {tool_name} not found",
                tool_call_id=tool_id
            ))
            continue

        try:
            tool = tool_map[tool_name]
            result = await tool.ainvoke(tool_args)

            # Handle different result formats
            if isinstance(result, list):
                result_text = str(result)
            elif isinstance(result, dict):
                result_text = str(result)
            else:
                result_text = str(result)

            tool_messages.append(ToolMessage(
                content=result_text,
                tool_call_id=tool_id
            ))

            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result_text
            })

            print(f"Executed tool: {tool_name} with args: {tool_args}")

        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_id
            ))

    return {
        "messages": tool_messages,
        "tool_results": tool_results,
        "steps": state.get("steps", []) + [{
            "node": "tool_executor",
            "result": f"Executed {len(tool_messages)} tool calls"
        }]
    }


def should_use_agent_tools(state: FAQState) -> str:
    """Check if we should use agent tools for enrichment.

    OPTIMIZATION: Skip agent_with_tools entirely when we already have comprehensive
    results (3+ documents) to avoid 5-10s of unnecessary LLM execution.

    This check happens AFTER search_documents and BEFORE agent_with_tools,
    allowing us to bypass the agent entirely when results are sufficient.

    Returns "agent" to use agent tools, "generate" to skip to response generation.
    """
    # OPTIMIZATION: Check if we have enough search results to skip agent entirely
    search_results = state.get("search_results", [])
    if len(search_results) >= 3:
        # We have 3+ documents - likely sufficient, skip agent entirely
        print(f"[OPTIMIZATION] Skipping agent tools - have {len(search_results)} good results")
        return "generate"

    return "agent"  # Need more info, use agent tools


def should_continue_to_tools(state: FAQState) -> str:
    """Check if last message has tool calls.

    Returns "tools" if the last message has tool calls, "continue" otherwise.
    """
    messages = state.get("messages", [])
    if not messages:
        return "continue"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "continue"


# ============================================================================
# Graph Construction
# ============================================================================

def create_faq_graph():
    """Create the FAQ workflow graph using LangGraph."""
    workflow = StateGraph(FAQState)

    # Add nodes
    workflow.add_node("expand_query", expand_query_node)
    workflow.add_node("search_documents", search_documents_node)
    workflow.add_node("agent_with_tools", agent_with_tools_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("generate_response", generate_response_node)

    # Set entry point
    workflow.set_entry_point("expand_query")

    # Linear flow for initial part
    workflow.add_edge("expand_query", "search_documents")

    # OPTIMIZATION: Conditional routing after search - skip agent if we have enough results
    workflow.add_conditional_edges(
        "search_documents",
        should_use_agent_tools,
        {
            "agent": "agent_with_tools",
            "generate": "generate_response"
        }
    )

    # Conditional routing for enrichment tools
    workflow.add_conditional_edges(
        "agent_with_tools",
        should_continue_to_tools,
        {
            "tools": "tool_executor",
            "continue": "generate_response"
        }
    )
    workflow.add_edge("tool_executor", "agent_with_tools")  # Loop for more enrichment
    workflow.add_edge("generate_response", END)

    return workflow.compile()


# ============================================================================
# Workflow Execution
# ============================================================================

async def load_mcp_search_tools(doc_search_url: str) -> list[StructuredTool]:
    """Load MCP tools from the document search server using langchain-mcp-adapters.

    Uses HTTP transport to connect to the Rust MCP server.
    """
    # Ensure URL ends with / for langchain-mcp-adapters
    url = doc_search_url if doc_search_url.endswith("/") else f"{doc_search_url}/"

    # Use MultiServerMCPClient to connect to the MCP server
    try:
        client = MultiServerMCPClient({
            "doc-search": {
                "url": url,
                "transport": "http",
            }
        })
        tools = await client.get_tools()
        print(f"Loaded {len(tools)} tools from MCP server at {url}")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        return tools
    except Exception as e:
        print(f"Failed to load MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return []


async def run_faq_workflow(question: str, doc_search_url: str, stream: bool = False) -> dict:
    """Run the FAQ workflow and return results."""
    graph = create_faq_graph()

    # Initialize LLM
    llm = get_llm()
    print(f"LLM initialized: {type(llm)}")

    # Load MCP tools from document search server
    all_tools = await load_mcp_search_tools(doc_search_url)

    # Filter tools for agent (all 5 tools available)
    agent_tools = [t for t in all_tools if t.name in [
        "search_documents",        # Can search again
        "list_documents",          # Browse all docs
        "get_document",            # Get full content
        "find_related_documents",  # Find "see also"
        "get_current_date"         # Time awareness
    ]]

    print(f"Agent tools: {[t.name for t in agent_tools]}")

    # Initial state with configuration embedded
    initial_state = FAQState({
        "messages": [],
        "question": question,
        "query_variants": [],
        "search_results": [],
        "final_answer": None,
        "steps": [],
        "llm": llm,
        "search_tools": all_tools,  # For search_documents node
        "agent_tools": agent_tools,  # For agent_with_tools node
        "tool_results": []
    })

    if stream:
        # Return streaming results
        result = {"steps": [], "final_answer": None, "question": question}

        async for event in graph.astream(initial_state):
            for node_name, node_output in event.items():
                step = {
                    "node": node_name,
                    "output": dict(node_output)
                }
                result["steps"].append(step)

                if "final_answer" in node_output and node_output["final_answer"]:
                    result["final_answer"] = node_output["final_answer"]

        return result
    else:
        # Run and return final result
        final_state = await graph.ainvoke(initial_state)

        return {
            "question": question,
            "steps": final_state.get("steps", []),
            "query_variants": final_state.get("query_variants", []),
            "search_results": final_state.get("search_results", []),
            "final_answer": final_state.get("final_answer", "")
        }


# ============================================================================
# Standalone Test
# ============================================================================

async def main():
    """Test the workflow standalone."""
    print("=== FAQ Workflow Test ===")
    print()

    test_question = "How do I reset my password?"
    print(f"Question: {test_question}")
    print()

    result = await run_faq_workflow(
        test_question,
        "http://localhost:8004/mcp",
        stream=True
    )

    print("Steps:")
    for step in result.get("steps", []):
        print(f"  - {step.get('node', 'unknown')}: {step.get('output', {})}")

    print()
    print(f"Final Answer:\n{result.get('final_answer', 'No answer generated')}")


if __name__ == "__main__":
    asyncio.run(main())
