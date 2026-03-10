"""
Polyglot Agent Labs — Use Case 14: Content Writing Agent (Blog Generator)
A multi-stage content generation pipeline with quality control loops using Tool API.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

Key Learning Goals:
- Tool API for structured output - tools accept typed parameters from LLM
- Multi-stage pipelines - chaining specialized agents for content creation
- Prompt chaining - passing outputs between stages as context
- Quality control loops - iterative improvement with scoring thresholds
- Long-form generation - handling article-length content

Pipeline Stages:
1. Outliner → Generate structured article outline
2. Researcher → Gather talking points for each section
3. Drafter → Write full prose for each section
4. Editor → Score quality 1-10, provide feedback
5. Quality Gate → Score >= 7 proceeds, < 7 triggers revision (max 2)
6. Finalizer → Format and save to output.md
"""

import asyncio
import json
import os
import sys
from typing import Annotated, Literal, Sequence

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

PROVIDERS = {
    "openai": ("gpt-4.1-nano", "openai"),
    "anthropic": ("claude-3-haiku-20240307", "anthropic"),
    "openrouter": ("stepfun/step-3.5-flash:free", "openrouter"),
}

DEMO_TOPIC = "Why developers should learn both Python and Rust"
QUALITY_THRESHOLD = 7
MAX_REVISIONS = 2


# ============================================================================
# Pydantic Models for Tool Arguments (Input Schema)
# ============================================================================


class ArticleOutlineArgs(BaseModel):
    """Arguments for submitting article outline."""
    title: str = Field(description="Article title")
    sections: list[dict] = Field(description="List of sections with heading and key_points")


class SectionResearchArgs(BaseModel):
    """Arguments for submitting section research."""
    sections: list[dict] = Field(description="List of research for each section")


class EditorReviewArgs(BaseModel):
    """Arguments for submitting editor review."""
    score: int = Field(description="Quality score 1-10", ge=1, le=10)
    feedback: str = Field(description="Specific feedback for improvement")
    grammar_issues: list[str] = Field(default_factory=list, description="Grammar issues found")
    coherence_issues: list[str] = Field(default_factory=list, description="Coherence issues found")


# ============================================================================
# Pydantic Models for Structured Output (Result Types)
# ============================================================================


class ArticleOutline(BaseModel):
    """Full article outline."""
    title: str
    sections: list[dict]


class ArticleResearch(BaseModel):
    """Complete research for all sections."""
    sections: list[dict]


class EditorReview(BaseModel):
    """Editor feedback and scoring."""
    score: int
    feedback: str
    grammar_issues: list[str]
    coherence_issues: list[str]


# ============================================================================
# Shared State for Tool Outputs
# =============================================================================

class ContentResults:
    """Shared state to store content generation results."""
    def __init__(self):
        self.outline: ArticleOutline | None = None
        self.research: ArticleResearch | None = None
        self.review: EditorReview | None = None

    def set_outline(self, result: ArticleOutline):
        self.outline = result

    def set_research(self, result: ArticleResearch):
        self.research = result

    def set_review(self, result: EditorReview):
        self.review = result

    def reset_outline(self):
        self.outline = None

    def reset_research(self):
        self.research = None

    def reset_review(self):
        self.review = None


# Global results container
content_results = ContentResults()


# ============================================================================
# Tool API - Content Generation Tools
# ============================================================================

def submit_outline_tool(
    title: str,
    sections: list[dict],
) -> str:
    """Submit article outline.

    Args:
        title: Article title
        sections: List of sections with heading and key_points

    Returns:
        Confirmation message
    """
    outline = ArticleOutline(title=title, sections=sections)
    content_results.set_outline(outline)
    return f"Outline created: {title} with {len(sections)} sections"


def submit_research_tool(sections: list[dict]) -> str:
    """Submit section research.

    Args:
        sections: List of research for each section

    Returns:
        Confirmation message
    """
    research = ArticleResearch(sections=sections)
    content_results.set_research(research)
    return f"Research generated for {len(sections)} sections"


def submit_review_tool(
    score: int,
    feedback: str,
    grammar_issues: list[str] | None = None,
    coherence_issues: list[str] | None = None,
) -> str:
    """Submit editor review.

    Args:
        score: Quality score 1-10
        feedback: Specific feedback for improvement
        grammar_issues: Grammar issues found
        coherence_issues: Coherence issues found

    Returns:
        Confirmation message
    """
    if grammar_issues is None:
        grammar_issues = []
    if coherence_issues is None:
        coherence_issues = []

    review = EditorReview(
        score=score,
        feedback=feedback,
        grammar_issues=grammar_issues,
        coherence_issues=coherence_issues,
    )
    content_results.set_review(review)
    return f"Review submitted: score {score}/10, {len(grammar_issues)} grammar issues, {len(coherence_issues)} coherence issues"


# Create tool definitions
outline_tool = StructuredTool.from_function(
    func=submit_outline_tool,
    name="submit_outline",
    description="Submit article outline with title and sections",
    args_schema=ArticleOutlineArgs,
)

research_tool = StructuredTool.from_function(
    func=submit_research_tool,
    name="submit_research",
    description="Submit research for article sections",
    args_schema=SectionResearchArgs,
)

review_tool = StructuredTool.from_function(
    func=submit_review_tool,
    name="submit_review",
    description="Submit editor review with score and feedback",
    args_schema=EditorReviewArgs,
)


# ============================================================================
# LangGraph State
# ============================================================================

class WritingState(dict):
    """State for content writing pipeline."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    outline: dict | None
    research: dict | None
    draft: str | None
    editor_review: dict | None
    revision_count: int
    max_revisions: int
    final_article: str | None


# ============================================================================
# Stage System Prompts
# ============================================================================

OUTLINER_SYSTEM = """You are a blog content strategist. Create compelling outlines for technical blog posts.

Your task:
1. Generate a catchy, SEO-friendly title
2. Create 4-6 section headings
3. Provide 2-3 key points for each section

Focus on: clarity, logical flow, and comprehensive coverage.

IMPORTANT: You MUST use the submit_outline tool to submit your outline."""

RESEARCHER_SYSTEM = """You are a technical researcher. Gather talking points and facts for each section.

For each section heading:
1. Generate 3-5 relevant talking points
2. Include specific technical details where applicable
3. Add examples or comparisons
4. Ensure factual accuracy

IMPORTANT: You MUST use the submit_research tool to submit your research."""

DRAFTER_SYSTEM = """You are a technical writer. Write engaging blog prose based on outline and research.

Your task:
1. Write 150-200 words per section
2. Maintain consistent, friendly tone
3. Use clear, accessible language
4. Include section transitions
5. Incorporate research talking points naturally

Style guidelines:
- Use active voice
- Include relevant examples
- Avoid jargon unless explained
- Keep paragraphs focused (3-5 sentences)

Write the full article as markdown."""

EDITOR_SYSTEM = """You are a content editor. Review blog drafts for quality and provide constructive feedback.

Evaluate on:
1. Coherence (20%): logical flow, transitions, structure
2. Tone (20%): consistency, voice, appropriateness
3. Grammar (20%): syntax, spelling, punctuation
4. Content (30%): accuracy, completeness, relevance
5. Engagement (10%): interest, readability, flow

Scoring rubric:
- 9-10: Excellent, ready to publish
- 7-8: Good, minor improvements
- 5-6: Fair, needs moderate revisions
- 1-4: Poor, needs major rewrite

IMPORTANT: You MUST use the submit_review tool to submit your assessment. The tool requires:
- score: integer 1-10
- feedback: specific feedback for improvement
- grammar_issues: array of grammar issues found (empty array if none)
- coherence_issues: array of coherence issues found (empty array if none)"""


# ============================================================================
# Provider Configuration
# ============================================================================


def create_chat_model(provider: str, model_id: str) -> BaseChatModel:
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
# Helper Functions
# ============================================================================

def extract_markdown_from_response(response: str) -> str:
    """Extract markdown from response."""
    if "```markdown" in response:
        start = response.find("```markdown") + 11
        end = response.find("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        return response[start:end].strip()
    return response


# ============================================================================
# Tool Calling Helper (for use in LangGraph nodes)
# ============================================================================

async def call_agent_with_tools(
    prompt: str,
    system_prompt: str,
    model: BaseChatModel,
    tools: list,
) -> str:
    """Call an agent with tools and return the response content."""
    model_with_tools = model.bind_tools(tools)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    for turn in range(5):  # max_turns = 5
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "")

                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    result = tool.func(**tool_args)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
        else:
            break

    return messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])


# ============================================================================
# Pipeline Nodes (LangGraph with Tool API)
# ============================================================================

async def outliner_node(state: WritingState, config):
    """Generate article outline using Tool API."""
    model = config["configurable"]["model"]

    # Reset global result
    content_results.reset_outline()

    prompt = f"""Create a comprehensive outline for a blog post on this topic:

Topic: {state['topic']}

Create an outline with a catchy title and 4-6 sections, each with 2-3 key points."""

    await call_agent_with_tools(prompt, OUTLINER_SYSTEM, model, [outline_tool])
    await asyncio.sleep(2)  # Rate limiting

    outline = content_results.outline
    if outline:
        return {
            "outline": {"title": outline.title, "sections": outline.sections},
            "messages": [AIMessage(content=f"Generated outline with {len(outline.sections)} sections")]
        }
    else:
        return {
            "outline": {"title": "Untitled", "sections": []},
            "messages": [AIMessage(content="Failed to generate outline")]
        }


async def researcher_node(state: WritingState, config):
    """Generate research for each section using Tool API."""
    model = config["configurable"]["model"]
    outline = state.get("outline", {})

    # Reset global result
    content_results.reset_research()

    sections_desc = "\n".join([
        f"{s['heading']}: {', '.join(s['key_points'])}"
        for s in outline.get("sections", [])
    ])

    prompt = f"""Generate talking points for each section:

Sections:
{sections_desc}

Generate 3-5 relevant talking points for each section."""

    await call_agent_with_tools(prompt, RESEARCHER_SYSTEM, model, [research_tool])
    await asyncio.sleep(2)  # Rate limiting

    research = content_results.research
    if research:
        return {
            "research": {"sections": research.sections},
            "messages": [AIMessage(content=f"Generated research for {len(research.sections)} sections")]
        }
    else:
        return {
            "research": {"sections": []},
            "messages": [AIMessage(content="Failed to generate research")]
        }


async def drafter_node(state: WritingState, config):
    """Write full article draft."""
    model = config["configurable"]["model"]
    outline = state.get("outline", {})
    research = state.get("research", {})
    revision_count = state.get("revision_count", 0)

    outline_str = json.dumps(outline, indent=2)
    research_str = json.dumps(research, indent=2)

    feedback_context = ""
    if revision_count > 0:
        review = state.get("editor_review", {})
        feedback_context = f"\n\nPREVIOUS FEEDBACK TO ADDRESS:\n{review.get('feedback', '')}\nGrammar issues: {review.get('grammar_issues', [])}\nCoherence issues: {review.get('coherence_issues', [])}"

    prompt = f"""Write a complete blog post based on this outline and research:

TOPIC: {state['topic']}

OUTLINE:
{outline_str}

RESEARCH:
{research_str}{feedback_context}

{DRAFTER_SYSTEM}

Write the full article as markdown. Each section should be 150-200 words."""

    response = await model.ainvoke([HumanMessage(content=prompt)])
    draft = response.content

    if "```" in draft:
        draft = extract_markdown_from_response(draft)

    await asyncio.sleep(2)  # Rate limiting

    return {
        "draft": draft,
        "messages": [AIMessage(content=f"Draft complete (~{len(draft.split())} words)")]
    }


async def editor_node(state: WritingState, config):
    """Review draft and provide score/feedback using Tool API."""
    model = config["configurable"]["model"]
    draft = state.get("draft", "")

    # Reset global result
    content_results.reset_review()

    prompt = f"""Review this blog post draft and provide quality assessment:

DRAFT:
{draft}

Provide an overall score (1-10), specific feedback, and list any grammar or coherence issues."""

    await call_agent_with_tools(prompt, EDITOR_SYSTEM, model, [review_tool])
    await asyncio.sleep(2)  # Rate limiting

    review = content_results.review
    if review:
        return {
            "editor_review": {
                "score": review.score,
                "feedback": review.feedback,
                "grammar_issues": review.grammar_issues,
                "coherence_issues": review.coherence_issues,
            },
            "messages": [AIMessage(content=f"Editor score: {review.score}/10")]
        }
    else:
        return {
            "editor_review": {"score": 5, "feedback": "Review failed", "grammar_issues": [], "coherence_issues": []},
            "messages": [AIMessage(content="Editor review failed")]
        }


def should_revise(state: WritingState) -> Literal["finalize", "revise"]:
    """Check if revision is needed based on editor score."""
    review = state.get("editor_review", {})
    score = review.get("score", 10)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", MAX_REVISIONS)

    if score >= QUALITY_THRESHOLD:
        return "finalize"
    elif revision_count < max_revisions:
        return "revise"
    else:
        return "finalize"


def increment_revision(state: WritingState):
    """Increment revision counter."""
    return {"revision_count": state.get("revision_count", 0) + 1}


def finalizer_node(state: WritingState, config):
    """Finalize and save article."""
    draft = state.get("draft", "")
    outline = state.get("outline", {})
    title = outline.get("title", "Blog Post")

    word_count = len(draft.split())

    final_article = f"""# {title}

*Generated by Polyglot Agent Labs - Content Writing Agent*
*Word count: {word_count}*
*Revisions: {state.get('revision_count', 0)}*

---

{draft}"""

    with open("output.md", "w") as f:
        f.write(final_article)

    return {
        "final_article": final_article,
        "messages": [AIMessage(content=f"Final article saved to output.md ({word_count} words, {state.get('revision_count', 0)} revisions)")]
    }


# ============================================================================
# Graph Building
# ============================================================================

def build_writing_graph(model):
    """Build the content writing pipeline graph."""
    workflow = StateGraph(WritingState)

    workflow.add_node("outliner", outliner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("drafter", drafter_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("increment", increment_revision)
    workflow.add_node("finalizer", finalizer_node)

    workflow.set_entry_point("outliner")

    workflow.add_edge("outliner", "researcher")
    workflow.add_edge("researcher", "drafter")
    workflow.add_edge("drafter", "editor")

    workflow.add_conditional_edges(
        "editor",
        should_revise,
        {
            "finalize": "finalizer",
            "revise": "increment"
        }
    )

    workflow.add_edge("increment", "drafter")
    workflow.add_edge("finalizer", END)

    return workflow.compile()


# ============================================================================
# Demo Execution
# ============================================================================

async def run_demo_async(model: BaseChatModel, provider_name: str, model_id: str):
    """Run content writing demo asynchronously."""
    print("=== Python — Content Writing Agent (Tool API) ===")
    print(f"Provider: {provider_name}")
    print(f"Model: {model_id}")
    print()

    graph = build_writing_graph(model)

    print(f"Topic: {DEMO_TOPIC}")
    print("-" * 60)
    print()

    initial_state = {
        "messages": [],
        "topic": DEMO_TOPIC,
        "outline": None,
        "research": None,
        "draft": None,
        "editor_review": None,
        "revision_count": 0,
        "max_revisions": MAX_REVISIONS,
        "final_article": None,
    }

    config = {"configurable": {"model": model}}

    # Track final state from stream
    final_state = {}
    final_article_content = None
    final_outline = None
    final_score = None

    # Stream progress
    async for step in graph.astream(initial_state, config):
        for node_name, node_output in step.items():
            # Track final state
            final_state = node_output if "__end__" not in step else final_state

            if node_name != "__end__":
                if node_name == "outliner":
                    outline = node_output.get("outline", {})
                    final_outline = outline
                    print("[1/5] Outliner")
                    print(f"  ✓ Title: {outline.get('title', 'N/A')}")
                    print(f"  ✓ Sections: {len(outline.get('sections', []))}")

                elif node_name == "researcher":
                    research = node_output.get("research", {})
                    print("[2/5] Researcher")
                    print(f"  ✓ Research generated for {len(research.get('sections', []))} sections")

                elif node_name == "drafter":
                    rev_count = node_output.get("revision_count", 0)
                    print(f"[3/5] Drafter (revision {rev_count})")
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            print(f"  ✓ {msg.content}")

                elif node_name == "editor":
                    review = node_output.get("editor_review", {})
                    final_score = review
                    score = review.get("score", 0)
                    print("[4/5] Editor")
                    print(f"  ✓ Score: {score}/10")
                    if score < QUALITY_THRESHOLD:
                        print(f"  → Feedback: {review.get('feedback', 'No feedback')[:80]}...")

                elif node_name == "increment":
                    new_count = node_output.get("revision_count", 0)
                    print(f"[5/5] Revision increment: {new_count}")

                elif node_name == "finalizer":
                    final_article_content = node_output.get("final_article")
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            print(f"[Finalizer]")
                            print(f"  ✓ {msg.content}")

    # Print summary
    print("\n=== Pipeline Summary ===")
    print(f"Title: {final_outline.get('title', 'N/A') if final_outline else 'N/A'}")
    print(f"Sections: {len(final_outline.get('sections', [])) if final_outline else 0}")
    print(f"Revisions: {final_state.get('revision_count', 0)}")
    print(f"Final Score: {final_score.get('score', 'N/A') if final_score else 'N/A'}/10")

    if final_article_content:
        word_count = len(final_article_content.split())
        print(f"Word Count: {word_count}")
        print(f"Output: output.md")


def run_demo(model: BaseChatModel, provider_name: str, model_id: str):
    """Synchronous wrapper for demo."""
    asyncio.run(run_demo_async(model, provider_name, model_id))


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
