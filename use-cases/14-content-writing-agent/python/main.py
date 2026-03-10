"""
Polyglot Agent Labs — Use Case 14: Content Writing Agent (Blog Generator)
A multi-stage content generation pipeline with quality control loops.
Switch provider with env var LLM_PROVIDER (default: openrouter).

Usage:
  python main.py

Key Learning Goals:
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

import json
import os
import re
import sys
from typing import Annotated, Literal, Sequence

import operator
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

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
# Pydantic Models for Structured Output
# ============================================================================


class SectionOutline(BaseModel):
    """Single section outline."""
    heading: str = Field(description="Section heading")
    key_points: list[str] = Field(description="2-3 key points for this section")


class ArticleOutline(BaseModel):
    """Full article outline."""
    title: str = Field(description="Article title")
    sections: list[SectionOutline] = Field(description="4-6 sections with key points")


class SectionResearch(BaseModel):
    """Research notes for a section."""
    section_heading: str = Field(description="Matching section heading")
    talking_points: list[str] = Field(description="3-5 relevant talking points")


class ArticleResearch(BaseModel):
    """Complete research for all sections."""
    sections: list[SectionResearch] = Field(description="Research for each section")


class EditorReview(BaseModel):
    """Editor feedback and scoring."""
    score: int = Field(description="Quality score 1-10", ge=1, le=10)
    feedback: str = Field(description="Specific feedback for improvement")
    grammar_issues: list[str] = Field(default_factory=list, description="Grammar issues found")
    coherence_issues: list[str] = Field(default_factory=list, description="Coherence issues found")

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

Focus on: clarity, logical flow, and comprehensive coverage."""

RESEARCHER_SYSTEM = """You are a technical researcher. Gather talking points and facts for each section.

For each section heading:
1. Generate 3-5 relevant talking points
2. Include specific technical details where applicable
3. Add examples or comparisons
4. Ensure factual accuracy"""

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
- Keep paragraphs focused (3-5 sentences)"""

EDITOR_SYSTEM = """You are a content editor. Review blog drafts for quality and provide constructive feedback.

Evaluate on:
1. Coherence (20%): logical flow, transitions, structure
2. Tone (20%): consistency, voice, appropriateness
3. Grammar (20%): syntax, spelling, punctuation
4. Content (30%): accuracy, completeness, relevance
5. Engagement (10%): interest, readability, flow

Provide:
- Overall score 1-10
- Specific feedback for improvement
- List of grammar issues (if any)
- List of coherence issues (if any)

Scoring rubric:
- 9-10: Excellent, ready to publish
- 7-8: Good, minor improvements
- 5-6: Fair, needs moderate revisions
- 1-4: Poor, needs major rewrite"""

FINALIZER_SYSTEM = """You are a final editor. Prepare the article for publication.

Your task:
1. Apply any final polish
2. Format as proper markdown
3. Ensure consistent formatting
4. Add meta information (title, word count)"""

# ============================================================================
# Helper Functions
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
        if candidate.startswith('{'):
            return candidate

    # Look for { and } as JSON boundaries
    start = response.find('{')
    if start != -1:
        brace_count = 0
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response[start:i+1]

    return response


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
# Provider Configuration
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
# Pipeline Nodes
# ============================================================================


def outliner_node(state: WritingState, config):
    """Generate article outline."""
    model = config["configurable"]["model"]

    prompt = f"""Create a comprehensive outline for a blog post on this topic:

Topic: {state['topic']}

{OUTLINER_SYSTEM}

Return valid JSON matching this schema:
{{
  "title": "Article title",
  "sections": [
    {{
      "heading": "Section heading",
      "key_points": ["point 1", "point 2", "point 3"]
    }}
  ]
}}"""

    response = model.invoke([HumanMessage(content=prompt)])
    outline_str = extract_json_from_response(response.content)
    outline = json.loads(outline_str)

    return {
        "outline": outline,
        "messages": [AIMessage(content=f"Generated outline with {len(outline.get('sections', []))} sections")]
    }


def researcher_node(state: WritingState, config):
    """Generate research for each section."""
    model = config["configurable"]["model"]
    outline = state.get("outline", {})
    sections = outline.get("sections", [])

    all_research = []
    for section in sections:
        prompt = f"""Generate talking points for this section:

Section: {section['heading']}
Key points to cover: {', '.join(section['key_points'])}

{RESEARCHER_SYSTEM}

Return valid JSON:
{{
  "section_heading": "{section['heading']}",
  "talking_points": ["point 1", "point 2", "point 3", "point 4"]
}}"""

        response = model.invoke([HumanMessage(content=prompt)])
        research_str = extract_json_from_response(response.content)
        all_research.append(json.loads(research_str))

    return {
        "research": {"sections": all_research},
        "messages": [AIMessage(content=f"Generated research for {len(all_research)} sections")]
    }


def drafter_node(state: WritingState, config):
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

    response = model.invoke([HumanMessage(content=prompt)])
    draft = response.content

    if "```" in draft:
        draft = extract_markdown_from_response(draft)

    return {
        "draft": draft,
        "messages": [AIMessage(content=f"Draft complete (~{len(draft.split())} words)")]
    }


def editor_node(state: WritingState, config):
    """Review draft and provide score/feedback."""
    model = config["configurable"]["model"]
    draft = state.get("draft", "")

    prompt = f"""Review this blog post draft and provide quality assessment:

DRAFT:
{draft}

{EDITOR_SYSTEM}

Return valid JSON:
{{
  "score": <1-10>,
  "feedback": "Specific feedback for improvement",
  "grammar_issues": ["issue 1", "issue 2"],
  "coherence_issues": ["issue 1", "issue 2"]
}}"""

    response = model.invoke([HumanMessage(content=prompt)])
    review_str = extract_json_from_response(response.content)
    review = json.loads(review_str)

    return {
        "editor_review": review,
        "messages": [AIMessage(content=f"Editor score: {review.get('score')}/10")]
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

    workflow.add_node("outliner", lambda state: outliner_node(state, {"configurable": {"model": model}}))
    workflow.add_node("researcher", lambda state: researcher_node(state, {"configurable": {"model": model}}))
    workflow.add_node("drafter", lambda state: drafter_node(state, {"configurable": {"model": model}}))
    workflow.add_node("editor", lambda state: editor_node(state, {"configurable": {"model": model}}))
    workflow.add_node("increment", increment_revision)
    workflow.add_node("finalizer", lambda state: finalizer_node(state, {"configurable": {"model": model}}))

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


def run_demo(model, provider_name: str, model_id: str):
    """Run content writing demo."""
    print("=== Python — Content Writing Agent ===")
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
    for step in graph.stream(initial_state, config):
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
