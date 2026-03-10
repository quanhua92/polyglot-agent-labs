//! Polyglot Agent Labs — Use Case 14: Content Writing Agent (Blog Generator)
//! A multi-stage content generation pipeline with quality control loops.
//! Uses Tool API for reliable structured output.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run
//!
//! Key Learning Goals:
//! - Multi-stage pipelines - chaining specialized agents for content creation
//! - Tool API for structured output - tools accept typed parameters from LLM
//! - Prompt chaining - passing outputs between stages as context
//! - Quality control loops - iterative improvement with scoring thresholds
//! - Long-form generation - handling article-length content
//!
//! Pipeline Stages:
//! 1. Outliner → Generate structured article outline (via SubmitOutline tool)
//! 2. Researcher → Gather talking points for each section (via SubmitResearch tool)
//! 3. Drafter → Write full prose for each section
//! 4. Editor → Score quality 1-10, provide feedback (via SubmitReview tool)
//! 5. Quality Gate → Score >= 7 proceeds, < 7 triggers revision (max 2)
//! 6. Finalizer → Format and save to output.md

use std::env;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Result};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::providers::{anthropic, openai, openrouter};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Configuration
// ============================================================================

const DEMO_TOPIC: &str = "Why developers should learn both Python and Rust";
const QUALITY_THRESHOLD: i32 = 7;
const MAX_REVISIONS: usize = 2;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SectionOutline {
    pub heading: String,
    pub key_points: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArticleOutline {
    pub title: String,
    pub sections: Vec<SectionOutline>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SectionResearch {
    pub section_heading: String,
    pub talking_points: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArticleResearch {
    pub sections: Vec<SectionResearch>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EditorReview {
    pub score: i32,
    pub feedback: String,
    #[serde(default)]
    pub grammar_issues: Vec<String>,
    #[serde(default)]
    pub coherence_issues: Vec<String>,
}

// ============================================================================
// Shared State for Tool Outputs
// ============================================================================

#[derive(Debug, Clone, Default)]
struct PipelineState {
    outline: Option<ArticleOutline>,
    research: Option<ArticleResearch>,
    review: Option<EditorReview>,
}

impl PipelineState {
    fn set_outline(&mut self, outline: ArticleOutline) {
        self.outline = Some(outline);
    }

    fn set_research(&mut self, research: ArticleResearch) {
        self.research = Some(research);
    }

    fn set_review(&mut self, review: EditorReview) {
        self.review = Some(review);
    }

    fn get_outline(&self) -> Result<ArticleOutline> {
        self.outline.clone().ok_or_else(|| anyhow::anyhow!("Outline not set"))
    }

    fn get_research(&self) -> Result<ArticleResearch> {
        self.research.clone().ok_or_else(|| anyhow::anyhow!("Research not set"))
    }

    fn get_review(&self) -> Result<EditorReview> {
        self.review.clone().ok_or_else(|| anyhow::anyhow!("Review not set"))
    }
}

// ============================================================================
// Tool API - Structured Output Tools
// ============================================================================

// ----------------------------------------------------------------------------
// SubmitOutline Tool
// ----------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error("Outline error")]
struct OutlineError;

#[derive(Debug, Deserialize)]
struct SectionOutlineData {
    heading: String,
    key_points: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SubmitOutlineArgs {
    title: String,
    sections: Vec<SectionOutlineData>,
}

#[derive(Debug, Serialize)]
struct OutlineResult(String);

struct SubmitOutline {
    state: Arc<Mutex<PipelineState>>,
}

impl Tool for SubmitOutline {
    const NAME: &'static str = "submit_outline";

    type Error = OutlineError;
    type Args = SubmitOutlineArgs;
    type Output = OutlineResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "submit_outline".to_string(),
            description: "Submit a complete article outline with title and sections".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The article title"
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {
                                    "type": "string",
                                    "description": "Section heading"
                                },
                                "key_points": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "2-3 key points for this section"
                                }
                            },
                            "required": ["heading", "key_points"]
                        }
                    }
                },
                "required": ["title", "sections"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let outline = ArticleOutline {
            title: args.title,
            sections: args.sections.into_iter().map(|s| SectionOutline {
                heading: s.heading,
                key_points: s.key_points,
            }).collect(),
        };

        let mut state = self.state.lock().unwrap();
        state.set_outline(outline.clone());

        Ok(OutlineResult(format!(
            "Outline created: '{}' with {} sections",
            outline.title,
            outline.sections.len()
        )))
    }
}

// ----------------------------------------------------------------------------
// SubmitResearch Tool
// ----------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error("Research error")]
struct ResearchError;

#[derive(Debug, Deserialize)]
struct SectionResearchData {
    section_heading: String,
    talking_points: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SubmitResearchArgs {
    sections: Vec<SectionResearchData>,
}

#[derive(Debug, Serialize)]
struct ResearchResult(String);

struct SubmitResearch {
    state: Arc<Mutex<PipelineState>>,
}

impl Tool for SubmitResearch {
    const NAME: &'static str = "submit_research";

    type Error = ResearchError;
    type Args = SubmitResearchArgs;
    type Output = ResearchResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "submit_research".to_string(),
            description: "Submit research notes with talking points for each section".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section_heading": {
                                    "type": "string",
                                    "description": "Matching section heading from outline"
                                },
                                "talking_points": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "3-5 relevant talking points"
                                }
                            },
                            "required": ["section_heading", "talking_points"]
                        }
                    }
                },
                "required": ["sections"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let research = ArticleResearch {
            sections: args.sections.into_iter().map(|s| SectionResearch {
                section_heading: s.section_heading,
                talking_points: s.talking_points,
            }).collect(),
        };

        let mut state = self.state.lock().unwrap();
        state.set_research(research.clone());

        Ok(ResearchResult(format!(
            "Research submitted for {} sections",
            research.sections.len()
        )))
    }
}

// ----------------------------------------------------------------------------
// SubmitReview Tool
// ----------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error("Review error")]
struct ReviewError;

#[derive(Debug, Deserialize)]
struct SubmitReviewArgs {
    score: i32,
    feedback: String,
    #[serde(default)]
    grammar_issues: Vec<String>,
    #[serde(default)]
    coherence_issues: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReviewResult(String);

struct SubmitReview {
    state: Arc<Mutex<PipelineState>>,
}

impl Tool for SubmitReview {
    const NAME: &'static str = "submit_review";

    type Error = ReviewError;
    type Args = SubmitReviewArgs;
    type Output = ReviewResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "submit_review".to_string(),
            description: "Submit a quality review with score and feedback".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "description": "Quality score from 1 to 10",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Specific feedback for improvement"
                    },
                    "grammar_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of grammar issues found"
                    },
                    "coherence_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of coherence issues found"
                    }
                },
                "required": ["score", "feedback"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let review = EditorReview {
            score: args.score,
            feedback: args.feedback,
            grammar_issues: args.grammar_issues,
            coherence_issues: args.coherence_issues,
        };

        let mut state = self.state.lock().unwrap();
        state.set_review(review.clone());

        Ok(ReviewResult(format!(
            "Review submitted: {}/10 - {}",
            review.score,
            if review.score >= 7 { "Good" } else { "Needs improvement" }
        )))
    }
}

// ============================================================================
// Stage System Prompts
// ============================================================================

const OUTLINER_SYSTEM: &str = r#"You are a blog content strategist. Create compelling outlines for technical blog posts.

Your task:
1. Generate a catchy, SEO-friendly title
2. Create 4-6 section headings
3. Provide 2-3 key points for each section

Focus on: clarity, logical flow, and comprehensive coverage.

Use the submit_outline tool to submit your outline."#;

const RESEARCHER_SYSTEM: &str = r#"You are a technical researcher. Gather talking points and facts for each section.

For each section heading:
1. Generate 3-5 relevant talking points
2. Include specific technical details where applicable
3. Add examples or comparisons
4. Ensure factual accuracy

Use the submit_research tool to submit your research for all sections at once."#;

const DRAFTER_SYSTEM: &str = r#"You are a technical writer. Write engaging blog prose based on outline and research.

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

Write the full article as markdown."#;

const EDITOR_SYSTEM: &str = r#"You are a content editor. Review blog drafts for quality and provide constructive feedback.

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
- coherence_issues: array of coherence issues found (empty array if none)"#;

// ============================================================================
// Helper Functions
// ============================================================================

fn extract_markdown_from_response(response: &str) -> String {
    if let Some(start) = response.find("```markdown") {
        let start = start + 11;
        if let Some(end) = response[start..].find("```") {
            return response[start..start + end].trim().to_string();
        }
    }

    if let Some(start) = response.find("```") {
        let start = start + 3;
        if let Some(end) = response[start..].find("```") {
            return response[start..start + end].trim().to_string();
        }
    }

    response.to_string()
}

// ============================================================================
// Pipeline Execution
// ============================================================================

pub async fn execute_pipeline<M>(model: M) -> Result<(String, usize, i32)>
where
    M: CompletionModel + Clone + Send + Sync + 'static,
{
    println!("=== Content Writing Pipeline ===");
    println!("Topic: {}\n", DEMO_TOPIC);

    let state = Arc::new(Mutex::new(PipelineState::default()));

    // Stage 1: Outline
    println!("[1/5] Generating outline...");
    {
        let outline_agent = rig::agent::AgentBuilder::new(model.clone())
            .preamble(OUTLINER_SYSTEM)
            .tool(SubmitOutline { state: state.clone() })
            .build();

        let prompt = format!(
            r#"Create a comprehensive outline for a blog post on this topic:

Topic: {DEMO_TOPIC}

Create an outline with a catchy title and 4-6 sections, each with 2-3 key points."#
        );

        let _response = outline_agent.prompt(&prompt).max_turns(5).await?;
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    let outline = {
        let s = state.lock().unwrap();
        s.get_outline()?
    };
    println!("  ✓ Title: {}", outline.title);
    println!("  ✓ {} sections", outline.sections.len());

    // Stage 2: Research
    println!("\n[2/5] Generating research...");
    {
        let research_agent = rig::agent::AgentBuilder::new(model.clone())
            .preamble(RESEARCHER_SYSTEM)
            .tool(SubmitResearch { state: state.clone() })
            .build();

        let sections_desc: Vec<String> = outline.sections.iter().map(|s| {
            format!("{}: {}", s.heading, s.key_points.join(", "))
        }).collect();

        let prompt = format!(
            r#"Generate talking points for each section:

Sections:
{}

Generate 3-5 relevant talking points for each section."#,
            sections_desc.join("\n")
        );

        let _response = research_agent.prompt(&prompt).max_turns(5).await?;
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    let research = {
        let s = state.lock().unwrap();
        s.get_research()?
    };
    println!("  ✓ Research complete");

    // Stage 3-5: Draft with revision loop
    let mut final_review: Option<EditorReview> = None;
    let mut revision_count = 0;

    let drafter_agent = rig::agent::AgentBuilder::new(model.clone())
        .preamble(DRAFTER_SYSTEM)
        .build();

    let draft = loop {
        // Stage 3: Draft
        println!("\n[3/5] Writing draft (revision {})...", revision_count);

        let feedback_str = match &final_review {
            Some(f) => format!(
                "\n\nPREVIOUS FEEDBACK TO ADDRESS:\n{}\nGrammar: {:?}\nCoherence: {:?}",
                f.feedback, f.grammar_issues, f.coherence_issues
            ),
            None => String::new(),
        };

        let outline_str = serde_json::to_string_pretty(&outline)?;
        let research_str = serde_json::to_string_pretty(&research)?;

        let prompt = format!(
            r#"Write a complete blog post based on this outline and research:

TOPIC: {DEMO_TOPIC}

OUTLINE:
{outline_str}

RESEARCH:
{research_str}{feedback_str}

Write the full article as markdown. Each section should be 150-200 words."#
        );

        let response = drafter_agent.prompt(&prompt).max_turns(3).await?;
        let current_draft = extract_markdown_from_response(&response);
        tokio::time::sleep(Duration::from_secs(2)).await;
        println!("  ✓ Draft complete (~{} words)", current_draft.split_whitespace().count());

        // Stage 4: Editor Review
        println!("\n[4/5] Editor review...");

        // Clear previous review
        {
            let mut s = state.lock().unwrap();
            s.review = None;
        }

        let editor_agent = rig::agent::AgentBuilder::new(model.clone())
            .preamble(EDITOR_SYSTEM)
            .tool(SubmitReview { state: state.clone() })
            .build();

        let review_prompt = format!(
            r#"Review this blog post draft and provide quality assessment:

DRAFT:
{current_draft}

Provide an overall score (1-10), specific feedback, and list any grammar or coherence issues."#
        );

        let _response = editor_agent.prompt(&review_prompt).max_turns(5).await?;
        tokio::time::sleep(Duration::from_secs(2)).await;

        let review = {
            let s = state.lock().unwrap();
            s.get_review()?
        };
        println!("  ✓ Score: {}/10", review.score);

        if review.score >= QUALITY_THRESHOLD {
            final_review = Some(review);
            break current_draft;
        } else if revision_count >= MAX_REVISIONS {
            println!("  ! Max revisions reached, proceeding");
            final_review = Some(review);
            break current_draft;
        }

        println!("  → Needs revision (feedback: {})", review.feedback);
        final_review = Some(review);
        revision_count += 1;
    };

    // Stage 5: Finalize
    println!("\n[5/5] Finalizing article...");
    let word_count = draft.split_whitespace().count();
    let final_article = format!(
        "# {}\n\n*Generated by Polyglot Agent Labs - Content Writing Agent*\n*Word count: {}*\n*Revisions: {}*\n\n---\n\n{}",
        outline.title, word_count, revision_count, draft
    );

    fs::write("output.md", &final_article)?;
    println!("  ✓ Saved to output.md");

    Ok((final_article, revision_count, final_review.unwrap().score))
}

// ============================================================================
// Provider Setup
// ============================================================================

async fn run_openai() -> Result<(String, usize, i32)> {
    let model_id = "gpt-4.1-nano";
    println!("=== Rust — Content Writing Agent ===");
    println!("Provider:  openai");
    println!("Model:     {model_id}\n");

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    execute_pipeline(model).await
}

async fn run_anthropic() -> Result<(String, usize, i32)> {
    let model_id = "claude-3-haiku-20240307";
    println!("=== Rust — Content Writing Agent ===");
    println!("Provider:  anthropic");
    println!("Model:     {model_id}\n");

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    execute_pipeline(model).await
}

async fn run_openrouter() -> Result<(String, usize, i32)> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("=== Rust — Content Writing Agent ===");
    println!("Provider:  openrouter");
    println!("Model:     {model_id}\n");

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    execute_pipeline(model).await
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    let (_article, revisions, score) = match provider.as_str() {
        "openai" => run_openai().await?,
        "anthropic" => run_anthropic().await?,
        "openrouter" => run_openrouter().await?,
        _ => {
            bail!("Unknown provider: '{}'. Supported: openai, anthropic, openrouter", provider);
        }
    };

    // Print summary
    println!("\n=== Pipeline Summary ===");
    println!("Revisions: {revisions}");
    println!("Final Score: {score}/10");

    Ok(())
}
