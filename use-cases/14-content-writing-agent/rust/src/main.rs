//! Polyglot Agent Labs — Use Case 14: Content Writing Agent (Blog Generator)
//! A multi-stage content generation pipeline with quality control loops.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run
//!
//! Key Learning Goals:
//! - Multi-stage pipelines - chaining specialized agents for content creation
//! - Prompt chaining - passing outputs between stages as context
//! - Quality control loops - iterative improvement with scoring thresholds
//! - Long-form generation - handling article-length content
//!
//! Pipeline Stages:
//! 1. Outliner → Generate structured article outline
//! 2. Researcher → Gather talking points for each section
//! 3. Drafter → Write full prose for each section
//! 4. Editor → Score quality 1-10, provide feedback
//! 5. Quality Gate → Score >= 7 proceeds, < 7 triggers revision (max 2)
//! 6. Finalizer → Format and save to output.md

use std::env;
use std::fs;

use anyhow::{bail, Result};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::extractor::{Extractor, ExtractorBuilder};
use rig::providers::{anthropic, openai, openrouter};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

// ============================================================================
// Configuration
// ============================================================================

const DEMO_TOPIC: &str = "Why developers should learn both Python and Rust";
const QUALITY_THRESHOLD: i32 = 7;
const MAX_REVISIONS: usize = 2;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SectionOutline {
    pub heading: String,
    pub key_points: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct ArticleOutline {
    pub title: String,
    pub sections: Vec<SectionOutline>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SectionResearch {
    pub section_heading: String,
    pub talking_points: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct ArticleResearch {
    pub sections: Vec<SectionResearch>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EditorReview {
    pub score: i32,
    pub feedback: String,
    #[serde(default)]
    pub grammar_issues: Vec<String>,
    #[serde(default)]
    pub coherence_issues: Vec<String>,
}

// ============================================================================
// Stage System Prompts
// ============================================================================

const OUTLINER_SYSTEM: &str = r#"You are a blog content strategist. Create compelling outlines for technical blog posts.

Your task:
1. Generate a catchy, SEO-friendly title
2. Create 4-6 section headings
3. Provide 2-3 key points for each section

Focus on: clarity, logical flow, and comprehensive coverage."#;

const RESEARCHER_SYSTEM: &str = r#"You are a technical researcher. Gather talking points and facts for each section.

For each section heading:
1. Generate 3-5 relevant talking points
2. Include specific technical details where applicable
3. Add examples or comparisons
4. Ensure factual accuracy"#;

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

Provide:
- Overall score 1-10
- Specific feedback for improvement
- List of grammar issues (if any)
- List of coherence issues (if any)"#;

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
// Stage Functions
// ============================================================================

async fn generate_outline(topic: &str, extractor: &Extractor<impl CompletionModel, ArticleOutline>) -> Result<ArticleOutline> {
    let prompt = format!(
        r#"Create a comprehensive outline for a blog post on this topic.

Topic: {topic}

{OUTLINER_SYSTEM}

Create an outline with a catchy title and 4-6 sections, each with 2-3 key points."#
    );

    extractor.extract(&prompt).await.map_err(Into::into)
}

async fn generate_research(
    outline: &ArticleOutline,
    extractor: &Extractor<impl CompletionModel, SectionResearch>,
) -> Result<ArticleResearch> {
    let mut all_research = Vec::new();

    for section in &outline.sections {
        let prompt = format!(
            r#"Generate talking points for this section.

Section: {}
Key points to cover: {}

{RESEARCHER_SYSTEM}

Generate 3-5 relevant talking points for this section."#,
            section.heading,
            section.key_points.join(", ")
        );

        let mut research = extractor.extract(&prompt).await
            .map_err(|e| anyhow::anyhow!("Failed to extract research: {}", e))?;
        // Ensure the section heading matches
        research.section_heading = section.heading.clone();
        all_research.push(research);
    }

    Ok(ArticleResearch {
        sections: all_research,
    })
}

async fn write_draft(
    topic: &str,
    outline: &ArticleOutline,
    research: &ArticleResearch,
    feedback: Option<&EditorReview>,
    agent: &impl Prompt,
) -> Result<String> {
    let outline_str = serde_json::to_string_pretty(outline)?;
    let research_str = serde_json::to_string_pretty(research)?;

    let feedback_str = match feedback {
        Some(f) => format!(
            "\n\nPREVIOUS FEEDBACK TO ADDRESS:\n{}\nGrammar: {:?}\nCoherence: {:?}",
            f.feedback, f.grammar_issues, f.coherence_issues
        ),
        None => String::new(),
    };

    let prompt = format!(
        r#"Write a complete blog post based on this outline and research:

TOPIC: {topic}

OUTLINE:
{outline_str}

RESEARCH:
{research_str}{feedback_str}

{DRAFTER_SYSTEM}"#
    );

    let response = agent.prompt(&prompt).await?;
    Ok(extract_markdown_from_response(&response))
}

async fn review_draft(draft: &str, extractor: &Extractor<impl CompletionModel, EditorReview>) -> Result<EditorReview> {
    let prompt = format!(
        r#"Review this blog post draft and provide quality assessment.

DRAFT:
{draft}

{EDITOR_SYSTEM}

Provide an overall score (1-10), specific feedback, and list any grammar or coherence issues."#
    );

    extractor.extract(&prompt).await.map_err(Into::into)
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

    // Create extractors
    let outline_extractor = ExtractorBuilder::new(model.clone()).build();
    let research_extractor = ExtractorBuilder::new(model.clone()).build();
    let editor_extractor = ExtractorBuilder::new(model.clone()).build();

    // Create agent for drafting (uses prompt, not extraction)
    let drafter_agent = rig::agent::AgentBuilder::new(model).build();

    // Stage 1: Outline
    println!("[1/5] Generating outline...");
    let outline = generate_outline(DEMO_TOPIC, &outline_extractor).await?;
    println!("  ✓ Title: {}", outline.title);
    println!("  ✓ {} sections", outline.sections.len());

    // Stage 2: Research
    println!("\n[2/5] Generating research...");
    let research = generate_research(&outline, &research_extractor).await?;
    println!("  ✓ Research complete");

    // Stage 3-5: Draft with revision loop
    let mut final_review: Option<EditorReview> = None;
    let mut revision_count = 0;

    let draft = loop {
        // Stage 3: Draft
        println!("\n[3/5] Writing draft (revision {})...", revision_count);
        let feedback = final_review.as_ref();
        let current_draft = write_draft(DEMO_TOPIC, &outline, &research, feedback, &drafter_agent).await?;
        println!("  ✓ Draft complete (~{} words)", current_draft.split_whitespace().count());

        // Stage 4: Editor Review
        println!("\n[4/5] Editor review...");
        let review = review_draft(&current_draft, &editor_extractor).await?;
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
