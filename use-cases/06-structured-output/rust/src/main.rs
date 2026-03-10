//! Polyglot Agent Labs — Use Case 06: Structured Output & Data Extraction
//! Extract typed, validated data from unstructured text using Tool API.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run
//!
//! This demo extracts structured data from 3 types of unstructured text:
//! - Job listings: title, company, location, salary_range, required_skills, employment_type, description
//! - Product reviews: product_name, rating, pros, cons, summary, would_recommend
//! - Emails: sender, recipients, subject, action_items, urgency, key_points, deadline

use std::env;
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
// Serde Structs for Validation
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobListing {
    pub title: String,
    pub company: String,
    pub location: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub salary_range: Option<String>,
    pub required_skills: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub employment_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProductReview {
    pub product_name: String,
    pub rating: f32,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub would_recommend: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmailInfo {
    pub sender: String,
    pub recipients: Vec<String>,
    pub subject: String,
    pub action_items: Vec<String>,
    pub urgency: String,
    pub key_points: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline: Option<String>,
}

// ============================================================================
// Hard-coded Sample Inputs
// ============================================================================

const SAMPLE_JOB_LISTING: &str = r#"
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
"#;

const SAMPLE_PRODUCT_REVIEW: &str = r#"
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
"#;

const SAMPLE_EMAIL: &str = r#"
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
"#;

// ============================================================================
// Tool API - Structured Output Tools
// ============================================================================

// ----------------------------------------------------------------------------
// ExtractJobListing Tool
// ----------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error("Job listing error")]
struct JobListingError;

#[derive(Debug, Deserialize)]
struct ExtractJobListingArgs {
    title: String,
    company: String,
    location: String,
    #[serde(default)]
    salary_range: Option<String>,
    required_skills: Vec<String>,
    #[serde(default)]
    employment_type: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Serialize)]
struct JobListingResult(String);

struct ExtractJobListing {
    result: Arc<Mutex<Option<JobListing>>>,
}

impl Tool for ExtractJobListing {
    const NAME: &'static str = "extract_job_listing";

    type Error = JobListingError;
    type Args = ExtractJobListingArgs;
    type Output = JobListingResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "extract_job_listing".to_string(),
            description: "Extract structured job listing data from unstructured text".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Job title"},
                    "company": {"type": "string", "description": "Company name"},
                    "location": {"type": "string", "description": "Job location"},
                    "salary_range": {"type": "string", "description": "Salary range if mentioned"},
                    "required_skills": {"type": "array", "items": {"type": "string"}, "description": "List of required skills"},
                    "employment_type": {"type": "string", "description": "Employment type if mentioned"},
                    "description": {"type": "string", "description": "Job description summary"}
                },
                "required": ["title", "company", "location", "required_skills"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let listing = JobListing {
            title: args.title,
            company: args.company,
            location: args.location,
            salary_range: args.salary_range,
            required_skills: args.required_skills,
            employment_type: args.employment_type,
            description: args.description,
        };

        let mut result = self.result.lock().unwrap();
        *result = Some(listing.clone());

        Ok(JobListingResult(format!(
            "Extracted: {} at {}",
            listing.title, listing.company
        )))
    }
}

// ----------------------------------------------------------------------------
// ExtractProductReview Tool
// ----------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error("Product review error")]
struct ProductReviewError;

#[derive(Debug, Deserialize)]
struct ExtractProductReviewArgs {
    product_name: String,
    rating: f32,
    pros: Vec<String>,
    cons: Vec<String>,
    summary: String,
    #[serde(default)]
    would_recommend: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ProductReviewResult(String);

struct ExtractProductReview {
    result: Arc<Mutex<Option<ProductReview>>>,
}

impl Tool for ExtractProductReview {
    const NAME: &'static str = "extract_product_review";

    type Error = ProductReviewError;
    type Args = ExtractProductReviewArgs;
    type Output = ProductReviewResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "extract_product_review".to_string(),
            description: "Extract structured product review data from unstructured text".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Name of the product"},
                    "rating": {"type": "number", "description": "Product rating (usually 1-5 or 1-10)"},
                    "pros": {"type": "array", "items": {"type": "string"}, "description": "List of pros mentioned"},
                    "cons": {"type": "array", "items": {"type": "string"}, "description": "List of cons mentioned"},
                    "summary": {"type": "string", "description": "Summary of the review"},
                    "would_recommend": {"type": "boolean", "description": "Whether reviewer recommends the product"}
                },
                "required": ["product_name", "rating", "pros", "cons", "summary"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let review = ProductReview {
            product_name: args.product_name,
            rating: args.rating,
            pros: args.pros,
            cons: args.cons,
            summary: args.summary,
            would_recommend: args.would_recommend,
        };

        let mut result = self.result.lock().unwrap();
        *result = Some(review.clone());

        Ok(ProductReviewResult(format!(
            "Extracted review for {} - {}/5",
            review.product_name, review.rating
        )))
    }
}

// ----------------------------------------------------------------------------
// ExtractEmailInfo Tool
// ----------------------------------------------------------------------------

#[derive(Debug, Error)]
#[error("Email info error")]
struct EmailInfoError;

#[derive(Debug, Deserialize)]
struct ExtractEmailInfoArgs {
    sender: String,
    recipients: Vec<String>,
    subject: String,
    action_items: Vec<String>,
    urgency: String,
    key_points: Vec<String>,
    #[serde(default)]
    deadline: Option<String>,
}

#[derive(Debug, Serialize)]
struct EmailInfoResult(String);

struct ExtractEmailInfo {
    result: Arc<Mutex<Option<EmailInfo>>>,
}

impl Tool for ExtractEmailInfo {
    const NAME: &'static str = "extract_email_info";

    type Error = EmailInfoError;
    type Args = ExtractEmailInfoArgs;
    type Output = EmailInfoResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "extract_email_info".to_string(),
            description: "Extract structured email information from unstructured text".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "sender": {"type": "string", "description": "Email sender"},
                    "recipients": {"type": "array", "items": {"type": "string"}, "description": "Email recipients"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "action_items": {"type": "array", "items": {"type": "string"}, "description": "Action items extracted from email"},
                    "urgency": {"type": "string", "description": "Urgency level (urgent, normal, low)"},
                    "key_points": {"type": "array", "items": {"type": "string"}, "description": "Key points from the email"},
                    "deadline": {"type": "string", "description": "Deadline if mentioned"}
                },
                "required": ["sender", "recipients", "subject", "action_items", "urgency", "key_points"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let info = EmailInfo {
            sender: args.sender,
            recipients: args.recipients,
            subject: args.subject,
            action_items: args.action_items,
            urgency: args.urgency,
            key_points: args.key_points,
            deadline: args.deadline,
        };

        let mut result = self.result.lock().unwrap();
        *result = Some(info.clone());

        Ok(EmailInfoResult(format!(
            "Extracted email from {} - urgency: {}",
            info.sender, info.urgency
        )))
    }
}

// ============================================================================
// Provider Setup
// ============================================================================

async fn run_openai() -> Result<(usize, usize)> {
    let model_id = "gpt-4.1-nano";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    run_demo(model, "openai").await
}

async fn run_anthropic() -> Result<(usize, usize)> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    run_demo(model, "anthropic").await
}

async fn run_openrouter() -> Result<(usize, usize)> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    run_demo(model, "openrouter").await
}

// ============================================================================
// Demo Execution
// ============================================================================

async fn run_demo<M>(model: M, provider_key: &str) -> Result<(usize, usize)>
where
    M: CompletionModel + Clone + Send + Sync + 'static,
{
    println!("=== Rust — Structured Output & Data Extraction ===");
    println!("Provider:  {provider_key}");
    println!();

    let mut success_count = 0;
    let mut failure_count = 0;

    // ========================================================================
    // Job Listing Extraction
    // ========================================================================
    println!("[1/3] Job Listing Extraction");
    println!("{}", "-".repeat(50));
    println!("Input: {}...", &SAMPLE_JOB_LISTING[..100].trim());
    println!();

    let job_result: Arc<Mutex<Option<JobListing>>> = Arc::new(Mutex::new(None));
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(ExtractJobListing { result: job_result.clone() })
        .build();

    let prompt = format!(
        r#"Extract structured job listing data from this text:

{}

Use the extract_job_listing tool to submit the extracted data."#,
        SAMPLE_JOB_LISTING.trim()
    );

    match agent.prompt(&prompt).max_turns(5).await {
        Ok(_) => {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let result = job_result.lock().unwrap();
            if let Some(ref listing) = *result {
                let json = serde_json::to_string_pretty(listing)?;
                println!("\nExtracted Data:");
                println!("{}", json);
                println!("\nValidation: ✓ All required fields present");
                success_count += 1;
            } else {
                println!("\n✗ Error: Tool was not called");
                failure_count += 1;
            }
        }
        Err(e) => {
            println!("\n✗ Error: {}", e);
            failure_count += 1;
        }
    }

    println!("{}", "=".repeat(50));
    println!();

    // ========================================================================
    // Product Review Extraction
    // ========================================================================
    println!("[2/3] Product Review Extraction");
    println!("{}", "-".repeat(50));
    println!("Input: {}...", &SAMPLE_PRODUCT_REVIEW[..100].trim());
    println!();

    let review_result: Arc<Mutex<Option<ProductReview>>> = Arc::new(Mutex::new(None));
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(ExtractProductReview { result: review_result.clone() })
        .build();

    let prompt = format!(
        r#"Extract structured product review data from this text:

{}

Use the extract_product_review tool to submit the extracted data."#,
        SAMPLE_PRODUCT_REVIEW.trim()
    );

    match agent.prompt(&prompt).max_turns(5).await {
        Ok(_) => {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let result = review_result.lock().unwrap();
            if let Some(ref review) = *result {
                let json = serde_json::to_string_pretty(review)?;
                println!("\nExtracted Data:");
                println!("{}", json);
                println!("\nValidation: ✓ All required fields present");
                success_count += 1;
            } else {
                println!("\n✗ Error: Tool was not called");
                failure_count += 1;
            }
        }
        Err(e) => {
            println!("\n✗ Error: {}", e);
            failure_count += 1;
        }
    }

    println!("{}", "=".repeat(50));
    println!();

    // ========================================================================
    // Email Info Extraction
    // ========================================================================
    println!("[3/3] Email Information Extraction");
    println!("{}", "-".repeat(50));
    println!("Input: {}...", &SAMPLE_EMAIL[..100].trim());
    println!();

    let email_result: Arc<Mutex<Option<EmailInfo>>> = Arc::new(Mutex::new(None));
    let agent = rig::agent::AgentBuilder::new(model)
        .tool(ExtractEmailInfo { result: email_result.clone() })
        .build();

    let prompt = format!(
        r#"Extract structured email information from this text:

{}

Use the extract_email_info tool to submit the extracted data."#,
        SAMPLE_EMAIL.trim()
    );

    match agent.prompt(&prompt).max_turns(5).await {
        Ok(_) => {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let result = email_result.lock().unwrap();
            if let Some(ref info) = *result {
                let json = serde_json::to_string_pretty(info)?;
                println!("\nExtracted Data:");
                println!("{}", json);
                println!("\nValidation: ✓ All required fields present");
                success_count += 1;
            } else {
                println!("\n✗ Error: Tool was not called");
                failure_count += 1;
            }
        }
        Err(e) => {
            println!("\n✗ Error: {}", e);
            failure_count += 1;
        }
    }

    println!("{}", "=".repeat(50));
    println!();

    // Session Summary
    println!("Session Summary");
    println!("  Extractions completed: 3/3");
    println!("  Successful: {}", success_count);
    println!("  Failed: {}", failure_count);

    Ok((success_count, failure_count))
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    match provider.as_str() {
        "openai" => {
            run_openai().await?;
        }
        "anthropic" => {
            run_anthropic().await?;
        }
        "openrouter" => {
            run_openrouter().await?;
        }
        _ => {
            bail!(
                "Unknown provider: '{}'. Supported: openai, anthropic, openrouter",
                provider
            );
        }
    }

    Ok(())
}
