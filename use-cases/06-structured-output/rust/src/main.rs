//! Polyglot Agent Labs — Use Case 06: Structured Output & Data Extraction
//! Extract typed, validated data from unstructured text using structured LLM output.
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

use anyhow::{bail, Result};
use rig::client::CompletionClient;
use rig::completion::Chat;
use rig::providers::{anthropic, openai, openrouter};
use serde::{Deserialize, Serialize};

// ============================================================================
// Serde Structs for Structured Extraction
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
    pub rating: i32,
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
// Extraction Functions
// ============================================================================

async fn extract_job_listing(text: &str, agent: &impl Chat) -> Result<JobListing> {
    let prompt = format!(
        "Extract structured data from the following job listing. Return valid JSON matching this schema:\n\
        {{\n\
          \"title\": \"string\",\n\
          \"company\": \"string\",\n\
          \"location\": \"string\",\n\
          \"salary_range\": \"string or null\",\n\
          \"required_skills\": [\"string\"],\n\
          \"employment_type\": \"string or null\",\n\
          \"description\": \"string or null\"\n\
        }}\n\n\
        Job listing:\n{}",
        text
    );

    let response = agent.chat(&prompt, vec![]).await?;

    // Try to extract JSON from the response
    let json_str = extract_json_from_response(&response);
    let parsed: JobListing = serde_json::from_str(&json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse job listing JSON: {}", e))?;

    Ok(parsed)
}

async fn extract_product_review(text: &str, agent: &impl Chat) -> Result<ProductReview> {
    let prompt = format!(
        "Extract structured data from the following product review. Return valid JSON matching this schema:\n\
        {{\n\
          \"product_name\": \"string\",\n\
          \"rating\": integer (1-5),\n\
          \"pros\": [\"string\"],\n\
          \"cons\": [\"string\"],\n\
          \"summary\": \"string\",\n\
          \"would_recommend\": boolean or null\n\
        }}\n\n\
        Product review:\n{}",
        text
    );

    let response = agent.chat(&prompt, vec![]).await?;
    let json_str = extract_json_from_response(&response);
    let parsed: ProductReview = serde_json::from_str(&json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse product review JSON: {}", e))?;

    Ok(parsed)
}

async fn extract_email_info(text: &str, agent: &impl Chat) -> Result<EmailInfo> {
    let prompt = format!(
        "Extract structured data from the following email. Return valid JSON matching this schema:\n\
        {{\n\
          \"sender\": \"string\",\n\
          \"recipients\": [\"string\"],\n\
          \"subject\": \"string\",\n\
          \"action_items\": [\"string\"],\n\
          \"urgency\": \"string\" (high/medium/low),\n\
          \"key_points\": [\"string\"],\n\
          \"deadline\": \"string or null\"\n\
        }}\n\n\
        Email:\n{}",
        text
    );

    let response = agent.chat(&prompt, vec![]).await?;
    let json_str = extract_json_from_response(&response);
    let parsed: EmailInfo = serde_json::from_str(&json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse email info JSON: {}", e))?;

    Ok(parsed)
}

fn extract_json_from_response(response: &str) -> String {
    // Try to find JSON in the response (handles cases where LLM adds extra text)
    let response = response.trim();

    // Look for JSON block between ```json and ```
    if let Some(start) = response.find("```json") {
        let json_start = start + 7; // Skip ```json
        if let Some(end) = response[json_start..].find("```") {
            return response[json_start..json_start + end].trim().to_string();
        }
    }

    // Look for JSON block between ``` and ```
    if let Some(start) = response.find("```") {
        let json_start = start + 3;
        if let Some(end) = response[json_start..].find("```") {
            return response[json_start..json_start + end].trim().to_string();
        }
    }

    // Look for { and } as JSON boundaries
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            return response[start..=end].to_string();
        }
    }

    // Return as-is if no JSON markers found
    response.to_string()
}

// ============================================================================
// Provider Setup
// ============================================================================

async fn run_openai() -> Result<(usize, usize)> {
    let model_id = "gpt-4.1-nano";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    run_demo(agent, "openai").await
}

async fn run_anthropic() -> Result<(usize, usize)> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    run_demo(agent, "anthropic").await
}

async fn run_openrouter() -> Result<(usize, usize)> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    run_demo(agent, "openrouter").await
}

// ============================================================================
// Demo Execution
// ============================================================================

async fn run_demo(agent: impl Chat, provider_key: &str) -> Result<(usize, usize)> {
    println!("=== Rust — Structured Output & Data Extraction ===");
    println!("Provider:  {}", provider_key);
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

    match extract_job_listing(SAMPLE_JOB_LISTING, &agent).await {
        Ok(result) => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("\nExtracted Data:");
            println!("{}", json);
            println!("\nValidation: ✓ All required fields present");
            success_count += 1;
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

    match extract_product_review(SAMPLE_PRODUCT_REVIEW, &agent).await {
        Ok(result) => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("\nExtracted Data:");
            println!("{}", json);
            println!("\nValidation: ✓ All required fields present");
            success_count += 1;
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

    match extract_email_info(SAMPLE_EMAIL, &agent).await {
        Ok(result) => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("\nExtracted Data:");
            println!("{}", json);
            println!("\nValidation: ✓ All required fields present");
            success_count += 1;
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
