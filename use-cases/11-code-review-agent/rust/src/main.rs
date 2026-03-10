/*!
Use Case 11: Code Review Agent

Demonstrates an agent that reads source code files, analyzes them,
and provides structured code review feedback including bugs, style issues,
security concerns, and improvement suggestions using Tool API for reliable structured output.

Key Learning Goals:
- Tool API for structured output - tools accept typed parameters from LLM
- Large context handling for code analysis
- Code analysis prompting techniques
- Multi-turn tool calling for reliable extraction
*/

use std::env;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::providers::{openai, openrouter};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// =============================================================================
// SAMPLE CODE FILES (with intentional issues)
// =============================================================================

const SAMPLE_FILES: &[(&str, &str)] = &[
    (
        "insecure_login.py",
        r#"# SQL Injection vulnerability
def login(username, password):
    query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
    return db.execute(query)

# Hardcoded credentials
API_KEY = "sk-1234567890abcdef"

# Missing input validation
def reset_email(email):
    send_email(email)"#,
    ),
    (
        "poor_error_handling.py",
        r#"# Poor error handling
def divide(a, b):
    return a / b  # No zero division check

# Unused import
import os
import sys
import json

# Magic number
def calculate_discount(price):
    return price * 0.15

# Inconsistent naming
def getUserData():
    pass

def process_item(item):
    x = item['value']  # Unreadable variable name
    return x"#,
    ),
    (
        "resource_leak.py",
        r#"# Resource leak (file not closed)
def read_config(path):
    f = open(path, 'r')
    return f.read()

# Missing docstring
def process(data):
    x = []
    for i in range(len(data)):
        x.append(data[i] * 2)
    return x

# Global variable mutable default
cache = {}

def add_to_cache(key, value, cache=cache):
    cache[key] = value"#,
    ),
];

// =============================================================================
// STRUCTS FOR STRUCTURED OUTPUT
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodeReview {
    pub summary: String,
    pub overall_score: i32,
    pub issues: Vec<String>,
    pub file_count: i32,
}

// =============================================================================
// TOOL API - Structured Output Tool
// =============================================================================

#[derive(Debug, Error)]
#[error("Code review error")]
struct CodeReviewError;

#[derive(Debug, Deserialize)]
struct SubmitCodeReviewArgs {
    summary: String,
    overall_score: i32,
    issues: Vec<String>,
    file_count: i32,
}

#[derive(Debug, Serialize)]
struct CodeReviewResult(String);

struct SubmitCodeReview {
    result: Arc<Mutex<Option<CodeReview>>>,
}

impl Tool for SubmitCodeReview {
    const NAME: &'static str = "submit_code_review";

    type Error = CodeReviewError;
    type Args = SubmitCodeReviewArgs;
    type Output = CodeReviewResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "submit_code_review".to_string(),
            description: "Submit structured code review feedback".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the code quality"
                    },
                    "overall_score": {
                        "type": "integer",
                        "description": "Overall score from 0-100",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of issues found (each as a brief description)"
                    },
                    "file_count": {
                        "type": "integer",
                        "description": "Number of files reviewed"
                    }
                },
                "required": ["summary", "overall_score", "issues", "file_count"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let review = CodeReview {
            summary: args.summary,
            overall_score: args.overall_score,
            issues: args.issues,
            file_count: args.file_count,
        };

        let mut result = self.result.lock().unwrap();
        *result = Some(review.clone());

        Ok(CodeReviewResult(format!(
            "Code review submitted: score {}/100, {} issues found",
            review.overall_score,
            review.issues.len()
        )))
    }
}

// =============================================================================
// CODE REVIEWER PROMPT
// =============================================================================

const CODE_REVIEWER_PROMPT: &str = r#"You are an expert code reviewer. Analyze the provided code and give structured feedback.

Provide:
- A summary of the code quality
- An overall score from 0-100
- A list of issues found (each as a brief description)
- The number of files reviewed

Focus on: bugs, security issues, style problems, and best practices.

IMPORTANT: You MUST use the submit_code_review tool to submit your assessment."#;

// =============================================================================
// REPORT FORMATTING
// =============================================================================

fn print_review_report(review: &CodeReview, provider_name: &str, model_id: &str) {
    println!("{}", "=".repeat(60));
    println!("CODE REVIEW REPORT");
    println!("{}", "=".repeat(60));
    println!();
    println!("Provider: {}", provider_name);
    println!("Model: {}", model_id);
    println!();
    println!("Summary: {}", review.summary);
    println!("Files Reviewed: {}", review.file_count);
    println!("Overall Score: {}/100", review.overall_score);
    println!();
    println!("{}", "─".repeat(60));
    println!("ISSUES FOUND");
    println!("{}", "─".repeat(60));

    if review.issues.is_empty() {
        println!("\n✓ No issues found!");
    } else {
        for (i, issue) in review.issues.iter().enumerate() {
            println!("\n[{}] {}", i + 1, issue);
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Total Issues: {}", review.issues.len());
    println!("{}", "=".repeat(60));
}

// =============================================================================
// DEMO EXECUTION
// =============================================================================

async fn run_openrouter() -> Result<()> {
    let model_id = env::var("MODEL_ID")
        .unwrap_or_else(|_| "stepfun/step-3.5-flash:free".to_string());

    println!("Provider: openrouter");
    println!("Model: {}", model_id);
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set");
    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(&model_id);

    run_demo(model, "openrouter", &model_id).await
}

async fn run_openai() -> Result<()> {
    let model_id = env::var("MODEL_ID")
        .unwrap_or_else(|_| "gpt-4.1-nano".to_string());

    println!("Provider: openai");
    println!("Model: {}", model_id);
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY not set");
    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(&model_id);

    run_demo(model, "openai", &model_id).await
}

async fn run_demo<M>(model: M, provider_name: &str, model_id: &str) -> Result<()>
where
    M: CompletionModel + Clone + Send + Sync + 'static,
{
    let review_result: Arc<Mutex<Option<CodeReview>>> = Arc::new(Mutex::new(None));

    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(CODE_REVIEWER_PROMPT)
        .tool(SubmitCodeReview { result: review_result.clone() })
        .build();

    // Build code text
    let code_text = SAMPLE_FILES
        .iter()
        .map(|(path, content)| {
            format!(
                "\n{}\nFILE: {}\n{}\n{}",
                "=".repeat(60),
                path,
                "=".repeat(60),
                content.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "{}\n\nCODE TO REVIEW:\n{}\n\nAnalyze the code and use the submit_code_review tool to submit your assessment.",
        CODE_REVIEWER_PROMPT, code_text
    );

    println!("Analyzing code...");
    let _response = agent.prompt(&prompt).max_turns(5).await?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let review = {
        let result = review_result.lock().unwrap();
        result
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Code review tool was not called"))?
            .clone()
    };

    // Print the report
    print_review_report(&review, provider_name, model_id);

    Ok(())
}

// =============================================================================
// MAIN
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Rust — Code Review Agent ===");
    println!();

    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    match provider.as_str() {
        "openrouter" => run_openrouter().await?,
        "openai" => run_openai().await?,
        _ => {
            eprintln!("Unknown provider: '{}'. Supported: openrouter, openai", provider);
            std::process::exit(1);
        }
    }

    Ok(())
}
