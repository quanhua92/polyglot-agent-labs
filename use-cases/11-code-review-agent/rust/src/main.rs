/*!
Use Case 11: Code Review Agent

Demonstrates an agent that reads source code files, analyzes them,
and provides structured code review feedback including bugs, style issues,
security concerns, and improvement suggestions.

Key Learning Goals:
- File I/O tools for reading source files
- Structured output for code review findings
- Large context handling for code analysis
- Code analysis prompting techniques
*/

use anyhow::Result;
use regex::Regex;
use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::providers::{openai, openrouter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
pub struct ReviewFinding {
    pub severity: String,
    pub category: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_number: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_snippet: Option<String>,
    pub message: String,
    pub suggestion: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodeReview {
    pub summary: String,
    pub findings: Vec<ReviewFinding>,
    pub overall_score: i32,
    pub file_count: i32,
    pub lines_reviewed: i32,
}

// =============================================================================
// CODE REVIEWER PROMPT
// =============================================================================

const CODE_REVIEWER_PROMPT: &str = r#"You are an expert code reviewer. Analyze the provided code and give structured feedback.

For each issue found, provide:
- severity: low, medium, high, or critical
- category: bugs, security, style, or best_practices
- line_number: The line where the issue occurs
- code_snippet: The relevant code snippet
- message: Clear description of the issue
- suggestion: Specific recommendation for fixing it

Scoring guidelines:
- Start at 100
- Critical: -20 points each
- High: -10 points each
- Medium: -5 points each
- Low: -2 points each

Provide your response as valid JSON matching this schema:
{
  "summary": "Overall assessment",
  "findings": [
    {
      "severity": "level",
      "category": "type",
      "line_number": number,
      "code_snippet": "code",
      "message": "description",
      "suggestion": "fix"
    }
  ],
  "overall_score": number,
  "file_count": number,
  "lines_reviewed": number
}"#;

// =============================================================================
// CODE REVIEW FUNCTION
// =============================================================================

fn extract_json_from_response(response: &str) -> String {
    // Extract JSON from a response that may contain extra text
    let response = response.trim();

    // Look for JSON block between ```json and ```
    let json_block_regex = Regex::new(r"```json\s*(.*?)\s*```").unwrap();
    if let Some(captures) = json_block_regex.captures(response) {
        return captures.get(1).unwrap().as_str().trim().to_string();
    }

    // Look for JSON block between ``` and ```
    let code_block_regex = Regex::new(r"```\s*(.*?)\s*```").unwrap();
    if let Some(captures) = code_block_regex.captures(response) {
        let candidate = captures.get(1).unwrap().as_str().trim();
        if candidate.starts_with('{') {
            return candidate.to_string();
        }
    }

    // Look for { and } as JSON boundaries
    if let Some(start) = response.find('{') {
        let mut brace_count = 0;
        for (i, ch) in response[start..].char_indices() {
            match ch {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        return response[start..start + i + 1].to_string();
                    }
                }
                _ => {}
            }
        }
    }

    response.to_string()
}

async fn review_code<T: Chat>(
    code_contents: &[(String, String)],
    agent: &T,
) -> Result<CodeReview> {
    let code_text = code_contents
        .iter()
        .map(|(path, content)| {
            format!(
                "\n{}\nFILE: {}\n{}\n{}",
                "=".repeat(60),
                path,
                "=".repeat(60),
                content
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let user_prompt = format!(
        "CODE TO REVIEW:\n{}\n\nAnalyze this code and return valid JSON.",
        code_text
    );

    let response = agent.chat(CODE_REVIEWER_PROMPT, vec![Message::user(&user_prompt)]).await?;
    let json_str = extract_json_from_response(&response);
    let parsed: CodeReview = serde_json::from_str(&json_str)?;

    Ok(parsed)
}

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
    println!("Lines Reviewed: {}", review.lines_reviewed);
    println!("Overall Score: {}/100", review.overall_score);
    println!();
    println!("{}", "─".repeat(60));
    println!("FINDINGS");
    println!("{}", "─".repeat(60));

    if review.findings.is_empty() {
        println!("\n✓ No issues found!");
    } else {
        // Group by severity
        let severity_order = ["critical", "high", "medium", "low"];
        let mut grouped: HashMap<&str, Vec<&ReviewFinding>> =
            severity_order.iter().map(|&s| (s, Vec::new())).collect();

        for finding in &review.findings {
            let sev = finding.severity.to_lowercase();
            if let Some(bucket) = grouped.get_mut(sev.as_str()) {
                bucket.push(finding);
            }
        }

        for severity in severity_order {
            if let Some(findings) = grouped.get(severity) {
                for (i, finding) in findings.iter().enumerate() {
                    println!(
                        "\n[{}] {} | {} | Line {}",
                        i + 1,
                        severity.to_uppercase(),
                        finding.category,
                        finding.line_number.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string())
                    );
                    println!("    {}", finding.message);

                    if let Some(snippet) = &finding.code_snippet {
                        let truncated = if snippet.len() > 60 {
                            format!("{}...", &snippet[..60])
                        } else {
                            snippet.clone()
                        };
                        println!("    Code: {}", truncated);
                    }

                    println!("    → {}", finding.suggestion);
                }
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Total Issues: {}", review.findings.len());
    println!("{}", "=".repeat(60));
}

// =============================================================================
// DEMO EXECUTION
// =============================================================================

async fn run_openrouter() -> Result<()> {
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "stepfun/step-3.5-flash:free".to_string());

    println!("Provider: openrouter");
    println!("Model: {}", model_id);
    println!();

    let api_key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY not set");
    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(&model_id);
    let agent = AgentBuilder::new(model).build();

    // Prepare sample code contents
    let code_contents: Vec<(String, String)> = SAMPLE_FILES
        .iter()
        .map(|(name, content)| (name.to_string(), content.trim().to_string()))
        .collect();

    // Perform code review
    let review = review_code(&code_contents, &agent).await?;

    // Print the report
    print_review_report(&review, "openrouter", &model_id);

    Ok(())
}

async fn run_openai() -> Result<()> {
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "gpt-4.1-nano".to_string());

    println!("Provider: openai");
    println!("Model: {}", model_id);
    println!();

    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY not set");
    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(&model_id);
    let agent = AgentBuilder::new(model).build();

    // Prepare sample code contents
    let code_contents: Vec<(String, String)> = SAMPLE_FILES
        .iter()
        .map(|(name, content)| (name.to_string(), content.trim().to_string()))
        .collect();

    // Perform code review
    let review = review_code(&code_contents, &agent).await?;

    // Print the report
    print_review_report(&review, "openai", &model_id);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Rust — Code Review Agent ===");
    println!();

    let provider = std::env::var("LLM_PROVIDER")
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
