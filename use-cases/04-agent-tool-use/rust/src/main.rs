//! Polyglot Agent Labs — Use Case 04: Agent with Tool Use (Function Calling)
//! An agent that can decide when to call external tools to answer user questions.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run
//!
//! The agent demonstrates function calling with 3 tools:
//! - calculator: Evaluate mathematical expressions
//! - get_current_time: Get current date/time
//! - string_length: Count characters in a string

use std::env;

use anyhow::{bail, Result};
use chrono::Local;
use rig::client::CompletionClient;
use rig::completion::{Chat, Message, ToolDefinition};
use rig::providers::{anthropic, openai, openrouter};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SYSTEM_PROMPT: &str = "You are a helpful assistant with access to tools. Use tools when needed to answer questions accurately.";

// Demo prompts that should trigger tool calls
const DEMO_PROMPTS: &[&str] = &[
    "What is 42 * 137?",
    "What time is it right now?",
    "How many characters in 'Polyglot Agent Labs'?",
];

// ============================================================================
// Tool Definitions
// ============================================================================

#[derive(Debug, Error)]
#[error("Calculator error")]
struct CalculatorError;

#[derive(Debug, Deserialize)]
struct CalculatorArgs {
    expression: String,
}

#[derive(Debug, Serialize)]
struct CalculatorOutput(String);

struct Calculator;

impl Tool for Calculator {
    const NAME: &'static str = "calculator";

    type Error = CalculatorError;
    type Args = CalculatorArgs;
    type Output = CalculatorOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".to_string(),
            description: "Evaluate a mathematical expression like '42 * 137'".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression like '42 * 137' or '2 + 2'"
                    }
                },
                "required": ["expression"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(CalculatorOutput(evaluate_expression(&args.expression)))
    }
}

#[derive(Debug, Error)]
#[error("Time error")]
struct TimeError;

#[derive(Debug, Deserialize)]
struct TimeArgs {}

#[derive(Debug, Serialize)]
struct TimeOutput(String);

struct GetCurrentTime;

impl Tool for GetCurrentTime {
    const NAME: &'static str = "get_current_time";

    type Error = TimeError;
    type Args = TimeArgs;
    type Output = TimeOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_current_time".to_string(),
            description: "Get the current date and time".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(TimeOutput(Local::now().format("%Y-%m-%d %H:%M:%S").to_string()))
    }
}

#[derive(Debug, Error)]
#[error("String error")]
struct StringError;

#[derive(Debug, Deserialize)]
struct StringLengthArgs {
    text: String,
}

#[derive(Debug, Serialize)]
struct StringLengthOutput(i32);

struct StringLength;

impl Tool for StringLength {
    const NAME: &'static str = "string_length";

    type Error = StringError;
    type Args = StringLengthArgs;
    type Output = StringLengthOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "string_length".to_string(),
            description: "Get the length of a string".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to measure"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(StringLengthOutput(args.text.len() as i32))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Safely evaluate a simple mathematical expression
fn evaluate_expression(expr: &str) -> String {
    let expr = expr.trim();

    // Simple parser for basic arithmetic
    if expr.contains('*') {
        let parts: Vec<&str> = expr.split('*').collect();
        if parts.len() == 2 {
            let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
            let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
            return (a * b).to_string();
        }
    } else if expr.contains('+') {
        let parts: Vec<&str> = expr.split('+').collect();
        if parts.len() == 2 {
            let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
            let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
            return (a + b).to_string();
        }
    } else if expr.contains('-') {
        let parts: Vec<&str> = expr.split('-').collect();
        if parts.len() == 2 {
            let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
            let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
            return (a - b).to_string();
        }
    } else if expr.contains('/') {
        let parts: Vec<&str> = expr.split('/').collect();
        if parts.len() == 2 {
            let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
            let b: f64 = parts[1].trim().parse().unwrap_or(1.0);
            if b == 0.0 {
                return "Error: Division by zero".to_string();
            }
            return (a / b).to_string();
        }
    }

    // Try to parse as a single number
    if let Ok(n) = expr.parse::<f64>() {
        return n.to_string();
    }

    format!("Error: Cannot evaluate '{}'", expr)
}

// ============================================================================
// Provider Setup
// ============================================================================

async fn run_openai() -> Result<usize> {
    let model_id = "gpt-4.1-nano";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(SYSTEM_PROMPT)
        .tool(Calculator)
        .tool(GetCurrentTime)
        .tool(StringLength)
        .build();

    run_demo(agent, "openai").await
}

async fn run_anthropic() -> Result<usize> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(SYSTEM_PROMPT)
        .tool(Calculator)
        .tool(GetCurrentTime)
        .tool(StringLength)
        .build();

    run_demo(agent, "anthropic").await
}

async fn run_openrouter() -> Result<usize> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(SYSTEM_PROMPT)
        .tool(Calculator)
        .tool(GetCurrentTime)
        .tool(StringLength)
        .build();

    run_demo(agent, "openrouter").await
}

// ============================================================================
// Demo Execution
// ============================================================================

async fn run_demo(agent: impl Chat, provider_key: &str) -> Result<usize> {
    println!("=== Rust — Agent with Tool Use ===");
    println!("Provider:  {provider_key}");
    println!();

    for (i, prompt) in DEMO_PROMPTS.iter().enumerate() {
        println!("[{}/{}] Question: {}", i + 1, DEMO_PROMPTS.len(), prompt);
        println!("{}", "-".repeat(50));

        let history = vec![Message::user(prompt.to_string())];
        match agent.chat(SYSTEM_PROMPT, history).await {
            Ok(response) => {
                println!("Response: {}", response);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }

        println!("{}", "=".repeat(50));
        println!();
    }

    // Print session summary
    println!("Session Summary");
    println!("  Prompts processed: {}", DEMO_PROMPTS.len());

    Ok(DEMO_PROMPTS.len())
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    let _turn_count = match provider.as_str() {
        "openai" => run_openai().await?,
        "anthropic" => run_anthropic().await?,
        "openrouter" => run_openrouter().await?,
        _ => {
            bail!("Unknown provider: '{}'. Supported: openai, anthropic, openrouter", provider);
        }
    };

    Ok(())
}
