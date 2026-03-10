//! Polyglot Agent Labs — Use Case 13: Workflow Automation Agent
//! An agent that uses tool calls to plan and execute workflows.
//!
//! Usage:
//!   cargo run

use std::env;

use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use rig::client::CompletionClient;
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::{anthropic, openai, openrouter};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SYSTEM_PROMPT: &str = "You are a workflow automation assistant. Use the available tools to complete the user's requests step by step.";

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub step_id: String,
    pub tool_name: String,
    pub success: bool,
    pub result: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct Contact {
    pub name: String,
    pub email: String,
}

// ============================================================================
// Tools
// ============================================================================

#[derive(Debug, Error)]
#[error("Search error")]
struct SearchError;

#[derive(Debug, Deserialize)]
struct SearchArgs {
    query: String,
}

#[derive(Debug, Serialize)]
struct SearchResult(String);

struct SearchContacts;

impl Tool for SearchContacts {
    const NAME: &'static str = "search_contacts";

    type Error = SearchError;
    type Args = SearchArgs;
    type Output = SearchResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "search_contacts".to_string(),
            description: "Search for people by name or email in the contact directory".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The person's name or email to search for"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let contacts = vec![
            Contact { name: "Alice Johnson".to_string(), email: "alice.johnson@company.com".to_string() },
            Contact { name: "Bob Smith".to_string(), email: "bob.smith@company.com".to_string() },
            Contact { name: "Carol Williams".to_string(), email: "carol.williams@company.com".to_string() },
            Contact { name: "David Brown".to_string(), email: "david.brown@company.com".to_string() },
        ];

        let query_lower = args.query.to_lowercase();
        let found: Vec<&Contact> = contacts
            .iter()
            .filter(|c| c.name.to_lowercase().contains(&query_lower) || c.email.to_lowercase().contains(&query_lower))
            .collect();

        if found.is_empty() {
            Ok(SearchResult(format!("No contacts found matching '{}'", args.query)))
        } else {
            let emails: Vec<&str> = found.iter().map(|c| c.email.as_str()).collect();
            Ok(SearchResult(format!("Found contacts: {}", emails.join(", "))))
        }
    }
}

#[derive(Debug, Error)]
#[error("Event error")]
struct EventError;

#[derive(Debug, Deserialize)]
struct EventArgs {
    title: String,
    #[serde(default)]
    date: String,
}

#[derive(Debug, Serialize)]
struct EventResult(String);

struct CreateCalendarEvent;

impl Tool for CreateCalendarEvent {
    const NAME: &'static str = "create_calendar_event";

    type Error = EventError;
    type Args = EventArgs;
    type Output = EventResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "create_calendar_event".to_string(),
            description: "Schedule a calendar event/meeting".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Meeting title"
                    },
                    "date": {
                        "type": "string",
                        "description": "Meeting date (e.g., 'next Tuesday', '2024-03-15')"
                    }
                },
                "required": ["title"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(EventResult(format!("✓ Calendar event created: '{}' on {}", args.title, args.date)))
    }
}

#[derive(Debug, Error)]
#[error("Email error")]
struct EmailError;

#[derive(Debug, Deserialize)]
struct EmailArgs {
    subject: String,
    #[serde(default)]
    recipients: String,
}

#[derive(Debug, Serialize)]
struct EmailResult(String);

struct SendEmail;

impl Tool for SendEmail {
    const NAME: &'static str = "send_email";

    type Error = EmailError;
    type Args = EmailArgs;
    type Output = EmailResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "send_email".to_string(),
            description: "Send an email to recipients".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "recipients": {
                        "type": "string",
                        "description": "Email addresses (comma-separated)"
                    }
                },
                "required": ["subject"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(EmailResult(format!("✓ Email sent: '{}' to {}", args.subject, args.recipients)))
    }
}

#[derive(Debug, Error)]
#[error("Task error")]
struct TaskError;

#[derive(Debug, Deserialize)]
struct TaskArgs {
    title: String,
    #[serde(default)]
    assignee: String,
    #[serde(default)]
    due_date: String,
}

#[derive(Debug, Serialize)]
struct TaskResult(String);

struct CreateTask;

impl Tool for CreateTask {
    const NAME: &'static str = "create_task";

    type Error = TaskError;
    type Args = TaskArgs;
    type Output = TaskResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "create_task".to_string(),
            description: "Create a task for someone".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Task title"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Person assigned to the task"
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date (e.g., 'Friday', '2024-03-15')"
                    }
                },
                "required": ["title"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(TaskResult(format!("✓ Task created: '{}' assigned to {}, due {}", args.title, args.assignee, args.due_date)))
    }
}

// ============================================================================
// Demo Scenarios
// ============================================================================

const DEMO_SCENARIOS: &[&str] = &[
    "Schedule a meeting with Alice and Bob next Tuesday about Q2 planning",
    "Create a task for Alice to review the design doc by Friday",
    "Send an email to Bob about the project update",
    "Schedule a meeting with Alice for Q2 planning and email her the agenda",
];

// ============================================================================
// Demo Execution
// ============================================================================

async fn run_demo(agent: rig::agent::Agent<impl rig::completion::CompletionModel>, provider_key: &str) -> Result<()> {
    println!("=== Rust — Workflow Automation Agent ===");
    println!("Provider: {}", provider_key);
    println!();

    for (i, instruction) in DEMO_SCENARIOS.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, DEMO_SCENARIOS.len(), instruction);
        println!("{}", "-".repeat(60));

        match agent.prompt(*instruction).max_turns(5).await {
            Ok(response) => {
                println!("Response:\n{}", response);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }

        println!("{}", "=".repeat(60));
        println!();
    }

    Ok(())
}

// ============================================================================
// Provider Setup
// ============================================================================

async fn run_openrouter() -> Result<()> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(SYSTEM_PROMPT)
        .tool(SearchContacts)
        .tool(CreateCalendarEvent)
        .tool(SendEmail)
        .tool(CreateTask)
        .build();

    run_demo(agent, "openrouter").await
}

async fn run_openai() -> Result<()> {
    let model_id = "gpt-4.1-nano";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(SYSTEM_PROMPT)
        .tool(SearchContacts)
        .tool(CreateCalendarEvent)
        .tool(SendEmail)
        .tool(CreateTask)
        .build();

    run_demo(agent, "openai").await
}

async fn run_anthropic() -> Result<()> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble(SYSTEM_PROMPT)
        .tool(SearchContacts)
        .tool(CreateCalendarEvent)
        .tool(SendEmail)
        .tool(CreateTask)
        .build();

    run_demo(agent, "anthropic").await
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
        "openai" => run_openai().await?,
        "anthropic" => run_anthropic().await?,
        "openrouter" => run_openrouter().await?,
        _ => {
            bail!("Unknown provider: '{}'. Supported: openai, anthropic, openrouter", provider);
        }
    }

    Ok(())
}
