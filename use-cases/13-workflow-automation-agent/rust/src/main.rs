//! Polyglot Agent Labs — Use Case 13: Workflow Automation Agent
//! An agent that decomposes high-level instructions into executable steps using mock tools.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run
//!
//! This demo demonstrates:
//! - Task decomposition: Breaking complex instructions into atomic steps
//! - Sequential tool execution: Executing tools in dependency order
//! - Error recovery: Handling partial failures gracefully
//! - Real-world API simulation: Mocking external service integrations
//!
//! The agent uses 4 mock tools:
//! - send_email: Send emails to recipients
//! - create_calendar_event: Schedule meetings
//! - create_task: Assign tasks to team members
//! - search_contacts: Find people in the contact database

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};

use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::providers::{anthropic, openai, openrouter};
use serde::{Deserialize, Serialize};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Contact {
    pub name: String,
    pub email: String,
    pub phone: Option<String>,
    pub department: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkflowStep {
    pub step_type: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExecutionResult {
    pub step_id: String,
    pub tool_name: String,
    pub success: bool,
    pub result: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct Decomposition {
    steps: Vec<WorkflowStep>,
}

// ============================================================================
// Mock Contact Database
// ============================================================================

fn get_contacts() -> Vec<Contact> {
    vec![
        Contact {
            name: "Alice Johnson".to_string(),
            email: "alice.johnson@company.com".to_string(),
            phone: Some("555-0101".to_string()),
            department: Some("Engineering".to_string()),
        },
        Contact {
            name: "Bob Smith".to_string(),
            email: "bob.smith@company.com".to_string(),
            phone: Some("555-0102".to_string()),
            department: Some("Product".to_string()),
        },
        Contact {
            name: "Carol Williams".to_string(),
            email: "carol.williams@company.com".to_string(),
            phone: Some("555-0103".to_string()),
            department: Some("Marketing".to_string()),
        },
        Contact {
            name: "David Brown".to_string(),
            email: "david.brown@company.com".to_string(),
            phone: Some("555-0104".to_string()),
            department: Some("Engineering".to_string()),
        },
        Contact {
            name: "Eva Martinez".to_string(),
            email: "eva.martinez@company.com".to_string(),
            phone: Some("555-0105".to_string()),
            department: Some("Sales".to_string()),
        },
    ]
}

// ============================================================================
// Mock Tools
// ============================================================================

pub async fn send_email(to: Vec<String>, subject: String, body: String) -> Result<String> {
    let mut hasher = DefaultHasher::new();
    format!("{}{}", to.join(""), subject).hash(&mut hasher);
    let email_id = hasher.finish() % 10000;

    Ok(format!(
        "✓ Email sent (ID: EMAIL-{:04}) to {} recipient(s)",
        email_id,
        to.len()
    ))
}

pub async fn create_calendar_event(
    title: String,
    date: String,
    attendees: Vec<String>,
) -> Result<String> {
    let mut hasher = DefaultHasher::new();
    format!("{}{}", title, date).hash(&mut hasher);
    let event_id = hasher.finish() % 10000;

    Ok(format!(
        "✓ Calendar event created (ID: EVT-{:04}): '{}' on {} with {} attendee(s)",
        event_id,
        title,
        date,
        attendees.len()
    ))
}

pub async fn create_task(
    title: String,
    assignee: String,
    context: String,
    due_date: String,
) -> Result<String> {
    let mut hasher = DefaultHasher::new();
    format!("{}{}", title, assignee).hash(&mut hasher);
    let task_id = hasher.finish() % 10000;

    Ok(format!(
        "✓ Task created (ID: TSK-{:04}): '{}' assigned to {}, due {}",
        task_id, title, assignee, due_date
    ))
}

pub fn search_contacts(query: &str) -> Vec<Contact> {
    let query_lower = query.to_lowercase();
    get_contacts()
        .iter()
        .filter(|c| {
            c.name.to_lowercase().contains(&query_lower)
                || c.email.to_lowercase().contains(&query_lower)
        })
        .cloned()
        .collect()
}

// ============================================================================
// Task Decomposer Prompt
// ============================================================================

const TASK_DECOMPOSER_PROMPT: &str = r#"You are a workflow planning assistant. Break down the user's instruction into specific steps.

Available tools:
- search_contacts(query): Find people by name or email
- create_calendar_event(title, date, attendees): Schedule a meeting
- send_email(to, subject, body): Send an email
- create_task(title, assignee, context, due_date): Create a task

Analyze the instruction and return a structured plan with steps in execution order.

IMPORTANT: Return ONLY valid JSON. No extra text.

Response format (valid JSON):
{
  "steps": [
    {
      "step_type": "search_contacts|create_calendar_event|send_email|create_task",
      "description": "What this step does",
      "parameters": {"param": "value"}
    }
  ]
}

Notes:
- For dates like "next Tuesday", calculate the actual date (today is {TODAY})
- For person names, you MUST search for them first using search_contacts
- After finding contacts, use their email addresses for subsequent operations
- For attendees in calendar events, use email addresses from contact search
- For task assignee, use email addresses from contact search
"#;

// ============================================================================
// JSON Extraction Helper
// ============================================================================

fn extract_json_from_response(response: &str) -> String {
    let response = response.trim();

    // Look for JSON block between ```json and ```
    if let Some(start) = response.find("```json") {
        let json_start = start + 7;
        if let Some(end) = response[json_start..].find("```") {
            return response[json_start..json_start + end].trim().to_string();
        }
    }

    // Look for JSON block between ``` and ```
    if let Some(start) = response.find("```") {
        let json_start = start + 3;
        if let Some(end) = response[json_start..].find("```") {
            let candidate = response[json_start..json_start + end].trim();
            if candidate.starts_with('{') {
                return candidate.to_string();
            }
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

// ============================================================================
// Workflow Execution
// ============================================================================

pub async fn execute_workflow(
    instruction: &str,
    agent: &impl Chat,
) -> Result<Vec<ExecutionResult>> {
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let decomposer_prompt = TASK_DECOMPOSER_PROMPT.replace("{TODAY}", &today);

    let full_prompt = format!("{}\n\nUser instruction: {}", decomposer_prompt, instruction);

    // Step 1: Decompose instruction
    let response = agent.chat(&full_prompt, vec![Message::user(instruction)]).await?;

    let decomposition_str = extract_json_from_response(&response);
    let decomposition: Decomposition = serde_json::from_str(&decomposition_str)
        .unwrap_or(Decomposition { steps: vec![] });

    // Step 2: Execute steps
    let mut results = Vec::new();
    let mut found_contacts: Vec<String> = Vec::new();

    if !decomposition.steps.is_empty() {
        for (index, step) in decomposition.steps.iter().enumerate() {
            let step_type = &step.step_type;
            let parameters = &step.parameters;

            println!("[Step {}/{}] Executing: {}", index + 1, decomposition.steps.len(), step_type);

            let result = match step_type.as_str() {
                "search_contacts" => {
                    let query = parameters
                        .get("query")
                        .and_then(|p| p.as_str())
                        .unwrap_or("");
                    let contacts = search_contacts(query);

                    if contacts.is_empty() {
                        // Create placeholder
                        let placeholder = Contact {
                            name: query.to_string(),
                            email: format!("{}.{}", query.to_lowercase().replace(' ', "."), "company.com"),
                            phone: None,
                            department: Some("Unknown".to_string()),
                        };
                        println!("  ⚠ Contact '{}' not found, using placeholder", query);
                        found_contacts.push(placeholder.email.clone());
                        Ok(format!("[PLACEHOLDER] Created placeholder for '{}': {}", query, placeholder.email))
                    } else {
                        for contact in &contacts {
                            found_contacts.push(contact.email.clone());
                        }
                        let email_list: Vec<&str> = contacts.iter().map(|c| c.email.as_str()).collect();
                        Ok(format!("Found {} contact(s): {}", contacts.len(),
                            email_list.join(", ")))
                    }
                }
                "create_calendar_event" => {
                    let title = parameters.get("title").and_then(|t| t.as_str()).unwrap_or("");
                    let date = parameters.get("date").and_then(|d| d.as_str()).unwrap_or("");
                    let attendees = parameters
                        .get("attendees")
                        .and_then(|a| a.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_else(|| found_contacts.clone());
                    create_calendar_event(title.to_string(), date.to_string(), attendees).await
                }
                "send_email" => {
                    let to = parameters
                        .get("to")
                        .and_then(|t| t.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_else(|| found_contacts.clone());
                    let subject = parameters.get("subject").and_then(|s| s.as_str()).unwrap_or("");
                    let body = parameters.get("body").and_then(|b| b.as_str()).unwrap_or("");
                    send_email(to, subject.to_string(), body.to_string()).await
                }
                "create_task" => {
                    let title = parameters.get("title").and_then(|t| t.as_str()).unwrap_or("");
                    let assignee = parameters.get("assignee").and_then(|a| a.as_str())
                        .unwrap_or_else(|| found_contacts.first().map(|s| s.as_str()).unwrap_or("unknown"));
                    let context = parameters.get("context").and_then(|c| c.as_str()).unwrap_or("");
                    let due_date = parameters.get("due_date").and_then(|d| d.as_str()).unwrap_or("");
                    create_task(title.to_string(), assignee.to_string(), context.to_string(), due_date.to_string()).await
                }
                _ => anyhow::bail!("Unknown step type: {}", step_type),
            };

            match result {
                Ok(output) => {
                    println!("  ✓ {}", output);
                    results.push(ExecutionResult {
                        step_id: format!("step-{}", index),
                        tool_name: step_type.clone(),
                        success: true,
                        result: Some(output),
                        timestamp: Utc::now(),
                    });
                }
                Err(e) => {
                    println!("  ✗ Error: {}", e);
                    results.push(ExecutionResult {
                        step_id: format!("step-{}", index),
                        tool_name: step_type.clone(),
                        success: false,
                        result: None,
                        timestamp: Utc::now(),
                    });
                }
            }
        }
    }

    Ok(results)
}

// ============================================================================
// Demo Scenarios
// ============================================================================

const DEMO_SCENARIOS: &[(&str, &str)] = &[
    (
        "Meeting Scheduling with Email",
        "Schedule a meeting with Alice and Bob next Tuesday about Q2 planning, then email them the agenda",
    ),
    (
        "Bulk Task Assignment",
        "Create a task for each team member to review the design doc by Friday. Team members: Alice, Carol, and David",
    ),
    (
        "Partial Failure Recovery",
        "Schedule a meeting with UnknownPerson and Alice about project kickoff",
    ),
];

async fn run_demo(agent: &impl Chat, provider_name: &str, model_id: &str) -> Result<()> {
    println!("=== Rust — Workflow Automation Agent ===");
    println!("Provider: {}", provider_name);
    println!("Model: {}", model_id);
    println!();

    for (i, (name, instruction)) in DEMO_SCENARIOS.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, DEMO_SCENARIOS.len(), name);
        println!("Instruction: {}", instruction);
        println!("{}", "-".repeat(60));

        match execute_workflow(instruction, agent).await {
            Ok(results) => {
                println!("\n--- Execution Summary ---");
                println!("Steps completed: {}", results.len());

                let success_count = results.iter().filter(|r| r.success).count();
                let error_count = results.iter().filter(|r| !r.success).count();
                println!("Successful: {}", success_count);
                println!("Errors: {}", error_count);

                println!("\n--- Tool Call Log ---");
                for result in &results {
                    println!("[{}] {}", result.tool_name, if result.success { "✓" } else { "✗" });
                    if let Some(ref output) = result.result {
                        println!("    {}", output);
                    }
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }

        println!("{}", "=".repeat(60));
        println!();
    }

    // Print session summary
    println!("Session Summary");
    println!("  Scenarios processed: {}", DEMO_SCENARIOS.len());

    Ok(())
}

// ============================================================================
// Provider Setup
// ============================================================================

async fn run_openai() -> Result<()> {
    let model_id = "gpt-4.1-nano";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = AgentBuilder::new(model).build();

    run_demo(&agent, "openai", model_id).await
}

async fn run_anthropic() -> Result<()> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = AgentBuilder::new(model).build();

    run_demo(&agent, "anthropic", model_id).await
}

async fn run_openrouter() -> Result<()> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = AgentBuilder::new(model).build();

    run_demo(&agent, "openrouter", model_id).await
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
