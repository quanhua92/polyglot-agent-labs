//! Polyglot Agent Labs — Use Case 03: Conversational Agent with Memory
//! A chatbot that maintains conversation history across multiple turns.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run                           # Non-interactive mode (predefined conversation)
//!   cargo run -- --interactive     # Enable interactive REPL mode

use std::env;
use std::io::Write;

use anyhow::{bail, Result};
use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::providers::{anthropic, openai, openrouter};

const SYSTEM_PROMPT: &str = "You are a helpful assistant. Be concise and friendly.";

// Predefined conversation for non-interactive mode
const PREDEFINED_CONVERSATION: &[&str] = &[
    "hello! my name is Alice.",
    "What's the weather in Tokyo?",
    "What's the weather like in London?",
    "Thanks! Goodbye!",
];

#[tokio::main]
async fn main() -> Result<()> {
    let interactive = std::env::args().any(|a| a == "--interactive");

    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    println!("=== Rust — Conversational Agent with Memory ===");
    println!("Provider:  {provider}");

    if interactive {
        println!("Mode:      interactive");
        println!("Commands: /quit, /exit, /q to end session");
    } else {
        println!("Mode:      non-interactive (predefined conversation)");
    }
    println!();

    let turn_count = match provider.as_str() {
        "openai" => run_openai(interactive).await?,
        "anthropic" => run_anthropic(interactive).await?,
        "openrouter" => run_openrouter(interactive).await?,
        _ => {
            bail!("Unknown provider: '{provider}'. Supported: openai, anthropic, openrouter");
        }
    };

    // Print session summary
    println!("{}", "=".repeat(50));
    println!("Session ended");
    println!("Total turns: {turn_count}");

    Ok(())
}

async fn run_openai(interactive: bool) -> Result<usize> {
    let model_id = "gpt-4.1-nano";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;
    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = AgentBuilder::new(model).build();
    run_chat_loop(agent, interactive).await
}

async fn run_anthropic(interactive: bool) -> Result<usize> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;
    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = AgentBuilder::new(model).build();
    run_chat_loop(agent, interactive).await
}

async fn run_openrouter(interactive: bool) -> Result<usize> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;
    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = AgentBuilder::new(model).build();
    run_chat_loop(agent, interactive).await
}

async fn run_chat_loop(agent: impl Chat, interactive: bool) -> Result<usize> {
    let mut history: Vec<Message> = Vec::new();
    let mut turn_count = 0;

    if interactive {
        // Interactive REPL loop
        loop {
            print!("You: ");
            std::io::stdout().flush()?;

            let mut input = String::new();
            match std::io::stdin().read_line(&mut input) {
                Ok(0) => {
                    // EOF
                    println!();
                    break;
                }
                Ok(_) => {}
                Err(_) => break,
            }

            let user_input = input.trim();

            // Check for exit commands
            if matches!(user_input.to_lowercase().as_str(), "/quit" | "/exit" | "/q") {
                break;
            }

            // Skip empty input
            if user_input.is_empty() {
                continue;
            }

            // Add user message to history
            history.push(Message::user(user_input.to_string()));

            match agent.chat(SYSTEM_PROMPT, history.clone()).await {
                Ok(response) => {
                    println!("Assistant: {}", response);
                    println!();

                    // Add assistant response to history
                    history.push(Message::assistant(&response));
                    turn_count += 1;
                }
                Err(e) => {
                    eprintln!("✗ Error: {e}");
                    println!();
                    // Remove the failed user message from history
                    history.pop();
                }
            }
        }
    } else {
        // Non-interactive mode with predefined conversation
        for user_input in PREDEFINED_CONVERSATION.iter() {
            println!("User: {}", user_input);

            // Add user message to history
            history.push(Message::user(user_input.to_string()));

            match agent.chat(SYSTEM_PROMPT, history.clone()).await {
                Ok(response) => {
                    println!("Assistant: {}", response);
                    println!();

                    // Add assistant response to history
                    history.push(Message::assistant(&response));
                    turn_count += 1;
                }
                Err(e) => {
                    eprintln!("✗ Error: {e}");
                    println!();
                    // Remove the failed user message from history
                    history.pop();
                    break;
                }
            }
        }
    }

    Ok(turn_count)
}
