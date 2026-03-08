//! Polyglot Agent Labs — Use Case 01: Simple LLM Completion (Multi-Provider)
//! Sends a prompt to an LLM via rig-core, supporting OpenAI, Anthropic, and OpenRouter.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).

use std::env;

use anyhow::{bail, Result};
use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::{anthropic, openai, openrouter};

const PROMPT_TEXT: &str = "Hello! Tell me a fun fact about programming.";

#[tokio::main]
async fn main() -> Result<()> {
    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    println!("=== Rust — Simple LLM Completion ===");
    println!("Provider:  {provider}");

    let response = match provider.as_str() {
        "openai" => {
            let model_id = "gpt-4.1-nano";
            println!("Model:     {model_id}");
            println!();
            let api_key = env::var("OPENAI_API_KEY")
                .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;
            let client = openai::Client::new(&api_key)?;
            let model = client.completion_model(model_id);
            let agent = AgentBuilder::new(model).build();
            agent.prompt(PROMPT_TEXT).await?
        }
        "anthropic" => {
            let model_id = "claude-3-haiku-20240307";
            println!("Model:     {model_id}");
            println!();
            let api_key = env::var("ANTHROPIC_API_KEY")
                .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;
            let client = anthropic::Client::new(&api_key)?;
            let model = client.completion_model(model_id);
            let agent = AgentBuilder::new(model).build();
            agent.prompt(PROMPT_TEXT).await?
        }
        "openrouter" => {
            let model_id = "stepfun/step-3.5-flash:free";
            println!("Model:     {model_id}");
            println!();
            let api_key = env::var("OPENROUTER_API_KEY")
                .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;
            let client = openrouter::Client::new(&api_key)?;
            let model = client.completion_model(model_id);
            let agent = AgentBuilder::new(model).build();
            agent.prompt(PROMPT_TEXT).await?
        }
        _ => {
            bail!("Unknown provider: '{provider}'. Supported: openai, anthropic, openrouter");
        }
    };

    println!("Prompt:    {PROMPT_TEXT}");
    println!("Response:  {response}");

    Ok(())
}
