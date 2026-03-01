//! Polyglot Agent Labs - Hello World (Rust)
//! Demonstrates environment variable loading for Rig agents.

use std::env;

fn main() {
    println!("=== Rust Environment Check ===");

    let openai_key = env::var("OPENAI_API_KEY");
    let anthropic_key = env::var("ANTHROPIC_API_KEY");
    let openrouter_key = env::var("OPENROUTER_API_KEY");
    let log_level = env::var("AGENT_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());

    println!(
        "OPENAI_API_KEY:     {}",
        if openai_key.is_ok() { "✓ Set" } else { "✗ Missing" }
    );
    println!(
        "ANTHROPIC_API_KEY:  {}",
        if anthropic_key.is_ok() { "✓ Set" } else { "✗ Missing" }
    );
    println!(
        "OPENROUTER_API_KEY: {}",
        if openrouter_key.is_ok() { "✓ Set" } else { "✗ Missing" }
    );
    println!("AGENT_LOG_LEVEL:    {}", log_level);

    // Example: Initialize a Rig agent
    // use rig::providers::openai;
    // let client = openai::Client::from_env();  // Uses OPENAI_API_KEY from env
    // let model = client.agent(openai::GPT_4O);
}
