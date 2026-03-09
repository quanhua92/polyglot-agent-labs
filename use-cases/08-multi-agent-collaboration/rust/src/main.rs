//! Polyglot Agent Labs — Use Case 08: Multi-Agent Collaboration (Researcher + Writer)
//! Demonstrates multi-agent collaboration where two specialized agents work together:
//! - Researcher: Uses ReAct loop to autonomously search and gather information
//! - Writer: Transforms research findings into a polished article
//!
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run
//!
//! Key Learning Goals:
//! - Agent composition and orchestration
//! - Role specialization with system prompts
//! - Inter-agent communication and data passing
//! - Multi-step workflows with manual chaining
//! - Tool use for knowledge base access

use std::env;

use anyhow::{bail, Result};
use rig::client::CompletionClient;
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::{anthropic, openai, openrouter};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Constants
// ============================================================================

const DEMO_TOPIC: &str = "Explain the benefits of Rust for systems programming";
const MAX_RESEARCH_ITERATIONS: usize = 5;

// ============================================================================
// Knowledge Base (Hard-coded Documents about Rust)
// ============================================================================

const KNOWLEDGE_BASE: &[(&str, &str)] = &[
    ("rust-performance", r#"## Rust Performance Characteristics

Rust provides zero-cost abstractions, memory safety without garbage collection, and predictable performance. This makes it ideal for systems programming where performance is critical.

Key performance benefits:
- No runtime garbage collector means consistent latency
- LLVM-based compiler generates optimized machine code
- Zero-cost abstractions allow high-level programming without performance penalty
- Efficient memory layout control with explicit lifetime management
- Compile-time optimizations eliminate runtime overhead

Rust's performance profile is comparable to C++ while providing stronger safety guarantees."#),
    ("rust-memory-safety", r#"## Rust Memory Safety Guarantees

Rust's ownership model ensures memory safety at compile time, eliminating entire classes of bugs that plague systems programming in C and C++.

Memory safety features:
- Ownership rules prevent data races at compile time
- Borrow checker ensures references are always valid
- No null pointers (Option<T> instead of null)
- No buffer overflows with bounds-checked slices
- No use-after-free errors due to ownership semantics
- No dangling pointers through lifetime annotations

The compiler enforces these rules, meaning memory safety issues are caught before code ever runs."#),
    ("rust-concurrency", r#"## Rust Concurrency Benefits

Rust makes concurrent programming safer and easier through its ownership system. The same rules that prevent memory errors also prevent data races.

Concurrency advantages:
- "Fearless concurrency" - compiler prevents data races
- Send and Sync traits mark thread-safe types
- Message passing with channels (mpsc, rpc)
- Async/await with tokio runtime for efficient I/O
- No data race guarantees at compile time

The ownership system ensures that either:
1. Only one thread can access data (mutable reference)
2. Multiple threads can read data (immutable references)"#),
    ("rust-ecosystem", r#"## Rust Ecosystem and Tooling

Rust has a modern, developer-friendly ecosystem that enhances productivity.

Key ecosystem features:
- Cargo: Integrated package manager and build system
- Crates.io: Central package registry with 100k+ packages
- rustfmt: Consistent code formatting
- clippy: Linter for catching common mistakes
- rustdoc: Documentation generator from code comments
- Excellent IDE support via rust-analyzer

The tooling "just works" out of the box, eliminating configuration headaches common in C++ projects."#),
    ("rust-use-cases", r#"## Rust Use Cases in Systems Programming

Rust is increasingly being adopted for systems programming across diverse domains.

Major use cases:
- Operating systems: Redox OS, components of Windows and Linux
- Embedded systems: Firmware, microcontrollers, IoT devices
- Network services: High-performance servers and proxies
- Blockchain: Solana, Polkadot, and many others
- WebAssembly: Compile to Wasm for browser and serverless
- CLI tools: ripgrep, bat, exa, and many replacements for GNU tools
- Database engines: TiKV, Detox, and database drivers

Companies using Rust include Microsoft, Amazon, Google, Mozilla, Dropbox, Cloudflare, and many more."#),
    ("rust-learning-curve", r#"## Rust Learning Curve Considerations

Rust has a steeper learning curve than many languages, but the investment pays off in code quality.

Learning challenges:
- Ownership, borrowing, and lifetimes are unique concepts
- No automatic garbage collection requires thinking about memory
- Pattern matching and algebraic data types (enums)
- Error handling with Result<T, E> instead of exceptions

However, once these concepts click:
- Debugging time decreases dramatically
- Refactoring becomes safer
- Code reviews focus on logic, not memory safety
- Production incidents related to memory vanish

Most developers report 2-3 months to become productive in Rust."#),
];

// ============================================================================
// Tool Definitions
// ============================================================================

#[derive(Debug, Error)]
#[error("Search error")]
struct SearchError;

#[derive(Debug, Deserialize)]
struct SearchArgs {
    query: String,
}

#[derive(Debug, Serialize)]
struct SearchOutput(String);

struct SearchNotes;

impl Tool for SearchNotes {
    const NAME: &'static str = "search_notes";

    type Error = SearchError;
    type Args = SearchArgs;
    type Output = SearchOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "search_notes".to_string(),
            description: "Search the knowledge base for information about Rust systems programming. Use specific keywords like 'performance', 'safety', 'concurrency', 'ecosystem', or 'use cases' to find relevant documents.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - use keywords like 'performance', 'safety', 'concurrency', 'ecosystem', or 'use cases'"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let query_lower = args.query.to_lowercase();

        // Build a keyword map to match queries to specific documents
        let keywords = [
            ("performance", "rust-performance"),
            ("speed", "rust-performance"),
            ("latency", "rust-performance"),
            ("safety", "rust-memory-safety"),
            ("memory", "rust-memory-safety"),
            ("security", "rust-memory-safety"),
            ("concurrency", "rust-concurrency"),
            ("parallel", "rust-concurrency"),
            ("thread", "rust-concurrency"),
            ("ecosystem", "rust-ecosystem"),
            ("tooling", "rust-ecosystem"),
            ("cargo", "rust-ecosystem"),
            ("use case", "rust-use-cases"),
            ("company", "rust-use-cases"),
            ("production", "rust-use-cases"),
            ("learning", "rust-learning-curve"),
            ("beginner", "rust-learning-curve"),
            ("curve", "rust-learning-curve"),
        ];

        // Find matching document based on keywords
        for (keyword, doc_id) in keywords.iter() {
            if query_lower.contains(keyword) {
                if let Some((_, content)) = KNOWLEDGE_BASE.iter().find(|(id, _)| id == doc_id) {
                    return Ok(SearchOutput(content.to_string()));
                }
            }
        }

        // Default: return first document for general queries
        Ok(SearchOutput(KNOWLEDGE_BASE[0].1.to_string()))
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

    println!("=== Rust — Multi-Agent Collaboration (ReAct Researcher + Writer) ===");
    println!("Provider:  openai");
    println!();
    println!("Topic: {}", DEMO_TOPIC);
    println!("{}", "-".repeat(50));

    // Research phase
    println!("\n[Research Phase - Agent with Tool Use]");

    let research_system = "You are a research assistant. Your task is to thoroughly research the topic using the search tool.

Use the search_notes tool to find relevant information. Each search returns one document, so you may need to search multiple times with different queries to gather comprehensive information covering:
- Performance characteristics
- Memory safety features
- Concurrency benefits
- Ecosystem and tooling
- Real-world use cases

When you have gathered sufficient information (3-5 key facts from different documents), provide a summary of your findings.";

    let agent = rig::agent::AgentBuilder::new(model.clone())
        .preamble(research_system)
        .tool(SearchNotes)
        .build();

    let research_prompt = format!("Research this topic: {DEMO_TOPIC}");

    println!("  [Agent calling tools automatically with max_turns({})]", MAX_RESEARCH_ITERATIONS);

    // Use the .prompt().max_turns().await pattern for rig-core 0.32
    let research_response = agent
        .prompt(&research_prompt)
        .max_turns(MAX_RESEARCH_ITERATIONS)
        .await?;

    println!("    ✓ Research complete");

    println!("\n--- Research Findings ---");
    let research_preview = if research_response.len() > 300 {
        format!("{}...", &research_response[..300])
    } else {
        research_response.clone()
    };
    println!("{}", research_preview);

    // Writing phase
    println!("\n[Writing Phase]");
    println!("{}", "-".repeat(50));

    let writer_system = "You are a technical writer. Your task is to turn research findings into a clear, well-structured markdown article.";
    let writer_agent = rig::agent::AgentBuilder::new(model)
        .preamble(writer_system)
        .build();

    let writer_prompt = format!(
        "Topic: {DEMO_TOPIC}

Research Findings:
{research_response}

Write a comprehensive article that:
1. Has a compelling title (use # heading)
2. Starts with an engaging introduction
3. Covers 3-5 main sections with proper ## headings
4. Each section is based on the research findings above
5. Has a conclusion that summarizes key takeaways
6. Is formatted in markdown with proper headings and structure

Return ONLY the markdown article, no additional commentary."
    );

    let article = writer_agent.prompt(&writer_prompt).await?;
    println!("{}", article);

    let word_count = article.split_whitespace().count();
    println!("\n--- Session Summary ---");
    println!("Research agent: Used search_notes tool with max_turns({})", MAX_RESEARCH_ITERATIONS);
    println!("Article word count: {}", word_count);

    Ok((1, word_count))
}

async fn run_anthropic() -> Result<(usize, usize)> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    println!("=== Rust — Multi-Agent Collaboration (ReAct Researcher + Writer) ===");
    println!("Provider:  anthropic");
    println!();
    println!("Topic: {}", DEMO_TOPIC);
    println!("{}", "-".repeat(50));

    // Research phase
    println!("\n[Research Phase - Agent with Tool Use]");

    let research_system = "You are a research assistant. Your task is to thoroughly research the topic using the search tool.

Use the search_notes tool to find relevant information. Each search returns one document, so you may need to search multiple times with different queries to gather comprehensive information covering:
- Performance characteristics
- Memory safety features
- Concurrency benefits
- Ecosystem and tooling
- Real-world use cases

When you have gathered sufficient information (3-5 key facts from different documents), provide a summary of your findings.";

    let agent = rig::agent::AgentBuilder::new(model.clone())
        .preamble(research_system)
        .tool(SearchNotes)
        .build();

    let research_prompt = format!("Research this topic: {DEMO_TOPIC}");

    println!("  [Agent calling tools automatically with max_turns({})]", MAX_RESEARCH_ITERATIONS);

    // Use the .prompt().max_turns().await pattern for rig-core 0.32
    let research_response = agent
        .prompt(&research_prompt)
        .max_turns(MAX_RESEARCH_ITERATIONS)
        .await?;

    println!("    ✓ Research complete");

    println!("\n--- Research Findings ---");
    let research_preview = if research_response.len() > 300 {
        format!("{}...", &research_response[..300])
    } else {
        research_response.clone()
    };
    println!("{}", research_preview);

    // Writing phase
    println!("\n[Writing Phase]");
    println!("{}", "-".repeat(50));

    let writer_system = "You are a technical writer. Your task is to turn research findings into a clear, well-structured markdown article.";
    let writer_agent = rig::agent::AgentBuilder::new(model)
        .preamble(writer_system)
        .build();

    let writer_prompt = format!(
        "Topic: {DEMO_TOPIC}

Research Findings:
{research_response}

Write a comprehensive article that:
1. Has a compelling title (use # heading)
2. Starts with an engaging introduction
3. Covers 3-5 main sections with proper ## headings
4. Each section is based on the research findings above
5. Has a conclusion that summarizes key takeaways
6. Is formatted in markdown with proper headings and structure

Return ONLY the markdown article, no additional commentary."
    );

    let article = writer_agent.prompt(&writer_prompt).await?;
    println!("{}", article);

    let word_count = article.split_whitespace().count();
    println!("\n--- Session Summary ---");
    println!("Research agent: Used search_notes tool with max_turns({})", MAX_RESEARCH_ITERATIONS);
    println!("Article word count: {}", word_count);

    Ok((1, word_count))
}

async fn run_openrouter() -> Result<(usize, usize)> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);

    println!("=== Rust — Multi-Agent Collaboration (ReAct Researcher + Writer) ===");
    println!("Provider:  openrouter");
    println!();
    println!("Topic: {}", DEMO_TOPIC);
    println!("{}", "-".repeat(50));

    // Research phase
    println!("\n[Research Phase - Agent with Tool Use]");

    let research_system = "You are a research assistant. Your task is to thoroughly research the topic using the search tool.

Use the search_notes tool to find relevant information. Each search returns one document, so you may need to search multiple times with different queries to gather comprehensive information covering:
- Performance characteristics
- Memory safety features
- Concurrency benefits
- Ecosystem and tooling
- Real-world use cases

When you have gathered sufficient information (3-5 key facts from different documents), provide a summary of your findings.";

    let agent = rig::agent::AgentBuilder::new(model.clone())
        .preamble(research_system)
        .tool(SearchNotes)
        .build();

    let research_prompt = format!("Research this topic: {DEMO_TOPIC}");

    println!("  [Agent calling tools automatically with max_turns({})]", MAX_RESEARCH_ITERATIONS);

    // Use the .prompt().max_turns().await pattern for rig-core 0.32
    let research_response = agent
        .prompt(&research_prompt)
        .max_turns(MAX_RESEARCH_ITERATIONS)
        .await?;

    println!("    ✓ Research complete");

    println!("\n--- Research Findings ---");
    let research_preview = if research_response.len() > 300 {
        format!("{}...", &research_response[..300])
    } else {
        research_response.clone()
    };
    println!("{}", research_preview);

    // Writing phase
    println!("\n[Writing Phase]");
    println!("{}", "-".repeat(50));

    let writer_system = "You are a technical writer. Your task is to turn research findings into a clear, well-structured markdown article.";
    let writer_agent = rig::agent::AgentBuilder::new(model)
        .preamble(writer_system)
        .build();

    let writer_prompt = format!(
        "Topic: {DEMO_TOPIC}

Research Findings:
{research_response}

Write a comprehensive article that:
1. Has a compelling title (use # heading)
2. Starts with an engaging introduction
3. Covers 3-5 main sections with proper ## headings
4. Each section is based on the research findings above
5. Has a conclusion that summarizes key takeaways
6. Is formatted in markdown with proper headings and structure

Return ONLY the markdown article, no additional commentary."
    );

    let article = writer_agent.prompt(&writer_prompt).await?;
    println!("{}", article);

    let word_count = article.split_whitespace().count();
    println!("\n--- Session Summary ---");
    println!("Research agent: Used search_notes tool with max_turns({})", MAX_RESEARCH_ITERATIONS);
    println!("Article word count: {}", word_count);

    Ok((1, word_count))
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    let (_iterations, _word_count) = match provider.as_str() {
        "openai" => run_openai().await?,
        "anthropic" => run_anthropic().await?,
        "openrouter" => run_openrouter().await?,
        _ => {
            bail!("Unknown provider: '{}'. Supported: openai, anthropic, openrouter", provider);
        }
    };

    Ok(())
}
