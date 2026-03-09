//! Polyglot Agent Labs — Use Case 05: RAG over Local Documents
//! Answer questions by retrieving relevant context from a knowledge base of documents.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run

use std::env;

use anyhow::{bail, Result};
use reqwest::Client;
use rig::client::CompletionClient;
use rig::completion::Chat;
use rig::providers::{anthropic, openai, openrouter};
use serde::{Deserialize, Serialize};

// ============================================================================
// Hard-coded Documents
// ============================================================================

// Documents as raw string literals
const FAQ_DOC: &str = r#"# Polyglot Agent Labs FAQ

## What is Polyglot Agent Labs?

Polyglot Agent Labs is a progressive roadmap of 16 use cases for building AI agents,
with each use case implemented in both Python and Rust side-by-side. The project
demonstrates how to build practical AI systems using modern frameworks and patterns.

## What are the main learning goals?

The project covers:
- Simple LLM completion and multi-provider support
- MCP (Model Context Protocol) servers
- Conversational agents with memory
- Tool use and function calling
- RAG (Retrieval-Augmented Generation)
- Structured output and data extraction
- Streaming responses and human-in-the-loop workflows
- Multi-agent collaboration and research agents
- Customer support agents with knowledge bases
- Code review and analysis agents
- Data analysis with CSV/SQL
- Workflow automation and content generation

## How do I get started?

Start with use case 00 (environment check), then progress through each use case
sequentially. Each use case builds upon concepts from previous ones.

## Why Python and Rust?

Python excels at rapid development and has the richest AI/ML ecosystem. Rust provides
memory safety, performance, and modern async capabilities. Learning both gives you
versatility in AI engineering.
"#;

const RUST_GUIDELINES_DOC: &str = r#"# Rust Coding Guidelines

## Naming Conventions

- **Structs**: Use `PascalCase` (e.g., `ChatMessage`, `ToolDefinition`)
- **Functions**: Use `snake_case` (e.g., `chat_prompt`, `tool_call`)
- **Constants**: Use `SCREAMING_SNAKE_CASE` (e.g., `MAX_TOKENS`, `DEFAULT_MODEL`)
- **Modules**: Use `snake_case` for file and module names

## Error Handling

- Use `anyhow::Result<T>` for application errors
- Use `thiserror` to define custom error types for libraries
- Prefer `?` operator over explicit match for error propagation
- Provide context with `.context()` or `.with_context()`

## Async Patterns

- Use `tokio` as the async runtime
- Mark async functions with `async fn`
- Use `.await` to call async functions
- Prefer async I/O over blocking operations

## Code Organization

- Keep main.rs focused on setup and orchestration
- Extract business logic into separate modules
- Use builder patterns for complex configuration (e.g., `AgentBuilder`)

## Style Tips

- Prefer `&str` over `String` for function arguments when ownership isn't needed
- Use `format!` for string building over concatenation
- Leverage rustfmt for consistent formatting
- Use clippy to catch common mistakes
"#;

const PYTHON_PRACTICES_DOC: &str = r#"# Python Best Practices for LangChain

## Error Handling

- Always handle API errors gracefully with try/except blocks
- Use specific exception types when possible
- Provide helpful error messages for API key issues
- Log errors for debugging while showing user-friendly messages

## LangChain Patterns

- Use `@tool` decorator for function calling tools
- Use `TypedDict` for state in LangGraph workflows
- Prefer `HumanMessage`, `AIMessage`, `SystemMessage` over raw strings
- Use `bind_tools()` to attach tools to models

## Memory Management

- Keep conversation history in memory for simple applications
- Use `ConversationBufferMemory` for persistent history
- Be mindful of token limits - summarize old messages when needed

## Provider Configuration

- Use environment variables for API keys
- Support multiple providers (OpenAI, Anthropic, OpenRouter)
- Provide clear error messages when API keys are missing
- Default to a free/low-cost option for development

## Type Hints

- Always add type hints to function signatures
- Use `from typing import ...` for generic types
- Return specific types, not `Any` when possible
- Use `|` syntax for unions in Python 3.12+
"#;

// Demo questions
const DEMO_QUESTIONS: &[&str] = &[
    "What are the main learning goals of Polyglot Agent Labs?",
    "What naming conventions should I follow in Rust?",
    "How should I handle errors in LangChain?",
    "What's the weather in Tokyo?",  // Edge case: not in docs
];

// Document structure
#[derive(Debug, Clone)]
struct Document {
    id: String,
    title: String,
    content: String,
}

// ============================================================================
// Embeddings API
// ============================================================================

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

async fn get_embeddings(texts: &[String], api_key: &str, base_url: &str) -> Result<Vec<Vec<f32>>> {
    let client = Client::new();
    let url = format!("{}/embeddings", base_url);

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("HTTP-Referer", "https://polyglot-agent-labs.com")
        .header("X-Title", "Polyglot Agent Labs")
        .json(&EmbeddingRequest {
            model: "text-embedding-3-small".to_string(),
            input: texts.to_vec(),
        })
        .send()
        .await?
        .error_for_status()?
        .json::<EmbeddingResponse>()
        .await?;

    Ok(response.data.into_iter().map(|d| d.embedding).collect())
}

// ============================================================================
// Vector Store
// ============================================================================

#[derive(Debug, Clone)]
struct DocumentChunk {
    source: String,
    title: String,
    content: String,
    embedding: Vec<f32>,
}

struct VectorStore {
    chunks: Vec<DocumentChunk>,
}

impl VectorStore {
    fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    fn add(&mut self, chunk: DocumentChunk) {
        self.chunks.push(chunk);
    }

    fn search(&self, query_embedding: &[f32], k: usize) -> Vec<&DocumentChunk> {
        let mut results: Vec<_> = self
            .chunks
            .iter()
            .map(|chunk| {
                let similarity = cosine_similarity(query_embedding, &chunk.embedding);
                (chunk, similarity)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(k).map(|(c, _)| c).collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

// ============================================================================
// Document Chunking
// ============================================================================

fn chunk_document(doc: &Document, chunk_size: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = doc
        .content
        .split("\n\n")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for para in paragraphs {
        if current_chunk.len() + para.len() > chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.clone());
            current_chunk = String::new();
        }
        current_chunk.push_str(para);
        current_chunk.push_str("\n\n");
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

// ============================================================================
// RAG Pipeline
// ============================================================================

async fn rag_query(
    agent: &impl Chat,
    vector_store: &VectorStore,
    api_key: &str,
    base_url: &str,
    question: &str,
) -> Result<String> {
    // 1. Embed the query
    let query_embeddings = get_embeddings(&[question.to_string()], api_key, base_url).await?;
    let query_embedding = &query_embeddings[0];

    // 2. Retrieve top-k chunks
    let retrieved = vector_store.search(query_embedding, 3);

    // 3. Format context with source metadata
    let context: String = retrieved
        .iter()
        .map(|chunk| {
            format!(
                "[Source: {}]\n{}",
                chunk.title,
                chunk.content.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    // 4. Build prompt with context
    let prompt = format!(
        "Answer the question based only on the following context:\n\n{}\n\nQuestion: {}\n\nIf the answer cannot be found in the context, say \"I don't have enough information to answer this question.\"",
        context, question
    );

    // 5. Get LLM response
    let response = agent.chat(&prompt, vec![]).await?;
    Ok(response)
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
    let embedding_key = env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| api_key.clone());
    let base_url = "https://api.openai.com/v1";

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    run_demo(agent, &embedding_key, base_url, "openai").await
}

async fn run_anthropic() -> Result<usize> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;
    // Note: Anthropic doesn't support the embeddings endpoint we're using
    // So we fall back to OpenAI embeddings
    let embedding_key = env::var("OPENAI_API_KEY")
        .or_else(|_| env::var("OPENROUTER_API_KEY"))
        .map_err(|_| anyhow::anyhow!("Need OPENAI_API_KEY or OPENROUTER_API_KEY for embeddings"))?;
    let embedding_base = if env::var("OPENAI_API_KEY").is_ok() {
        "https://api.openai.com/v1"
    } else {
        "https://openrouter.ai/api/v1"
    };

    let client = anthropic::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    run_demo(agent, &embedding_key, embedding_base, "anthropic").await
}

async fn run_openrouter() -> Result<usize> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {model_id}");
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;
    let base_url = "https://openrouter.ai/api/v1";

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    run_demo(agent, &api_key, base_url, "openrouter").await
}

// ============================================================================
// Demo Execution
// ============================================================================

async fn run_demo(
    agent: impl Chat,
    embedding_key: &str,
    embedding_base: &str,
    provider_key: &str,
) -> Result<usize> {
    println!("=== Rust — RAG over Local Documents ===");
    println!("Provider:  {provider_key}");
    println!();

    // Create documents
    let documents = vec![
        Document {
            id: "faq".to_string(),
            title: "Polyglot Agent Labs FAQ".to_string(),
            content: FAQ_DOC.to_string(),
        },
        Document {
            id: "rust-guidelines".to_string(),
            title: "Rust Coding Guidelines".to_string(),
            content: RUST_GUIDELINES_DOC.to_string(),
        },
        Document {
            id: "python-practices".to_string(),
            title: "Python Best Practices".to_string(),
            content: PYTHON_PRACTICES_DOC.to_string(),
        },
    ];

    // Chunk documents
    let mut all_chunks = Vec::new();
    let mut _chunk_id = 0;

    for doc in &documents {
        let chunks = chunk_document(doc, 500);
        for chunk_text in chunks {
            all_chunks.push((doc, chunk_text));
            _chunk_id += 1;
        }
    }

    // Get embeddings for all chunks
    println!("Indexing {} chunks...", all_chunks.len());
    let chunk_texts: Vec<String> = all_chunks
        .iter()
        .map(|(_, text)| text.clone())
        .collect();

    let embeddings = get_embeddings(&chunk_texts, embedding_key, embedding_base).await?;

    // Build vector store
    let mut vector_store = VectorStore::new();
    for ((doc, chunk_text), embedding) in all_chunks
        .iter()
        .zip(embeddings.iter())
    {
        vector_store.add(DocumentChunk {
            source: doc.id.clone(),
            title: doc.title.clone(),
            content: chunk_text.clone(),
            embedding: embedding.clone(),
        });
    }

    println!("Documents indexed: {}", documents.len());
    println!("Total chunks: {}", vector_store.chunks.len());
    println!();

    // Process demo questions
    for (i, question) in DEMO_QUESTIONS.iter().enumerate() {
        println!(
            "[{}/{}] Question: {}",
            i + 1,
            DEMO_QUESTIONS.len(),
            question
        );
        println!("{}", "-".repeat(50));

        // Get query embedding for retrieval display
        let query_embeddings = get_embeddings(&[question.to_string()], embedding_key, embedding_base).await?;
        let retrieved = vector_store.search(&query_embeddings[0], 3);

        println!("\nRetrieved {} chunks:", retrieved.len());
        for (j, chunk) in retrieved.iter().enumerate() {
            println!("  [{}] {} ({})", j + 1, chunk.title, chunk.source);
            let preview = if chunk.content.len() > 100 {
                format!("{}...", &chunk.content[..100])
            } else {
                chunk.content.clone()
            };
            println!("      {}", preview);
        }

        // Get RAG response
        match rag_query(&agent, &vector_store, embedding_key, embedding_base, question).await {
            Ok(answer) => {
                println!("\nAnswer: {}", answer);
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
    println!("  Questions processed: {}", DEMO_QUESTIONS.len());

    Ok(DEMO_QUESTIONS.len())
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
            bail!(
                "Unknown provider: '{}'. Supported: openai, anthropic, openrouter",
                provider
            );
        }
    };

    Ok(())
}
