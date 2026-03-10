//! Polyglot Agent Labs — Use Case 10: Customer Support Agent
//! A complete customer support pipeline with intent classification, KB retrieval,
//! response generation, and escalation logic with multi-turn conversation support.
//! Switch provider with env var LLM_PROVIDER (default: openrouter).
//!
//! Usage:
//!   cargo run -- --interactive    # Interactive REPL mode
//!   cargo run -- --demo          # Demo mode with predefined scenarios

use std::collections::HashMap;
use std::env;
use std::io::Write;

use anyhow::{bail, Result};
use reqwest::Client;
use rig::client::CompletionClient;
use rig::completion::Chat;
use rig::providers::{anthropic, openai, openrouter};
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

const INTENT_CLASSIFIER_PROMPT: &str = r#"You are an intent classifier for a customer support system.

Analyze the customer's message and classify it into one of these intents:
- billing: Questions about charges, invoices, payments, refunds
- shipping: Questions about delivery, tracking, shipping options
- returns: Questions about return policy, exchanges, refunds
- account: Questions about login, password, profile, settings
- general: General inquiries, product info, company info
- escalate: Explicit request to speak to a human/manager

Extract relevant entities (order_id, email, product_name, etc.)

Return valid JSON with this exact format:
{
  "intent": "category",
  "confidence": 0.95,
  "entities": {"key": "value"},
  "urgency": "medium"
}"#;

const RESPONSE_GENERATOR_PROMPT: &str = r#"You are a helpful customer support agent.

Your task is to provide a friendly, helpful response using the knowledge base context provided.

Guidelines:
- Be warm and empathetic
- Provide specific, actionable information
- Cite the source articles you're using
- If you're unsure, say so and offer to connect them with a human
- Keep responses concise but complete
- Use formatting (bullet points, numbered lists) for readability"#;

const ESCALATION_CONFIDENCE_THRESHOLD: f32 = 0.6;

// ============================================================================
// Hard-coded Knowledge Base
// ============================================================================

const KNOWLEDGE_BASE: &[(&str, &str, &str)] = &[
    ("return-policy", "Return Policy", r#"We offer a 30-day return policy for all unused items in original packaging.

To return an item:
1. Log into your account and go to Order History
2. Select the item you wish to return
3. Print the prepaid return shipping label
4. Pack the item securely in original packaging
5. Drop off at any authorized shipping location

Refunds are processed within 5-7 business days after we receive the return. The refund will be credited to the original payment method.

Items that cannot be returned:
- Personalized or custom items
- Perishable goods
- Gift cards
- Items marked as "Final Sale"

Exchanges are available for different sizes of the same product."#),
    ("shipping-info", "Shipping Information", r#"We offer several shipping options:

Standard Shipping (5-7 business days): $4.99
Express Shipping (2-3 business days): $9.99
Overnight Shipping (1 business day): $19.99

FREE standard shipping on orders over $50!

Orders placed before 2 PM EST ship the same day. Orders placed after 2 PM EST ship the next business day.

We ship to all 50 US states. International shipping is not currently available.

Track your order:
- Use the tracking number in your shipping confirmation email
- Or log into your account and view Order Status

Delivery delays may occur during holidays and severe weather. We'll notify you of any significant delays."#),
    ("billing-payments", "Billing and Payments", r#"We accept the following payment methods:
- Credit/Debit Cards (Visa, MasterCard, American Express, Discover)
- PayPal
- Apple Pay
- Google Pay

Payment is charged at the time of order confirmation. A detailed invoice is emailed to you immediately.

To request a duplicate invoice:
- Log into your account
- Go to Order History
- Click "Download Invoice" next to the order

For billing questions:
- Email: billing@example.com
- Phone: 1-800-BILLING
- Hours: Mon-Fri 9 AM - 6 PM EST

If a payment fails:
- Check your card details are correct
- Ensure sufficient funds
- Try an alternative payment method
- Contact your bank if the issue persists"#),
    ("account-management", "Account Management", r#"Managing Your Account:

Password Reset:
1. Click "Forgot Password" on the login page
2. Enter your email address
3. Check your email for reset link (expires in 24 hours)
4. Create a new password

Profile Updates:
- Go to Account Settings
- Update your email, phone, shipping address
- Changes save automatically

Order History:
- View all past orders
- Track current orders
- Reorder previous purchases
- Download invoices

Account Deletion:
- Go to Account Settings
- Scroll to "Delete Account"
- Confirm deletion
- All data will be permanently removed within 30 days

For account issues not resolved here, contact customer support."#),
    ("order-status", "Order Status and Tracking", r#"Track Your Order:

1. Log into your account
2. Go to Order History
3. Click on the order number
4. View real-time tracking information

Order Status Meanings:
- Processing: We're preparing your order
- Shipped: Order is on its way
- Delivered: Order has been delivered
- Delayed: Shipping delay has occurred
- Cancelled: Order was cancelled

Delivery Times:
- Standard: 5-7 business days
- Express: 2-3 business days
- Overnight: 1 business day

Cancellations:
- Cancel within 1 hour of placing the order
- Go to Order History and click "Cancel"
- After 1 hour, contact customer support

Missing Orders:
- Check tracking for delivery status
- Verify shipping address is correct
- Allow extra time for standard shipping
- Contact support if not received within 10 business days"#),
    ("contact-info", "Contact Customer Support", r#"How to Reach Us:

Phone Support:
- 1-800-SUPPORT
- Mon-Fri: 8 AM - 8 PM EST
- Sat-Sun: 10 AM - 6 PM EST

Email Support:
- support@example.com
- Response within 24 hours

Live Chat:
- Available on our website
- Mon-Fri: 9 AM - 9 PM EST

Social Media:
- Twitter: @SupportTeam
- Facebook: /SupportTeam

For fastest response, use live chat during business hours.

When contacting us, please have:
- Your order number (if applicable)
- Your account email
- A brief description of your issue

Urgent issues (cancellation, wrong item): Call us directly.

General inquiries: Email or live chat is preferred."#),
];

// ============================================================================
// Demo Scenarios
// ============================================================================

#[derive(Debug, Clone)]
struct DemoScenario {
    name: &'static str,
    input: &'static str,
}

const DEMO_SCENARIOS: &[DemoScenario] = &[
    DemoScenario {
        name: "Return Policy Inquiry",
        input: "How do I return a product I bought last week?",
    },
    DemoScenario {
        name: "Explicit Escalation",
        input: "I want to speak to a manager!",
    },
    DemoScenario {
        name: "Order Status",
        input: "What's the status of order #12345?",
    },
];

// ============================================================================
// Structs for Structured Output
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IntentClassification {
    pub intent: String,
    pub confidence: f32,
    #[serde(default)]
    pub entities: HashMap<String, String>,
    #[serde(default = "default_urgency")]
    pub urgency: String,
}

fn default_urgency() -> String {
    "medium".to_string()
}

#[derive(Debug, Clone)]
pub struct SupportResponse {
    pub response: String,
    pub sources: Vec<String>,
    pub escalation_reason: Option<String>,
    pub should_escalate: bool,
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
    api_key: String,
    base_url: String,
}

impl VectorStore {
    fn new(api_key: String, base_url: String) -> Self {
        Self {
            chunks: Vec::new(),
            api_key,
            base_url,
        }
    }

    async fn initialize(&mut self) -> Result<usize> {
        // Chunk documents
        let mut all_chunks = Vec::new();

        for (id, title, content) in KNOWLEDGE_BASE {
            let chunks = chunk_document(*content, 500);
            for chunk_text in chunks {
                all_chunks.push((*id, *title, chunk_text));
            }
        }

        // Get embeddings for all chunks
        let chunk_texts: Vec<String> = all_chunks
            .iter()
            .map(|(_, _, text)| text.clone())
            .collect();

        let embeddings = get_embeddings(&chunk_texts, &self.api_key, &self.base_url).await?;

        // Build vector store
        for ((id, title, content), embedding) in all_chunks.iter().zip(embeddings.iter()) {
            self.chunks.push(DocumentChunk {
                source: id.to_string(),
                title: title.to_string(),
                content: content.clone(),
                embedding: embedding.clone(),
            });
        }

        Ok(self.chunks.len())
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

fn chunk_document(content: &str, chunk_size: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = content
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
            return response[json_start..json_start + end].trim().to_string();
        }
    }

    // Look for { and } as JSON boundaries
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            return response[start..=end].to_string();
        }
    }

    response.to_string()
}

// ============================================================================
// Support Agent Pipeline
// ============================================================================

async fn classify_intent(
    user_input: &str,
    agent: &impl Chat,
) -> Result<IntentClassification> {
    let prompt = format!(
        r#"{}

Customer message: {}

Classify this message and return structured JSON."#,
        INTENT_CLASSIFIER_PROMPT, user_input
    );

    let response = agent.chat(&prompt, vec![]).await?;
    let json_str = extract_json_from_response(&response);
    let parsed: IntentClassification = serde_json::from_str(&json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse intent classification JSON: {}", e))?;

    Ok(parsed)
}

async fn generate_response(
    user_input: &str,
    kb_context: &str,
    agent: &impl Chat,
) -> Result<String> {
    let prompt = format!(
        "{}\n\nCustomer question: {}\n\nRelevant information from our knowledge base:\n{}\n\nProvide a helpful response. Include citations to the source articles.",
        RESPONSE_GENERATOR_PROMPT, user_input, kb_context
    );

    Ok(agent.chat(&prompt, vec![]).await?)
}

async fn process_customer_message(
    user_input: &str,
    agent: &impl Chat,
    vector_store: &VectorStore,
) -> Result<(IntentClassification, SupportResponse)> {
    // Step 1: Classify intent
    let classification = classify_intent(user_input, agent).await?;

    // Step 2: Check escalation conditions
    let should_escalate = classification.confidence < ESCALATION_CONFIDENCE_THRESHOLD
        || classification.intent == "escalate";

    if should_escalate {
        let reason = if classification.intent == "escalate" {
            "Customer explicitly requested human assistance".to_string()
        } else {
            format!(
                "Low confidence score ({:.2} < {})",
                classification.confidence, ESCALATION_CONFIDENCE_THRESHOLD
            )
        };

        let response = SupportResponse {
            response: format!(
                "🚨 Transferring to human agent...\n\nReason for escalation: {}\n\nA human agent will be with you shortly. Your conversation has been logged and the agent will have full context.",
                reason
            ),
            sources: vec![],
            escalation_reason: Some(reason),
            should_escalate: true,
        };

        return Ok((classification, response));
    }

    // Step 3: Retrieve KB articles
    let query_embeddings = get_embeddings(&[user_input.to_string()], &vector_store.api_key, &vector_store.base_url).await?;
    let retrieved = vector_store.search(&query_embeddings[0], 2);

    let kb_context = retrieved
        .iter()
        .map(|chunk| format!("[Source: {}]\n{}", chunk.title, chunk.content))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let sources: Vec<String> = retrieved.iter().map(|c| c.source.clone()).collect();

    // Step 4: Generate response
    let response_text = generate_response(user_input, &kb_context, agent).await?;

    let response = SupportResponse {
        response: response_text,
        sources,
        escalation_reason: None,
        should_escalate: false,
    };

    Ok((classification, response))
}

// ============================================================================
// Interactive REPL
// ============================================================================

async fn run_interactive_repl(
    agent: &impl Chat,
    vector_store: &VectorStore,
    provider_key: &str,
) -> Result<usize> {
    println!("=== Rust — Customer Support Agent ===");
    println!("Provider:  {}", provider_key);
    println!("Mode:      interactive");
    println!();
    println!("Type your message to chat with the support agent.");
    println!("Commands: /quit, /exit, /q to quit");
    println!("{}", "-".repeat(50));
    println!();

    let mut turn_count = 0;

    loop {
        print!("You: ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        match std::io::stdin().read_line(&mut input) {
            Ok(0) => {
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

        // Process the message
        match process_customer_message(user_input, agent, vector_store).await {
            Ok((classification, response)) => {
                println!(
                    "\n[Intent: {} | Confidence: {:.2}]",
                    classification.intent.to_uppercase(),
                    classification.confidence
                );
                println!("\nAgent: {}", response.response);

                if !response.sources.is_empty() {
                    println!("\n[Sources: {}]", response.sources.join(", "));
                }

                println!();
                turn_count += 1;
            }
            Err(e) => {
                eprintln!("\n✗ Error: {}\n", e);
            }
        }
    }

    println!("{}", "-".repeat(50));
    println!("Session Summary");
    println!("Total turns: {}", turn_count);

    Ok(turn_count)
}

// ============================================================================
// Demo Mode
// ============================================================================

async fn run_demo_scenarios(
    agent: &impl Chat,
    vector_store: &VectorStore,
    provider_key: &str,
) -> Result<usize> {
    println!("=== Rust — Customer Support Agent ===");
    println!("Provider:  {}", provider_key);
    println!("Mode:      demo");
    println!("{}", "-".repeat(50));
    println!();

    for (i, scenario) in DEMO_SCENARIOS.iter().enumerate() {
        println!(
            "[{}/{}] Scenario: {}",
            i + 1,
            DEMO_SCENARIOS.len(),
            scenario.name
        );
        println!("Input: {}", scenario.input);
        println!("{}", "-".repeat(50));

        match process_customer_message(scenario.input, agent, vector_store).await {
            Ok((classification, response)) => {
                println!("Intent: {}", classification.intent.to_uppercase());
                println!("Confidence: {:.2}", classification.confidence);
                println!("\nResponse:\n{}", response.response);

                if let Some(reason) = &response.escalation_reason {
                    println!("\n⚠️ Escalated: {}", reason);
                }

                if !response.sources.is_empty() {
                    println!("\nKB Sources: {}", response.sources.join(", "));
                }
            }
            Err(e) => {
                eprintln!("✗ Error: {}", e);
            }
        }

        println!("{}", "=".repeat(50));
        println!();
    }

    // Session summary
    println!("Session Summary");
    println!("Scenarios processed: {}", DEMO_SCENARIOS.len());

    Ok(DEMO_SCENARIOS.len())
}

// ============================================================================
// Provider Setup
// ============================================================================

async fn run_openai(interactive: bool) -> Result<usize> {
    let model_id = "gpt-4.1-nano";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;
    let embedding_key = env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| api_key.clone());
    let base_url = "https://api.openai.com/v1";

    let client = openai::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    let mut vector_store = VectorStore::new(embedding_key, base_url.to_string());
    println!("Indexing knowledge base...");
    let chunk_count = vector_store.initialize().await?;
    println!("Documents indexed: {}", KNOWLEDGE_BASE.len());
    println!("Total chunks: {}", chunk_count);
    println!();

    if interactive {
        run_interactive_repl(&agent, &vector_store, "openai").await
    } else {
        run_demo_scenarios(&agent, &vector_store, "openai").await
    }
}

async fn run_anthropic(interactive: bool) -> Result<usize> {
    let model_id = "claude-3-haiku-20240307";
    println!("Model:     {}", model_id);
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

    let mut vector_store = VectorStore::new(embedding_key, embedding_base.to_string());
    println!("Indexing knowledge base...");
    let chunk_count = vector_store.initialize().await?;
    println!("Documents indexed: {}", KNOWLEDGE_BASE.len());
    println!("Total chunks: {}", chunk_count);
    println!();

    if interactive {
        run_interactive_repl(&agent, &vector_store, "anthropic").await
    } else {
        run_demo_scenarios(&agent, &vector_store, "anthropic").await
    }
}

async fn run_openrouter(interactive: bool) -> Result<usize> {
    let model_id = "stepfun/step-3.5-flash:free";
    println!("Model:     {}", model_id);
    println!();

    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENROUTER_API_KEY not set"))?;
    let base_url = "https://openrouter.ai/api/v1";

    let client = openrouter::Client::new(&api_key)?;
    let model = client.completion_model(model_id);
    let agent = rig::agent::AgentBuilder::new(model).build();

    let mut vector_store = VectorStore::new(api_key.clone(), base_url.to_string());
    println!("Indexing knowledge base...");
    let chunk_count = vector_store.initialize().await?;
    println!("Documents indexed: {}", KNOWLEDGE_BASE.len());
    println!("Total chunks: {}", chunk_count);
    println!();

    if interactive {
        run_interactive_repl(&agent, &vector_store, "openrouter").await
    } else {
        run_demo_scenarios(&agent, &vector_store, "openrouter").await
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let interactive = args.contains(&"--interactive".to_string());
    let demo = args.contains(&"--demo".to_string());

    if !interactive && !demo {
        println!("Usage:");
        println!("  cargo run -- --interactive    # Interactive REPL mode");
        println!("  cargo run -- --demo          # Demo mode with predefined scenarios");
        std::process::exit(1);
    }

    let provider = env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "openrouter".to_string())
        .to_lowercase();

    let _turn_count = match provider.as_str() {
        "openai" => run_openai(interactive).await?,
        "anthropic" => run_anthropic(interactive).await?,
        "openrouter" => run_openrouter(interactive).await?,
        _ => {
            bail!(
                "Unknown provider: '{}'. Supported: openai, anthropic, openrouter",
                provider
            );
        }
    };

    Ok(())
}
