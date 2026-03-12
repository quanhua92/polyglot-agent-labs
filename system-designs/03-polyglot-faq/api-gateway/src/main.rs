use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

mod mcp_client;
mod streaming;

use mcp_client::McpClient;
use streaming::create_stream_response;

/// Configuration for the API gateway
#[derive(Clone)]
struct AppConfig {
    workflow_mcp_url: String,
}

/// Ask request payload
#[derive(Debug, Deserialize)]
struct AskRequest {
    question: String,
    #[serde(default)]
    stream: bool,
}

/// Ask response
#[derive(Debug, Serialize)]
struct AskResponse {
    question: String,
    answer: String,
    metadata: ResponseMetadata,
}

#[derive(Debug, Serialize)]
struct ResponseMetadata {
    steps_taken: usize,
    documents_found: usize,
    processing_time_ms: u64,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    services: ServiceStatus,
}

#[derive(Debug, Serialize)]
struct ServiceStatus {
    workflow_server: String,
    doc_search_server: String,
}

/// Error response
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

/// Handler for POST /ask - non-streaming
async fn ask_handler(
    State(config): State<Arc<AppConfig>>,
    Json(payload): Json<AskRequest>,
) -> Result<Json<AskResponse>, ErrorResponse> {
    let start = std::time::Instant::now();

    // Validate input
    if payload.question.trim().is_empty() {
        return Err(ErrorResponse {
            error: "invalid_input".to_string(),
            message: "Question cannot be empty".to_string(),
        });
    }

    // Call the MCP workflow server (create new client per request for simplicity)
    let client = McpClient::new(config.workflow_mcp_url.clone());

    match client.call_faq_workflow(&payload.question, false).await {
        Ok(workflow_result) => {
            let processing_time = start.elapsed().as_millis() as u64;

            Ok(Json(AskResponse {
                question: workflow_result.question,
                answer: workflow_result.final_answer,
                metadata: ResponseMetadata {
                    steps_taken: workflow_result.steps.len(),
                    documents_found: workflow_result.search_results.len(),
                    processing_time_ms: processing_time,
                },
            }))
        }
        Err(e) => Err(ErrorResponse {
            error: "workflow_failed".to_string(),
            message: format!("Failed to process question: {}", e),
        }),
    }
}

/// Handler for POST /ask/stream - streaming response
async fn ask_stream_handler(
    State(config): State<Arc<AppConfig>>,
    Json(payload): Json<AskRequest>,
) -> Response {
    // Validate input
    if payload.question.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_input".to_string(),
                message: "Question cannot be empty".to_string(),
            }),
        )
            .into_response();
    }

    // For demo purposes, return a simulated stream
    // In production, this would connect to the MCP server's streaming endpoint
    create_stream_response().into_response()
}

/// Handler for GET /health - health check
async fn health_handler(
    State(config): State<Arc<AppConfig>>,
) -> Result<Json<HealthResponse>, ErrorResponse> {
    // Simple health check - in production, actually ping the services
    let _client = McpClient::new(config.workflow_mcp_url.clone());

    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        services: ServiceStatus {
            workflow_server: "unknown".to_string(),
            doc_search_server: "unknown".to_string(),
        },
    }))
}

/// Handler for GET / - root endpoint with API info
async fn root_handler() -> &'static str {
    r#"
FAQ API Gateway
===============

A high-performance Rust API gateway for the FAQ system.

Endpoints:
- POST /ask          - Get answer to a FAQ question (non-streaming)
- POST /ask/stream   - Get answer with streaming intermediate results
- GET  /health       - Health check

Example:
  curl -X POST http://localhost:3000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "How do I reset my password?"}'
"#
}

/// Convert ErrorResponse into axum Response
impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let status = match self.error.as_str() {
            "invalid_input" => StatusCode::BAD_REQUEST,
            "workflow_failed" => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(self)).into_response()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Rust Axum — FAQ API Gateway ===");

    // Load configuration from environment
    let workflow_mcp_url =
        std::env::var("WORKFLOW_MCP_URL").unwrap_or_else(|_| "http://workflow-server:8003".to_string());

    let config = Arc::new(AppConfig { workflow_mcp_url });

    println!("Workflow MCP URL: {}", config.workflow_mcp_url);
    println!();

    // Build the router
    let app = Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/ask", post(ask_handler))
        .route("/ask/stream", post(ask_stream_handler))
        .with_state(config);

    // Start the server
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    let listener = TcpListener::bind(addr).await?;

    println!("API Gateway listening on http://{}", addr);
    println!();
    println!("Available endpoints:");
    println!("  GET  /              - API information");
    println!("  GET  /health        - Health check");
    println!("  POST /ask           - Ask a question (non-streaming)");
    println!("  POST /ask/stream    - Ask a question (streaming)");
    println!();

    axum::serve(listener, app).await?;

    Ok(())
}
