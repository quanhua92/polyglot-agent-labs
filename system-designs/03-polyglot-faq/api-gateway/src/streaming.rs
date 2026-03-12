use axum::response::sse::{Event, Sse};
use futures::Stream;
use serde::{Serialize, Deserialize};
use serde_json::json;
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

/// Streaming event for FAQ workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    #[serde(rename = "type")]
    pub event_type: StreamEventType,
    pub node: Option<String>,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamEventType {
    NodeStarted,
    NodeCompleted,
    Error,
    Complete,
}

/// Create a streaming response for the FAQ workflow
pub fn create_stream_response() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);

    // Spawn a task to simulate streaming
    // In a real implementation, this would connect to the MCP server
    // and stream the actual workflow execution
    tokio::spawn(async move {
        // Simulate workflow steps
        let events = vec![
            StreamEvent {
                event_type: StreamEventType::NodeStarted,
                node: Some("expand_query".to_string()),
                data: json!({"message": "Expanding query into variants..."}),
            },
            StreamEvent {
                event_type: StreamEventType::NodeCompleted,
                node: Some("expand_query".to_string()),
                data: json!({"variants": ["password reset", "how to reset password", "forgot password"]}),
            },
            StreamEvent {
                event_type: StreamEventType::NodeStarted,
                node: Some("search_documents".to_string()),
                data: json!({"message": "Searching FAQ documents..."}),
            },
            StreamEvent {
                event_type: StreamEventType::NodeCompleted,
                node: Some("search_documents".to_string()),
                data: json!({"found": 3, "documents": ["password-reset", "account-creation", "profile-update"]}),
            },
            StreamEvent {
                event_type: StreamEventType::NodeStarted,
                node: Some("generate_response".to_string()),
                data: json!({"message": "Generating response..."}),
            },
            StreamEvent {
                event_type: StreamEventType::NodeCompleted,
                node: Some("generate_response".to_string()),
                data: json!({"answer": "To reset your password..."}),
            },
            StreamEvent {
                event_type: StreamEventType::Complete,
                node: None,
                data: json!({"status": "success"}),
            },
        ];

        for event in events {
            if let Err(_) = tx.send(event).await {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    });

    // Convert channel receiver to SSE stream
    let stream = ReceiverStream::new(rx).map(move |event| {
        let json = serde_json::to_string(&event).unwrap_or_default();
        Ok::<_, Infallible>(Event::default().data(json))
    });

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(1))
            .text("keepalive"),
    )
}

/// Stream from MCP workflow server (real implementation)
pub async fn stream_from_mcp(
    _workflow_url: String,
    _question: String,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, anyhow::Error> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);

    // In a real implementation, this would:
    // 1. Connect to the MCP server using SseTransport
    // 2. Call the faq_workflow tool with stream=true
    // 3. Parse incoming SSE events and forward them

    tokio::spawn(async move {
        // Placeholder for real MCP streaming implementation
        let _ = tx.send(StreamEvent {
            event_type: StreamEventType::NodeStarted,
            node: Some("expand_query".to_string()),
            data: json!({"message": "Starting workflow..."}),
        }).await;

        let _ = tx.send(StreamEvent {
            event_type: StreamEventType::Complete,
            node: None,
            data: json!({"answer": "Answer would come from MCP server"}),
        }).await;
    });

    let stream = ReceiverStream::new(rx).map(move |event| {
        let json = serde_json::to_string(&event).unwrap_or_default();
        Ok::<_, Infallible>(Event::default().data(json))
    });

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(1))
            .text("keepalive"),
    ))
}
