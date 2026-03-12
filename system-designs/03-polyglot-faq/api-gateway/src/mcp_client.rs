use anyhow::{anyhow, Result};
use rmcp::{
    model::CallToolRequestParams,
    ServiceExt,
    transport::streamable_http_client::StreamableHttpClientTransport,
};
use serde::{Deserialize, Serialize};

/// Result from the FAQ workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub question: String,
    pub steps: Vec<WorkflowStep>,
    pub query_variants: Vec<String>,
    pub search_results: Vec<SearchResult>,
    pub final_answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub node: String,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub title: String,
    pub content: String,
    pub score: f64,
}

/// MCP client using rmcp's StreamableHttpClientTransport
pub struct McpClient {
    base_url: String,
}

impl McpClient {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    pub async fn call_faq_workflow(&self, question: &str, _stream: bool) -> Result<WorkflowResult> {
        // Create HTTP transport using rmcp's built-in reqwest support
        let transport = StreamableHttpClientTransport::from_uri(format!("{}/mcp", self.base_url));

        // Create and serve the client
        let mut client = ().serve(transport).await?;

        // Call the faq_workflow tool
        let result = client
            .call_tool(
                CallToolRequestParams::new("faq_workflow")
                    .with_arguments(
                        serde_json::json!({
                            "question": question,
                            "stream": false
                        }).as_object()
                        .unwrap()
                        .clone()
                    ),
            )
            .await
            .map_err(|e| anyhow!("Tool call failed: {}", e))?;

        // Extract the result from the tool call response
        let content = result.content;
        if content.is_empty() {
            return Err(anyhow!("Empty response from workflow tool"));
        }

        // Get the first content item
        let first_content = &content[0];

        // Get the text content
        let text = first_content.raw.as_text()
            .ok_or_else(|| anyhow!("Response is not text"))?
            .text.as_str();

        // Parse the JSON result
        let workflow_result: WorkflowResult = serde_json::from_str(text)
            .map_err(|e| anyhow!("Failed to parse workflow result: {}", e))?;

        Ok(workflow_result)
    }
}
