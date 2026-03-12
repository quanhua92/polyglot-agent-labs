use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerInfo, ServerCapabilities};
use rmcp::service::RequestContext;
use rmcp::ServerHandler;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager,
    tower::{StreamableHttpServerConfig, StreamableHttpService},
};
use rmcp::{tool, tool_handler, tool_router, RoleServer, schemars};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::net::TcpListener;

mod documents;
mod scoring;
mod related;
mod time;

use documents::load_documents;
use scoring::search_and_score;
use related::find_related;
use time::get_current_time;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchInput {
    /// The search query to find relevant documents
    query: String,
    /// Maximum number of results to return (default: 5)
    #[serde(default = "default_top_n")]
    top_n: usize,
}

fn default_top_n() -> usize {
    5
}

#[derive(Debug, Serialize)]
struct DocumentResult {
    id: String,
    title: String,
    content: String,
    score: f64,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetDocumentInput {
    /// The ID of the document to retrieve (e.g., "password-reset", "return-policy")
    document_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct FindRelatedDocumentsInput {
    /// List of document IDs to find related documents for
    document_ids: Vec<String>,
    /// Maximum number of related documents to return (default: 3)
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn default_max_results() -> usize {
    3
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetCurrentDateInput {
    /// Timezone for the current time (e.g., "UTC", "local"). Default: "local"
    timezone: Option<String>,
}

#[derive(Clone)]
struct DocSearchServer {
    documents: Vec<documents::Document>,
    tool_router: rmcp::handler::server::router::tool::ToolRouter<Self>,
}

impl DocSearchServer {
    fn new() -> Self {
        let documents = load_documents();
        Self {
            documents,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl DocSearchServer {
    #[tool(description = "Search FAQ documents by query with TF-IDF scoring")]
    async fn search_documents(
        &self,
        Parameters(input): Parameters<SearchInput>,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let results = search_and_score(&self.documents, &input.query, input.top_n);

        let doc_results: Vec<DocumentResult> = results
            .into_iter()
            .map(|(doc, score)| DocumentResult {
                id: doc.id.clone(),
                title: doc.title.clone(),
                content: doc.content.clone(),
                score,
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&doc_results).unwrap_or_else(|_| "Error serializing results".to_string()),
        )]))
    }

    #[tool(description = "List all available FAQ documents")]
    async fn list_documents(
        &self,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let docs: Vec<String> = self
            .documents
            .iter()
            .map(|doc| format!("{}: {}", doc.id, doc.title))
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            docs.join("\n"),
        )]))
    }

    #[tool(description = "Get full document content by document ID")]
    async fn get_document(
        &self,
        Parameters(input): Parameters<GetDocumentInput>,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let doc = self
            .documents
            .iter()
            .find(|d| d.id == input.document_id)
            .ok_or_else(|| {
                rmcp::ErrorData::new(
                    rmcp::model::ErrorCode::INVALID_PARAMS,
                    format!("Document not found: {}", input.document_id),
                    None,
                )
            })?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&doc).unwrap_or_else(|_| "Error serializing document".to_string()),
        )]))
    }

    #[tool(description = "Find documents related to the given document IDs based on category and topic similarity")]
    async fn find_related_documents(
        &self,
        Parameters(input): Parameters<FindRelatedDocumentsInput>,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let related = find_related(&self.documents, &input.document_ids, input.max_results);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&related).unwrap_or_else(|_| "Error serializing related documents".to_string()),
        )]))
    }

    #[tool(description = "Get current date and time for time-sensitive FAQ responses")]
    async fn get_current_date(
        &self,
        Parameters(input): Parameters<GetCurrentDateInput>,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let current_time = get_current_time(input.timezone.as_deref());

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&current_time).unwrap_or_else(|_| "Error serializing time".to_string()),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for DocSearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(rmcp::model::Implementation::default())
        .with_instructions(
            "FAQ Document Search MCP Server. Tools: search_documents(query, top_n), list_documents, \
             get_document(document_id), find_related_documents(document_ids, max_results), get_current_date(timezone). \
             Provides TF-IDF based document search, full document retrieval, related document discovery, and current time awareness for FAQ content."
                .to_string(),
        )
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Rust MCP — FAQ Document Search Server ===");

    let server = DocSearchServer::new();
    println!("Loaded {} FAQ documents", server.documents.len());

    let service = StreamableHttpService::new(
        || Ok(DocSearchServer::new()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    let app = axum::Router::new().nest_service("/mcp", service);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8004));
    let listener = TcpListener::bind(addr).await?;
    println!("MCP server listening on http://{}", addr);
    println!("  Endpoint: http://{}/mcp", addr);
    println!();
    println!("Available tools:");
    println!("  - search_documents(query, top_n): Search FAQ documents");
    println!("  - list_documents(): List all documents");
    println!("  - get_document(document_id): Get full document by ID");
    println!("  - find_related_documents(document_ids, max_results): Find related docs");
    println!("  - get_current_date(timezone): Get current date/time");
    println!();

    axum::serve(listener, app).await?;
    Ok(())
}
