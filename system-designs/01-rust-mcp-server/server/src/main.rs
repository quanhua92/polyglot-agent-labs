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
use serde::Deserialize;
use std::net::SocketAddr;
use tokio::net::TcpListener;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct AddInput {
    /// First number to add
    a: i32,
    /// Second number to add
    b: i32,
}

#[derive(Clone)]
struct AddServer {
    tool_router: rmcp::handler::server::router::tool::ToolRouter<Self>,
}

impl AddServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl AddServer {
    #[tool(description = "Add two numbers together")]
    async fn add(
        &self,
        Parameters(input): Parameters<AddInput>,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = input.a + input.b;
        Ok(CallToolResult::success(vec![Content::text(
            format!("{} + {} = {}", input.a, input.b, result),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for AddServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(rmcp::model::Implementation::default())
        .with_instructions("A simple MCP server that adds two numbers".to_string())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create the MCP server
    let service = StreamableHttpService::new(
        || Ok(AddServer::new()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    // Set up Axum routes using nest_service - this properly integrates the service
    let app = axum::Router::new().nest_service("/mcp", service);

    // Start the HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], 3001));
    let listener = TcpListener::bind(addr).await?;
    println!("MCP server listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
