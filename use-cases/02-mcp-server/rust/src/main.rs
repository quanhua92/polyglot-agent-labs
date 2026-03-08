//! Polyglot Agent Labs — Use Case 02: Simple MCP Server + Client
//!
//! Single binary that runs as either server or client:
//!   - `cargo run`            → client mode (default): spawns itself as server, calls get_weather
//!   - `cargo run -- --server` → server mode: runs MCP server over stdio

use std::env;

use anyhow::Result;
use rmcp::model::{CallToolRequestParams, ServerInfo};
#[allow(unused_imports)]
use std::ops::Deref;
use rmcp::schemars;
use rmcp::serde;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

// ─── Tool Input Schema ──────────────────────────────────────────────────────

/// Input for the get_weather tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetWeatherInput {
    /// Name of the city to get weather for.
    city: String,
}

// ─── MCP Server ─────────────────────────────────────────────────────────────

/// Weather server that exposes a `get_weather` tool.
#[derive(Clone)]
struct WeatherServer {
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}

impl WeatherServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl WeatherServer {
    /// Get the current weather for a given city.
    #[tool(description = "Get the current weather for a given city")]
    fn get_weather(
        &self,
        input: rmcp::handler::server::wrapper::Parameters<GetWeatherInput>,
    ) -> String {
        let city = input.0.city.trim().to_lowercase();
        let weather = match city.as_str() {
            "tokyo" => "☀️ Sunny, 22°C — clear skies with light breeze",
            "london" => "🌧️ Rainy, 14°C — overcast with intermittent showers",
            "new york" => "⛅ Partly cloudy, 18°C — mild with occasional sun",
            "paris" => "🌤️ Mostly sunny, 20°C — warm with gentle winds",
            "sydney" => "🌡️ Hot, 30°C — bright sunshine, UV index high",
            _ => "🌍 No data available — try Tokyo, London, New York, Paris, or Sydney.",
        };
        format!("Weather in {}: {}", input.0.city, weather)
    }
}

#[tool_handler]
impl ServerHandler for WeatherServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }
}

// ─── Server Mode ────────────────────────────────────────────────────────────

async fn run_server() -> Result<()> {
    eprintln!("=== Rust — MCP Weather Server (stdio) ===");
    let server = WeatherServer::new();
    let transport = rmcp::transport::stdio();
    let service = server.serve(transport).await?;
    service.waiting().await?;
    Ok(())
}

// ─── Client Mode ────────────────────────────────────────────────────────────

async fn run_client() -> Result<()> {
    println!("=== Rust — MCP Client → Weather Server ===");
    println!();

    // Spawn ourselves with --server flag as the MCP server child process
    let exe = env::current_exe()?;
    let transport = TokioChildProcess::new(
        Command::new(&exe).configure(|cmd| {
            cmd.arg("--server");
        }),
    )?;

    let client = ().serve(transport).await?;

    // List available tools
    let tools = client.list_tools(None).await?;
    let tool_names: Vec<&str> = tools.tools.iter().map(|t| &*t.name).collect();
    println!("Available tools: {tool_names:?}");
    println!();

    // Call with known cities
    for city in ["Tokyo", "London", "Sydney"] {
        let result = client
            .call_tool(CallToolRequestParams::new("get_weather")
                .with_arguments(json!({"city": city}).as_object().unwrap().clone()))
            .await?;
        if let Some(content) = result.content.first() {
            println!(
                "  {}",
                content
                    .raw
                    .as_text()
                    .map(|t| t.text.as_str())
                    .unwrap_or("???")
            );
        }
    }

    // Call with unknown city
    println!();
    let result = client
        .call_tool(CallToolRequestParams::new("get_weather")
            .with_arguments(json!({"city": "Mars"}).as_object().unwrap().clone()))
        .await?;
    if let Some(content) = result.content.first() {
        println!(
            "  {}",
            content
                .raw
                .as_text()
                .map(|t| t.text.as_str())
                .unwrap_or("???")
        );
    }

    println!();
    println!("✓ MCP client-server communication successful!");

    client.cancel().await?;
    Ok(())
}

// ─── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--server") {
        run_server().await
    } else {
        run_client().await
    }
}
