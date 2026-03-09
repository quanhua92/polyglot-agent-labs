//! Polyglot Agent Labs — Use Case 02: Full-Featured MCP Server + Client
//!
//! Demonstrates tools, prompts, resources, and stateful server — mirroring the
//! counter.rs example from the official rust-sdk.
//!
//!   - `cargo run`             → client mode (default): spawns itself as server
//!   - `cargo run -- --server` → server mode: runs MCP server over stdio

use std::env;
use std::sync::Arc;

use anyhow::Result;
use rmcp::handler::server::router::prompt::PromptRouter;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::serde;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{
    prompt, prompt_handler, prompt_router, tool, tool_handler, tool_router, ServerHandler,
    ServiceExt,
};
use rmcp::{service::RequestContext, ErrorData as McpError, RoleServer};
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;
use tokio::sync::Mutex;

// ─── Tool Input Schemas ─────────────────────────────────────────────────────

/// Input for the get_weather tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetWeatherInput {
    /// Name of the city to get weather for.
    city: String,
}

/// Input for the sum tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SumInput {
    /// First number
    a: i32,
    /// Second number
    b: i32,
}

/// Arguments for the example_prompt.
#[derive(Debug, serde::Serialize, Deserialize, schemars::JsonSchema)]
struct ExamplePromptArgs {
    /// A message to put in the prompt
    message: String,
}

/// Arguments for the weather_analysis prompt.
#[derive(Debug, serde::Serialize, Deserialize, schemars::JsonSchema)]
struct WeatherAnalysisArgs {
    /// City to analyze weather for
    city: String,
    /// Preferred strategy: 'brief' or 'detailed'
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<String>,
}

// ─── MCP Server ─────────────────────────────────────────────────────────────

/// Full-featured weather & counter server.
#[derive(Clone)]
struct WeatherServer {
    counter: Arc<Mutex<i32>>,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
    prompt_router: PromptRouter<Self>,
}

// Mock weather data
fn weather_data(city: &str) -> Option<&'static str> {
    match city.trim().to_lowercase().as_str() {
        "tokyo" => Some("☀️ Sunny, 22°C — clear skies with light breeze"),
        "london" => Some("🌧️ Rainy, 14°C — overcast with intermittent showers"),
        "new york" => Some("⛅ Partly cloudy, 18°C — mild with occasional sun"),
        "paris" => Some("🌤️ Mostly sunny, 20°C — warm with gentle winds"),
        "sydney" => Some("🌡️ Hot, 30°C — bright sunshine, UV index high"),
        _ => None,
    }
}

impl WeatherServer {
    fn new() -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        }
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }
}

// ─── Tools ──────────────────────────────────────────────────────────────────

#[tool_router]
impl WeatherServer {
    /// Get the current weather for a given city.
    #[tool(description = "Get the current weather for a given city")]
    fn get_weather(
        &self,
        input: rmcp::handler::server::wrapper::Parameters<GetWeatherInput>,
    ) -> Result<CallToolResult, McpError> {
        let city = &input.0.city;
        let text = match weather_data(city) {
            Some(w) => format!("Weather in {}: {}", city, w),
            None => format!(
                "Weather in {}: 🌍 No data available — try Tokyo, London, New York, Paris, or Sydney.",
                city
            ),
        };
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Increment the counter by 1.
    #[tool(description = "Increment the counter by 1")]
    async fn increment(&self) -> Result<CallToolResult, McpError> {
        let mut counter = self.counter.lock().await;
        *counter += 1;
        Ok(CallToolResult::success(vec![Content::text(
            counter.to_string(),
        )]))
    }

    /// Decrement the counter by 1.
    #[tool(description = "Decrement the counter by 1")]
    async fn decrement(&self) -> Result<CallToolResult, McpError> {
        let mut counter = self.counter.lock().await;
        *counter -= 1;
        Ok(CallToolResult::success(vec![Content::text(
            counter.to_string(),
        )]))
    }

    /// Get the current counter value.
    #[tool(description = "Get the current counter value")]
    async fn get_value(&self) -> Result<CallToolResult, McpError> {
        let counter = self.counter.lock().await;
        Ok(CallToolResult::success(vec![Content::text(
            counter.to_string(),
        )]))
    }

    /// Say hello to the client.
    #[tool(description = "Say hello to the client")]
    fn say_hello(&self) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text("hello")]))
    }

    /// Repeat what you say.
    #[tool(description = "Repeat what you say")]
    fn echo(
        &self,
        rmcp::handler::server::wrapper::Parameters(object): rmcp::handler::server::wrapper::Parameters<JsonObject>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::Value::Object(object).to_string(),
        )]))
    }

    /// Calculate the sum of two numbers.
    #[tool(description = "Calculate the sum of two numbers")]
    fn sum(
        &self,
        rmcp::handler::server::wrapper::Parameters(SumInput { a, b }): rmcp::handler::server::wrapper::Parameters<SumInput>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(
            (a + b).to_string(),
        )]))
    }
}

// ─── Prompts ────────────────────────────────────────────────────────────────

#[prompt_router]
impl WeatherServer {
    /// A simple example prompt that takes one required argument.
    #[prompt(name = "example_prompt")]
    async fn example_prompt(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<ExamplePromptArgs>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let prompt = format!(
            "This is an example prompt with your message here: '{}'",
            args.message
        );
        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            prompt,
        )])
    }

    /// Analyze the weather for a city, combining with counter state.
    #[prompt(name = "weather_analysis")]
    async fn weather_analysis(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<WeatherAnalysisArgs>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        let style = args.style.unwrap_or_else(|| "brief".to_string());
        let counter_val = *self.counter.lock().await;
        let weather_text = weather_data(&args.city)
            .unwrap_or("No weather data available for this city");

        let messages = vec![
            PromptMessage::new_text(
                PromptMessageRole::Assistant,
                "I'll analyze the weather situation and suggest the best approach.",
            ),
            PromptMessage::new_text(
                PromptMessageRole::User,
                format!(
                    "City: {}\nWeather: {}\nCounter value: {}\nStyle preference: {}\n\nPlease analyze the weather and suggest activities.",
                    args.city, weather_text, counter_val, style
                ),
            ),
        ];

        Ok(GetPromptResult::new(messages).with_description(format!(
            "Weather analysis for {}",
            args.city
        )))
    }
}

// ─── ServerHandler (resources, info, prompts) ───────────────────────────────

#[prompt_handler]
#[tool_handler]
impl ServerHandler for WeatherServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_prompts()
                .enable_resources()
                .enable_tools()
                .build(),
        )
        .with_server_info(Implementation::from_build_env())
        .with_instructions(
            "Weather & Counter MCP server. Tools: get_weather, increment, decrement, \
             get_value, say_hello, echo, sum. Prompts: example_prompt, weather_analysis. \
             Resources: weather://cities, counter://value."
                .to_string(),
        )
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text("weather://cities", "Available Cities"),
                self._create_resource_text("counter://value", "Counter Value"),
            ],
            next_cursor: None,
            meta: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let uri = &request.uri;
        match uri.as_str() {
            "weather://cities" => {
                let cities = "Available cities: Tokyo, London, New York, Paris, Sydney";
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    cities,
                    uri.clone(),
                )]))
            }
            "counter://value" => {
                let val = self.counter.lock().await;
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    val.to_string(),
                    uri.clone(),
                )]))
            }
            _ => Err(McpError::resource_not_found(
                "resource_not_found",
                Some(json!({ "uri": uri })),
            )),
        }
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
            meta: None,
        })
    }
}

// ─── Server Mode ────────────────────────────────────────────────────────────

async fn run_server() -> Result<()> {
    eprintln!("=== Rust — MCP Weather & Counter Server (stdio) ===");
    let server = WeatherServer::new();
    let transport = rmcp::transport::stdio();
    let service = server.serve(transport).await?;
    service.waiting().await?;
    Ok(())
}

// ─── Client Mode ────────────────────────────────────────────────────────────

async fn run_client() -> Result<()> {
    println!("=== Rust — MCP Client → Weather & Counter Server ===");
    println!();

    // Spawn ourselves with --server flag as the MCP server child process
    let exe = env::current_exe()?;
    let transport = TokioChildProcess::new(
        Command::new(&exe).configure(|cmd| {
            cmd.arg("--server");
        }),
    )?;

    let client = ().serve(transport).await?;

    // ── 1. List available tools ─────────────────────────────────────────
    let tools = client.list_tools(None).await?;
    let tool_names: Vec<&str> = tools.tools.iter().map(|t| &*t.name).collect();
    println!("📦 Available tools: {tool_names:?}");
    println!();

    // ── 2. Call get_weather ─────────────────────────────────────────────
    println!("── Weather Tools ──");
    for city in ["Tokyo", "London", "Sydney", "Mars"] {
        let result = client
            .call_tool(
                CallToolRequestParams::new("get_weather")
                    .with_arguments(json!({"city": city}).as_object().unwrap().clone()),
            )
            .await?;
        if let Some(content) = result.content.first() {
            println!(
                "  {}",
                content.raw.as_text().map(|t| t.text.as_str()).unwrap_or("???")
            );
        }
    }

    // ── 3. Counter tools ────────────────────────────────────────────────
    println!();
    println!("── Counter Tools ──");

    // Increment 3 times
    for _ in 0..3 {
        client
            .call_tool(CallToolRequestParams::new("increment"))
            .await?;
    }
    let result = client
        .call_tool(CallToolRequestParams::new("get_value"))
        .await?;
    println!(
        "  After 3 increments: {}",
        result.content.first().and_then(|c| c.raw.as_text().map(|t| t.text.as_str())).unwrap_or("???")
    );

    // Decrement once
    client
        .call_tool(CallToolRequestParams::new("decrement"))
        .await?;
    let result = client
        .call_tool(CallToolRequestParams::new("get_value"))
        .await?;
    println!(
        "  After 1 decrement:  {}",
        result.content.first().and_then(|c| c.raw.as_text().map(|t| t.text.as_str())).unwrap_or("???")
    );

    // ── 4. say_hello ────────────────────────────────────────────────────
    println!();
    println!("── Misc Tools ──");
    let result = client
        .call_tool(CallToolRequestParams::new("say_hello"))
        .await?;
    println!(
        "  say_hello → {}",
        result.content.first().and_then(|c| c.raw.as_text().map(|t| t.text.as_str())).unwrap_or("???")
    );

    // ── 5. echo ─────────────────────────────────────────────────────────
    let result = client
        .call_tool(
            CallToolRequestParams::new("echo")
                .with_arguments(json!({"msg": "Hello MCP!", "n": 42}).as_object().unwrap().clone()),
        )
        .await?;
    println!(
        "  echo      → {}",
        result.content.first().and_then(|c| c.raw.as_text().map(|t| t.text.as_str())).unwrap_or("???")
    );

    // ── 6. sum ──────────────────────────────────────────────────────────
    let result = client
        .call_tool(
            CallToolRequestParams::new("sum")
                .with_arguments(json!({"a": 17, "b": 25}).as_object().unwrap().clone()),
        )
        .await?;
    println!(
        "  sum(17,25) → {}",
        result.content.first().and_then(|c| c.raw.as_text().map(|t| t.text.as_str())).unwrap_or("???")
    );

    // ── 7. Resources ────────────────────────────────────────────────────
    println!();
    println!("── Resources ──");
    let resources = client.list_resources(None).await?;
    for r in &resources.resources {
        println!("  📄 {} ({})", r.name, r.uri);
    }

    let cities = client
        .read_resource(serde_json::from_value(json!({ "uri": "weather://cities" }))?)
        .await?;
    if let Some(c) = cities.contents.first() {
        if let ResourceContents::TextResourceContents { text, .. } = c {
            println!("  → {}", text);
        }
    }

    let counter_res = client
        .read_resource(serde_json::from_value(json!({ "uri": "counter://value" }))?)
        .await?;
    if let Some(c) = counter_res.contents.first() {
        if let ResourceContents::TextResourceContents { text, .. } = c {
            println!("  → Counter value: {}", text);
        }
    }

    // ── 8. Prompts ──────────────────────────────────────────────────────
    println!();
    println!("── Prompts ──");
    let prompts = client.list_prompts(None).await?;
    for p in &prompts.prompts {
        let arg_names: Vec<&str> = p
            .arguments
            .as_ref()
            .map(|args| args.iter().map(|a| a.name.as_str()).collect())
            .unwrap_or_default();
        println!("  💬 {} (args: {:?})", p.name, arg_names);
    }

    let prompt_result = client
        .get_prompt(serde_json::from_value(json!({
            "name": "example_prompt",
            "arguments": { "message": "Hello from client!" }
        }))?)
        .await?;
    if let Some(msg) = prompt_result.messages.first() {
        let text = match &msg.content {
            PromptMessageContent::Text { text } => text.as_str(),
            _ => "???",
        };
        println!("  → example_prompt: {}", text);
    }

    let prompt_result = client
        .get_prompt(serde_json::from_value(json!({
            "name": "weather_analysis",
            "arguments": { "city": "Tokyo", "style": "detailed" }
        }))?)
        .await?;
    if let Some(msg) = prompt_result.messages.last() {
        let text = match &msg.content {
            PromptMessageContent::Text { text } => text.as_str(),
            _ => "???",
        };
        println!("  → weather_analysis: {}", text);
    }

    println!();
    println!("✓ Full MCP client-server communication successful!");

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
