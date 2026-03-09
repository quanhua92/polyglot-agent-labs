# TODO — Polyglot Agent Labs Use Cases

A progressive roadmap of use cases, each implemented in **Python** and **Rust** side-by-side.

---

## ✅ 00 — Hello World (Environment Check)

- **Status:** Done
- **Python:** `dotenv` + env var check
- **Rust:** `std::env` + env var check
- Validates that the project setup and API keys are working.

---

## 📋 01 — Simple LLM Completion (Multi-Provider)

> Send a simple prompt to an LLM and print the response. Support switching between providers (OpenAI, OpenRouter, Anthropic) via an env var like `LLM_PROVIDER`.

**Key learning:** Provider-agnostic LLM calls, API key routing, response parsing.

### Python (`litellm`)
- [x] Set up `pyproject.toml` with `litellm`, `python-dotenv`
- [x] Read `LLM_PROVIDER` env var to pick provider (openai / anthropic / openrouter)
- [x] Build a provider config map: `{"openai": "gpt-4.1-nano", "anthropic": "claude-3-haiku-20240307", "openrouter": "stepfun/step-3.5-flash:free"}`
- [x] Send a `"Hello! Tell me a fun fact about programming."` prompt via the chosen provider
- [x] Print the model name, provider, and full response text
- [x] Handle errors gracefully (missing API key, network error, invalid provider)

### Rust (`rig-core 0.32`)
- [x] Set up `Cargo.toml` with `rig-core 0.32`, `tokio`, `anyhow`
- [x] Read `LLM_PROVIDER` env var
- [x] Initialize the appropriate `rig` provider client (`openai::Client`, `anthropic::Client`, `openrouter::Client`)
- [x] Send the same prompt and print the response
- [x] Handle errors with `anyhow`

---

## 📋 02 — Simple MCP Server (Tool Serving)

> Build a minimal MCP (Model Context Protocol) server that exposes a `get_weather(city)` tool, then call it from a client.

**Key learning:** MCP protocol basics, defining tools, stdio/SSE transport, cross-language interop.

### Python (`fastmcp`)
- [x] Set up `pyproject.toml` with `fastmcp` dependency
- [x] Define a `get_weather(city: str) -> str` tool using `@mcp.tool()` decorator
- [x] Implement mock weather data (return hardcoded or random weather for a given city)
- [x] Configure the server to run over **stdio** transport
- [x] Write a simple client script (`client.py`) that connects to the server and calls the tool
- [x] Print the tool result from the client side
- [x] Test: start server, run client, verify the response

### Rust (`rmcp`)
- [x] Set up `Cargo.toml` with `rmcp` crate + `tokio`
- [x] Define a `GetWeather` tool struct implementing the `rmcp` tool trait
- [x] Implement the handler returning mock weather data
- [x] Serve over **stdio** transport
- [x] Write a client binary (or test) that connects and calls `get_weather`
- [x] Verify cross-language interop: Python client → Rust server (and vice versa)

---

## ✅ 03 — Conversational Agent with Memory

- **Status:** Done
- **Python:** `langchain>=1.0.10` + conversation REPL
- **Rust:** `rig-core>=0.32` + prompt-based conversation history

> Build a chatbot that maintains conversation history across multiple turns. Supports both non-interactive (predefined conversation) and interactive REPL modes.

**Key learning:** State management, conversation loops, context window handling, graceful exit.

### Python (`langchain`)
- [x] Set up `pyproject.toml` with `langchain>=1.0.10`, `langchain-openai`, `langchain-anthropic`
- [x] Define a message history list with `SystemMessage`, `HumanMessage`, `AIMessage`
- [x] Build a simple conversation loop with LLM calls
- [x] Implement `--interactive` flag for REPL mode (`while True` with `input()`)
- [x] Default to non-interactive mode with predefined conversation
- [x] Accumulate messages in history across turns
- [x] Support `/quit`, `/exit`, `/q` commands to break the loop (interactive mode)
- [x] Support multi-provider via `LLM_PROVIDER` env var (openai, anthropic, openrouter)
- [x] Print turn count on exit

### Rust (`rig-core`)
- [x] Set up `Cargo.toml` with `rig-core>=0.32`, `tokio`
- [x] Use `AgentBuilder` to create agent from completion model
- [x] Implement prompt-based conversation history (appending to prompt string)
- [x] Implement `--interactive` flag for REPL mode (loop with stdin)
- [x] Default to non-interactive mode with predefined conversation
- [x] Support `/quit`, `/exit`, `/q` to break (interactive mode)
- [x] Support multi-provider via `LLM_PROVIDER` env var (openai, anthropic, openrouter)
- [x] Print turn count on exit

---

## ✅ 04 — Agent with Tool Use (Function Calling)

- **Status:** Done
- **Python:** `langgraph>=1.0.10` + tool definitions with `@tool` decorator
- **Rust:** `rig-core>=0.32` + Tool trait implementation

> Create an agent that can decide when to call external tools (calculator, date/time, string utilities) to answer user questions. The LLM chooses which tool to invoke.

**Key learning:** Function calling / tool-use APIs, tool schemas, agent decision loops, structured output.

### Python (`langgraph` + `@tool`)
- [x] Set up `pyproject.toml` with `langgraph`, `langchain-openai`
- [x] Define 3 tools using `@tool` decorator:
  - [x] `calculator(expression: str) -> str` — evaluates a math expression
  - [x] `get_current_time() -> str` — returns current date/time
  - [x] `string_length(text: str) -> int` — returns length of a string
- [x] Build a LangGraph graph with tool nodes: `agent → should_call_tool? → tool_node → agent`
- [x] Implement the conditional edge (if LLM response has tool calls → route to tool node, else → end)
- [x] Run 3 demo prompts:
  - [x] `"What is 42 * 137?"` → should use calculator
  - [x] `"What time is it right now?"` → should use get_current_time
  - [x] `"How many characters in 'Polyglot Agent Labs'?"` → should use string_length
- [x] Print the agent's reasoning chain (tool calls + final answer)

### Rust (`rig-core` + Tool trait)
- [x] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`
- [x] Define the same 3 tools using rig's `Tool` trait
- [x] Attach tools to a rig agent via `.tool(...)` builder
- [x] Run the same 3 demo prompts
- [x] Print tool invocations and final responses
- [x] Compare output format with Python version

---

## ✅ 05 — RAG over Local Documents

- **Status:** Done
- **Python:** `langchain` + FAISS with hard-coded documents
- **Rust:** `rig-core` + custom embeddings via OpenAI API

> Index hard-coded documents as strings, then answer questions using retrieval-augmented generation.

**Key learning:** Embeddings, chunking strategies, vector similarity search, retrieval chains, prompt engineering for grounded answers.

**Implementation Notes:**
- Documents are hard-coded as strings instead of loading from files (simplification)
- Multi-provider support for both LLM and embeddings (default: openrouter)
- Uses cosine similarity for vector search

### Shared Setup
- [x] 3 hard-coded documents (FAQ, Rust Guidelines, Python Practices) at ~200-300 words each
- [x] Same 4 demo questions for cross-language verification (including edge case)

### Python (`langchain` + FAISS)
- [x] Set up `pyproject.toml` with `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-community`, `faiss-cpu`
- [x] Hard-coded documents as constant strings
- [x] Split documents into chunks using `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50)
- [x] Generate embeddings via `OpenAIEmbeddings` with multi-provider support
- [x] Store embeddings in a FAISS vector store (in-memory)
- [x] Build a RAG pipeline: query → retrieve top-3 chunks → format context → LLM → answer
- [x] Run 4 demo questions that require information from the docs
- [x] Print: retrieved chunks (with source file) + final answer
- [x] Handle edge case: question not answerable from docs → agent says "I don't have enough information"

### Rust (`rig-core`)
- [x] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`, `reqwest`
- [x] Hard-coded same 3 documents as static string literals
- [x] Manual chunking by paragraphs with configurable chunk size
- [x] Generate embeddings using direct OpenAI API calls via `reqwest`
- [x] Custom in-memory vector store with cosine similarity search
- [x] Build a RAG pipeline: embed query → search top-3 → format context → LLM call
- [x] Run the same 4 demo questions
- [x] Print retrieved chunks and final answer
- [x] Multi-provider support for LLM and embeddings

---

## ✅ 06 — Structured Output & Data Extraction

- **Status:** Done
- **Python:** `langchain` + Pydantic with hard-coded sample inputs
- **Rust:** `rig-core` + `serde` with hard-coded sample inputs

> Given unstructured text (job listing, product review, email), extract structured data into a typed schema. The LLM outputs validated JSON matching the schema.

**Key learning:** JSON mode / structured output APIs, schema validation, error recovery, typed LLM outputs.

### Shared Setup
- [x] Create 3 sample inputs as hard-coded strings: a job listing, a product review, and an email
- [x] Define the target schemas (e.g., `JobListing { title, company, salary_range, skills[] }`)

### Python (`langchain` + Pydantic)
- [x] Set up `pyproject.toml` with `langchain`, `langchain-openai`, `pydantic`
- [x] Define Pydantic models for each extraction target:
  - [x] `JobListing` — title, company, location, salary_range, required_skills, employment_type, description
  - [x] `ProductReview` — product_name, rating, pros, cons, summary, would_recommend
  - [x] `EmailInfo` — sender, recipients, subject, action_items, urgency, key_points, deadline
- [x] Use `llm.with_structured_output(Model)` to extract data
- [x] Run extraction on each sample input and pretty-print the resulting model
- [x] Validate: check that all required fields are populated
- [x] Handle malformed output gracefully with retry logic (max 2 retries)

### Rust (`rig-core` + `serde`)
- [x] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`
- [x] Define equivalent Rust structs with `#[derive(Deserialize, Serialize)]`
- [x] Use agent.chat() with explicit JSON schema prompts to parse LLM response into structs
- [x] Run the same 3 extractions and print with `serde_json::to_string_pretty`
- [x] Handle JSON extraction from LLM responses with helper function

---

## 📋 07 — Streaming Responses + Human-in-the-Loop

> Stream LLM tokens to the terminal in real-time. Then add an approval gate: the agent proposes an action and pauses for user confirmation before executing.

**Key learning:** Token streaming, async iteration, interrupt/resume patterns, approval workflows.

### Python (`langgraph`)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`
- [ ] **Part A — Streaming:** Send a prompt and stream tokens to stdout using `astream_events` or `stream`
- [ ] Print tokens as they arrive (no buffering), show total time elapsed
- [ ] **Part B — Human-in-the-Loop:**
  - [ ] Build a graph: `generate_plan → human_approval → execute_plan`
  - [ ] Use LangGraph's `interrupt()` to pause before execution
  - [ ] Prompt: `"Draft an email to cancel my gym membership"`
  - [ ] Agent generates the email draft → prints it → asks `"Send this? [y/n]"`
  - [ ] If approved → print "✓ Sent!" / If rejected → print "✗ Cancelled"
- [ ] Handle timeout (auto-reject after 30 seconds)

### Rust (`rig-core`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `futures`
- [ ] **Part A — Streaming:** Use rig's streaming completion, iterate over the stream and print chunks
- [ ] **Part B — Human-in-the-Loop:**
  - [ ] Implement a manual approval gate: LLM generates plan → print → read stdin → proceed or abort
  - [ ] Same email cancellation prompt
- [ ] Compare streaming latency (time-to-first-token) with Python

---

## 📋 08 — Multi-Agent Collaboration (Researcher + Writer)

> Two agents work together: a **Researcher** gathers information using tools, then a **Writer** takes the findings and produces a polished document. Orchestrated as a pipeline.

**Key learning:** Agent composition, role specialization, inter-agent communication, workflow orchestration.

### Python (`langgraph`)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`
- [ ] Define **Researcher** agent:
  - [ ] System prompt: "You are a research assistant. Gather key facts about the given topic."
  - [ ] Has access to a `search_notes(query)` tool (mock: searches a local knowledge base)
  - [ ] Returns structured findings: `{ facts: [...], sources: [...] }`
- [ ] Define **Writer** agent:
  - [ ] System prompt: "You are a technical writer. Turn research findings into a clear, well-structured article."
  - [ ] Takes Researcher output as context
  - [ ] Returns a formatted markdown document
- [ ] Build a LangGraph: `researcher_node → writer_node → output`
  - [ ] Optionally add a **Reviewer** node that sends feedback back to Writer (loop)
- [ ] Demo topic: `"Explain the benefits of Rust for systems programming"`
- [ ] Print both the raw research findings and the final polished article

### Rust (`rig-core`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`
- [ ] Create two separate rig agents with different system prompts and tools
- [ ] Pipe the output of the Researcher into the Writer's prompt
- [ ] Implement the same demo topic
- [ ] Compare article quality and structure with Python output

---

## 📋 09 — Web Research Agent

> Given a question, the agent searches the web, reads results, and produces a cited summary. Multi-step: search → read → evaluate → synthesize → cite.

**Key learning:** External API integration, iterative reasoning, source citation, multi-step tool chains.

### Python (`langgraph` + Tavily)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`, `tavily-py`
- [ ] Add `TAVILY_API_KEY` to `.env.example`
- [ ] Define tools:
  - [ ] `web_search(query: str) -> list[SearchResult]` — calls Tavily API
  - [ ] `read_page(url: str) -> str` — fetches and extracts text from a URL
- [ ] Build a LangGraph agent loop:
  - [ ] Agent decides: search → read specific results → search again if needed → synthesize
  - [ ] Max 3 search iterations to prevent infinite loops
- [ ] Demo questions:
  - [ ] `"What are the latest developments in Rust async runtime?"`
  - [ ] `"Compare FastAPI vs Django for building REST APIs in 2025"`
- [ ] Output: summary with inline `[1]`, `[2]` citations + source list at the bottom
- [ ] Print total number of searches and pages read

### Rust (`rig-core` + `reqwest`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `reqwest`, `serde`
- [ ] Implement a `WebSearch` tool that calls Tavily (or SerpAPI) via HTTP
- [ ] Implement a `ReadPage` tool using `reqwest` + basic HTML-to-text extraction
- [ ] Build the same iterative agent loop with max 3 iterations
- [ ] Run the same demo questions and compare citation quality

---

## 📋 10 — Customer Support Agent

> A complete support pipeline: classify intent → retrieve knowledge base articles → generate response → escalate if needed. Handles multi-turn conversations.

**Key learning:** Intent classification, conditional routing, knowledge bases, escalation patterns, conversation flows.

### Shared Setup
- [ ] Create a `knowledge_base/` folder with 5–10 FAQ-style articles (returns policy, shipping info, billing, account issues, etc.)
- [ ] Define intent categories: `billing`, `shipping`, `returns`, `account`, `general`, `escalate`

### Python (`langgraph`)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`, `faiss-cpu`
- [ ] Build the graph with these nodes:
  - [ ] **Intent Classifier** — determines intent + confidence from user message
  - [ ] **KB Retriever** — RAG lookup in knowledge base (reuses patterns from use case 05)
  - [ ] **Response Generator** — drafts a helpful reply using KB context
  - [ ] **Escalation Check** — if confidence < threshold or user explicitly asks → route to human
  - [ ] **Human Escalation** — prints "🚨 Transferring to human agent..." and logs the conversation
- [ ] Implement as an interactive REPL (multi-turn, remembers context)
- [ ] Demo scenarios:
  - [ ] `"How do I return a product?"` → retrieves return policy → generates answer
  - [ ] `"I want to speak to a manager"` → escalates
  - [ ] `"What's the status of order #12345?"` → responds with info or escalates
- [ ] Print: detected intent, confidence score, retrieved articles, final response

### Rust (`rig-core`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`
- [ ] Implement intent classification using structured output (returns `Intent` enum + confidence)
- [ ] Reuse RAG pattern from use case 05 for KB retrieval
- [ ] Build conditional routing: intent → handler function → response or escalation
- [ ] Same REPL-style demo with multi-turn support

---

## 📋 11 — Code Review Agent

> An agent that reads source code files, analyzes them, and provides structured code review feedback (bugs, style issues, security concerns, improvement suggestions).

**Key learning:** Large context handling, code analysis prompts, structured feedback, file I/O tools.

### Python (`langgraph`)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`
- [ ] Define tools:
  - [ ] `read_file(path: str) -> str` — reads a source file
  - [ ] `list_files(directory: str) -> list[str]` — lists files in a directory
- [ ] Define structured output schema:
  - [ ] `ReviewFinding { file, line, severity, category, message, suggestion }`
  - [ ] `CodeReview { summary, findings[], overall_score }`
- [ ] Build an agent that:
  - [ ] Reads one or more files from a sample `code_samples/` directory
  - [ ] Analyzes each file for bugs, style, security, and performance
  - [ ] Returns a structured `CodeReview`
- [ ] Provide 2–3 sample code files with intentional issues (e.g., SQL injection, unused variables, poor error handling)
- [ ] Print a formatted review report (table of findings + overall score)

### Rust (`rig-core`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`, `schemars`
- [ ] Implement `ReadFile` and `ListFiles` tools
- [ ] Define equivalent Rust structs for review output
- [ ] Run on the same sample files and compare feedback quality

---

## 📋 12 — Data Analysis Agent (CSV / SQL)

> An agent that can load a CSV file, understand its schema, and answer analytical questions by writing and executing code (Python) or SQL queries.

**Key learning:** Code generation + execution, sandboxing, data reasoning, tabular data handling.

### Shared Setup
- [ ] Provide a sample CSV file (e.g., `sales_data.csv` with ~100 rows: date, product, region, revenue, units)

### Python (`langgraph` + `pandas`)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`, `pandas`
- [ ] Define tools:
  - [ ] `load_csv(path: str) -> str` — loads CSV and returns schema + first 5 rows
  - [ ] `run_pandas_query(code: str) -> str` — executes a pandas snippet and returns the result
- [ ] Build an agent loop: user question → generate pandas code → execute → interpret result → answer
- [ ] Demo questions:
  - [ ] `"What was the total revenue in Q1?"` → agent writes `df[df['date']...].revenue.sum()`
  - [ ] `"Which product had the highest sales?"` → agent writes groupby + sort
  - [ ] `"Show me a monthly trend summary"` → agent writes resample + agg
- [ ] Print: generated code + execution result + natural language answer
- [ ] Handle errors: if generated code fails, retry with error feedback

### Rust (`rig-core` + `polars`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `polars`, `serde`
- [ ] Implement a `LoadCsv` tool and a `RunQuery` tool (generate + execute polars expressions)
- [ ] Since Rust can't easily eval arbitrary code, consider:
  - [ ] Pre-defined query templates the LLM selects from, OR
  - [ ] Agent generates a polars expression as JSON → interpret and execute it
- [ ] Run the same demo questions and compare approaches

---

## 📋 13 — Workflow Automation Agent

> An agent that takes a high-level instruction (e.g., "Schedule a meeting with the team about the Q2 roadmap") and breaks it into steps, executing each via tool calls. Simulates real-world integrations.

**Key learning:** Task decomposition, sequential tool execution, error recovery, real-world API simulation.

### Python (`langgraph`)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`
- [ ] Define mock tools simulating external services:
  - [ ] `send_email(to, subject, body) -> str` — logs email details, returns confirmation
  - [ ] `create_calendar_event(title, date, attendees) -> str` — logs event, returns ID
  - [ ] `create_task(title, assignee, due_date) -> str` — logs task, returns ID
  - [ ] `search_contacts(query) -> list[Contact]` — returns mock contact list
- [ ] Build an agent that decomposes complex instructions into tool calls:
  - [ ] `"Schedule a meeting with Alice and Bob next Tuesday about Q2 planning, then email them the agenda"`
  - [ ] Agent should: search contacts → create event → draft agenda → send emails
  - [ ] `"Create a task for each team member to review the design doc by Friday"`
  - [ ] Agent should: search contacts → create multiple tasks
- [ ] Print the full execution plan (steps taken, tools called, results)
- [ ] Handle partial failures (e.g., contact not found → skip and report)

### Rust (`rig-core`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`, `chrono`
- [ ] Implement the same mock tools as Rust structs
- [ ] Build the same multi-step workflow execution
- [ ] Compare how each language handles the decomposition and error recovery

---

## 📋 14 — Content Writing Agent (Blog Generator)

> An agent that takes a topic and writes a full blog post through multiple stages: outline → research → draft → edit → final. Each stage is a distinct agent or prompt.

**Key learning:** Multi-stage pipelines, prompt chaining, quality control loops, long-form generation.

### Python (`langgraph`)
- [ ] Set up `pyproject.toml` with `langgraph`, `langchain-openai`
- [ ] Build a multi-stage pipeline graph:
  - [ ] **Outliner** — generates a structured outline (title, sections, key points per section)
  - [ ] **Researcher** — for each section, generates relevant talking points / facts
  - [ ] **Drafter** — writes the full prose for each section based on outline + research
  - [ ] **Editor** — reviews the draft, checks for coherence, tone, grammar; suggests edits
  - [ ] **Finalizer** — applies edits and produces the final markdown document
- [ ] Add a quality gate: Editor scores the draft 1–10; if < 7, send back to Drafter (max 2 revisions)
- [ ] Demo: `"Write a blog post about why developers should learn both Python and Rust"`
- [ ] Output: save final article as `output.md`, print word count and revision count

### Rust (`rig-core`)
- [ ] Set up `Cargo.toml` with `rig-core`, `tokio`, `serde`
- [ ] Implement each stage as a separate rig agent with a specialized system prompt
- [ ] Chain them together, passing structured output between stages
- [ ] Implement the same quality gate loop
- [ ] Save output to `output.md` and compare with Python version

---

## 📋 15 — Personal Assistant Agent (Capstone)

> A full-featured personal assistant that combines everything: conversational memory, tool use, RAG, structured output, streaming, and human-in-the-loop. The capstone project.

**Key learning:** Integrating all previous patterns, complex state management, graceful degradation, production-ready agent architecture.

### Python (`langgraph`)
- [ ] Set up `pyproject.toml` with all necessary dependencies
- [ ] Integrate capabilities from previous use cases:
  - [ ] 🗣️ **Conversational** — remembers context across turns (use case 03)
  - [ ] 🔧 **Tool use** — calculator, date/time, web search (use cases 04, 09)
  - [ ] 📚 **Knowledge base** — RAG over a personal notes directory (use case 05)
  - [ ] 📋 **Task management** — create/list/complete tasks (use case 13)
  - [ ] ✉️ **Email drafting** — drafts and sends emails with approval (use case 07)
  - [ ] 📊 **Data queries** — answer questions about CSV data (use case 12)
- [ ] Build a LangGraph with a central **Router** that classifies user intent and dispatches to the right sub-graph
- [ ] Implement streaming output for all responses
- [ ] Add human-in-the-loop for destructive actions (sending emails, creating events)
- [ ] Interactive REPL with rich output formatting
- [ ] Demo session: walk through 5+ diverse queries in a single conversation
- [ ] Print session summary on exit (tools called, tasks created, queries answered)

### Rust (`rig-core`)
- [ ] Set up `Cargo.toml` with all necessary dependencies
- [ ] Build a modular architecture: each capability is a separate module with its own tools
- [ ] Implement a central router agent that dispatches to capability modules
- [ ] Streaming + approval gates for destructive actions
- [ ] Same interactive REPL and demo session
- [ ] Compare: code complexity, performance, DX between Python and Rust
