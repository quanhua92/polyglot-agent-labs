FROM rust:latest

WORKDIR /app

# Copy source code
COPY . .

# Build the application
RUN cargo build --release

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 3000

ENV WORKFLOW_MCP_URL=http://workflow-server:8003

CMD ["/app/target/release/faq-api-gateway"]
