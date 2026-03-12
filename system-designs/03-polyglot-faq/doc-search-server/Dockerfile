FROM rust:latest

WORKDIR /app

# Copy source code
COPY . .

# Build the application
RUN cargo build --release

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8004

CMD ["/app/target/release/faq-doc-search"]
