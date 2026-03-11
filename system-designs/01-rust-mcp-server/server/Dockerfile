FROM rust:1.94 as builder

WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/mcp-add-server /app/
EXPOSE 3001
ENTRYPOINT ["/app/mcp-add-server"]
