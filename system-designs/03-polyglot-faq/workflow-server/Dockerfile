FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install Python dependencies using pip
RUN pip install --no-cache-dir \
    fastmcp>=3.1.0 \
    langgraph>=0.2.0 \
    langchain-openai>=0.2.0 \
    langchain-anthropic>=0.2.0 \
    langchain-mcp-adapters>=0.1.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.27.0

# Copy application code
COPY server.py workflow.py ./

EXPOSE 8003

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
