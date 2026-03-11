FROM python:3.13-slim

WORKDIR /app

# Install uv for faster package installation
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system fastmcp uvicorn

# Copy server code
COPY server.py ./

EXPOSE 8002

# Run the server with uvicorn (HTTP transport)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
