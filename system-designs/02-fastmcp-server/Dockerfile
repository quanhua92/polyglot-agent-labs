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

# Run the server
CMD ["python", "server.py"]
