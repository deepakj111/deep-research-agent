# mcp_servers/web_search/Dockerfile
# Web Search MCP Server — Tavily-backed, SSE transport, JWT auth.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cache DB will live inside the container (ephemeral) or on a volume
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8001/health"]

CMD ["python", "server.py"]
