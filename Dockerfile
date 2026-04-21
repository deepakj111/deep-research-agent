# Dockerfile — FastAPI Agent Gateway
# Serves the research agent API on port 8080.
# Built with uv for fast, reproducible installs.

FROM python:3.11-slim AS base

# Install curl for Docker HEALTHCHECK probes
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — production security best practice
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev group)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY agent/ agent/
COPY api/ api/
COPY config/ config/
COPY observability/ observability/
COPY evaluation/ evaluation/
COPY utils/ utils/
COPY mcp_servers/__init__.py mcp_servers/__init__.py

# Ensure the project package is installed
RUN uv sync --frozen --no-dev

# Switch to non-root user
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8080/health"]

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
