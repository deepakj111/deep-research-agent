# app/Dockerfile — Streamlit Frontend
# Serves the research UI on port 8501.

FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy root dependency files (Streamlit app imports agent/config modules)
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Copy necessary modules
COPY agent/ agent/
COPY api/ api/
COPY config/ config/
COPY observability/ observability/
COPY utils/ utils/
COPY app/ app/
COPY mcp_servers/__init__.py mcp_servers/__init__.py

RUN uv sync --frozen --no-dev

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8501/_stcore/health"]

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
