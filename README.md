# DeepResearch Agent

An autonomous deep research agent that accepts a natural language query and produces a structured, cited, multi-source research report — drawing from live web search, academic papers (arXiv), and GitHub repositories simultaneously.

## Architecture

- **Agent Orchestration**: LangGraph (stateful, resumable, conditional graphs)
- **MCP Servers**: FastMCP over HTTP/SSE transport (Web Search, arXiv, GitHub)
- **LLMs**: OpenAI GPT-4o (primary) + Anthropic Claude Sonnet (synthesis)
- **Frontend**: Streamlit with live SSE streaming
- **API Gateway**: FastAPI async SSE endpoints
- **Observability**: LangSmith tracing + custom SQLite run logger
- **Evaluation**: LLM-as-judge pipeline (DeepEval)

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Docker Desktop

### Setup

```bash
git clone https://github.com/deepakj111/deep-research-agent.git
cd deep-research-agent
uv sync --dev
source .venv/bin/activate
cp .env.example .env
nano .env
```

### Required API Keys

| Key | Get it from |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |
| `GITHUB_TOKEN` | GitHub → Settings → Developer Settings → Tokens |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) |
| `MCP_JWT_SECRET` | Any random string, 32+ characters |

### Commands

```bash
make run        # Start all services via Docker
make test       # Run tests with coverage
make lint       # Run ruff + mypy
make security   # Run bandit + pip-audit + uv-secure
make benchmark  # Run LLM-as-judge eval (Phase 3+)
```

## Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Foundations & Scaffolding | ✅ Complete |
| Phase 1 | Core Agent & MCP Servers | 🔄 In Progress |
| Phase 2 | Advanced Agent Intelligence | ⏳ Pending |
| Phase 3 | Observability & Evaluation | ⏳ Pending |
| Phase 4 | Production Infrastructure | ⏳ Pending |
| Phase 5 | UI/UX & Streaming | ⏳ Pending |
| Phase 6 | Enterprise Features & Polish | ⏳ Pending |

## Development

```bash
# Install pre-commit hooks (once after cloning)
uv run pre-commit install

# Run all quality checks
make lint

# Run security audit
make security
```
