# DeepResearch Agent

An autonomous deep research agent that accepts a natural-language query and produces a **structured, cited, multi-source research report** — drawing from **live web search**, **academic papers (arXiv)**, and **GitHub repositories** simultaneously.

Built with **LangGraph** for stateful agent orchestration, **Model Context Protocol (MCP)** for standardised tool integration, and a **multi-model synthesis** pipeline (GPT-5-mini) with automated contradiction detection.

---

## Key Features

| Feature | Description |
|---|---|
| **MCP Tool Servers** | Three custom `FastMCP` servers (Web Search, arXiv, GitHub) running as independent microservices over HTTP/SSE with JWT authentication and SQLite caching. |
| **LangGraph Orchestration** | Stateful, resumable agent graph with `SqliteSaver` checkpointing and Human-in-the-Loop (HITL) plan approval via `interrupt_before`. |
| **Parallel Fan-Out** | Supervisor dispatches all sub-questions to all 3 agents concurrently using LangGraph's `Send` API — *N* questions × 3 agents = *3N* parallel tasks. |
| **Multi-Model Synthesis** | Primary and secondary model synthesis (GPT-5-mini default), followed by an automated fallback and contradiction tracking step. |
| **Critic Loop** | Quality-gated iteration: a critic node scores coverage, recency, depth, and source diversity, then decides whether to loop for more research or proceed to synthesis. |
| **Budget Guard** | Graph-integrated kill switch enforcing hard limits on iteration count and estimated USD cost — prevents runaway API spend. |
| **Per-Tool Retry Policies** | Configurable exponential/linear backoff per tool with graceful degradation: non-critical tools (arXiv, GitHub) degrade with notes; critical tools (web search) propagate failures. |
| **PII Filtering** | Regex-based middleware scrubbing SSN, credit card, email, phone, and IP address patterns from all MCP tool outputs before they enter agent state. |
| **Observability** | Custom async-safe SQLite tracer logging every tool call, node execution, token count, cost estimate, and evaluation score — independent of LangSmith. |
| **LLM-as-Judge Evaluation** | Automated 5-dimension scoring (faithfulness, answer relevancy, source coverage, citation accuracy, coherence) with CI quality gates. |
| **Full Docker Stack** | 5-service `docker-compose.yml`: 3 MCP servers → FastAPI agent API → Streamlit frontend, with health checks, memory limits, and non-root containers. |

---

## Architecture Overview

```
                    ┌─────────────┐
                    │  Streamlit  │  ← SSE streaming UI
                    │  :8501      │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  FastAPI    │  ← Agent gateway, SSE endpoints, HITL
                    │  API :8080  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────┴───┐ ┌──────┴──────┐ ┌──┴──────────┐
     │ Web Search │ │   arXiv     │ │   GitHub     │
     │ MCP :8001  │ │  MCP :8002  │ │  MCP :8003   │
     │ (SQLite    │ │ (SQLite     │ │ (SQLite      │
     │  cache)    │ │  cache)     │ │  cache)      │
     └────────────┘ └─────────────┘ └──────────────┘
```

> For a detailed architecture walkthrough, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Quick Start

### Prerequisites

- **Python 3.11** (exact match required)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Docker Desktop** — for the containerised MCP servers

### Setup

```bash
# Clone and install
git clone https://github.com/deepakj111/deep-research-agent.git
cd deep-research-agent
uv sync --dev
cp .env.example .env
```

Edit `.env` with your API keys:

| Key | Purpose | Get it from |
|---|---|---|
| `OPENAI_API_KEY` | Primary & Secondary LLMs (`gpt-5-mini` default) | [platform.openai.com](https://platform.openai.com/api-keys) |
| `TAVILY_API_KEY` | Web search | [app.tavily.com](https://app.tavily.com) |
| `GITHUB_TOKEN` | GitHub API | GitHub → Settings → Developer Settings → Tokens |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) | [smith.langchain.com](https://smith.langchain.com) |
| `MCP_JWT_SECRET` | MCP server authentication | Any random string, 32+ characters |
| `FRONTEND_PASSWORD` | Streamlit login gate | Any password string |

### Running

```bash
# Production — full Docker stack (5 services)
make run

# Development — hot-reload with volume mounts
make dev

# Tests
make test

# Linting (ruff + mypy)
make lint

# Security audit (uv audit + bandit)
make security

# LLM-as-judge benchmark
make benchmark

# Docker image build verification
make docker-test
```

---

## Project Structure

```
deep-research-agent/
├── agent/                        # LangGraph agent core
│   ├── graph.py                  # StateGraph definition with HITL + budget guard
│   ├── state.py                  # Pydantic models + TypedDict state
│   ├── budget_guard.py           # Cost/iteration kill switch
│   ├── circuit_breaker.py        # Circuit breaker (closed/open/half-open)
│   ├── retry_policy.py           # Per-tool retry + backoff policies
│   ├── middleware/
│   │   └── pii_filter.py         # PII redaction (SSN, CC, email, phone, IP)
│   ├── nodes/
│   │   ├── classifier.py         # Query difficulty classification
│   │   ├── planner.py            # HITL sub-question generation
│   │   ├── supervisor.py         # Parallel fan-out via Send API
│   │   ├── web_agent.py          # Web search sub-agent
│   │   ├── arxiv_agent.py        # arXiv sub-agent
│   │   ├── github_agent.py       # GitHub sub-agent
│   │   ├── critic.py             # Quality evaluation + loop decision
│   │   ├── synthesizer.py        # Primary synthesis with secondary fallback
│   │   └── writer.py             # Citation builder + report finalisation
│   └── prompts/                  # YAML prompt templates
├── api/
│   └── main.py                   # FastAPI gateway (SSE streaming, HITL)
├── app/
│   └── streamlit_app.py          # Streamlit frontend (Phase 5)
├── mcp_servers/
│   ├── web_search/               # Tavily-backed web search MCP
│   ├── arxiv/                    # arXiv Atom XML parser MCP
│   └── github/                   # GitHub REST API MCP
├── config/
│   ├── settings.py               # Pydantic Settings (env vars)
│   ├── profiles.py               # Centralized YAML profile loader (cached)
│   └── profiles/                 # fast.yaml / deep.yaml research profiles
├── observability/
│   ├── tracer.py                 # Async SQLite run tracer
│   └── schema.sql                # Tracer schema
├── evaluation/
│   ├── evaluator.py              # LLM-as-judge (5 dimensions)
│   ├── run_benchmark.py          # Benchmark runner with CI gate
│   └── benchmark_queries.json    # 10-query evaluation dataset
├── utils/
│   ├── auth.py                   # JWT token generation for MCP server auth
│   ├── callbacks.py              # LangChain token usage callback
│   ├── cost_estimator.py         # Dynamic LLM pricing (LiteLLM community data)
│   ├── logger.py                 # Centralized logging setup
│   └── report_formatter.py       # ReportOutput → Markdown / HTML / PDF
├── tests/
│   ├── unit/                     # 131 unit tests
│   └── integration/              # Full pipeline integration test
├── .github/workflows/
│   ├── ci.yml                    # Lint + Test + Docker build
│   ├── eval.yml                  # Weekly LLM-as-judge benchmark
│   ├── security.yml              # uv audit + bandit
│   └── secrets-check.yml         # Gitleaks secret scanning
├── docker-compose.yml            # Production 5-service stack
├── docker-compose.dev.yml        # Dev overrides (hot-reload)
├── Dockerfile                    # Agent API container
├── Makefile                      # Build/test/deploy commands
└── pyproject.toml                # uv project config
```

---

## Documentation

| Document | Description |
|---|---|
| **[Architecture](docs/ARCHITECTURE.md)** | LangGraph workflow, MCP protocol, state management, multi-model synthesis |
| **[Evaluation](docs/EVALUATION.md)** | LLM-as-judge pipeline, scoring rubric, benchmark dataset, CI integration |
| **[Infrastructure](docs/INFRASTRUCTURE.md)** | Docker stack, retry policies, circuit breakers, PII filter, budget guard |

---

## Development

```bash
# Install pre-commit hooks (once)
uv run pre-commit install

# Run all quality checks
make lint

# Run security audit
make security

# Run the full test suite
make test
```

### Pre-Commit Hooks

| Hook | Tool |
|---|---|
| Trailing whitespace, EOF fixes | built-in |
| Merge conflict detection | built-in |
| Private key detection | built-in |
| Import sorting | isort |
| Linting | ruff |
| Formatting | ruff format |
| Type checking | mypy |
| Security analysis | bandit |

---

## License

This project is developed as a portfolio demonstration of production-grade AI agent engineering.
