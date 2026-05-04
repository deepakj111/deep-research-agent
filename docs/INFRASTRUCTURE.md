# Infrastructure

This document covers the production infrastructure of the DeepResearch Agent — Docker orchestration, fault tolerance patterns, security hardening, and cost controls.

---

## Table of Contents

- [Docker Stack](#docker-stack)
- [Service Configuration](#service-configuration)
- [Fault Tolerance](#fault-tolerance)
- [Security & Privacy](#security--privacy)
- [Cost Controls](#cost-controls)
- [CI/CD Pipelines](#cicd-pipelines)
- [Local Development](#local-development)

---

## Docker Stack

The production deployment is a **5-service Docker Compose stack** defined in `docker-compose.yml`:

```
┌──────────────────────────────────────────────────────────┐
│                    docker compose up                      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ streamlit (:8501) ──depends──▸ agent             │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ agent (:8080)                                     │   │
│  │   depends ──▸ web-search-mcp (healthy)            │   │
│  │   depends ──▸ arxiv-mcp (healthy)                 │   │
│  │   depends ──▸ github-mcp (healthy)                │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌────────────────┐ ┌──────────┐ ┌───────────────┐     │
│  │ web-search-mcp │ │arxiv-mcp │ │ github-mcp    │     │
│  │ :8001          │ │ :8002    │ │ :8003         │     │
│  └────────────────┘ └──────────┘ └───────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Startup Order

Docker Compose dependency conditions enforce strict ordering:

1. **MCP servers** start first (no dependencies)
2. **Agent API** starts after all three MCP servers are healthy (`curl /health` returns 200)
4. **Streamlit** starts after the agent API is healthy

---

## Service Configuration

### Dockerfiles

All Dockerfiles follow production best practices:

| Practice | Implementation |
|---|---|
| **Non-root user** | `useradd appuser` + `USER appuser` — containers never run as root |
| **Health check tooling** | `curl` installed for `HEALTHCHECK` probes |
| **Docker HEALTHCHECK** | Each container has a `HEALTHCHECK` instruction (not just compose-level) |
| **Layer caching** | `COPY requirements.txt` before `COPY . .` for optimal build cache |
| **Minimal base** | `python:3.11-slim` with `--no-install-recommends` |

### Resource Limits

| Service | Memory Limit | Restart Policy |
|---|---|---|
| Web Search MCP | 256M | `unless-stopped` |
| arXiv MCP | 256M | `unless-stopped` |
| GitHub MCP | 256M | `unless-stopped` |
| Agent API | 1G | `unless-stopped` |
| Streamlit | 512M | `unless-stopped` |

### Health Checks

| Service | Probe | Interval | Retries |
|---|---|---|---|
| MCP servers | `curl -f http://localhost:{port}/health` | 10s | 3 |
| Agent API | `curl -f http://localhost:8080/health` | 15s | 3 |
| Streamlit | `curl -f http://localhost:8501/_stcore/health` | 15s | 3 |

### Environment Variables

All secrets are injected via environment variables from a `.env` file (never baked into images):

```bash
# .env (not committed to git)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
GITHUB_TOKEN=ghp_...
MCP_JWT_SECRET=your-super-secret-jwt-key-min-32-chars
```

Inside Docker Compose, MCP URLs are overridden to use container DNS:

```yaml
WEB_SEARCH_MCP_URL: http://web-search-mcp:8001/sse
ARXIV_MCP_URL: http://arxiv-mcp:8002/sse
GITHUB_MCP_URL: http://github-mcp:8003/sse
```

---

## Fault Tolerance

The agent implements a **three-layer defence** against tool failures:

### Layer 1: Retry Policies (`agent/retry_policy.py`)

Each MCP tool has an explicit error policy:

| Tool | Max Retries | Backoff | Fallback |
|---|---|---|---|
| `search_web` | 3 | Exponential (1s, 2s, 4s) | Raise (critical tool) |
| `fetch_papers` | 2 | Linear (2s, 2s) | Degrade with note |
| `search_repos` | 2 | Linear (2s, 2s) | Degrade with note |

**Critical vs non-critical tools**: Web search failures propagate to the agent because web results are essential for most queries. arXiv and GitHub failures degrade gracefully — the agent continues with a `ToolDegradedError` that inserts a note into the report (e.g., "[arXiv unavailable — academic sources omitted from this section]").

### Layer 2: Circuit Breaker (`agent/circuit_breaker.py`)

Each tool has a singleton `CircuitBreaker` with three states:

```
CLOSED ──(failures ≥ threshold)──▸ OPEN ──(recovery timeout)──▸ HALF-OPEN
   ▲                                                               │
   └──────────────(success)────────────────────────────────────────┘
```

| Tool | Failure Threshold | Recovery Timeout |
|---|---|---|
| `search_web` | 3 failures | 60 seconds |
| `fetch_papers` | 2 failures | 60 seconds |
| `search_repos` | 2 failures | 60 seconds |

When a circuit is **OPEN**, calls fail immediately with `RuntimeError("CircuitBreaker[name] OPEN")` without attempting the network call.

### Layer 3: Observability Swallowing

All observability calls (tracer logging) use `contextlib.suppress(Exception)` so that tracing failures never propagate to the agent. This is enforced in:
- Agent node tool call logging
- The `ResearchTracer._safe()` method (wraps all DB writes)
- The `trace_tool_call()` async context manager

### Composition

The three layers compose in this order:

```
retry_with_policy()            ← Layer 1: retries with backoff
  └─▸ circuit_breaker.call()   ← Layer 2: circuit state check
       └─▸ tool.ainvoke()      ← Actual MCP call
```

---

## Security & Privacy

### Container Security

- **Non-root execution**: All containers run as `appuser` (UID 1000), not root
- **No secrets in images**: Environment variables are injected at runtime via `.env`
- **Gitleaks scanning**: The `secrets-check.yml` GitHub Action uses Gitleaks to detect accidentally committed secrets in every push

### JWT Authentication

MCP server authentication uses HS256 JWTs:

1. Agent generates a token with 1-hour expiry via `utils.auth.get_jwt_token()`
2. MCP servers validate via `@require_auth` decorator in `auth.py`
3. Invalid/expired tokens return `PermissionError` which the agent catches gracefully

### PII Filtering (`agent/middleware/pii_filter.py`)

All MCP tool outputs are scrubbed for personally identifiable information before entering the agent state:

| Pattern | Replacement | Regex |
|---|---|---|
| Social Security Number | `[SSN REDACTED]` | `\d{3}-\d{2}-\d{4}` |
| Credit Card Number | `[CARD REDACTED]` | 13–19 digits with optional separators |
| Email Address | `[EMAIL REDACTED]` | Standard email pattern |
| Phone Number | `[PHONE REDACTED]` | US format with optional country code |
| IP Address | `[IP REDACTED]` | IPv4 dotted quad |

**Audit trail**: `filter_pii()` returns a `RedactionResult` with `total_redactions` and per-pattern `redaction_counts` for compliance logging.

### Static Analysis

| Tool | Scope | CI Workflow |
|---|---|---|
| **Bandit** | Python security linter (agent, MCP servers, API) | `security.yml` |
| **uv audit** | Dependency vulnerability scanning against advisory databases | `security.yml` |
| **Gitleaks** | Secret detection in git history | `secrets-check.yml` |

---

## Cost Controls

### Budget Guard (`agent/budget_guard.py`)

The budget guard sits as a **conditional edge** in the LangGraph graph, wrapping the critic's `should_continue` decision:

```python
workflow.add_conditional_edges(
    "critic",
    check_budget,         # ← Budget guard function
    {
        "continue": "planner",
        "synthesize": "synthesizer",
    },
)
```

**Check order:**

1. **Iteration limit**: `meta.iteration_count >= settings.max_iterations` → force synthesis
2. **Cost limit**: `meta.estimated_cost_usd >= settings.max_cost_per_run_usd` → force synthesis
3. **Critic decision**: If budget OK, delegate to `critic.should_continue()`

**Default limits** (from `config/settings.py`):

| Limit | Default | Environment Variable |
|---|---|---|
| Max iterations | 15 | `MAX_ITERATIONS` |
| Max cost per run | $2.00 | `MAX_COST_PER_RUN_USD` |

### Cost Estimation

Token costs are estimated dynamically using `utils/cost_estimator.py`, which reads pricing data from the **LiteLLM community-maintained pricing database** (2,600+ models).

#### Design Choice: LiteLLM Pricing Without the LiteLLM Package

The industry-standard approach for LLM cost estimation is [LiteLLM](https://github.com/BerriAI/litellm), which maintains a [community-curated pricing JSON](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) covering 2,600+ models across all major providers. However, adding `litellm` as a runtime dependency is impractical for this project because:

1. **Aggressive version pinning**: litellm pins exact versions of core dependencies (`pydantic==2.12.5`, `python-dotenv==1.0.1`, `openai==2.24.0`) that conflict with our stack
2. **Supply chain risk**: litellm v1.82.7/1.82.8 were compromised in a [March 2026 supply chain attack](https://snyk.io/blog/litellm-supply-chain-attack/)
3. **Slow import**: `import litellm` makes a network call to fetch pricing data, causing hangs in CI/CD environments
4. **Heavy transitive dependencies**: Adds 20+ transitive packages to the dependency tree

**Our solution**: Fetch litellm's community pricing JSON directly from GitHub at runtime, cache it locally, and never `import litellm`. This gives us:

- ✅ Dynamic pricing for 2,600+ models (not a hardcoded dictionary)
- ✅ Automatic refresh every 7 days with local disk caching
- ✅ Zero additional runtime dependencies
- ✅ Graceful offline fallback to conservative GPT-4o pricing
- ✅ No security vulnerabilities from litellm's dependency tree

#### How It Works

```
First call to estimate_cost()
  │
  ├─ In-process cache hit? → return immediately
  │
  ├─ Local disk cache fresh (< 7 days)? → load from ~/.cache/deep-research-agent/
  │
  ├─ Fetch from GitHub (10s timeout) → cache to disk
  │
  ├─ Stale disk cache exists? → use it (better than nothing)
  │
  └─ No data at all → conservative GPT-4o fallback ($2.50/$10.00 per 1M tokens)
```

Unknown models always fall back to GPT-4o pricing (conservative estimate — never underestimates cost).

### Token Usage Callback

The `utils/callbacks.py` `TokenCostCallback` is a LangChain callback handler that accumulates token counts and estimated costs across all LLM calls within a run. It handles both OpenAI and Anthropic token usage response formats.

---

## CI/CD Pipelines

Four GitHub Actions workflows automate quality assurance:

### 1. CI — Code Quality (`ci.yml`)

**Trigger**: Every push to `main`/`dev`, every PR to `main`

| Step | Tool | Failure Mode |
|---|---|---|
| Linting | `ruff check` | Blocks merge |
| Formatting | `ruff format --check` | Blocks merge |
| Type checking | `mypy` | Blocks merge |
| Tests | `pytest` with coverage | Blocks merge |
| Docker build | `docker compose build` | Blocks merge |
| Coverage upload | `actions/upload-artifact` | Non-blocking |

### 2. Agent Quality Evaluation (`eval.yml`)

**Trigger**: Manual dispatch or weekly (Sunday 10:00 UTC)

Runs the full benchmark pipeline with the Docker Compose stack. See [Evaluation Pipeline](EVALUATION.md) for details.

### 3. Security — Dependency Audit (`security.yml`)

**Trigger**: Every push + weekly Monday 9:00 UTC

| Step | Tool |
|---|---|
| Dependency vulnerability scan | `uv audit` |
| Static analysis | `bandit -ll -ii` |

### 4. Security — Secret Scanning (`secrets-check.yml`)

**Trigger**: Every push/PR

Uses Gitleaks with full git history scanning (`fetch-depth: 0`).

---

## Local Development

### Hot-Reload Setup

```bash
# Start with volume mounts for hot reload
make dev
# Equivalent to:
# docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The `docker-compose.dev.yml` override:
- Mounts source directories as volumes into each container
- Enables `WATCHFILES_FORCE_POLLING=true` for Docker volume change detection
- Overrides the agent command to use `uvicorn --reload`

### Running Without Docker

For agent development without Docker:

```bash
# Start MCP servers individually (each in a separate terminal)
cd mcp_servers/web_search && python server.py
cd mcp_servers/arxiv && python server.py
cd mcp_servers/github && python server.py

# Start the agent API
uv run uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Makefile Reference

| Target | Command | Description |
|---|---|---|
| `make run` | `docker compose up --build` | Full production stack |
| `make dev` | `docker compose -f ... up --build` | Hot-reload dev stack |
| `make test` | `uv run pytest tests/ -v --cov=...` | Tests with coverage |
| `make benchmark` | `uv run python evaluation/run_benchmark.py ...` | LLM-as-judge eval |
| `make lint` | `ruff check . && ruff format --check . && mypy ...` | All linters |
| `make security` | `uv audit && bandit` | Security scans |
| `make build` | `docker compose build` | Build images only |
| `make docker-test` | Build + health check all services + teardown | Docker smoke test |
| `make clean` | `docker compose down -v` + cleanup | Remove containers + caches |
