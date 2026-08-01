# Architecture

This document describes the technical architecture of the DeepResearch Agent — a LangGraph-based autonomous system that orchestrates multiple MCP tool servers and LLMs to produce structured, cited research reports.

---

## Table of Contents

- [System Overview](#system-overview)
- [Agent Graph Workflow](#agent-graph-workflow)
- [Node Descriptions](#node-descriptions)
- [State Management](#state-management)
- [Model Context Protocol (MCP)](#model-context-protocol-mcp)
- [Multi-Model Synthesis](#multi-model-synthesis)
- [Observability Layer](#observability-layer)
- [API Gateway](#api-gateway)

---

## System Overview

The system follows a **microservices architecture** with clear separation of concerns:

```
┌──────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  Streamlit UI (:8501)  ← SSE streaming ←  FastAPI API (:8080)   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
┌────────────────────────────────┼─────────────────────────────────┐
│                         Agent Layer                              │
│                                │                                 │
│  ┌──────────┐  ┌──────────┐  ┌┴─────────┐  ┌────────────────┐  │
│  │Classifier│→ │ Planner  │→ │Supervisor│→ │   Sub-Agents   │  │
│  │(Primary  │  │ (Primary │  │(Fan-out) │  │ web / arXiv /  │  │
│  │ model)   │  │  [HITL]  │  │  [Send]  │  │    github      │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────┤           │
│  │                                                  ↓           │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────┐                 │
│  │  │  Critic  │← │Synthesizer │← │  Writer  │→ ReportOutput   │
│  │  │(Primary) │  │ (Primary + │  │(Citation │                 │
│  │  │ [Loop?]  │  │ Fallback)  │  │ builder) │                 │
│  │  └──────────┘  └────────────┘  └──────────┘                 │
│  │       ↑                                                      │
│  │       └── Budget Guard (iteration + cost limits)             │
│  └──────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼─────────────────────────────────┐
│                         Tool Layer (MCP)                         │
│                                │                                 │
│  ┌────────────────┐  ┌────────┴───────┐  ┌───────────────────┐  │
│  │ Web Search MCP │  │  arXiv MCP     │  │  GitHub MCP       │  │
│  │ :8001 (Tavily) │  │  :8002 (Atom)  │  │  :8003 (REST API) │  │
│  │ JWT + Cache    │  │  JWT + Cache   │  │  JWT + Cache      │  │
│  └────────────────┘  └────────────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Tech stack:**

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph `StateGraph` with `HybridSqliteSaver` checkpointing |
| Primary LLM | OpenAI GPT-5-mini (`langchain-openai`) |
| Secondary LLM | OpenAI GPT-5-mini (`langchain-openai`) |
| Tool protocol | Model Context Protocol (MCP) via `FastMCP` over SSE |
| API gateway | FastAPI with Server-Sent Events (SSE) |
| Frontend | Streamlit |
| Configuration | Pydantic Settings + YAML profiles |
| Cost estimation | Dynamic pricing via [LiteLLM community database](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) (2,600+ models, auto-cached) |
| Observability | Custom SQLite tracer + LangSmith |
| Evaluation | LLM-as-judge with structured output |

---

## Agent Graph Workflow

The agent is defined as a **LangGraph `StateGraph`** in `agent/graph.py`. The graph is compiled with:

- **`HybridSqliteSaver` checkpointer** — persists state across process restarts with dual sync and async SQLite access (in WAL mode), enabling reliable state management and resumable runs.
- **`interrupt_before=["supervisor"]`** — implements Human-in-the-Loop (HITL) by running the classifier and planner to generate initial sub-questions, then pausing the graph before the supervisor node dispatches parallel agents. This allows the user to review and optionally edit the research plan.

### Graph Flow

```
┌─────────┐     ┌─────────┐     ┌────────────┐
│Classifier│────▸│ Planner │────▸│ Supervisor │
└─────────┘     └─────────┘     └────────────┘
                                  [HITL ↑]
                                  interrupt
                                     │ Send() × 3N
                                     ▼
                              ┌─────────────┐
                              │  web_agent   │─┐
                              │  arxiv_agent │─┤──▸ reconverge
                              │  github_agent│─┘
                              └─────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │   Critic     │
                              │  [Budget     │
                              │   Guard]     │
                              └──────┬──────┘
                                     │
                          ┌──────────┼──────────┐
                          │                     │
                    "continue"            "synthesize"
                          │                     │
                          ▼                     ▼
                   ┌────────────┐       ┌──────────────┐
                   │  Planner   │       │ Synthesizer  │
                   │ (loop back)│       │  (Primary +  │
                   └────────────┘       │   Fallback)  │
                                        └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────┐
                                        │  Writer  │
                                        │  [END]   │
                                        └──────────┘
```

The graph uses **conditional edges** after the critic node, mediated by the `check_budget()` function from `agent/budget_guard.py`:

1. **Budget check first**: If the iteration count has reached `settings.max_iterations` (default: 10) or the estimated cost has exceeded `settings.max_cost_per_run_usd` (default: $0.50), the graph routes directly to synthesis.
2. **Critic decision**: If budget is OK, the critic's `should_continue` flag determines whether to loop (route back to planner for another targeted research round) or proceed to synthesis.

---

## Node Descriptions

### Classifier (`agent/nodes/classifier.py`)

- **Model**: Configurable via `settings.default_model` (lazy-initialized)
- **Purpose**: Classifies the query difficulty as `narrow`, `broad`, or `ambiguous`
- **Output**: Sets `query_difficulty` in state, suggests number of sub-questions (3/4/6)
- **Design choice**: Uses structured output (`ClassifierOutput` Pydantic model) for deterministic parsing

### Planner (`agent/nodes/planner.py`)

- **Model**: Configurable via `settings.default_model`
- **Purpose**: Generates research sub-questions based on the classified difficulty and user profile
- **HITL**: The graph is interrupted *before* the supervisor node (after the classifier and planner generate the research plan). The API emits an SSE `hitl_interrupt` event on the first iteration. The user can:
  - **Approve**: The graph resumes execution to the supervisor node
  - **Edit**: The user provides `edited_subquestions` via `POST /research/approve`, which are injected directly into state before resuming
  - **Reject**: The run is marked as `rejected` in the tracer
- **Iterative Drill-down**: On subsequent loops (`iteration_count > 0`), the planner dynamically generates follow-up sub-questions based on the critic's `missing_areas` feedback to independently drill deeper into the content without repeated HITL prompting.
- **Prompts**: Loaded from `agent/prompts/planner.yaml`

### Supervisor (`agent/nodes/supervisor.py`)

- **Purpose**: Parallel task dispatch using LangGraph's `Send` API
- **Behaviour**: For *N* sub-questions, it creates *3N* `Send` objects (one per sub-question per agent: web, arXiv, GitHub)
- **Edge case**: If `subquestions` is empty, routes directly to the critic

```python
# Core fan-out logic
for subquestion in subquestions:
    for agent in ["web_agent", "arxiv_agent", "github_agent"]:
        sends.append(Send(agent, {**state, "subquestions": [subquestion]}))
```

### Sub-Agents (`web_agent.py`, `arxiv_agent.py`, `github_agent.py`)

Each sub-agent follows the same pattern:

1. **Connect** to its MCP server via `MultiServerMCPClient` with JWT auth
2. **Call** the tool with retry policy wrapping a circuit breaker
3. **Filter** string fields through the PII middleware
4. **Log** the tool call to the observability tracer (fire-and-forget)
5. **Return** `ResearchFindings` with results and any `tool_errors`

The retry and degradation layers compose as:

```
retry_with_policy() → circuit_breaker.call() → MCP tool.ainvoke()
```

### Critic (`agent/nodes/critic.py`)

- **Model**: Configurable via `settings.default_model` with structured output (`CritiqueOutput`)
- **Scores**: `coverage_score`, `recency_score`, `depth_score`, `source_diversity_score` (each 0.0–1.0)
- **Relevant Sources Filter**: Evaluates `source_diversity_score` against `relevant_sources` selected by the classifier so omitted non-relevant source types (e.g. arXiv for web queries) are not penalized.
- **Decision**: Sets `should_continue: bool` — if `True` and budget permits, the graph loops back to planner
- **Source trust scoring**: `score_source_trust()` evaluates each source based on type-specific heuristics (citation count for arXiv, star count for GitHub, domain trustworthiness for web)

### Synthesizer (`agent/nodes/synthesizer.py`)

- **Model Architecture**: Streamlined execution with primary model synthesis (`settings.default_model`) and automatic fallback to secondary model (`settings.secondary_model`) if primary fails.
- **Performance**: Eliminates redundant multi-model latency while maintaining full structured report output (`ReportOutput`).
- **Output**: `ReportOutput` with key findings, citations, executive summary, emerging trends, and next steps.

### Writer (`agent/nodes/writer.py`)

- **Purpose**: Builds the citation list from all `ResearchFindings` and attaches it to the report
- **Source trust**: Each citation's trust score is computed via the critic's `score_source_trust()` function
- **Versioning**: Sets `report.version = 1` (placeholder for future report revision tracking)

---

## State Management

The agent state is a `TypedDict` defined in `agent/state.py`:

```python
class ResearchState(TypedDict):
    query: str  # Input
    profile: str  # "fast" or "deep"
    run_id: str
    query_difficulty: str  # "narrow" | "broad" | "ambiguous"
    relevant_sources: list[str]  # Selected source types: ["web", "arxiv", "github"]
    subquestions: list[str]
    approved_plan: bool
    findings: Annotated[list[ResearchFindings], operator.add]  # Parallel-safe append
    critique: CritiqueOutput | None
    iteration_count: int
    final_report: ReportOutput | None
    run_metadata: RunMetadata
    error_log: Annotated[list[str], operator.add]  # Parallel-safe append
    thought_log: Annotated[list[str], operator.add]  # Parallel-safe append
```

**Key design decisions:**

- **`Annotated[list, operator.add]`**: Fields modified by parallel nodes (`findings`, `error_log`, `thought_log`) use LangGraph's additive reducer. This ensures that results from concurrent sub-agents are merged without race conditions.
- **Pydantic models**: All data structures (`WebResult`, `ArxivPaper`, `GitHubRepo`, `ReportOutput`, etc.) are Pydantic `BaseModel` subclasses with field validation.
- **`RunMetadata`**: Accumulates operational metrics (token counts, cost, iteration count) across the run's lifecycle.

### Research Profiles

Two profile configurations (`config/profiles/fast.yaml` and `deep.yaml`) control:

| Parameter | Fast | Deep |
|---|---|---|
| `max_web_results` | 3 | 8 |
| `max_arxiv_papers` | 2 | 5 |
| `max_github_repos` | 3 | 5 |
| `max_iterations` | 2 | 4 |
| `synthesis_depth` | brief | comprehensive |
| `query_decomposition` | breadth-first | depth-first |

> **Note:** The LLM model is not per-profile — it is a global setting controlled by `settings.default_model` (default: `gpt-5-mini`) and `settings.secondary_model` (default: `gpt-5-mini`). Per-node temperatures (`classifier`: 0.0, `planner`: 0.2, `synthesis`: 0.3, `critic`: 0.0) fine-tune creativity versus determinism per node. A third query decomposition strategy, `hypothesis-driven`, is also implemented and can be used in custom profiles.

---

## Model Context Protocol (MCP)

The three MCP servers are independent microservices built with `FastMCP`:

### Server Architecture

Each server follows an identical pattern, with shared authentication and caching modules:

```
mcp_servers/
├── shared/
│   ├── auth.py            # JWT Bearer token validation decorator
│   └── cache.py           # SQLite-backed result cache with TTL
├── web_search/
│   ├── server.py          # Tool registration + health endpoint
│   ├── Dockerfile         # Non-root user, curl for HEALTHCHECK
│   └── requirements.txt   # Minimal dependencies
├── arxiv/
│   └── (same structure)
└── github/
    └── (same structure)
```

### Server Details

| Server | Port | Tool | Data Source | Cache TTL |
|---|---|---|---|---|
| Web Search | 8001 | `search_web` | Tavily REST API | 1 hour |
| arXiv | 8002 | `fetch_papers` | arXiv Atom XML API | 24 hours |
| GitHub | 8003 | `search_repos` | GitHub REST API | 2 hours |

### Authentication

All MCP servers use JWT authentication:

1. The agent generates a short-lived HS256 JWT via `utils.auth.get_jwt_token()`.
2. The token is sent as `Authorization: Bearer <token>` in the SSE connection headers
3. Each MCP server validates the token via its `@require_auth` decorator before executing the tool
4. The shared secret is configured via `MCP_JWT_SECRET` environment variable

### Transport

All servers use **SSE (Server-Sent Events)** transport, which is the standard MCP transport for HTTP-based servers. The agent connects via `langchain-mcp-adapters`' `MultiServerMCPClient`.

---

## Multi-Model Synthesis

The synthesizer implements a **streamlined synthesis architecture with automated model fallback**:

```
                 ┌─────────────────────┐
                 │    Primary Model    │ ── (success) ──▸ Final Report
                 │    (settings.       │
                 │    default_model)   │
                 └──────────┬──────────┘
                            │ (on error/timeout)
                            ▼
                 ┌─────────────────────┐
                 │   Secondary Model   │ ── (fallback) ──▸ Final Report
                 │   (settings.        │
                 │   secondary_model)  │
                 └─────────────────────┘
```

1. **Primary Generation**: The primary model (`settings.default_model`, e.g. `gpt-5-mini`) receives the synthesis prompt containing formatted findings from all sub-questions.
2. **Automated Fallback**: If the primary model fails or times out (`settings.synthesis_timeout_seconds`), execution automatically falls back to the secondary model (`settings.secondary_model`, e.g. `gpt-5-mini`).
3. **Structured Verification**: The result is validated against the `ReportOutput` schema (including executive summary, key findings with citations, emerging trends, and next steps).

---

## Observability Layer

### Custom SQLite Tracer (`observability/tracer.py`)

The tracer records four types of events:

| Table | Records | Key Fields |
|---|---|---|
| `runs` | Run lifecycle | `run_id`, `query`, `status`, `total_cost_usd`, `final_score` |
| `tool_calls` | MCP tool invocations | `tool_name`, `success`, `latency_ms`, `error_message` |
| `node_executions` | LLM node calls | `model_name`, `input_tokens`, `output_tokens`, `estimated_cost_usd` |
| `eval_scores` | LLM-as-judge results | `faithfulness`, `answer_relevancy`, `source_coverage`, `citation_accuracy`, `coherence` |

Cost values in `node_executions` are computed dynamically via `utils/cost_estimator.py`, which reads per-token pricing from the LiteLLM community pricing database (see [Infrastructure → Cost Estimation](INFRASTRUCTURE.md#cost-estimation) for design details).

**Design constraints:**

1. **Non-blocking**: All DB writes run via `asyncio.to_thread()` so they never stall the LangGraph event loop
2. **Non-fatal**: Every public method swallows exceptions with `contextlib.suppress(Exception)` — observability must never crash the agent
3. **Singleton**: `get_tracer()` returns a process-level instance

### LangSmith Integration

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set, all LangChain/LangGraph operations are automatically traced to LangSmith. This provides a complementary view with:
- Full prompt/completion logging
- Token-level streaming traces
- Run grouping by `LANGCHAIN_PROJECT`

---

## API Gateway

The FastAPI gateway (`api/main.py`) exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `POST /research/stream` | POST | Start a new research run, stream SSE events |
| `POST /research/approve` | POST | Resume a run paused at the HITL supervisor interrupt |
| `GET /research/state/{thread_id}` | GET | Get current graph state for a thread |
| `GET /research/report/{thread_id}` | GET | Get the final completed report as structured JSON |
| `GET /research/report/{thread_id}/pdf` | GET | Download the final report as a styled PDF |
| `GET /research/report/{thread_id}/markdown` | GET | Download the final report as Markdown |
| `GET /research/report/{thread_id}/html` | GET | Download the final report as styled HTML |
| `GET /research/runs` | GET | List recent runs from observability DB |
| `GET /research/runs/{run_id}` | GET | Get full detail for a single run |
| `GET /health` | GET | Health check |
| `GET /health/deep` | GET | Deep health check (validates SQLite tracer, MCP servers, API keys) |

### SSE Event Types

| Event Type | When | Payload |
|---|---|---|
| `node_start` | A graph node begins execution | `{ node: string, input: string, timestamp: string }` |
| `node_end` | A graph node finishes execution | `{ node: string, output: string, timestamp: string }` |
| `tool_call` | An MCP tool is invoked | `{ tool: string, input: string, timestamp: string }` |
| `tool_result` | An MCP tool returns | `{ tool: string, count: int, full_output: string, timestamp: string }` |
| `llm_start` | LLM invocation starts | `{ model: string, prompt: string, timestamp: string }` |
| `llm_end` | LLM invocation completes | `{ model: string, response: string, timestamp: string }` |
| `token` | LLM streaming chunk | `{ content: string }` |
| `hitl_interrupt` | Graph paused before supervisor | `{ thread_id, query_difficulty, relevant_sources, subquestions, estimated_cost_usd }` |
| `complete` | Writer finished, report ready | `{ run_id: string, timestamp: string }` |

### Rate Limiting

The `/research/stream` endpoint is rate-limited to **5 requests per minute** per client IP using [slowapi](https://github.com/laurentS/slowapi). Exceeding the limit returns HTTP 429.
