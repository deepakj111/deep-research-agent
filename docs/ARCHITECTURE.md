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
│  │(GPT-4o-  │  │ (GPT-4o) │  │(Fan-out) │  │ web / arXiv /  │  │
│  │  mini)   │  │  [HITL]  │  │  [Send]  │  │    github      │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────┤           │
│  │                                                  ↓           │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────┐                 │
│  │  │  Critic  │← │Synthesizer │← │  Writer  │→ ReportOutput   │
│  │  │(GPT-4o)  │  │(GPT-4o +   │  │(Citation │                 │
│  │  │ [Loop?]  │  │  Claude)   │  │ builder) │                 │
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
│                                │                                 │
│                         ┌──────┴──────┐                          │
│                         │    Redis    │                          │
│                         │    :6379    │                          │
│                         └─────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

**Tech stack:**

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph `StateGraph` with `SqliteSaver` checkpointing |
| Primary LLM | OpenAI GPT-4o (`langchain-openai`) |
| Secondary LLM | Anthropic Claude Sonnet (`langchain-anthropic`) |
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

- **`SqliteSaver` checkpointer** — persists state across process restarts, enabling resumable runs.
- **`interrupt_before=["planner"]`** — implements Human-in-the-Loop (HITL) by pausing the graph before the planner node runs, allowing the user to review and optionally edit the research plan.

### Graph Flow

```
┌─────────┐     ┌─────────┐     ┌────────────┐
│Classifier│────▸│ Planner │────▸│ Supervisor │
└─────────┘     └─────────┘     └────────────┘
                  [HITL ↑]           │
                  interrupt          │ Send() × 3N
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
                   │ Supervisor │       │ Synthesizer  │
                   │ (loop back)│       │ (GPT + Claude│
                   └────────────┘       │  parallel)   │
                                        └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────┐
                                        │  Writer  │
                                        │  [END]   │
                                        └──────────┘
```

The graph uses **conditional edges** after the critic node, mediated by the `check_budget()` function from `agent/budget_guard.py`:

1. **Budget check first**: If the iteration count has reached `settings.max_iterations` (default: 15) or the estimated cost has exceeded `settings.max_cost_per_run_usd` (default: $2.00), the graph routes directly to synthesis.
2. **Critic decision**: If budget is OK, the critic's `should_continue` flag determines whether to loop (route back to supervisor for another research round) or proceed to synthesis.

---

## Node Descriptions

### Classifier (`agent/nodes/classifier.py`)

- **Model**: GPT-4o-mini (fast, cheap)
- **Purpose**: Classifies the query difficulty as `narrow`, `broad`, or `ambiguous`
- **Output**: Sets `query_difficulty` in state, suggests number of sub-questions (3/4/6)
- **Design choice**: Uses structured output (`ClassifierOutput` Pydantic model) for deterministic parsing

### Planner (`agent/nodes/planner.py`)

- **Model**: GPT-4o
- **Purpose**: Generates research sub-questions based on the classified difficulty and user profile
- **HITL**: The graph is interrupted *before* this node. The API emits an SSE `hitl_interrupt` event. The user can:
  - **Approve**: The planner runs normally
  - **Edit**: The user provides `edited_subquestions` via `POST /research/approve`, which are injected directly into state (bypassing the planner LLM entirely)
  - **Reject**: The run is marked as `rejected` in the tracer
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

- **Model**: GPT-4o with structured output (`CritiqueOutput`)
- **Scores**: `coverage_score`, `recency_score`, `depth_score`, `source_diversity_score` (each 0.0–1.0)
- **Decision**: Sets `should_continue: bool` — if `True` and budget permits, the graph loops back to supervisor
- **Source trust scoring**: `score_source_trust()` evaluates each source based on type-specific heuristics (citation count for arXiv, star count for GitHub, domain trustworthiness for web)

### Synthesizer (`agent/nodes/synthesizer.py`)

- **Models**: GPT-4o and Claude Sonnet run **in parallel** via `asyncio.gather()`
- **Reconciliation**: When both models succeed, a third LLM call (`ReconcileOutput`) detects contradictions and records model disagreements
- **Fallback**: If one model fails, the other's output is used directly. If both fail, `final_report` is set to `None` and the error is logged
- **Output**: `ReportOutput` with `contradictions` and `model_disagreements` fields

### Writer (`agent/nodes/writer.py`)

- **Purpose**: Builds the citation list from all `ResearchFindings` and attaches it to the report
- **Source trust**: Each citation's trust score is computed via the critic's `score_source_trust()` function
- **Versioning**: Sets `report.version = 1` (placeholder for future report revision tracking)

---

## State Management

The agent state is a `TypedDict` defined in `agent/state.py`:

```python
class ResearchState(TypedDict):
    query: str                                                  # Input
    profile: str                                                # "fast" or "deep"
    run_id: str
    query_difficulty: str                                       # "narrow" | "broad" | "ambiguous"
    subquestions: list[str]
    approved_plan: bool
    findings: Annotated[list[ResearchFindings], operator.add]   # Parallel-safe append
    critique: CritiqueOutput | None
    iteration_count: int
    final_report: ReportOutput | None
    run_metadata: RunMetadata
    error_log: Annotated[list[str], operator.add]               # Parallel-safe append
    thought_log: Annotated[list[str], operator.add]             # Parallel-safe append
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
| `llm_model` | gpt-4o-mini | gpt-4o |
| `max_iterations` | 8 | 15 |
| `synthesis_depth` | brief | comprehensive |
| `query_decomposition` | breadth-first | depth-first |

---

## Model Context Protocol (MCP)

The three MCP servers are independent microservices built with `FastMCP`:

### Server Architecture

Each server follows an identical pattern:

```
FastMCP server
├── server.py          # Tool registration + health endpoint
├── auth.py            # JWT Bearer token validation decorator
├── cache.py           # SQLite-backed result cache with TTL
├── Dockerfile         # Non-root user, curl for HEALTHCHECK
└── requirements.txt   # Minimal dependencies
```

### Server Details

| Server | Port | Tool | Data Source | Cache TTL |
|---|---|---|---|---|
| Web Search | 8001 | `search_web` | Tavily REST API | 1 hour |
| arXiv | 8002 | `fetch_papers` | arXiv Atom XML API | 1 hour |
| GitHub | 8003 | `search_repos` | GitHub REST API | 1 hour |

### Authentication

All MCP servers use JWT authentication:

1. The agent generates a short-lived HS256 JWT via `utils/cost_estimator.get_jwt_token()`
2. The token is sent as `Authorization: Bearer <token>` in the SSE connection headers
3. Each MCP server validates the token via its `@require_auth` decorator before executing the tool
4. The shared secret is configured via `MCP_JWT_SECRET` environment variable

### Transport

All servers use **SSE (Server-Sent Events)** transport, which is the standard MCP transport for HTTP-based servers. The agent connects via `langchain-mcp-adapters`' `MultiServerMCPClient`.

---

## Multi-Model Synthesis

The synthesizer implements a **parallel dual-model architecture**:

```
                 ┌──────────┐
     ┌──────────▸│  GPT-4o  │──────────┐
     │           └──────────┘          │
     │                                  │  asyncio.gather()
  Prompt                                ▼
     │           ┌──────────┐    ┌──────────────┐
     └──────────▸│  Claude  │───▸│ Reconciler   │──▸ Final Report
                 │  Sonnet  │    │ (GPT-4o)     │
                 └──────────┘    │ Contradictions│
                                 └──────────────┘
```

1. **Parallel generation**: Both models receive the same synthesis prompt (context from all findings)
2. **Reconciliation**: A third LLM call compares the two executive summaries, identifying contradictions as `ContradictionRecord` objects
3. **Fallback**: If one model fails, the other's report is used directly. If both fail, the writer logs an error and produces no report
4. **Transparency**: Model disagreements are surfaced in the final report's `model_disagreements` field

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
| `POST /research/approve` | POST | Resume a run paused at the HITL planner interrupt |
| `GET /research/state/{thread_id}` | GET | Get current graph state for a thread |
| `GET /research/report/{thread_id}` | GET | Get the final completed report as structured JSON |
| `GET /research/report/{thread_id}/pdf` | GET | Download the final report as a styled PDF |
| `GET /research/report/{thread_id}/markdown` | GET | Download the final report as Markdown |
| `GET /research/report/{thread_id}/html` | GET | Download the final report as styled HTML |
| `GET /research/runs` | GET | List recent runs from observability DB |
| `GET /research/runs/{run_id}` | GET | Get full detail for a single run |
| `GET /health` | GET | Health check |

### SSE Event Types

| Event Type | When | Payload |
|---|---|---|
| `node_start` | A graph node begins execution | `{ node: string }` |
| `tool_call` | An MCP tool is invoked | `{ tool: string, input: string }` |
| `tool_result` | An MCP tool returns | `{ tool: string, count: int }` |
| `token` | LLM streaming chunk | `{ content: string }` |
| `hitl_interrupt` | Graph paused before planner | `{ thread_id, query_difficulty, estimated_cost_usd }` |
| `complete` | Writer finished, report ready | `{ run_id: string }` |
