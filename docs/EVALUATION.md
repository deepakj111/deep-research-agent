# Evaluation Pipeline

This document describes the automated evaluation and benchmarking infrastructure used to measure the quality of the DeepResearch Agent's research reports.

---

## Table of Contents

- [Overview](#overview)
- [LLM-as-Judge Evaluator](#llm-as-judge-evaluator)
- [Scoring Rubric](#scoring-rubric)
- [Benchmark Dataset](#benchmark-dataset)
- [Benchmark Runner](#benchmark-runner)
- [CI Integration](#ci-integration)
- [DeepEval Integration](#deepeval-integration)
- [Interpreting Results](#interpreting-results)

---

## Overview

The evaluation pipeline answers a critical question: **"Is the agent producing good research reports?"**

Unlike traditional software testing, AI agent output quality cannot be verified with deterministic assertions. Instead, we use an **LLM-as-judge** approach where GPT-4o evaluates each report across five quality dimensions, producing a normalised score between 0 and 1.

This score is used as:
- A **CI quality gate** — the GitHub Actions `eval.yml` workflow fails if the average score drops below 0.70
- A **regression detector** — weekly scheduled runs catch quality drift
- An **observability metric** — scores are persisted to the SQLite tracer for historical analysis

---

## LLM-as-Judge Evaluator

**File**: `evaluation/evaluator.py`

The evaluator uses GPT-4o with `temperature=0` and structured output (`EvalScores` Pydantic model) to ensure deterministic, parseable scoring.

### How It Works

1. The completed `ReportOutput` is flattened into plain text
2. The source inventory is compiled from `report.sources` (capped at 40 sources to stay within context limits)
3. A detailed rubric prompt is sent to GPT-4o for each dimension
4. The model returns a structured JSON response with five float scores (0.0–5.0) and evaluator notes
5. Scores are normalised to [0, 1] by dividing the total (out of 25) by 25

### Design Decisions

- **Lazy LLM instantiation**: The evaluator LLM is created on first call, not at import time. This prevents `OpenAIError` during test collection when `OPENAI_API_KEY` is not set.
- **Report truncation**: Reports longer than 12,000 characters are truncated with a `[TRUNCATED FOR EVALUATION]` marker to prevent context window overflow.
- **Timeout**: Each evaluation call has a 90-second hard timeout via `asyncio.wait_for()`.

---

## Scoring Rubric

Each dimension is scored on a 0.0–5.0 scale (floats allowed):

### Faithfulness

> Are all factual claims grounded in the retrieved sources?

| Score | Meaning |
|---|---|
| 5.0 | Every claim is traceable to a source with no fabrications |
| 0.0 | Many unsupported or hallucinated claims |

### Answer Relevancy

> Does the report fully and directly address the original query?

| Score | Meaning |
|---|---|
| 5.0 | Query answered comprehensively with appropriate depth |
| 0.0 | Report is off-topic, superficial, or misses the question |

### Source Coverage

> Does the report draw substantively from all three source types?

| Score | Meaning |
|---|---|
| 5.0 | Web, arXiv, AND GitHub sources used with meaningful content |
| 0.0 | Only one type used, or none at all |

This metric is unique to the DeepResearch Agent — it specifically validates the multi-source architecture by ensuring all three MCP servers contribute to the final report.

### Citation Accuracy

> Are inline citations accurate and linked to actually-fetched content?

| Score | Meaning |
|---|---|
| 5.0 | All citations verified against the source list |
| 0.0 | Citations are fabricated, missing, or link to unfetched pages |

The evaluator receives the full list of actually-retrieved sources alongside the report, enabling it to cross-reference citations against real data.

### Coherence

> Is the report logically structured, non-repetitive, and well-written?

| Score | Meaning |
|---|---|
| 5.0 | Professional quality, flows naturally, clear section hierarchy |
| 0.0 | Incoherent, disorganised, heavily repetitive |

### Normalised Average

The five raw scores (out of 5) are summed and divided by 25 to produce a single **normalised average** in the range [0.0, 1.0]. This is the metric used for CI gating.

---

## Benchmark Dataset

**File**: `evaluation/benchmark_queries.json`

The dataset contains **10 curated queries** spanning 6 domains:

| ID | Domain | Query | Purpose |
|---|---|---|---|
| q01 | Physics | Quantum error correction breakthroughs 2025 | Standard multi-source query |
| q02 | Biology | Protein structure prediction post-AlphaFold3 | Should cite specific papers |
| q03 | AI | Multimodal LLMs and agent capabilities 2025 | High-volume domain test |
| q04 | Hardware | Neuromorphic computing and spiking neural networks | Sparse source handling |
| q05 | AI | Open-source AI coding assistants landscape 2026 | GitHub-heavy query |
| q06 | Physics | Nuclear fusion energy timelines 2024–2025 | News vs academic balance |
| q07 | Biology | CRISPR base editing clinical trials | High faithfulness requirement |
| q08 | Robotics | Autonomous vehicle perception 2025 | Technical depth test |
| q09 | Security | Post-quantum cryptography NIST PQC adoption | Niche domain with implementations |
| q10 | AI | LangGraph vs AutoGen vs CrewAI comparison 2026 | GitHub agent routing test |

Each query specifies:
- `expected_source_types`: Which MCP servers should contribute
- `notes`: What the evaluation should look for
- `domain`: For per-domain breakdown in results

---

## Benchmark Runner

**File**: `evaluation/run_benchmark.py`

### Usage

```bash
# Quick CI run (3 queries, fast profile)
uv run python evaluation/run_benchmark.py --queries 3

# Full benchmark suite
uv run python evaluation/run_benchmark.py --profile deep --fail-below 0.80

# Single query debug
uv run python evaluation/run_benchmark.py --query-id q01

# Via Makefile
make benchmark
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--profile` | `fast` | Research profile (`fast` or `deep`) |
| `--fail-below` | `0.75` | Exit 1 if mean score < threshold |
| `--queries` | all 10 | Limit to first N queries (cost control) |
| `--query-id` | — | Run a single query by ID |
| `--concurrency` | `1` | Parallel queries (keep at 1 to respect MCP rate limits) |

### Two-Phase Execution

The LangGraph graph is compiled with `interrupt_before=["planner"]`, requiring two `ainvoke()` calls per benchmark query:

1. **Phase 1**: `graph.ainvoke(initial_state)` → classifier runs, graph pauses before planner
2. **Phase 2**: `graph.ainvoke(None)` → resumes through planner → supervisor → sub-agents → critic → synthesizer → writer

This auto-approves all plans without human review. The benchmark measures output quality, not HITL UX.

### Output

Results are written to `evaluation/results/benchmark_{profile}_{timestamp}.md` as a formatted Markdown table:

```
| ID | Domain | Query | Faith. | Relev. | Coverage | Citation | Coh. | Avg | Latency | Cost | Status |
```

With summary statistics and per-domain breakdowns.

### Observability Integration

After evaluating each query, the scores are persisted to the SQLite tracer:

```python
await tracer.log_eval_scores(EvalScoreRecord(
    run_id=run_id,
    faithfulness=scores.faithfulness,
    answer_relevancy=scores.answer_relevancy,
    source_coverage=scores.source_coverage,
    citation_accuracy=scores.citation_accuracy,
    coherence=scores.coherence,
    normalized_average=scores.normalized_average,
    overall_notes=scores.overall_notes,
))
```

This allows historical score tracking and trend analysis via the observability dashboard.

---

## CI Integration

**File**: `.github/workflows/eval.yml`

The evaluation pipeline is integrated into GitHub Actions as a **manually-triggered + weekly scheduled** workflow.

### Trigger Modes

| Trigger | When | Default Config |
|---|---|---|
| `workflow_dispatch` | Manual via GitHub UI or `gh workflow run` | Configurable profile, query limit, threshold |
| `schedule` | Every Sunday 10:00 UTC | `fast` profile, 3 queries, 0.70 threshold |

### Pipeline Steps

1. **Setup**: Install uv, Python 3.11, project dependencies
2. **Docker stack**: Build and start the full 5-service Docker Compose stack
3. **Health checks**: Wait for all services (3 MCP servers, agent API) to report healthy
4. **Benchmark**: Run `evaluation/run_benchmark.py` with configured parameters
5. **Artifacts**: Upload results to GitHub Actions artifacts (90-day retention)
6. **Teardown**: `docker compose down -v`

### Quality Gate

The workflow exits with code 1 if the mean normalised score falls below the configured threshold (default: 0.70). This prevents quality regressions from being merged unnoticed.

### Required Secrets

| Secret | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT-4o for synthesis + evaluation |
| `ANTHROPIC_API_KEY` | Claude Sonnet for synthesis |
| `TAVILY_API_KEY` | Web search MCP server |
| `MCP_JWT_SECRET` | MCP server JWT authentication |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) |

---

## DeepEval Integration

**File**: `evaluation/evaluator.py` → `evaluate_with_deepeval()`

An optional integration with [DeepEval](https://github.com/confident-ai/deepeval) provides RAGAs-style metrics:

- **FaithfulnessMetric**: Cross-references claims against retrieval context
- **AnswerRelevancyMetric**: Measures response relevance to the input

This integration is **gracefully degraded** — if `deepeval` is not installed or not configured, the function returns `None` and the benchmark falls back to the primary LLM-as-judge evaluator.

---

## Interpreting Results

### Score Ranges

| Range | Quality | Action |
|---|---|---|
| 0.85–1.00 | Excellent | No action needed |
| 0.70–0.84 | Good | Acceptable for CI pass |
| 0.55–0.69 | Needs Improvement | Investigate prompt quality, source coverage |
| Below 0.55 | Poor | Critical — check MCP server health and LLM responses |

### Common Failure Patterns

| Low Dimension | Likely Cause |
|---|---|
| Faithfulness | LLM hallucinating beyond retrieved sources |
| Source Coverage | One or more MCP servers failing (check circuit breaker state) |
| Citation Accuracy | Writer not attaching citations properly |
| Coherence | Synthesis prompt needs refinement |
| Answer Relevancy | Classifier/planner decomposing query poorly |
