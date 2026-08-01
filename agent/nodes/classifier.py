import asyncio
import functools
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from agent.middleware.input_sanitizer import sanitize_query
from agent.state import ResearchState
from config.settings import settings
from utils.callbacks import TokenCostCallback


class ClassifierOutput(BaseModel):
    difficulty: Literal["narrow", "broad", "ambiguous"]
    reasoning: str
    relevant_sources: list[Literal["web", "arxiv", "github"]]


@functools.lru_cache(maxsize=1)
def _get_llm():
    return init_chat_model(
        settings.default_model, temperature=settings.classifier_temperature
    ).with_structured_output(ClassifierOutput)


CLASSIFIER_PROMPT = """Classify this research query:

Query: {query}

1. Difficulty levels:
- narrow: specific, well-defined topic (e.g. "silver investment", "current Bitcoin price")
- broad: covers multiple domains or time periods (e.g. "history of AI regulation")
- ambiguous: unclear intent, needs decomposition

2. Relevant sources — apply STRICT selection rules:

- web: ALWAYS include for any query. Covers news, market data, financial analysis, product reviews, general research.

- arxiv: ONLY select when the query is explicitly about scientific/academic research: physics papers, ML algorithms,
  medical studies, academic theory. NEVER select for:
  * Financial queries (investments, stocks, commodities, ETFs, funds, portfolios)
  * Market analysis or economic queries
  * Business, industry, or company research
  * General technology questions that are not about academic ML/AI research
  * News and current events
  Examples of queries that should NOT use arxiv: "silver investment", "best ETFs 2025", "AI startups to watch",
  "housing market trends", "how does GPT work", "cloud computing cost comparison"

- github: ALWAYS select when the query involves software implementations, developer tools, open-source libraries, AI frameworks (e.g., LangGraph, LangChain, PyTorch, React, Next.js), APIs, release updates, or code.
  Examples of queries that SHOULD use github: "langgraph latest updates", "fastapi release notes", "react 19 features", "open source LLM repos".
  NEVER select for finance, market research, news, or non-software topics.

Output JSON matching the schema."""


async def run(state: ResearchState) -> dict:
    # ── Prompt injection defense ──────────────────────────────────────────
    # Sanitize before any LLM sees the query. The classifier is the first
    # LLM-touching node, so this protects the entire downstream pipeline.
    sanitized = sanitize_query(state["query"])
    query = sanitized.text

    thought_entries: list[str] = []
    if sanitized.was_modified:
        thought_entries.append(
            f"[InputSanitizer] Detected {sanitized.total_detections} prompt injection "
            f"pattern(s): {sanitized.detection_counts}. Query sanitized before classification."
        )

    cb = TokenCostCallback()
    result: ClassifierOutput = await asyncio.wait_for(  # type: ignore[assignment]
        _get_llm().ainvoke(
            CLASSIFIER_PROMPT.format(query=query),
            config={"callbacks": [cb]},
        ),
        timeout=settings.classifier_timeout_seconds,
    )

    # Fallback to web if LLM or mock returns empty/missing list
    sources = getattr(result, "relevant_sources", ["web"]) or ["web"]

    # ── Update run metadata with cost ─────────────────────────────────────
    meta = state.get("run_metadata")
    if meta:
        meta.total_input_tokens += cb.total_input_tokens
        meta.total_output_tokens += cb.total_output_tokens
        meta.estimated_cost_usd += cb.total_cost_usd

    thought_entries.append(
        f"[Classifier] Query classified as '{result.difficulty}'. "
        f"Selected sources: {sources}. "
        f"Reason: {result.reasoning}"
    )

    return {
        "query": query,  # propagate sanitized query to all downstream nodes
        "query_difficulty": result.difficulty,
        "relevant_sources": sources,
        "run_metadata": meta,
        "thought_log": thought_entries,
    }
