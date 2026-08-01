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
    suggested_num_questions: int
    relevant_sources: list[Literal["web", "arxiv", "github"]]


@functools.lru_cache(maxsize=1)
def _get_llm():
    return init_chat_model(
        settings.default_model, temperature=settings.classifier_temperature
    ).with_structured_output(ClassifierOutput)


CLASSIFIER_PROMPT = """Classify this research query:

Query: {query}

1. Difficulty levels:
- narrow: specific, well-defined topic → suggest 3 sub-questions
- broad: covers multiple domains or time periods → suggest 5-6 sub-questions
- ambiguous: unclear intent, needs decomposition → suggest 4 sub-questions

2. Relevant sources (select only what is appropriate):
- web: web search (market trends, news, financial recommendations, general knowledge). Included for almost all queries.
- arxiv: academic papers (scientific research, physics, ML papers, academic algorithms). DO NOT select for general news, financial, market, or simple coding questions.
- github: code repositories & libraries (software engineering, open-source repos, API implementations). DO NOT select for non-software topics like news, finance, science without code, etc.

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
        f"Suggested {result.suggested_num_questions} sub-questions. "
        f"Reason: {result.reasoning}"
    )

    return {
        "query": query,  # propagate sanitized query to all downstream nodes
        "query_difficulty": result.difficulty,
        "relevant_sources": sources,
        "run_metadata": meta,
        "thought_log": thought_entries,
    }
