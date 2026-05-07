import asyncio
import functools
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from agent.middleware.input_sanitizer import sanitize_query
from agent.state import ResearchState
from config.settings import settings
from utils.callbacks import TokenCostCallback

# Default timeout for LLM calls (seconds). Prevents hangs when the
# upstream provider is slow or unresponsive.
_LLM_TIMEOUT_SECONDS = 60.0


class ClassifierOutput(BaseModel):
    difficulty: Literal["narrow", "broad", "ambiguous"]
    reasoning: str
    suggested_num_questions: int


@functools.lru_cache(maxsize=1)
def _get_llm():
    return init_chat_model(
        settings.default_model, temperature=settings.classifier_temperature
    ).with_structured_output(ClassifierOutput)


CLASSIFIER_PROMPT = """Classify this research query:

Query: {query}

Difficulty levels:
- narrow: specific, well-defined topic → suggest 3 sub-questions
- broad: covers multiple domains or time periods → suggest 5-6 sub-questions
- ambiguous: unclear intent, needs decomposition → suggest 4 sub-questions

Output JSON only. Be concise."""


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
        timeout=_LLM_TIMEOUT_SECONDS,
    )

    # ── Update run metadata with cost ─────────────────────────────────────
    meta = state.get("run_metadata")
    if meta:
        meta.total_input_tokens += cb.total_input_tokens
        meta.total_output_tokens += cb.total_output_tokens
        meta.estimated_cost_usd += cb.total_cost_usd

    thought_entries.append(
        f"[Classifier] Query classified as '{result.difficulty}'. "
        f"Suggested {result.suggested_num_questions} sub-questions. "
        f"Reason: {result.reasoning}"
    )

    return {
        "query": query,  # propagate sanitized query to all downstream nodes
        "query_difficulty": result.difficulty,
        "run_metadata": meta,
        "thought_log": thought_entries,
    }
