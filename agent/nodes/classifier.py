import functools
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from agent.state import ResearchState
from config.settings import settings


class ClassifierOutput(BaseModel):
    difficulty: Literal["narrow", "broad", "ambiguous"]
    reasoning: str
    suggested_num_questions: int


@functools.lru_cache(maxsize=1)
def _get_llm():
    return init_chat_model(settings.default_model).with_structured_output(ClassifierOutput)


CLASSIFIER_PROMPT = """Classify this research query:

Query: {query}

Difficulty levels:
- narrow: specific, well-defined topic → suggest 3 sub-questions
- broad: covers multiple domains or time periods → suggest 5-6 sub-questions
- ambiguous: unclear intent, needs decomposition → suggest 4 sub-questions

Output JSON only. Be concise."""


async def run(state: ResearchState) -> dict:
    result: ClassifierOutput = await _get_llm().ainvoke(  # type: ignore[assignment]
        CLASSIFIER_PROMPT.format(query=state["query"])
    )
    return {
        "query_difficulty": result.difficulty,
        "thought_log": [
            f"[Classifier] Query classified as '{result.difficulty}'. "
            f"Suggested {result.suggested_num_questions} sub-questions. "
            f"Reason: {result.reasoning}"
        ],
    }
