import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.state import ResearchState
from config.settings import settings

with open("agent/prompts/planner.yaml") as f:
    _prompts = yaml.safe_load(f)

PLANNER_SYSTEM = _prompts["system"]
PLANNER_USER = _prompts["user"]


class PlanOutput(BaseModel):
    subquestions: list[str]
    reasoning: str


_llm = ChatOpenAI(model=settings.default_model).with_structured_output(PlanOutput)

NUM_QUESTIONS = {
    "narrow": 3,
    "broad": 6,
    "ambiguous": 4,
}


async def run(state: ResearchState) -> dict:
    difficulty = state.get("query_difficulty", "narrow")
    n = NUM_QUESTIONS.get(difficulty, 4)
    profile = state.get("profile", "fast")

    result: PlanOutput = await _llm.ainvoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": PLANNER_USER.format(
                    query=state["query"],
                    n=n,
                    profile=profile,
                    difficulty=difficulty,
                ),
            },
        ]
    )

    return {
        "subquestions": result.subquestions,
        "approved_plan": True,
        "thought_log": [
            f"[Planner] Generated {len(result.subquestions)} sub-questions for '{difficulty}' query. "
            f"Reasoning: {result.reasoning}"
        ],
    }
