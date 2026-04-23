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

STRATEGIES = {
    "breadth-first": (
        "Generate {n} independent, parallel sub-questions that together cover "
        "all major aspects of the topic. Each question should be investigable "
        "on its own without reference to the others."
    ),
    "depth-first": (
        "Generate 1 core question. After research, generate follow-up "
        "questions based on what was discovered. Drill progressively deeper "
        "into the most important finding."
    ),
    "hypothesis-driven": (
        "Generate 1 central hypothesis about the topic. "
        "Then generate: (a) 2 questions seeking evidence FOR the hypothesis, "
        "and (b) 2 questions seeking evidence AGAINST it. "
        "The final report must weigh both sides."
    ),
}

_profile_cache: dict = {}


def load_profile(name: str) -> dict:
    if name not in _profile_cache:
        with open(f"config/profiles/{name}.yaml") as f:
            _profile_cache[name] = yaml.safe_load(f)
    return _profile_cache[name]


async def run(state: ResearchState) -> dict:
    difficulty = state.get("query_difficulty", "narrow")
    n = NUM_QUESTIONS.get(difficulty, 4)
    profile_name = state.get("profile", "fast")

    profile_cfg = load_profile(profile_name)
    strategy_key = profile_cfg.get("query_decomposition", "breadth-first")
    strategy_text = STRATEGIES.get(strategy_key, STRATEGIES["breadth-first"]).format(n=n)

    result: PlanOutput = await _llm.ainvoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": PLANNER_USER.format(
                    query=state["query"],
                    n=n,
                    profile=profile_name,
                    difficulty=difficulty,
                    strategy=strategy_text,
                ),
            },
        ]
    )

    return {
        "subquestions": result.subquestions,
        "approved_plan": True,
        "thought_log": [
            f"[Planner] Generated {len(result.subquestions)} sub-questions for '{difficulty}' query using {strategy_key} strategy. "
            f"Reasoning: {result.reasoning}"
        ],
    }
