import yaml
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.state import ResearchState
from config.profiles import load_profile
from config.settings import settings

with open("agent/prompts/planner.yaml") as f:
    _prompts = yaml.safe_load(f)

PLANNER_SYSTEM = _prompts["system"]
PLANNER_USER = _prompts["user"]


class PlanOutput(BaseModel):
    subquestions: list[str]
    reasoning: str


_planner_llm = None


def _get_planner_llm():
    global _planner_llm
    if _planner_llm is None:
        _planner_llm = init_chat_model(settings.default_model).with_structured_output(PlanOutput)
    return _planner_llm


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _invoke_planner_llm(llm, messages):
    return await llm.ainvoke(messages)


async def run(state: ResearchState) -> dict:
    difficulty = state.get("query_difficulty", "narrow")
    n = NUM_QUESTIONS.get(difficulty, 4)
    profile_name = state.get("profile", "fast")

    profile_cfg = load_profile(profile_name)
    strategy_key = profile_cfg.get("query_decomposition", "breadth-first")
    strategy_text = STRATEGIES.get(strategy_key, STRATEGIES["breadth-first"]).format(n=n)

    meta = state.get("run_metadata")
    iteration = meta.iteration_count if meta else 0
    critique = state.get("critique")

    if iteration > 0 and critique and critique.missing_areas:
        user_prompt = _prompts.get("replan_user", "").format(
            query=state["query"],
            iteration=iteration,
            max_iterations=settings.max_iterations,
            missing_areas="\n".join([f"- {m}" for m in critique.missing_areas]),
            n=n,
        )
        task_label = (
            f"Generated {n} follow-up questions for iteration {iteration} targeting missing areas."
        )
    else:
        user_prompt = PLANNER_USER.format(
            query=state["query"],
            n=n,
            profile=profile_name,
            difficulty=difficulty,
            strategy=strategy_text,
        )
        task_label = (
            f"Generated {n} sub-questions for '{difficulty}' query using {strategy_key} strategy."
        )

    llm = _get_planner_llm()
    result: PlanOutput = await _invoke_planner_llm(
        llm,
        [  # type: ignore[assignment]
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "subquestions": result.subquestions,
        "approved_plan": True,
        "thought_log": [f"[Planner] {task_label} Reasoning: {result.reasoning}"],
    }
