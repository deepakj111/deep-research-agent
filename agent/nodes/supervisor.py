from typing import Literal

import yaml
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel

from agent.state import ResearchState

with open("agent/prompts/supervisor.yaml") as f:
    _prompts = yaml.safe_load(f)

SUPERVISOR_PROMPT = _prompts["routing_prompt"]


class RouterDecision(BaseModel):
    next: Literal["web_agent", "arxiv_agent", "github_agent", "critic"]
    reasoning: str
    current_subquestion: str


_llm = ChatOpenAI(model="gpt-4o").with_structured_output(RouterDecision)


async def run(state: ResearchState) -> Command:
    covered = len(state.get("findings", []))
    total = len(state.get("subquestions", []))

    if covered >= total:
        return Command(goto="critic")

    current_q = state["subquestions"][covered]

    decision: RouterDecision = await _llm.ainvoke(  # type: ignore[assignment]
        SUPERVISOR_PROMPT.format(
            subquestion=current_q,
            covered=covered,
            total=total,
        )
    )

    return Command(
        goto=decision.next,
        update={
            "thought_log": [
                f"[Supervisor] Dispatching sub-question {covered + 1}/{total} "
                f"to '{decision.next}': {current_q[:80]}... "
                f"Reason: {decision.reasoning}"
            ]
        },
    )
