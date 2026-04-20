# agent/nodes/supervisor.py
from langgraph.types import Command, Send

from agent.state import ResearchState


async def run(state: ResearchState) -> Command:
    subquestions = state.get("subquestions", [])
    total = len(subquestions)

    if total == 0:
        return Command(goto="critic")

    # Fan-out: dispatch ALL subquestions to all 3 agents in one shot.
    # We always do this — supervisor is only called once per research cycle.
    # After all Send tasks complete, each agent routes back to critic.
    sends = []
    for subquestion in subquestions:
        for agent in ["web_agent", "arxiv_agent", "github_agent"]:
            sends.append(Send(agent, {**state, "subquestions": [subquestion]}))

    log = (
        f"[Supervisor] Parallel fan-out: {total} subquestions "
        f"× 3 agents = {len(sends)} concurrent tasks."
    )
    return Command(
        goto=sends,
        update={"thought_log": [log]},
    )
