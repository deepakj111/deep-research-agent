# agent/nodes/supervisor.py
from langgraph.types import Command, Send

from agent.state import ResearchState


async def run(state: ResearchState) -> Command:
    subquestions = state.get("subquestions", [])
    total = len(subquestions)

    if total == 0:
        return Command(goto="critic")

    # Map state's relevant_sources to target node names
    source_map = {
        "web": "web_agent",
        "arxiv": "arxiv_agent",
        "github": "github_agent",
    }
    relevant_sources = state.get("relevant_sources", ["web", "arxiv", "github"])
    target_agents = [source_map[s] for s in relevant_sources if s in source_map]
    if not target_agents:
        target_agents = ["web_agent"]

    # Fan-out: dispatch subquestions to selected target agents
    sends = []
    for subquestion in subquestions:
        for agent in target_agents:
            sends.append(Send(agent, {**state, "subquestions": [subquestion]}))

    log = (
        f"[Supervisor] Smart fan-out: {total} subquestions "
        f"× {len(target_agents)} selected agents ({', '.join(target_agents)}) = {len(sends)} concurrent tasks."
    )
    return Command(
        goto=sends,
        update={"thought_log": [log]},
    )
