# agent/nodes/supervisor.py
from langgraph.types import Command, Send

from agent.state import ResearchState

_ARXIV_KEYWORDS = {
    "paper",
    "arxiv",
    "academic",
    "theory",
    "model",
    "architecture",
    "benchmark",
    "survey",
    "algorithm",
    "method",
    "study",
    "publication",
    "citation",
    "theorem",
    "proof",
    "journal",
    "experiment",
    "novel",
}

_GITHUB_KEYWORDS = {
    "code",
    "repo",
    "github",
    "library",
    "framework",
    "implementation",
    "package",
    "sdk",
    "tool",
    "repository",
    "software",
    "open-source",
    "git",
    "script",
    "module",
    "api",
    "pip",
}


def _matches_keywords(text: str, keywords: set[str]) -> bool:
    words = text.lower().split()
    return any(kw in text.lower() for kw in keywords) or any(w in keywords for w in words)


async def run(state: ResearchState) -> dict | Command:
    subquestions = state.get("subquestions", [])
    total = len(subquestions)

    if total == 0:
        return Command(goto="critic")

    relevant_sources = state.get("relevant_sources", ["web", "arxiv", "github"])
    payload_base = {
        "profile": state.get("profile", "deep"),
        "run_id": state.get("run_id", ""),
    }

    sends = []
    arxiv_assigned = False
    github_assigned = False

    for subquestion in subquestions:
        # Web agent processes all subquestions if web source is enabled
        if "web" in relevant_sources or not relevant_sources:
            sends.append(Send("web_agent", {**payload_base, "subquestions": [subquestion]}))

        # Arxiv agent processes academic subquestions (or all if arxiv is sole source)
        if "arxiv" in relevant_sources and (
            relevant_sources == ["arxiv"] or _matches_keywords(subquestion, _ARXIV_KEYWORDS)
        ):
            sends.append(Send("arxiv_agent", {**payload_base, "subquestions": [subquestion]}))
            arxiv_assigned = True

        # GitHub agent processes code/repo subquestions (or all if github is sole source)
        if "github" in relevant_sources and (
            relevant_sources == ["github"] or _matches_keywords(subquestion, _GITHUB_KEYWORDS)
        ):
            sends.append(Send("github_agent", {**payload_base, "subquestions": [subquestion]}))
            github_assigned = True

    # Fallback: if arxiv/github were specified in relevant_sources but no subquestion matched keywords,
    # route the first subquestion to ensure requested sources are queried.
    if "arxiv" in relevant_sources and not arxiv_assigned and subquestions:
        sends.append(Send("arxiv_agent", {**payload_base, "subquestions": [subquestions[0]]}))
    if "github" in relevant_sources and not github_assigned and subquestions:
        sends.append(Send("github_agent", {**payload_base, "subquestions": [subquestions[0]]}))

    if not sends:
        sends.append(Send("web_agent", {**payload_base, "subquestions": [subquestions[0]]}))

    log = (
        f"[Supervisor] Targeted routing: {total} subquestions → {len(sends)} targeted tasks "
        f"across sources ({', '.join(relevant_sources)})."
    )
    return Command(
        goto=sends,
        update={"thought_log": [log]},
    )
