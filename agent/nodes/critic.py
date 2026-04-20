import yaml
from langchain_openai import ChatOpenAI

from agent.state import CritiqueOutput, ResearchState
from config.settings import settings

with open("agent/prompts/critic.yaml") as f:
    _prompts = yaml.safe_load(f)

CRITIC_PROMPT = _prompts["evaluation_prompt"]

_llm = ChatOpenAI(model=settings.default_model).with_structured_output(CritiqueOutput)

MAX_ITERATIONS = 15


def score_source_trust(source: dict, source_type: str) -> float:
    score = 0.0
    if source_type == "arxiv":
        score = 0.4
        if source.get("citation_count", 0) > 50:
            score += 0.2
        pub_year = int(str(source.get("published_date", "2020"))[:4])
        if pub_year >= 2024:
            score += 0.2
        score += 0.2
    elif source_type == "web":
        trusted_domains = {"arxiv.org", "github.com", ".edu", ".gov", "nature.com"}
        url = source.get("url", "")
        if any(d in url for d in trusted_domains):
            score = 0.4
        score = min(0.6, score + 0.1 * source.get("relevance_score", 0) * 6)
    elif source_type == "github":
        stars = source.get("stars", 0)
        score = min(0.6, stars / 1000 * 0.6)
        if str(source.get("last_updated", ""))[:4] >= "2024":
            score += 0.4
    return min(1.0, score)


def should_continue(state: ResearchState) -> str:
    critique = state.get("critique")
    meta = state.get("run_metadata")
    if meta and meta.iteration_count >= MAX_ITERATIONS:
        return "synthesize"
    if critique and critique.should_continue:
        return "continue"
    return "synthesize"


async def run(state: ResearchState) -> dict:
    all_findings = state.get("findings", [])
    meta = state.get("run_metadata")
    iteration_count = meta.iteration_count if meta else 0

    critique: CritiqueOutput = await _llm.ainvoke(  # type: ignore[assignment]
        CRITIC_PROMPT.format(
            query=state["query"],
            subquestions=state.get("subquestions", []),
            web_count=sum(len(f.web_results) for f in all_findings),
            paper_count=sum(len(f.papers) for f in all_findings),
            repo_count=sum(len(f.repos) for f in all_findings),
            errors=[e for f in all_findings for e in f.tool_errors],
            iteration=iteration_count,
            max_iterations=MAX_ITERATIONS,
        )
    )

    if meta:
        meta.iteration_count = iteration_count + 1

    return {
        "critique": critique,
        "run_metadata": meta,
        "thought_log": [
            f"[Critic] Coverage={critique.coverage_score:.2f} | "
            f"Recency={critique.recency_score:.2f} | "
            f"Depth={critique.depth_score:.2f} | "
            f"Diversity={critique.source_diversity_score:.2f} | "
            f"{'LOOP AGAIN' if critique.should_continue else 'SYNTHESIZE'}"
        ],
    }
