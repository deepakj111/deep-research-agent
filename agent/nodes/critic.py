# agent/nodes/critic.py
import asyncio
import functools
from pathlib import Path

import yaml
from langchain.chat_models import init_chat_model

from agent.state import CritiqueOutput, ResearchState, RunMetadata
from config.settings import settings
from utils.callbacks import TokenCostCallback

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

with open(_PROMPTS_DIR / "critic.yaml") as f:
    _prompts = yaml.safe_load(f)

CRITIC_PROMPT = _prompts["evaluation_prompt"]


@functools.lru_cache(maxsize=1)
def _get_llm():
    return init_chat_model(
        settings.default_model, temperature=settings.critic_temperature
    ).with_structured_output(CritiqueOutput)


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
    if critique and critique.should_continue:
        return "continue"
    return "synthesize"


async def run(state: ResearchState) -> dict:
    all_findings = state.get("findings", [])
    meta = state.get("run_metadata")
    iteration_count = meta.iteration_count if meta else 0

    profile_name = state.get("profile", "deep")

    try:
        from config.profiles import load_profile  # noqa: PLC0415

        max_iters = load_profile(profile_name).get("max_iterations", settings.max_iterations)
    except Exception:
        max_iters = settings.max_iterations

    total_results = (
        sum(len(f.web_results) for f in all_findings)
        + sum(len(f.papers) for f in all_findings)
        + sum(len(f.repos) for f in all_findings)
    )

    # Short-circuit LLM call on terminal iteration, sufficient results count, or budget limit
    if (
        iteration_count + 1 >= max_iters
        or total_results >= 10
        or (meta and meta.estimated_cost_usd >= settings.max_cost_per_run_usd)
    ):
        short_circuit_critique = CritiqueOutput(
            coverage_score=1.0,
            recency_score=1.0,
            depth_score=1.0,
            source_diversity_score=1.0,
            missing_areas=[],
            should_continue=False,
            reasoning="Max iterations, sufficient findings count, or budget limit reached.",
        )
        updated_meta = RunMetadata(
            **(
                meta.model_dump()
                if meta
                else {"run_id": state.get("run_id", ""), "profile": profile_name}
            ),
        )
        updated_meta.iteration_count = iteration_count + 1
        return {
            "critique": short_circuit_critique,
            "run_metadata": updated_meta,
            "thought_log": [
                "[Critic] Terminal iteration / budget / findings threshold reached — short-circuiting to SYNTHESIZE"
            ],
        }

    cb = TokenCostCallback()
    critique: CritiqueOutput = await asyncio.wait_for(  # type: ignore[assignment]
        _get_llm().ainvoke(
            CRITIC_PROMPT.format(
                query=state["query"],
                subquestions=state.get("subquestions", []),
                relevant_sources=state.get("relevant_sources", ["web"]),
                web_count=sum(len(f.web_results) for f in all_findings),
                paper_count=sum(len(f.papers) for f in all_findings),
                repo_count=sum(len(f.repos) for f in all_findings),
                errors=[e for f in all_findings for e in f.tool_errors],
                iteration=iteration_count,
                max_iterations=max_iters,
            ),
            config={"callbacks": [cb]},
        ),
        timeout=settings.critic_timeout_seconds,
    )

    # Force synthesis if coverage score is high (>=0.75) or total results >= 6
    if critique.coverage_score >= 0.75 or total_results >= 6:
        critique.should_continue = False

    # Build a new RunMetadata with incremented iteration_count.
    updated_meta = RunMetadata(
        **(
            meta.model_dump()
            if meta
            else {"run_id": state.get("run_id", ""), "profile": profile_name}
        ),
    )
    updated_meta.iteration_count = iteration_count + 1
    updated_meta.total_input_tokens += cb.total_input_tokens
    updated_meta.total_output_tokens += cb.total_output_tokens
    updated_meta.estimated_cost_usd += cb.total_cost_usd

    return {
        "critique": critique,
        "run_metadata": updated_meta,
        "thought_log": [
            f"[Critic] Coverage={critique.coverage_score:.2f} | "
            f"Recency={critique.recency_score:.2f} | "
            f"Depth={critique.depth_score:.2f} | "
            f"Diversity={critique.source_diversity_score:.2f} | "
            f"{'LOOP AGAIN' if critique.should_continue else 'SYNTHESIZE'}"
        ],
    }
