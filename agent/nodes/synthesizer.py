# agent/nodes/synthesizer.py
import asyncio
import functools
from pathlib import Path
from typing import Any

import yaml
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.state import ContradictionRecord, Finding, ReportOutput, ResearchState
from config.profiles import load_profile
from config.settings import settings
from utils.callbacks import TokenCostCallback

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

with open(_PROMPTS_DIR / "synthesizer.yaml") as f:
    _prompts = yaml.safe_load(f)

SYNTHESIS_PROMPT = _prompts["synthesis_prompt"]


class SynthesisOutput(BaseModel):
    title: str
    executive_summary: str
    introduction: str = Field(default="", description="Contextual background and research scope")
    detailed_analysis: str = Field(
        default="",
        description="Comprehensive multi-section narrative breakdown with analytical prose and data synthesis",
    )
    key_findings: list[Finding]
    emerging_trends: list[str]
    model_disagreements: list[str] = Field(default_factory=list)
    contradictions: list[ContradictionRecord] = Field(default_factory=list)


@functools.lru_cache(maxsize=1)
def _get_primary():
    return init_chat_model(
        settings.default_model, temperature=settings.synthesis_temperature
    ).with_structured_output(SynthesisOutput)


@functools.lru_cache(maxsize=1)
def _get_secondary():
    return init_chat_model(
        settings.secondary_model, temperature=settings.synthesis_temperature
    ).with_structured_output(SynthesisOutput)  # type: ignore[call-arg]


def build_synthesis_context(findings: list[Any]) -> str:
    sections = []
    source_idx = 1
    seen_urls: set[str] = set()

    for f in findings:
        section = [f"### Sub-question: {f.subquestion}"]
        if f.web_results:
            web_lines = []
            for w in f.web_results:
                if w.url in seen_urls:
                    continue
                seen_urls.add(w.url)
                web_lines.append(f"- [{source_idx}] [{w.title}]({w.url}): {w.snippet[:300]}")
                source_idx += 1
            if web_lines:
                section.append("**Web Sources:**")
                section.extend(web_lines)

        if f.papers:
            paper_lines = []
            for p in f.papers:
                if p.url in seen_urls:
                    continue
                seen_urls.add(p.url)
                paper_lines.append(
                    f"- [{source_idx}] [{p.title}]({p.url}) ({p.published_date}): {p.abstract[:300]}"
                )
                source_idx += 1
            if paper_lines:
                section.append("**Academic Papers:**")
                section.extend(paper_lines)

        if f.repos:
            repo_lines = []
            for r in f.repos:
                if r.url in seen_urls:
                    continue
                seen_urls.add(r.url)
                repo_lines.append(
                    f"- [{source_idx}] [{r.name}]({r.url}) ★{r.stars}: {r.description[:200]}"
                )
                source_idx += 1
            if repo_lines:
                section.append("**GitHub Repos:**")
                section.extend(repo_lines)

        if len(section) > 1:
            sections.append("\n".join(section))

    return "\n\n".join(sections)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _invoke_synth_llm(llm, prompt, callbacks=None):
    return await llm.ainvoke(prompt, config={"callbacks": callbacks or []})


async def run(state: ResearchState) -> dict:
    profile_name = state.get("profile", "fast")
    profile_cfg = load_profile(profile_name)
    synthesis_depth = profile_cfg.get("synthesis_depth", "brief")

    context = build_synthesis_context(state.get("findings", []))
    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
        context=context,
        synthesis_depth=synthesis_depth,
    )

    cb_primary = TokenCostCallback()
    cb_secondary = TokenCostCallback()

    synth_output: SynthesisOutput | None = None
    primary_failed = False
    secondary_failed = False
    primary_err: Exception | None = None
    secondary_err: Exception | None = None

    # Try primary LLM first
    try:
        synth_output = await asyncio.wait_for(
            _invoke_synth_llm(_get_primary(), prompt, callbacks=[cb_primary]),
            timeout=settings.synthesis_timeout_seconds,
        )
    except Exception as exc:
        primary_failed = True
        primary_err = exc

    # Fallback to secondary LLM if primary failed
    if primary_failed:
        try:
            synth_output = await asyncio.wait_for(
                _invoke_synth_llm(_get_secondary(), prompt, callbacks=[cb_secondary]),
                timeout=settings.synthesis_timeout_seconds,
            )
        except Exception as exc:
            secondary_failed = True
            secondary_err = exc

    if (primary_failed and secondary_failed) or synth_output is None:
        return {
            "final_report": None,
            "error_log": [
                f"[Synthesizer] Both models failed. "
                f"Primary: {str(primary_err)[:100]} | Secondary: {str(secondary_err)[:100]}"
            ],
            "thought_log": ["[Synthesizer] 0/2 models succeeded. Cannot produce report."],
        }

    final_report = ReportOutput(
        title=synth_output.title,
        executive_summary=synth_output.executive_summary,
        introduction=synth_output.introduction,
        detailed_analysis=synth_output.detailed_analysis,
        key_findings=synth_output.key_findings,
        emerging_trends=synth_output.emerging_trends,
        model_disagreements=synth_output.model_disagreements,
        contradictions=synth_output.contradictions,
        sources=[],
    )

    # ── Accumulate cost from LLM calls ──────────────────────────────────────
    meta = state.get("run_metadata")
    if meta:
        for cb in (cb_primary, cb_secondary):
            meta.total_input_tokens += cb.total_input_tokens
            meta.total_output_tokens += cb.total_output_tokens
            meta.estimated_cost_usd += cb.total_cost_usd

    status_msg = (
        "[Synthesizer] Generated report using primary model."
        if not primary_failed
        else "[Synthesizer] Used 1/2 models (primary failed, secondary fallback succeeded)."
    )

    return {
        "final_report": final_report,
        "run_metadata": meta,
        "thought_log": [status_msg],
    }
