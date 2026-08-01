# agent/nodes/synthesizer.py
import asyncio
import functools
from pathlib import Path
from typing import Any

import yaml
from langchain.chat_models import init_chat_model
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.state import ReportOutput, ResearchState
from config.settings import settings
from utils.callbacks import TokenCostCallback

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

with open(_PROMPTS_DIR / "synthesizer.yaml") as f:
    _prompts = yaml.safe_load(f)

SYNTHESIS_PROMPT = _prompts["synthesis_prompt"]


@functools.lru_cache(maxsize=1)
def _get_primary():
    return init_chat_model(
        settings.default_model, temperature=settings.synthesis_temperature
    ).with_structured_output(ReportOutput)


@functools.lru_cache(maxsize=1)
def _get_secondary():
    return init_chat_model(
        settings.secondary_model, temperature=settings.synthesis_temperature
    ).with_structured_output(ReportOutput)  # type: ignore[call-arg]


def build_synthesis_context(findings: list[Any]) -> str:
    sections = []
    for f in findings:
        section = [f"### Sub-question: {f.subquestion}"]
        if f.web_results:
            section.append("**Web Sources:**")
            for w in f.web_results:
                section.append(f"- [{w.title}]({w.url}): {w.snippet[:200]}")
        if f.papers:
            section.append("**Academic Papers:**")
            for p in f.papers:
                section.append(f"- {p.title} ({p.published_date}): {p.abstract[:200]}")
        if f.repos:
            section.append("**GitHub Repos:**")
            for r in f.repos:
                section.append(f"- [{r.name}]({r.url}) ★{r.stars}: {r.description[:150]}")
        if f.tool_errors:
            section.append(f"**Errors (graceful degradation):** {', '.join(f.tool_errors)}")
        sections.append("\n".join(section))
    return "\n\n".join(sections)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _invoke_synth_llm(llm, prompt, callbacks=None):
    return await llm.ainvoke(prompt, config={"callbacks": callbacks or []})


async def run(state: ResearchState) -> dict:
    context = build_synthesis_context(state.get("findings", []))
    prompt = SYNTHESIS_PROMPT.format(query=state["query"], context=context)

    cb_primary = TokenCostCallback()
    cb_secondary = TokenCostCallback()

    final_report: ReportOutput | None = None
    primary_failed = False
    secondary_failed = False
    primary_err: Exception | None = None
    secondary_err: Exception | None = None

    # Try primary LLM first
    try:
        final_report = await asyncio.wait_for(
            _invoke_synth_llm(_get_primary(), prompt, callbacks=[cb_primary]),
            timeout=settings.synthesis_timeout_seconds,
        )
    except Exception as exc:
        primary_failed = True
        primary_err = exc

    # Fallback to secondary LLM if primary failed
    if primary_failed:
        try:
            final_report = await asyncio.wait_for(
                _invoke_synth_llm(_get_secondary(), prompt, callbacks=[cb_secondary]),
                timeout=settings.synthesis_timeout_seconds,
            )
        except Exception as exc:
            secondary_failed = True
            secondary_err = exc

    if primary_failed and secondary_failed:
        return {
            "final_report": None,
            "error_log": [
                f"[Synthesizer] Both models failed. "
                f"Primary: {str(primary_err)[:100]} | Secondary: {str(secondary_err)[:100]}"
            ],
            "thought_log": ["[Synthesizer] 0/2 models succeeded. Cannot produce report."],
        }

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
