# agent/nodes/synthesizer.py
import asyncio
import functools
from pathlib import Path
from typing import Any

import yaml
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.state import ContradictionRecord, ReportOutput, ResearchState
from config.settings import settings
from utils.callbacks import TokenCostCallback

_LLM_TIMEOUT_SECONDS = 120.0  # synthesis is the most expensive call

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

with open(_PROMPTS_DIR / "synthesizer.yaml") as f:
    _prompts = yaml.safe_load(f)

SYNTHESIS_PROMPT = _prompts["synthesis_prompt"]
RECONCILE_PROMPT = _prompts["reconcile_prompt"]


class ReconcileOutput(BaseModel):
    contradictions: list[ContradictionRecord]
    summary: str


@functools.lru_cache(maxsize=1)
def _get_gpt4o():
    return init_chat_model(
        settings.default_model, temperature=settings.synthesis_temperature
    ).with_structured_output(ReportOutput)


@functools.lru_cache(maxsize=1)
def _get_claude():
    return init_chat_model(
        settings.secondary_model, temperature=settings.synthesis_temperature
    ).with_structured_output(ReportOutput)  # type: ignore[call-arg]


@functools.lru_cache(maxsize=1)
def _get_reconciler():
    return init_chat_model(
        settings.default_model, temperature=settings.synthesis_temperature
    ).with_structured_output(ReconcileOutput)


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

    cb_gpt = TokenCostCallback()
    cb_claude = TokenCostCallback()

    results = await asyncio.wait_for(
        asyncio.gather(
            _invoke_synth_llm(_get_gpt4o(), prompt, callbacks=[cb_gpt]),
            _invoke_synth_llm(_get_claude(), prompt, callbacks=[cb_claude]),
            return_exceptions=True,
        ),
        timeout=_LLM_TIMEOUT_SECONDS,
    )
    gpt_report: ReportOutput | Exception = results[0]  # type: ignore[assignment]
    claude_report: ReportOutput | Exception = results[1]  # type: ignore[assignment]

    gpt_failed = isinstance(gpt_report, Exception)
    claude_failed = isinstance(claude_report, Exception)

    contradictions: list[ContradictionRecord] = []
    cb_reconcile = TokenCostCallback()

    if gpt_failed and claude_failed:
        # Both models failed — return None so writer.py handles it gracefully
        # via its existing "No report to write" error_log path.
        return {
            "final_report": None,
            "error_log": [
                f"[Synthesizer] Both models failed. "
                f"GPT: {str(gpt_report)[:100]} | Claude: {str(claude_report)[:100]}"
            ],
            "thought_log": ["[Synthesizer] 0/2 models succeeded. Cannot produce report."],
        }

    if gpt_failed:
        final = claude_report
    elif claude_failed:
        final = gpt_report
    else:
        reconcile: ReconcileOutput = await asyncio.wait_for(  # type: ignore[assignment]
            _invoke_synth_llm(
                _get_reconciler(),
                RECONCILE_PROMPT.format(
                    query=state["query"],
                    summary_a=gpt_report.executive_summary,  # type: ignore[union-attr]
                    summary_b=claude_report.executive_summary,  # type: ignore[union-attr]
                ),
                callbacks=[cb_reconcile],
            ),
            timeout=_LLM_TIMEOUT_SECONDS,
        )
        final = gpt_report
        contradictions = reconcile.contradictions
        final.contradictions = contradictions  # type: ignore[union-attr]
        final.model_disagreements = [reconcile.summary]  # type: ignore[union-attr]

    # ── Accumulate cost from all LLM calls ──────────────────────────────
    meta = state.get("run_metadata")
    if meta:
        for cb in (cb_gpt, cb_claude, cb_reconcile):
            meta.total_input_tokens += cb.total_input_tokens
            meta.total_output_tokens += cb.total_output_tokens
            meta.estimated_cost_usd += cb.total_cost_usd

    failed = sum([gpt_failed, claude_failed])
    return {
        "final_report": final,
        "run_metadata": meta,
        "thought_log": [
            f"[Synthesizer] Used {2 - failed}/2 models. "
            f"{len(contradictions)} contradictions detected."
        ],
    }
