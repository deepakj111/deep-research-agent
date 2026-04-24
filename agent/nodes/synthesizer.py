# agent/nodes/synthesizer.py
import asyncio
from typing import Any

import yaml
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from agent.state import ContradictionRecord, ReportOutput, ResearchState
from config.settings import settings

with open("agent/prompts/synthesizer.yaml") as f:
    _prompts = yaml.safe_load(f)

SYNTHESIS_PROMPT = _prompts["synthesis_prompt"]
RECONCILE_PROMPT = _prompts["reconcile_prompt"]

_gpt4o = init_chat_model(settings.default_model).with_structured_output(ReportOutput)
_claude = init_chat_model(settings.secondary_model).with_structured_output(ReportOutput)  # type: ignore[call-arg]


class ReconcileOutput(BaseModel):
    contradictions: list[ContradictionRecord]
    summary: str


_reconciler = init_chat_model(settings.default_model).with_structured_output(ReconcileOutput)


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


async def run(state: ResearchState) -> dict:
    context = build_synthesis_context(state.get("findings", []))
    prompt = SYNTHESIS_PROMPT.format(query=state["query"], context=context)

    results = await asyncio.gather(
        _gpt4o.ainvoke(prompt),
        _claude.ainvoke(prompt),
        return_exceptions=True,
    )
    gpt_report: ReportOutput | Exception = results[0]  # type: ignore[assignment]
    claude_report: ReportOutput | Exception = results[1]  # type: ignore[assignment]

    gpt_failed = isinstance(gpt_report, Exception)
    claude_failed = isinstance(claude_report, Exception)

    contradictions: list[ContradictionRecord] = []

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
        reconcile: ReconcileOutput = await _reconciler.ainvoke(  # type: ignore[assignment]
            RECONCILE_PROMPT.format(
                query=state["query"],
                summary_a=gpt_report.executive_summary,  # type: ignore[union-attr]
                summary_b=claude_report.executive_summary,  # type: ignore[union-attr]
            )
        )
        final = gpt_report
        contradictions = reconcile.contradictions
        final.contradictions = contradictions  # type: ignore[union-attr]
        final.model_disagreements = [reconcile.summary]  # type: ignore[union-attr]

    failed = sum([gpt_failed, claude_failed])
    return {
        "final_report": final,
        "thought_log": [
            f"[Synthesizer] Used {2 - failed}/2 models. "
            f"{len(contradictions)} contradictions detected."
        ],
    }
