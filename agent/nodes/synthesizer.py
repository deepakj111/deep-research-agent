import asyncio

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from agent.state import ReportOutput, ResearchState
from config.settings import settings

with open("agent/prompts/synthesizer.yaml") as f:
    _prompts = yaml.safe_load(f)

SYNTHESIS_PROMPT = _prompts["synthesis_prompt"]
RECONCILE_PROMPT = _prompts["reconcile_prompt"]

_gpt4o = ChatOpenAI(model=settings.default_model).with_structured_output(ReportOutput)
_claude = ChatAnthropic(model_name=settings.secondary_model).with_structured_output(ReportOutput)  # type: ignore[call-arg]
_reconciler = ChatOpenAI(model=settings.default_model)


def build_synthesis_context(findings) -> str:
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

    disagreements: list[str] = []

    if isinstance(gpt_report, Exception):
        final = claude_report
        disagreements.append(f"GPT-4o failed: {str(gpt_report)[:100]}")
    elif isinstance(claude_report, Exception):
        final = gpt_report
        disagreements.append(f"Claude failed: {str(claude_report)[:100]}")
    else:
        reconcile_response = await _reconciler.ainvoke(
            RECONCILE_PROMPT.format(
                query=state["query"],
                summary_a=gpt_report.executive_summary,  # type: ignore[union-attr]
                summary_b=claude_report.executive_summary,  # type: ignore[union-attr]
            )
        )
        final = gpt_report
        final.model_disagreements = [str(reconcile_response.content)[:500]]  # type: ignore[union-attr]

    failed = sum([isinstance(gpt_report, Exception), isinstance(claude_report, Exception)])
    return {
        "final_report": final,
        "thought_log": [
            f"[Synthesizer] Used {2 - failed}/2 models. {len(disagreements)} disagreements flagged."
        ],
    }
