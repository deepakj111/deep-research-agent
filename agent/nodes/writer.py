# agent/nodes/writer.py
from agent.nodes.critic import score_source_trust
from agent.state import Citation, ResearchState


def _build_citations(findings) -> list[Citation]:
    citations: list[Citation] = []
    for f in findings:
        for w in f.web_results:
            trust = score_source_trust(w.model_dump(), "web")
            citations.append(
                Citation(
                    source_url=w.url,
                    title=w.title,
                    exact_snippet=w.snippet[:300],
                    source_type="web",
                    trust_score=trust,
                )
            )
        for p in f.papers:
            trust = score_source_trust(p.model_dump(), "arxiv")
            citations.append(
                Citation(
                    source_url=p.url,
                    title=p.title,
                    exact_snippet=p.abstract[:300],
                    source_type="arxiv",
                    trust_score=trust,
                )
            )
        for r in f.repos:
            trust = score_source_trust(r.model_dump(), "github")
            citations.append(
                Citation(
                    source_url=r.url,
                    title=r.name,
                    exact_snippet=r.description[:300],
                    source_type="github",
                    trust_score=trust,
                )
            )
    return citations


async def run(state: ResearchState) -> dict:
    report = state.get("final_report")
    if not report:
        return {
            "error_log": ["[Writer] No report to write — synthesizer produced None."],
            "thought_log": ["[Writer] Skipped — no report available."],
        }

    citations = _build_citations(state.get("findings", []))
    report.sources = citations
    report.version = 1

    return {
        "final_report": report,
        "thought_log": [
            f"[Writer] Report finalized. "
            f"{len(citations)} citations attached. "
            f"Version {report.version}."
        ],
    }


async def _identify_section(instruction: str, current_report) -> str:
    """Mock implementation to identify which section of the report to refine based on instruction."""
    # In a full implementation, an LLM call would parse the instruction and map it to a specific report section.
    return "example_section"


async def _targeted_research(section: str, state: ResearchState) -> list:
    """Mock implementation to fetch additional targeted research."""
    # In a full implementation, this routes back to sub-agents (e.g. web_agent) with dynamic localized queries.
    return state.get("findings", [])


async def _patch_report(current_report, additional_findings: list, instruction: str):
    """Mock implementation to patch the report with new findings."""
    # Process new findings and integrate them into the AST of the report.
    return current_report


async def refine_report(state: ResearchState, instruction: str) -> dict:
    """
    Re-run only the affected sub-agent and patch the report.
    Uses LangGraph checkpointing to resume without full re-run.
    """
    current_report = state.get("final_report")
    if not current_report:
        return {"error_log": ["[Writer] Cannot refine — no existing report."]}

    new_version = current_report.version + 1

    # Identify which section to refine
    section_to_refine = await _identify_section(instruction, current_report)

    # Re-run only that sub-agent
    additional_findings = await _targeted_research(section_to_refine, state)

    # Patch the report
    patched = await _patch_report(current_report, additional_findings, instruction)
    patched.version = new_version

    return {
        "final_report": patched,
        "thought_log": [f"[Writer] Report refined (v{new_version}): {instruction[:80]}"],
    }
