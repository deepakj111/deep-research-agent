# agent/nodes/writer.py
from langchain.chat_models import init_chat_model
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.nodes.critic import score_source_trust
from agent.state import Citation, ReportOutput, ResearchState

_writer_llm = None


def _get_writer_llm():
    global _writer_llm
    if _writer_llm is None:
        from config.settings import settings

        _writer_llm = init_chat_model(settings.default_model)
    return _writer_llm


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _invoke_writer_llm(llm, prompt):
    return await llm.ainvoke(prompt)


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


async def _identify_section(instruction: str, current_report: ReportOutput) -> str:
    """Uses LLM to generate a targeted search query based on the refinement instruction."""
    prompt = f"Given this feedback for a report: '{instruction}', generate a single, concise web search query to gather the missing information. Output ONLY the query."
    llm = _get_writer_llm()
    res = await _invoke_writer_llm(llm, prompt)
    return str(res.content).strip()


async def _targeted_research(query: str, state: ResearchState) -> list:
    """Dispatches the query to the web_agent for live targeted research."""
    from agent.nodes.web_agent import run as run_web_agent

    temp_state = state.copy()
    temp_state["subquestions"] = [query]
    result = await run_web_agent(temp_state)
    return result.get("findings", [])


async def _patch_report(
    current_report: ReportOutput, additional_findings: list, instruction: str
) -> ReportOutput:
    """Uses structured LLM output to patch the entire report JSON logically based on new findings."""
    llm = _get_writer_llm().with_structured_output(ReportOutput)
    findings_str = "\n".join([str(f.model_dump()) for f in additional_findings])

    prompt = f"""
    You are an expert research editor. Update the report based on the user instruction and new findings.
    Only modify the sections that need updating.

    Instruction: {instruction}

    New Findings Context:
    {findings_str}

    Current Report JSON:
    {current_report.model_dump_json()}
    """

    patched = await _invoke_writer_llm(llm, prompt)
    if isinstance(patched, ReportOutput):
        # Preserve original complex nested metadata
        patched.sources = current_report.sources
        patched.contradictions = current_report.contradictions
        patched.model_disagreements = current_report.model_disagreements
        return patched
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

    search_query = await _identify_section(instruction, current_report)

    # Re-run targeted research
    additional_findings = await _targeted_research(search_query, state)

    # Patch the report
    patched = await _patch_report(current_report, additional_findings, instruction)
    patched.version = new_version

    return {
        "final_report": patched,
        "thought_log": [f"[Writer] Report refined (v{new_version}): {instruction[:80]}"],
    }
