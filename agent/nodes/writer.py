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
