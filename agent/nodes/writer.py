from agent.state import Citation, ResearchState


def _build_citations(findings) -> list[Citation]:
    citations: list[Citation] = []
    for f in findings:
        for w in f.web_results:
            citations.append(
                Citation(
                    source_url=w.url,
                    title=w.title,
                    exact_snippet=w.snippet[:300],
                    source_type="web",
                    trust_score=min(1.0, 0.3 + w.relevance_score * 0.7),
                )
            )
        for p in f.papers:
            citations.append(
                Citation(
                    source_url=p.url,
                    title=p.title,
                    exact_snippet=p.abstract[:300],
                    source_type="arxiv",
                    trust_score=p.trust_score or 0.6,
                )
            )
        for r in f.repos:
            citations.append(
                Citation(
                    source_url=r.url,
                    title=r.name,
                    exact_snippet=r.description[:300],
                    source_type="github",
                    trust_score=r.trust_score or 0.5,
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
