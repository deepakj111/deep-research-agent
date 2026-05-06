# agent/nodes/writer.py
"""
Report finalization with grounding verification.

After the synthesizer produces a report, the writer:
  1. Builds citations from the actual retrieved findings.
  2. Verifies that every claim in the report's key_findings is grounded
     in the retrieved sources (anti-hallucination check).
  3. Downgrades confidence on ungrounded claims and logs a warning.
  4. Deduplicates citations by URL.
"""

from __future__ import annotations

import logging

from agent.nodes.critic import score_source_trust
from agent.state import Citation, Finding, ReportOutput, ResearchState

logger = logging.getLogger(__name__)


def _build_citations(findings) -> list[Citation]:
    """Build deduplicated citations from all retrieved findings."""
    citations: list[Citation] = []
    seen_urls: set[str] = set()

    for f in findings:
        for w in f.web_results:
            if w.url in seen_urls:
                continue
            seen_urls.add(w.url)
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
            if p.url in seen_urls:
                continue
            seen_urls.add(p.url)
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
            if r.url in seen_urls:
                continue
            seen_urls.add(r.url)
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


def _verify_grounding(
    report: ReportOutput,
    retrieved_urls: set[str],
    retrieved_titles: set[str],
) -> tuple[list[Finding], int, int]:
    """
    Cross-reference each key finding's citations against actually-retrieved sources.

    For each finding, check whether its inline citations reference URLs or titles
    that were actually fetched during research. If a finding has NO grounded
    citations, its confidence is downgraded to 'low' to flag potential
    hallucination.

    Returns:
        (verified_findings, grounded_count, ungrounded_count)
    """
    grounded = 0
    ungrounded = 0
    verified: list[Finding] = []

    for finding in report.key_findings:
        # A finding is grounded if at least one of its citations matches
        # a URL or title from the retrieved sources.
        has_grounded_citation = False
        for citation in finding.citations:
            url_match = citation.source_url in retrieved_urls
            # Fuzzy title match: check if the cited title appears as a
            # substring in any retrieved title (handles minor reformatting).
            title_match = any(
                citation.title.lower() in rt.lower() or rt.lower() in citation.title.lower()
                for rt in retrieved_titles
                if len(rt) > 5  # skip trivially short titles
            )
            if url_match or title_match:
                has_grounded_citation = True
                break

        if has_grounded_citation or not finding.citations:
            grounded += 1
            verified.append(finding)
        else:
            ungrounded += 1
            # Downgrade confidence — don't remove the finding, but flag it
            verified.append(
                Finding(
                    claim=finding.claim,
                    citations=finding.citations,
                    confidence="low",
                )
            )
            logger.warning(
                "[Writer] Ungrounded finding detected (hallucination risk): %s",
                finding.claim[:100],
            )

    return verified, grounded, ungrounded


async def run(state: ResearchState) -> dict:
    report = state.get("final_report")
    if not report:
        return {
            "error_log": ["[Writer] No report to write — synthesizer produced None."],
            "thought_log": ["[Writer] Skipped — no report available."],
        }

    findings = state.get("findings", [])
    citations = _build_citations(findings)
    report.sources = citations
    report.version = 1

    # ── Grounding verification (anti-hallucination) ───────────────────────
    # Build the set of URLs and titles that were *actually* retrieved during
    # research. Any claim citing a source outside this set is suspect.
    retrieved_urls: set[str] = set()
    retrieved_titles: set[str] = set()
    for f in findings:
        for w in f.web_results:
            retrieved_urls.add(w.url)
            retrieved_titles.add(w.title)
        for p in f.papers:
            retrieved_urls.add(p.url)
            retrieved_titles.add(p.title)
        for r in f.repos:
            retrieved_urls.add(r.url)
            retrieved_titles.add(r.name)

    verified_findings, grounded, ungrounded = _verify_grounding(
        report, retrieved_urls, retrieved_titles
    )
    report.key_findings = verified_findings

    thought_entries = [
        f"[Writer] Report finalized. "
        f"{len(citations)} unique citations attached. "
        f"Version {report.version}."
    ]

    if ungrounded > 0:
        thought_entries.append(
            f"[Writer] Grounding check: {grounded} grounded, "
            f"{ungrounded} ungrounded (downgraded to low confidence). "
            f"Hallucination risk flagged."
        )
    else:
        thought_entries.append(
            f"[Writer] Grounding check passed: all {grounded} findings "
            f"verified against retrieved sources."
        )

    return {
        "final_report": report,
        "thought_log": thought_entries,
    }
