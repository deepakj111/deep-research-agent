"""
tests/unit/test_report_formatter.py

Unit tests for the report formatter: markdown, HTML, and PDF export.
"""

from __future__ import annotations

import pytest

from agent.state import Citation, ContradictionRecord, Finding, ReportOutput
from utils.report_formatter import export_to_pdf, to_html, to_markdown

# ────────────────────────── Fixtures ──────────────────────────────────────────


@pytest.fixture
def sample_report() -> ReportOutput:
    return ReportOutput(
        title="Advances in Quantum Error Correction 2025",
        executive_summary="This report surveys the latest breakthroughs in quantum error correction.",
        key_findings=[
            Finding(
                claim="Surface codes have achieved a new threshold of 99.9% fidelity.",
                citations=[
                    Citation(
                        source_url="https://arxiv.org/abs/2501.12345",
                        title="Surface Code Improvements",
                        exact_snippet="We demonstrate a 99.9% fidelity threshold...",
                        source_type="arxiv",
                        trust_score=0.85,
                    ),
                    Citation(
                        source_url="https://example.com/quantum",
                        title="Quantum Computing News",
                        exact_snippet="Researchers report breaking the error threshold...",
                        source_type="web",
                        trust_score=0.55,
                    ),
                ],
                confidence="high",
            ),
            Finding(
                claim="Google's Willow chip demonstrates below-threshold performance.",
                citations=[
                    Citation(
                        source_url="https://github.com/google/quantum",
                        title="google/quantum",
                        exact_snippet="Willow chip benchmark results",
                        source_type="github",
                        trust_score=0.72,
                    ),
                ],
                confidence="medium",
            ),
        ],
        emerging_trends=[
            "Topological qubits gaining traction",
            "LDPC codes outperforming surface codes in simulation",
        ],
        recommended_next_steps=[
            "Monitor IBM's next error correction milestone",
            "Track topological qubit progress at Microsoft",
        ],
        model_disagreements=[
            "GPT-4o and Claude disagreed on timeline estimates for fault tolerance."
        ],
        contradictions=[
            ContradictionRecord(
                claim_a="Fault tolerance by 2027",
                claim_b="Fault tolerance by 2030",
                resolution="Timeline depends on qubit type; surface codes favor 2027",
                preferred_source="gpt4o",
            )
        ],
        sources=[
            Citation(
                source_url="https://arxiv.org/abs/2501.12345",
                title="Surface Code Improvements",
                exact_snippet="We demonstrate a 99.9% fidelity threshold...",
                source_type="arxiv",
                trust_score=0.85,
            ),
            Citation(
                source_url="https://example.com/quantum",
                title="Quantum Computing News",
                exact_snippet="Researchers report breaking the error threshold...",
                source_type="web",
                trust_score=0.55,
            ),
            Citation(
                source_url="https://github.com/google/quantum",
                title="google/quantum",
                exact_snippet="Willow chip benchmark results",
                source_type="github",
                trust_score=0.72,
            ),
        ],
        version=1,
    )


@pytest.fixture
def minimal_report() -> ReportOutput:
    """A report with minimal content — tests empty-list edge cases."""
    return ReportOutput(
        title="Minimal Report",
        executive_summary="Brief summary.",
        key_findings=[],
        emerging_trends=[],
        recommended_next_steps=[],
        model_disagreements=[],
        contradictions=[],
        sources=[],
        version=1,
    )


# ────────────────────────── Markdown Tests ────────────────────────────────────


class TestToMarkdown:
    def test_contains_title(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "# Advances in Quantum Error Correction 2025" in md

    def test_contains_executive_summary(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "## Executive Summary" in md
        assert "latest breakthroughs" in md

    def test_contains_key_findings(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "## Key Findings" in md
        assert "Surface codes" in md
        assert "Willow chip" in md

    def test_contains_citations_with_trust_scores(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "trust: 0.85" in md
        assert "arxiv" in md

    def test_contains_emerging_trends(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "## Emerging Trends" in md
        assert "Topological qubits" in md

    def test_contains_model_disagreements(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "## Model Disagreements" in md

    def test_contains_contradictions(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "## Contradictions Detected" in md
        assert "Fault tolerance by 2027" in md

    def test_contains_sources_with_trust_indicators(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "## Sources" in md
        # High trust = green circle
        assert "🟢" in md
        # Medium trust
        assert "🟡" in md

    def test_contains_version_footer(self, sample_report: ReportOutput):
        md = to_markdown(sample_report)
        assert "Report version 1" in md

    def test_minimal_report_no_crash(self, minimal_report: ReportOutput):
        md = to_markdown(minimal_report)
        assert "# Minimal Report" in md
        assert "## Key Findings" in md
        # No trends/steps sections when empty
        assert "## Emerging Trends" not in md


# ────────────────────────── HTML Tests ────────────────────────────────────────


class TestToHtml:
    def test_valid_html_structure(self, sample_report: ReportOutput):
        html = to_html(sample_report)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<body>" in html

    def test_contains_title_in_head(self, sample_report: ReportOutput):
        html = to_html(sample_report)
        assert f"<title>{sample_report.title}</title>" in html

    def test_contains_report_content(self, sample_report: ReportOutput):
        html = to_html(sample_report)
        assert "Surface codes" in html
        assert "Executive Summary" in html

    def test_contains_css_styling(self, sample_report: ReportOutput):
        html = to_html(sample_report)
        assert "font-family" in html
        assert "Inter" in html

    def test_minimal_report(self, minimal_report: ReportOutput):
        html = to_html(minimal_report)
        assert "Minimal Report" in html


# ────────────────────────── PDF Tests ─────────────────────────────────────────


class TestExportToPdf:
    def test_pdf_output_is_bytes(self, sample_report: ReportOutput):
        try:
            pdf = export_to_pdf(sample_report)
            assert isinstance(pdf, bytes)
            assert len(pdf) > 0
        except ImportError:
            pytest.skip("WeasyPrint not installed — skipping PDF test")

    def test_pdf_starts_with_header(self, sample_report: ReportOutput):
        try:
            pdf = export_to_pdf(sample_report)
            # PDF files start with %PDF
            assert pdf[:5] == b"%PDF-"
        except ImportError:
            pytest.skip("WeasyPrint not installed — skipping PDF test")

    def test_minimal_report_pdf(self, minimal_report: ReportOutput):
        try:
            pdf = export_to_pdf(minimal_report)
            assert pdf[:5] == b"%PDF-"
        except ImportError:
            pytest.skip("WeasyPrint not installed — skipping PDF test")
