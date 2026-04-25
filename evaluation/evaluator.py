"""
evaluation/evaluator.py

LLM-as-judge evaluation pipeline for the DeepResearch Agent.

Scoring model: GPT-4o with structured output (temperature=0 for determinism).
Five dimensions, each rated 0–5, then normalized to [0, 1].

Design choices:
  • Lazy LLM instantiation — no import-time OpenAI client creation, which would
    fail if OPENAI_API_KEY is not set (e.g. during unit test collection).
  • evaluate_report() is the single public entry point; the benchmark runner
    and API both call it.
  • An optional DeepEval integration is provided for teams that want RAGAs-
    style metrics; it degrades gracefully if deepeval is unconfigured.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from agent.state import ReportOutput

logger = logging.getLogger(__name__)

# ──────────────────────────── Score Model ────────────────────────────────────


class EvalScores(BaseModel):
    """Per-dimension quality scores for a single research report."""

    faithfulness: float = Field(
        ge=0,
        le=5,
        description="All claims grounded in retrieved sources, not hallucinated.",
    )
    answer_relevancy: float = Field(
        ge=0,
        le=5,
        description="Report fully and directly addresses the original query.",
    )
    source_coverage: float = Field(
        ge=0,
        le=5,
        description="Report draws from all three source types: web, arXiv, GitHub.",
    )
    citation_accuracy: float = Field(
        ge=0,
        le=5,
        description="In-text citations link to sources that were actually fetched.",
    )
    coherence: float = Field(
        ge=0,
        le=5,
        description="Logical structure, no repetition, clear professional narrative.",
    )
    overall_notes: str = Field(
        default="",
        description="Brief evaluator notes on the report's key strengths and weaknesses.",
    )

    @property
    def normalized_average(self) -> float:
        """Mean score normalized to [0, 1] (divides by max possible score of 25)."""
        total = (
            self.faithfulness
            + self.answer_relevancy
            + self.source_coverage
            + self.citation_accuracy
            + self.coherence
        )
        return round(total / 25.0, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "source_coverage": self.source_coverage,
            "citation_accuracy": self.citation_accuracy,
            "coherence": self.coherence,
            "overall_notes": self.overall_notes,
            "normalized_average": self.normalized_average,
        }


# ──────────────────────────── Prompt ─────────────────────────────────────────

_EVAL_RUBRIC = """\
You are an expert research quality evaluator. Score the following report on five
dimensions. Be strict — a 5 requires near-perfection. Output JSON only.

## Original Research Query
{query}

## Report Under Evaluation
{report}

## Sources Actually Retrieved During Research
{sources}

## Scoring Rubric (each dimension: 0.0 – 5.0, floats allowed)

**faithfulness**
  Are all factual claims grounded in the retrieved sources listed above?
  5 = Every claim is traceable to a source with no fabrications.
  0 = Many unsupported or hallucinated claims.

**answer_relevancy**
  Does the report fully and directly address the original query?
  5 = Query answered comprehensively with appropriate depth.
  0 = Report is off-topic, superficial, or misses the question.

**source_coverage**
  Does the report draw substantively from all three source types?
  (web sources, arXiv academic papers, AND GitHub repositories)
  5 = All three types used with meaningful content.
  0 = Only one type used, or none at all.

**citation_accuracy**
  Are inline citations accurate and linked to actually-fetched content?
  5 = All citations verified against the source list above.
  0 = Citations are fabricated, missing, or link to unfetched pages.

**coherence**
  Is the report logically structured, non-repetitive, and well-written?
  5 = Professional quality, flows naturally, clear section hierarchy.
  0 = Incoherent, disorganized, heavily repetitive.

Output only a JSON object matching this schema — no prose outside the JSON:
{{
  "faithfulness": <float>,
  "answer_relevancy": <float>,
  "source_coverage": <float>,
  "citation_accuracy": <float>,
  "coherence": <float>,
  "overall_notes": "<string>"
}}
"""

# ──────────────────────── Lazy LLM Instance ──────────────────────────────────

_eval_llm = None


def _get_eval_llm():  # type: ignore[return]
    """Lazily construct the evaluation LLM to avoid import-time side effects."""
    global _eval_llm
    if _eval_llm is None:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        _eval_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(EvalScores)
    return _eval_llm


# ──────────────────────── Report Formatting Helpers ──────────────────────────


def _build_report_text(report: ReportOutput) -> str:
    """Flatten a ReportOutput into a plain-text string for the evaluator prompt."""
    parts = [
        f"# {report.title}",
        f"\n## Executive Summary\n{report.executive_summary}",
        "\n## Key Findings",
    ]
    for i, finding in enumerate(report.key_findings[:15], 1):
        parts.append(f"\n### {i}. {finding.claim}")
        parts.append(f"Confidence: {finding.confidence}")
        for citation in finding.citations[:3]:
            parts.append(f"  - Source: {citation.title} ({citation.source_url})")

    if report.emerging_trends:
        parts.append("\n## Emerging Trends")
        parts.extend(f"- {t}" for t in report.emerging_trends)

    if report.recommended_next_steps:
        parts.append("\n## Recommended Next Steps")
        parts.extend(f"- {s}" for s in report.recommended_next_steps)

    if report.model_disagreements:
        parts.append("\n## Model Disagreements")
        parts.extend(f"> {d}" for d in report.model_disagreements)

    return "\n".join(parts)


def _build_source_list(report: ReportOutput) -> str:
    """Produce a compact source inventory for faithfulness checking."""
    if not report.sources:
        return "(no sources recorded)"
    lines = []
    for src in report.sources[:40]:  # cap to stay within token limits
        lines.append(
            f"[{src.source_type.upper()}] {src.title} — {src.source_url} "
            f"(trust: {src.trust_score:.2f})"
        )
    return "\n".join(lines)


# ──────────────────────────── Public API ─────────────────────────────────────


async def evaluate_report(
    query: str,
    report: ReportOutput,
    *,
    timeout_seconds: float = 90.0,
) -> EvalScores:
    """
    Evaluate a research report using GPT-4o as the judge.

    Args:
        query:           The original natural-language research query.
        report:          The completed ReportOutput from the agent's writer node.
        timeout_seconds: Hard timeout for the evaluation LLM call.

    Returns:
        EvalScores with per-dimension ratings and a normalized_average property.

    Raises:
        asyncio.TimeoutError: If the evaluation LLM does not respond in time.
        Exception:            Any LLM call errors are propagated to the caller.
    """
    llm = _get_eval_llm()

    report_text = _build_report_text(report)
    # Guard against enormous reports that would blow past the eval model's context
    if len(report_text) > 12_000:
        report_text = report_text[:12_000] + "\n\n[TRUNCATED FOR EVALUATION]"

    prompt = _EVAL_RUBRIC.format(
        query=query,
        report=report_text,
        sources=_build_source_list(report),
    )

    result = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout_seconds)
    return result  # type: ignore[return-value]


# ──────────────────────── Optional DeepEval Integration ──────────────────────


async def evaluate_with_deepeval(
    query: str,
    report: ReportOutput,
) -> dict[str, Any] | None:
    """
    Optional DeepEval-based evaluation (RAGAs-style metrics).

    Returns None if deepeval is not configured or an error occurs.
    The benchmark runner falls back to evaluate_report() in that case.
    """
    try:
        from deepeval import evaluate  # noqa: PLC0415
        from deepeval.metrics import (  # noqa: PLC0415
            AnswerRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase  # noqa: PLC0415
    except ImportError:
        logger.debug("deepeval not available — skipping DeepEval evaluation")
        return None

    try:
        report_text = _build_report_text(report)
        sources = [src.exact_snippet for src in report.sources[:20]]

        test_case = LLMTestCase(
            input=query,
            actual_output=report_text,
            retrieval_context=sources,
        )

        metrics = [FaithfulnessMetric(threshold=0.7), AnswerRelevancyMetric(threshold=0.7)]

        # DeepEval's evaluate() is synchronous — run in thread pool
        result: Any = await asyncio.to_thread(evaluate, [test_case], metrics)  # type: ignore

        return {
            "deepeval_results": [
                {
                    "metric": m.__class__.__name__,
                    "score": getattr(m, "score", None),
                    "passed": getattr(m, "is_successful", lambda: None)(),
                }
                for m in metrics
            ],
            "raw": result,
        }
    except Exception as exc:
        logger.warning("DeepEval evaluation failed: %s", exc)
        return None
