# agent/state.py
import operator
from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class WebResult(BaseModel):
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0, le=1, default=0.5)
    fetched_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ArxivPaper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published_date: str
    url: str
    citation_count: int = 0
    trust_score: float = 0.0


class GitHubRepo(BaseModel):
    name: str
    url: str
    description: str
    stars: int
    language: str | None
    last_updated: str
    trust_score: float = 0.0


class ResearchFindings(BaseModel):
    subquestion: str
    web_results: list[WebResult] = Field(default_factory=list)
    papers: list[ArxivPaper] = Field(default_factory=list)
    repos: list[GitHubRepo] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)


class CritiqueOutput(BaseModel):
    coverage_score: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)
    depth_score: float = Field(ge=0, le=1)
    source_diversity_score: float = Field(ge=0, le=1)
    missing_areas: list[str] = Field(default_factory=list)
    should_continue: bool
    reasoning: str


class Citation(BaseModel):
    source_url: str
    title: str
    exact_snippet: str
    source_type: Literal["web", "arxiv", "github"]
    trust_score: float


class Finding(BaseModel):
    claim: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]


# Phase 2 — structured contradiction tracking
class ContradictionRecord(BaseModel):
    claim_a: str
    claim_b: str
    resolution: str
    preferred_source: Literal["gpt4o", "claude", "unresolved"]


class ReportOutput(BaseModel):
    title: str
    executive_summary: str
    key_findings: list[Finding]
    emerging_trends: list[str]
    recommended_next_steps: list[str]
    model_disagreements: list[str] = Field(default_factory=list)
    contradictions: list[ContradictionRecord] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
    version: int = 1


class RunMetadata(BaseModel):
    run_id: str
    profile: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    iteration_count: int = 0


class ResearchState(TypedDict):
    # Input
    query: str
    profile: str
    run_id: str

    # Planning
    query_difficulty: str
    subquestions: list[str]
    approved_plan: bool

    # Research — parallel nodes safely append
    findings: Annotated[list[ResearchFindings], operator.add]

    # Evaluation
    critique: CritiqueOutput | None
    iteration_count: int

    # Output
    final_report: ReportOutput | None

    # Observability
    run_metadata: RunMetadata
    error_log: Annotated[list[str], operator.add]
    thought_log: Annotated[list[str], operator.add]
