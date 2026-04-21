"""
evaluation/run_benchmark.py

Benchmark runner for the DeepResearch Agent.

Runs standard queries through the full pipeline, evaluates with LLM-as-judge,
writes a Markdown results table, and exits 1 if quality drops below a threshold.

This enables the GitHub Actions eval.yml to gate merges on report quality.

Usage::

    # Run 3 queries with fast profile (CI default)
    uv run python evaluation/run_benchmark.py --queries 3

    # Full benchmark suite
    uv run python evaluation/run_benchmark.py --profile deep --fail-below 0.80

    # Evaluate a single query by ID
    uv run python evaluation/run_benchmark.py --query-id q01

Architecture note:
    The graph is compiled with interrupt_before=["planner"].  The benchmark
    auto-approves all plans without human review.  This is intentional — the
    benchmark measures output quality, not HITL UX.  Two ainvoke() calls are
    required per query: one runs the classifier (stopping before planner), the
    second resumes through the full pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import RunMetadata  # noqa: E402
from evaluation.evaluator import EvalScores, evaluate_report  # noqa: E402
from observability.tracer import EvalScoreRecord, get_tracer  # noqa: E402

BENCHMARK_PATH = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


# ──────────────────────────── Query Loading ───────────────────────────────────


def load_queries(
    limit: int | None = None,
    query_id: str | None = None,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = json.loads(BENCHMARK_PATH.read_text())
    if query_id:
        queries = [q for q in queries if q["id"] == query_id]
        if not queries:
            print(f"ERROR: Query ID '{query_id}' not found in benchmark_queries.json")
            sys.exit(1)
    return queries[:limit] if limit else queries


# ──────────────────────────── Result Writer ───────────────────────────────────


def write_results_markdown(results: list[dict[str, Any]], profile: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_{profile}_{timestamp}.md"

    lines = [
        f"# Benchmark Results — Profile: `{profile}`",
        f"\n*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}*\n",
        "| ID | Domain | Query | Faith. | Relev. | Coverage | Citation | Coh. | **Avg** | Latency | Cost | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        s = r.get("scores")
        query_short = r["query"][:45].rstrip() + "..."
        if s:
            avg = s["normalized_average"]
            status = "✅" if avg >= 0.70 else "⚠️"
            lines.append(
                f"| {r['id']} | {r['domain']} | {query_short} | "
                f"{s['faithfulness']:.1f} | {s['answer_relevancy']:.1f} | "
                f"{s['source_coverage']:.1f} | {s['citation_accuracy']:.1f} | "
                f"{s['coherence']:.1f} | **{avg:.2f}** | "
                f"{r['latency_seconds']}s | ${r['cost_usd'] or 0:.3f} | {status} |"
            )
        else:
            error_short = (r.get("error") or "unknown error")[:60]
            lines.append(
                f"| {r['id']} | {r['domain']} | {query_short} | "
                f"— | — | — | — | — | **ERR** | "
                f"{r['latency_seconds']}s | — | ❌ `{error_short}` |"
            )

    scored = [r for r in results if r.get("scores")]
    if scored:
        overall = sum(r["scores"]["normalized_average"] for r in scored) / len(scored)
        total_cost = sum(r["cost_usd"] or 0 for r in results)
        avg_latency = sum(r["latency_seconds"] or 0 for r in results) / len(results)
        lines += [
            "",
            "## Summary Statistics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Queries evaluated | {len(results)} ({len(scored)} successful) |",
            f"| Overall average score | **{overall:.3f}** / 1.00 |",
            f"| Total cost | ${total_cost:.4f} USD |",
            f"| Average latency | {avg_latency:.1f}s |",
            f"| Profile | `{profile}` |",
        ]

    if results:
        # Per-domain breakdown
        domains: dict[str, list[float]] = {}
        for r in scored:
            d = r.get("domain", "unknown")
            domains.setdefault(d, []).append(r["scores"]["normalized_average"])
        if domains:
            lines += [
                "",
                "## Domain Breakdown",
                "",
                "| Domain | Avg Score | Queries |",
                "|---|---|---|",
            ]
            for domain, scores_list in sorted(domains.items()):
                avg_d = sum(scores_list) / len(scores_list)
                lines.append(f"| {domain} | {avg_d:.3f} | {len(scores_list)} |")

    out_path.write_text("\n".join(lines))
    return out_path


# ──────────────────────────── Core Runner ────────────────────────────────────


async def run_single_query(
    query_obj: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    """
    Run the full agent pipeline on one benchmark query and evaluate the result.

    Handles the two-phase invocation required by the HITL planner interrupt:
      Phase 1: graph.ainvoke(initial_state) → classifier runs, stops before planner
      Phase 2: graph.ainvoke(None)         → resumes, planner through writer runs
    """
    # Import here to avoid circular imports at module load time
    from agent.graph import graph  # noqa: PLC0415

    run_id = str(uuid.uuid4())
    result: dict[str, Any] = {
        "id": query_obj["id"],
        "query": query_obj["query"],
        "domain": query_obj.get("domain", "unknown"),
        "run_id": run_id,
        "scores": None,
        "error": None,
        "latency_seconds": None,
        "cost_usd": None,
    }

    thread_config: dict[str, Any] = {"configurable": {"thread_id": run_id}}
    start = time.perf_counter()

    try:
        initial_state: dict[str, Any] = {
            "query": query_obj["query"],
            "profile": profile,
            "run_id": run_id,
            "query_difficulty": "",
            "subquestions": [],
            "approved_plan": False,
            "findings": [],
            "critique": None,
            "iteration_count": 0,
            "final_report": None,
            "run_metadata": RunMetadata(run_id=run_id, profile=profile),
            "error_log": [],
            "thought_log": [],
        }

        # Phase 1: run classifier, graph pauses before planner
        await graph.ainvoke(initial_state, config=thread_config)

        # Phase 2: auto-approve — resume through planner → supervisor → ... → writer
        final_state = await graph.ainvoke(None, config=thread_config)

        result["latency_seconds"] = round(time.perf_counter() - start, 1)

        # Extract metadata
        meta = final_state.get("run_metadata")
        if meta and hasattr(meta, "estimated_cost_usd"):
            result["cost_usd"] = round(meta.estimated_cost_usd, 4)

        report = final_state.get("final_report")
        if report is None:
            errors = final_state.get("error_log", [])
            result["error"] = (
                f"No report produced. Last error: {errors[-1][:200] if errors else 'none'}"
            )
            return result

        # Evaluate the report
        scores: EvalScores = await evaluate_report(query_obj["query"], report)
        result["scores"] = scores.to_dict()

        # Persist eval scores to the observability DB
        tracer = get_tracer()
        await tracer.log_eval_scores(
            EvalScoreRecord(
                run_id=run_id,
                faithfulness=scores.faithfulness,
                answer_relevancy=scores.answer_relevancy,
                source_coverage=scores.source_coverage,
                citation_accuracy=scores.citation_accuracy,
                coherence=scores.coherence,
                normalized_average=scores.normalized_average,
                overall_notes=scores.overall_notes,
            )
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        result["latency_seconds"] = round(time.perf_counter() - start, 1)

    return result


async def run_benchmark(
    profile: str = "fast",
    fail_below: float = 0.75,
    query_limit: int | None = None,
    query_id: str | None = None,
    concurrency: int = 1,
) -> bool:
    """
    Execute the full benchmark suite.

    Args:
        profile:      Research profile ("fast" or "deep").
        fail_below:   Minimum acceptable normalized_average. CI fails if below this.
        query_limit:  Cap on number of queries (cost control for CI).
        query_id:     Run a single query by ID (for debugging).
        concurrency:  Parallel queries (keep at 1 to avoid MCP rate limits).

    Returns:
        True if mean score ≥ fail_below, False otherwise.
    """
    queries = load_queries(limit=query_limit, query_id=query_id)

    print(f"\n{'═' * 65}")
    print("  DeepResearch Agent — Benchmark Evaluation")
    print(f"  Profile : {profile}  |  Queries : {len(queries)}  |  Threshold : {fail_below:.0%}")
    print(f"{'═' * 65}\n")

    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore(q: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            print(f"  ▶  [{q['id']}] {q['query'][:70]}...")
            r = await run_single_query(q, profile)
            if r["scores"]:
                avg = r["scores"]["normalized_average"]
                print(
                    f"  ✓  [{r['id']}] score={avg:.2f}  "
                    f"latency={r['latency_seconds']}s  "
                    f"cost=${r['cost_usd'] or 0:.3f}"
                )
            else:
                print(f"  ✗  [{r['id']}] ERROR: {r['error'][:80]}")
            return r

    results = list(await asyncio.gather(*(_run_with_semaphore(q) for q in queries)))

    # Write results file
    out_path = write_results_markdown(results, profile)
    print(f"\n  Results → {out_path}\n")

    # Compute verdict
    scored = [r for r in results if r.get("scores")]
    if not scored:
        print("  BENCHMARK FAILED — no queries produced evaluable results.\n")
        return False

    overall_avg = sum(r["scores"]["normalized_average"] for r in scored) / len(scored)
    passed = overall_avg >= fail_below

    print(f"{'═' * 65}")
    print(f"  Overall average : {overall_avg:.3f} / 1.00")
    print(f"  Threshold       : {fail_below:.3f}")
    print(f"  Result          : {'✅  PASSED' if passed else '❌  FAILED'}")
    print(f"{'═' * 65}\n")

    return passed


# ────────────────────────────── Entry Point ──────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run DeepResearch Agent benchmark evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--profile", default="fast", choices=["fast", "deep"], help="Research profile")
    p.add_argument(
        "--fail-below",
        type=float,
        default=0.75,
        metavar="THRESHOLD",
        help="Exit 1 if mean score < THRESHOLD (0-1)",
    )
    p.add_argument(
        "--queries",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N queries (cost control for CI)",
    )
    p.add_argument(
        "--query-id",
        type=str,
        default=None,
        metavar="ID",
        help="Run a single query by ID (e.g. q01)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel queries; keep at 1 to respect MCP rate limits",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    passed = asyncio.run(
        run_benchmark(
            profile=args.profile,
            fail_below=args.fail_below,
            query_limit=args.queries,
            query_id=args.query_id,
            concurrency=args.concurrency,
        )
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
