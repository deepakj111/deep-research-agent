"""
agent/budget_guard.py

Budget enforcement for the research agent graph.

Provides a check function that can be used as a conditional edge in the
LangGraph graph. Prevents runaway cost by hard-stopping the research
loop when either the iteration count or estimated USD cost exceeds the
configured limits.

When a budget limit is exceeded, the graph routes directly to the
synthesizer with whatever findings have been collected so far. This
ensures the user always gets *some* output, even if the research was
cut short. The thought_log records what happened so the report can
note the truncation.

Integration point:
  - Used in graph.py as a conditional edge check after the critic node.
  - Reads limits from config/settings.py (max_iterations, max_cost_per_run_usd).
"""

from __future__ import annotations

import logging

from agent.state import ResearchState
from config.settings import settings

logger = logging.getLogger(__name__)


def check_budget(state: ResearchState) -> str:
    """
    Evaluate whether the current run has exceeded its budget.

    Returns:
        "budget_exceeded" — route to synthesizer immediately
        "continue"        — allow the critic's decision to stand
        "synthesize"      — critic says done, proceed normally

    This function is designed to wrap the critic's should_continue decision.
    It's called from the graph's conditional edge after the critic node.
    """
    from agent.nodes.critic import should_continue  # noqa: PLC0415

    meta = state.get("run_metadata")

    # Hard iteration limit — prevents infinite loops
    if meta and meta.iteration_count >= settings.max_iterations:
        logger.warning(
            "[BudgetGuard] Iteration limit reached (%d/%d). "
            "Forcing synthesis with current findings.",
            meta.iteration_count,
            settings.max_iterations,
        )
        return "synthesize"

    # Cost limit — prevents runaway API spend
    if meta and meta.estimated_cost_usd >= settings.max_cost_per_run_usd:
        logger.warning(
            "[BudgetGuard] Cost limit reached ($%.3f/$%.2f). "
            "Forcing synthesis with current findings.",
            meta.estimated_cost_usd,
            settings.max_cost_per_run_usd,
        )
        return "synthesize"

    # Budget OK — delegate to the critic's decision
    return should_continue(state)
