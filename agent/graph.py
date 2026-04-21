# agent/graph.py
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent.budget_guard import check_budget
from agent.nodes import (
    arxiv_agent,
    classifier,
    critic,
    github_agent,
    planner,
    supervisor,
    synthesizer,
    web_agent,
    writer,
)
from agent.state import ResearchState

workflow = StateGraph(ResearchState)

workflow.add_node("classifier", classifier.run)
workflow.add_node("planner", planner.run)
workflow.add_node("supervisor", supervisor.run)
workflow.add_node("web_agent", web_agent.run)
workflow.add_node("arxiv_agent", arxiv_agent.run)
workflow.add_node("github_agent", github_agent.run)
workflow.add_node("critic", critic.run)
workflow.add_node("synthesizer", synthesizer.run)
workflow.add_node("writer", writer.run)

workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "planner")
workflow.add_edge("planner", "supervisor")

# Agent nodes feed back to critic after each individual call
# (Send-based fan-out means all agents run in parallel, then reconverge at critic)
workflow.add_edge("web_agent", "critic")
workflow.add_edge("arxiv_agent", "critic")
workflow.add_edge("github_agent", "critic")

# Budget guard wraps the critic's should_continue decision:
# - checks iteration count against settings.max_iterations
# - checks estimated_cost_usd against settings.max_cost_per_run_usd
# - if budget OK, delegates to critic.should_continue
workflow.add_conditional_edges(
    "critic",
    check_budget,
    {
        "continue": "supervisor",
        "synthesize": "synthesizer",
    },
)

workflow.add_edge("synthesizer", "writer")
workflow.add_edge("writer", END)

# SqliteSaver persists state across process restarts — required for HITL resume
conn = sqlite3.connect(".checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["planner"],
)
