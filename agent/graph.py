# agent/graph.py
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

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

workflow.add_edge("web_agent", "supervisor")
workflow.add_edge("arxiv_agent", "supervisor")
workflow.add_edge("github_agent", "supervisor")

workflow.add_conditional_edges(
    "critic",
    critic.should_continue,
    {
        "continue": "supervisor",
        "synthesize": "synthesizer",
    },
)

workflow.add_edge("synthesizer", "writer")
workflow.add_edge("writer", END)

# In-memory checkpointer
memory = MemorySaver()

graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["planner"],
)
