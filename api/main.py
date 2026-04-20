# api/main.py
import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agent.graph import graph
from agent.state import RunMetadata
from utils.cost_estimator import estimate_cost

app = FastAPI(title="DeepResearch Agent API")


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


class ResearchRequest(BaseModel):
    query: str
    profile: str = "fast"


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool
    edited_subquestions: list[str] | None = None


@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    run_id = str(uuid.uuid4())
    thread_config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    async def event_generator():
        async for event in graph.astream_events(
            {
                "query": request.query,
                "profile": request.profile,
                "run_id": run_id,
                # Planning — empty until classifier + planner run
                "query_difficulty": "",
                "subquestions": [],
                "approved_plan": False,
                # Research
                "findings": [],
                # Evaluation
                "critique": None,
                "iteration_count": 0,
                # Output
                "final_report": None,
                # Observability
                "run_metadata": RunMetadata(run_id=run_id, profile=request.profile),
                "error_log": [],
                "thought_log": [],
            },
            version="v2",
            config=thread_config,
        ):
            kind = event["event"]
            if kind == "on_chain_start":
                yield sse({"type": "node_start", "node": event["name"]})
            elif kind == "on_tool_start":
                yield sse(
                    {
                        "type": "tool_call",
                        "tool": event["name"],
                        "input": str(event.get("data", {}).get("input", ""))[:200],
                    }
                )
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", [])
                yield sse(
                    {
                        "type": "tool_result",
                        "tool": event["name"],
                        "count": len(output) if isinstance(output, list) else 1,
                    }
                )
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield sse({"type": "token", "content": chunk.content})
            elif kind == "on_chain_end" and event["name"] == "writer":
                yield sse({"type": "complete", "run_id": run_id})

        # After classifier runs, graph is paused at planner interrupt
        state_snapshot = graph.get_state(thread_config)
        if state_snapshot and state_snapshot.next == ("planner",):
            values = state_snapshot.values
            difficulty = values.get("query_difficulty", "narrow")
            n_questions = {"narrow": 3, "broad": 6, "ambiguous": 4}.get(difficulty, 4)
            estimated_cost = estimate_cost("gpt-4o", n_questions * 800, n_questions * 400)
            yield sse(
                {
                    "type": "hitl_interrupt",
                    "thread_id": run_id,
                    "query_difficulty": difficulty,
                    "estimated_subquestions": n_questions,
                    "estimated_cost_usd": round(estimated_cost, 4),
                    "message": "Plan ready for approval. POST /research/approve to continue.",
                }
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/research/approve")
async def approve_plan(request: ApproveRequest):
    thread_config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}
    state_snapshot = graph.get_state(thread_config)

    if not state_snapshot:
        raise HTTPException(status_code=404, detail=f"Thread {request.thread_id} not found.")

    if state_snapshot.next != ("planner",):
        raise HTTPException(
            status_code=409,
            detail=f"Thread is not paused at planner. Current next: {state_snapshot.next}",
        )

    if not request.approved:
        return {"status": "rejected", "thread_id": request.thread_id}

    if request.edited_subquestions:
        graph.update_state(
            thread_config,
            {"subquestions": request.edited_subquestions, "approved_plan": True},
            as_node="planner",
        )

    async def resume_generator():
        async for event in graph.astream_events(
            None,
            version="v2",
            config=thread_config,
        ):
            kind = event["event"]
            if kind == "on_chain_start":
                yield sse({"type": "node_start", "node": event["name"]})
            elif kind == "on_tool_start":
                yield sse(
                    {
                        "type": "tool_call",
                        "tool": event["name"],
                        "input": str(event.get("data", {}).get("input", ""))[:200],
                    }
                )
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", [])
                yield sse(
                    {
                        "type": "tool_result",
                        "tool": event["name"],
                        "count": len(output) if isinstance(output, list) else 1,
                    }
                )
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield sse({"type": "token", "content": chunk.content})
            elif kind == "on_chain_end" and event["name"] == "writer":
                yield sse({"type": "complete", "run_id": request.thread_id})

    return StreamingResponse(resume_generator(), media_type="text/event-stream")


@app.get("/research/state/{thread_id}")
async def get_research_state(thread_id: str):
    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state_snapshot = graph.get_state(thread_config)
    if not state_snapshot:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return {
        "thread_id": thread_id,
        "next_node": state_snapshot.next,
        "query": state_snapshot.values.get("query"),
        "query_difficulty": state_snapshot.values.get("query_difficulty"),
        "subquestions": state_snapshot.values.get("subquestions", []),
        "findings_count": len(state_snapshot.values.get("findings", [])),
        "approved_plan": state_snapshot.values.get("approved_plan"),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "deep-research-agent-api"}
